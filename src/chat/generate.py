# -*- coding: utf-8 -*-
"""
自定义生成模块
实现自回归生成 + 记忆机制
"""
import logging
import threading
import torch
from typing import Optional
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from training.model_utils import forward_backbone, ensure_last_hidden_state, build_causal_lm_output

_log = logging.getLogger(__name__)


def memory_head(query_vector, memory_db, debug=False):
    """
    Memory Head: 从记忆库中检索记忆向量，输出logits（相似度分数）
    
    设计理念：将记忆检索视为特殊的"head"，完全类似于lm_head用于生成token。
    - lm_head: hidden_state -> logits (vocab_size) - 对所有vocab计算logits
    - memory_head: query_vector -> logits (memory_candidates) - 对所有记忆向量计算相似度
    
    与lm_head完全一致：
    - 都计算所有候选的分数（vocab_size 或 所有记忆向量）
    - 都只输出logits，不进行softmax和采样
    - softmax和采样将在生成流程中统一处理
    - top-k截断在logits_warper中进行，与token生成完全一致
    
    Args:
        query_vector: 查询向量 [hidden_dim]，来自<recall>位置的last_hidden_state
        memory_db: 记忆向量数据库
        debug: 是否输出调试信息
    
    Returns:
        memory_logits: 记忆向量的logits（相似度分数）[num_candidates]，如果未找到则返回None
        memory_candidates: 候选记忆向量列表，每个元素包含 {'embedding': tensor, 'score': float, 'index': int}
        如果未找到则返回 (None, None)
    """
    if memory_db is None or len(memory_db) == 0:
        _log.info("🔍 [Memory Head] 记忆向量库为空，无法进行匹配")
        return None, None

    _log.info(f"🔍 [Memory Head] 开始搜索记忆库，查询向量shape: {query_vector.shape}, 记忆库大小: {len(memory_db)}")
    # 检索所有记忆向量（与lm_head对所有vocab计算logits一致）
    # memory_db.search内部会计算所有向量的相似度，然后返回所有结果
    # top-k截断将在logits_warper中进行
    search_results = memory_db.search(
        query_vector.detach().clone(),
        top_k=len(memory_db),  # 检索所有向量，与lm_head对所有vocab计算logits一致
        debug=debug
    )
    if not search_results:
        _log.info("🔍 [Memory Head] 未找到匹配的记忆向量")
        return None, None

    _log.info(f"🔍 [Memory Head] 找到 {len(search_results)} 个候选记忆向量")
    for i, result in enumerate(search_results):
        score = result.get('score', 0.0)
        _log.info(f"  [{i+1}] 相似度={score:.4f}")

    # 提取logits（相似度分数），与lm_head输出logits完全一致
    memory_logits = torch.tensor(
        [item['score'] for item in search_results],
        dtype=torch.float32,
        device=query_vector.device
    )
    
    _log.debug(f"🔍 [Memory Head] 输出logits shape: {memory_logits.shape}, 范围: [{memory_logits.min():.4f}, {memory_logits.max():.4f}]")
    return memory_logits, search_results


def memory_embedding(memory_vector, model, device=None, dtype=None):
    """
    Memory Embedding: 准备记忆向量用于注入，跳过embedding层
    
    设计理念：将记忆向量视为特殊的embedding，类似于input_embeddings用于token ID，
    但这里直接使用记忆向量，不经过embedding层计算。记忆向量是已经计算好的hidden state，
    直接作为下一个位置的输入。
    
    Args:
        memory_vector: 记忆向量 [hidden_dim] 或 [1, hidden_dim] 或 [1, 1, hidden_dim]
        model: 模型实例（用于获取设备和数据类型）
        device: 目标设备（如果为None，则从model获取）
        dtype: 目标数据类型（如果为None，则从model获取）
    
    Returns:
        memory_embedding: 准备好的记忆向量 [1, 1, hidden_dim]，可直接作为inputs_embeds使用
    """
    if memory_vector is None:
        return None
    
    # 调整形状为 [1, 1, hidden_dim]
    if memory_vector.dim() == 1:
        memory_vector = memory_vector.unsqueeze(0)  # [1, hidden_dim]
    if memory_vector.dim() == 2:
        memory_vector = memory_vector.unsqueeze(0)  # [1, 1, hidden_dim]
    
    # 获取设备和数据类型
    if device is None or dtype is None:
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        if device is None:
            device = model_device
        if dtype is None:
            dtype = model_dtype
    
    # 移动到目标设备和数据类型
    memory_embedding = memory_vector.to(device=device, dtype=dtype)
    
    _log.debug(f"🔧 [Memory Embedding] 准备记忆向量，shape: {memory_embedding.shape}, device: {device}, dtype: {dtype}")
    return memory_embedding


def custom_generate(
    model,
    processor,
    memory_db,
    recall_token_ids,
    config,
    inputs,
    max_new_tokens: int = 1000,
    stopping_criteria: StoppingCriteriaList = None,
    logits_processor: LogitsProcessorList = None,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    do_sample: bool = True,
    pad_token_id: int = None,
    eos_token_id: int = None,
    interrupt_event: threading.Event = None,
    early_stop_on_tool_call: bool = False,
):
    """
    自定义生成函数：保持官方generate流程，但将记忆检索视为特殊head，将记忆向量视为特殊嵌入
    
    Args:
        model: 模型实例
        processor: 处理器实例
        memory_db: 记忆向量数据库
        recall_token_ids: 特殊token ID映射
        config: 配置字典
        inputs: 输入字典
        max_new_tokens: 最大生成token数
        stopping_criteria: 停止条件列表
        logits_processor: logits处理器列表
        temperature: 温度参数
        top_k: top-k采样参数
        top_p: top-p采样参数
        do_sample: 是否使用采样
        pad_token_id: padding token ID
        eos_token_id: EOS token ID
        interrupt_event: 中断事件
        early_stop_on_tool_call: 是否在工具调用时提前停止
    
    Returns:
        生成的token IDs（如果有记忆注入，还会返回注入位置信息）
    """
    input_ids = inputs.get('input_ids')
    attention_mask = inputs.get('attention_mask', None)

    batch_size = input_ids.shape[0]
    cur_len = input_ids.shape[-1]
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    if stopping_criteria is None:
        stopping_criteria = StoppingCriteriaList()
    if logits_processor is None:
        logits_processor = LogitsProcessorList()

    # 配置logits warper
    logits_warper = None
    if do_sample:
        logits_warper_list = []
        if temperature is not None and temperature != 1.0:
            logits_warper_list.append(TemperatureLogitsWarper(temperature=temperature))
        if top_k is not None and top_k > 0:
            logits_warper_list.append(TopKLogitsWarper(top_k=top_k))
        if top_p is not None and top_p < 1.0:
            logits_warper_list.append(TopPLogitsWarper(top_p=top_p))
        if logits_warper_list:
            logits_warper = LogitsProcessorList(logits_warper_list)

    # 准备model_kwargs
    model_kwargs = {k: v for k, v in inputs.items() if k not in ['input_ids', 'attention_mask']}
    if attention_mask is not None:
        model_kwargs['attention_mask'] = attention_mask
    if 'use_cache' not in model_kwargs:
        model_kwargs['use_cache'] = True

    # 配置cache_position
    if not model_kwargs.get("use_cache", True):
        model_kwargs["cache_position"] = None
    else:
        past_length = 0
        if "past_key_values" in model_kwargs and model_kwargs["past_key_values"] is not None:
            try:
                from transformers.cache_utils import Cache
                if isinstance(model_kwargs["past_key_values"], Cache):
                    past_length = model_kwargs["past_key_values"].get_seq_length()
                else:
                    past_length = model_kwargs["past_key_values"][0][0].shape[2]
            except (ImportError, AttributeError):
                past_length = model_kwargs["past_key_values"][0][0].shape[2]
        if "inputs_embeds" in model_kwargs:
            input_seq_len = model_kwargs["inputs_embeds"].shape[1]
        else:
            input_seq_len = input_ids.shape[-1]
        model_kwargs["cache_position"] = torch.arange(past_length, input_seq_len, device=input_ids.device)

    # 处理EOS token
    if eos_token_id is not None:
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_ids = torch.tensor(list(eos_token_id), device=input_ids.device)
        else:
            eos_token_ids = torch.tensor([eos_token_id], device=input_ids.device)
    else:
        eos_token_ids = None
    has_eos_stopping_criteria = eos_token_ids is not None

    # 获取记忆相关的token ID
    recall_token_id = recall_token_ids.get("<recall>") if recall_token_ids else None
    memory_pad_token_id = recall_token_ids.get("<|memory_pad|>") if recall_token_ids else None

    # 记忆注入位置记录
    memory_injection_positions = []
    
    # 记忆配置
    memory_cfg = config.get("memory", {}).get("autoregressive_recall", {})
    autorecall_enabled = bool(memory_cfg.get("enabled", False))
    autorecall_top_k = max(1, int(memory_cfg.get("top_k", 5)))
    autorecall_temperature = float(memory_cfg.get("temperature", 1.0))
    autorecall_top_p = float(memory_cfg.get("top_p", 1.0))
    autorecall_use_sampling = bool(memory_cfg.get("use_sampling", True))
    autorecall_debug = bool(memory_cfg.get("debug", False))

    def _update_model_kwargs_helper(outputs_obj):
        """更新model_kwargs"""
        nonlocal model_kwargs
        try:
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs_obj,
                model_kwargs,
                is_encoder_decoder=False,
                standardize_cache_format=True,
            )
        except TypeError:
            try:
                model_kwargs = model._update_model_kwargs_for_generation(
                    outputs_obj,
                    model_kwargs,
                    is_encoder_decoder=False,
                )
            except TypeError:
                model_kwargs = model._update_model_kwargs_for_generation(
                    outputs_obj,
                    model_kwargs,
                )

    def _forward_with_last_hidden_state(forward_inputs):
        """执行前向传播并获取最后的隐藏状态"""
        local_inputs = dict(forward_inputs)
        use_cache_flag = local_inputs.pop("use_cache", True)
        output_hidden_flag = local_inputs.pop("output_hidden_states", False)

        backbone_outputs = forward_backbone(
            model,
            use_cache=use_cache_flag,
            output_hidden_states=output_hidden_flag,
            return_dict=True,
            **local_inputs,
        )
        outputs = build_causal_lm_output(model, backbone_outputs)
        last_hidden_state = ensure_last_hidden_state(backbone_outputs)
        outputs.last_hidden_state = last_hidden_state
        return outputs


    # 生成循环
    override_next_embed = None  # 用于覆盖下一步的embedding（记忆向量）
    forced_next_token_id = None

    while cur_len < max_new_tokens:
        # 检查中断
        if interrupt_event and interrupt_event.is_set():
            break

        # 准备输入
        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            **model_kwargs
        )

        # 检查是否触发回忆
        current_input_ids = model_inputs.get('input_ids', input_ids)
        recall_triggered = False
        if current_input_ids.shape[-1] > 0:
            last_token_id = current_input_ids[0, -1].item()
            if (
                autorecall_enabled
                and recall_token_id is not None
                and last_token_id == recall_token_id
            ):
                recall_triggered = True

        # 前向传播
        forward_inputs = dict(model_inputs)
        forward_inputs.setdefault("use_cache", model_kwargs.get("use_cache", True))
        if override_next_embed is not None:
            # 当提供inputs_embeds时，必须移除input_ids以避免框架报错
            forward_inputs.pop("input_ids", None)
            forward_inputs["inputs_embeds"] = override_next_embed
        outputs = _forward_with_last_hidden_state(forward_inputs)
        last_hidden_state = outputs.last_hidden_state
        override_next_embed = None
        forced_next_token_id = None

        # 处理回忆触发
        memory_logits = None
        memory_candidates = None
        if recall_triggered:
            _log.info("🔄 [回忆触发] 检测到<recall> token，准备检索记忆向量")
            if last_hidden_state is None:
                _log.warning("⚠️ [回忆触发] 无法获取<recall>隐藏向量，继续普通生成")
            elif memory_db is None or len(memory_db) == 0:
                _log.info("ℹ️ [回忆触发] 记忆向量库为空，<recall> 按普通token处理")
            else:
                # Memory Head: 从记忆库中检索，输出logits（与lm_head完全一致）
                query_vector = last_hidden_state[0, -1, :]
                # 检索所有记忆向量（与lm_head对所有vocab计算logits一致）
                # top-k截断将在logits_warper中进行
                memory_logits, memory_candidates = memory_head(
                    query_vector=query_vector,
                    memory_db=memory_db,
                    debug=autorecall_debug
                )
                
                if memory_logits is None or memory_candidates is None:
                    _log.info("ℹ️ [回忆触发] 未找到可用记忆，<recall> 按普通token处理")
                else:
                    _log.info(f"🎯 [回忆触发] Memory Head输出logits，候选数: {len(memory_candidates)}")

        # 处理记忆检索（与token生成完全统一的流程）
        if memory_logits is not None and memory_candidates is not None:
            # 1. 应用logits processor（如果需要，可以对记忆logits应用相同的处理）
            # 注意：这里暂时不应用logits_processor，因为它是为token设计的
            # 如果需要，可以创建专门的memory_logits_processor
            memory_scores = memory_logits
            
            # 2. 应用logits warper（温度、top-k、top-p）- 与token生成完全一致
            if autorecall_use_sampling:
                # 创建记忆专用的logits warper（使用记忆配置的温度、top-k、top-p等）
                memory_warper_list = []
                if autorecall_temperature is not None and autorecall_temperature != 1.0:
                    memory_warper_list.append(TemperatureLogitsWarper(temperature=autorecall_temperature))
                if autorecall_top_k is not None and autorecall_top_k > 0:
                    memory_warper_list.append(TopKLogitsWarper(top_k=autorecall_top_k))
                if autorecall_top_p is not None and autorecall_top_p < 1.0:
                    memory_warper_list.append(TopPLogitsWarper(top_p=autorecall_top_p))
                if memory_warper_list:
                    memory_warper = LogitsProcessorList(memory_warper_list)
                    # 注意：logits_warper需要input_ids，这里传入dummy input_ids
                    dummy_input_ids = torch.zeros((1, 1), dtype=torch.long, device=memory_scores.device)
                    memory_scores = memory_warper(dummy_input_ids, memory_scores.unsqueeze(0)).squeeze(0)
                    _log.debug(f"🔍 [Memory Head] 应用logits_warper后，候选数: {memory_scores.shape[0]}")
            
            # 3. 采样或贪婪选择（与token生成完全一致）
            if autorecall_use_sampling:
                probs = torch.nn.functional.softmax(memory_scores, dim=-1)
                choice_idx = torch.multinomial(probs, num_samples=1).item()
                _log.info(f"🔍 [Memory Head] 使用采样方式选择记忆，选择索引: {choice_idx}, 概率: {probs[choice_idx]:.4f}")
            else:
                choice_idx = torch.argmax(memory_scores).item()
                _log.info(f"🔍 [Memory Head] 使用贪婪方式选择记忆，选择索引: {choice_idx}, 最高相似度: {memory_scores[choice_idx]:.4f}")
            
            # 4. 获取选中的记忆向量
            selected = memory_candidates[choice_idx]
            memory_vector = selected['embedding']
            memory_score = selected.get('score', 0.0)
            _log.info(f"✅ [Memory Head] 已选择记忆向量，相似度={memory_score:.4f}")
            
            # 5. 准备记忆注入
            if memory_pad_token_id is not None:
                forced_next_token_id = memory_pad_token_id
                injection_pos = input_ids.shape[-1]
                memory_injection_positions.append((injection_pos, memory_score))
                
                # Memory Embedding: 准备记忆向量，跳过embedding层
                override_next_embed = memory_embedding(
                    memory_vector=memory_vector,
                    model=model
                )
                
                # 更新attention mask
                if 'attention_mask' in model_kwargs and model_kwargs['attention_mask'] is not None:
                    model_kwargs['attention_mask'] = torch.cat(
                        [model_kwargs['attention_mask'], torch.ones((1, 1), device=model_kwargs['attention_mask'].device, dtype=model_kwargs['attention_mask'].dtype)],
                        dim=1
                    )
                _log.info("✅ [回忆触发] Memory Embedding已准备，将强制生成<|memory_pad|>并用记忆向量覆盖其embedding")
            else:
                _log.warning("⚠️ [回忆触发] 未找到<|memory_pad|> token，无法插入记忆向量，继续普通生成")
        
        # 获取下一个token（与记忆检索完全统一的流程）
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if do_sample and logits_warper is not None:
            next_token_scores = logits_warper(input_ids, next_token_scores)

        # 生成next token
        if forced_next_token_id is not None:
            next_tokens = torch.full(
                (batch_size,),
                forced_next_token_id,
                device=input_ids.device,
                dtype=input_ids.dtype
            )
        else:
            if do_sample:
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

        # 处理EOS
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # 添加新token
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # 更新model_kwargs
        _update_model_kwargs_helper(outputs)

        # 检查EOS
        if eos_token_ids is not None:
            eos_in_sentence = (next_tokens.unsqueeze(-1) == eos_token_ids.unsqueeze(0)).any(dim=-1)
            unfinished_sequences = unfinished_sequences & ~eos_in_sentence

        cur_len += 1

        # 检查停止条件
        should_stop = stopping_criteria(input_ids, next_token_scores)
        if isinstance(should_stop, bool):
            should_stop_tensor = torch.tensor([should_stop], device=unfinished_sequences.device, dtype=torch.bool)
            if batch_size > 1:
                should_stop_tensor = should_stop_tensor.expand(batch_size)
        else:
            should_stop_tensor = should_stop.bool() if should_stop.dtype != torch.bool else should_stop
        unfinished_sequences = unfinished_sequences & ~should_stop_tensor

        if unfinished_sequences.max() == 0:
            if interrupt_event and interrupt_event.is_set():
                _log.info("⚠️ 生成因中断而停止")
            else:
                _log.debug("生成因StoppingCriteria而停止（正常停止）")
            break

        # 提前停止（工具调用）
        if early_stop_on_tool_call:
            try:
                decoded_so_far = processor.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                open_idx = decoded_so_far.rfind("<tool_call")
                if open_idx != -1:
                    close_idx = decoded_so_far.rfind("</tool_call>")
                    if close_idx != -1 and close_idx > open_idx:
                        _log.info("🔧 检测到工具调用闭合标签，提前结束首轮生成")
                        break
            except Exception:
                pass

    # 返回结果
    if memory_injection_positions:
        return input_ids, memory_injection_positions
    return input_ids

