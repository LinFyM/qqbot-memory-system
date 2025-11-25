# -*- coding: utf-8 -*-
"""
回复处理模块
包含 generate_reply 等核心生成逻辑
"""
import logging
import threading
import torch
from typing import Dict, List, Any, Tuple, Optional

from chat.generate import custom_generate
from chat.prompting import build_system_prompt, extract_final_reply
from chat.history_manager import save_chat_history_to_storage
from utils.media_utils import is_image_url_valid
from transformers.generation.stopping_criteria import StoppingCriteriaList

_log = logging.getLogger(__name__)


class InterruptStoppingCriteria:
    """中断停止条件"""
    def __init__(self, interrupt_event):
        self.interrupt_event = interrupt_event
    
    def __call__(self, input_ids, scores, **kwargs):
        if self.interrupt_event and self.interrupt_event.is_set():
            return True
        return False


def truncate_history_by_tokens(
    processor,
    chat_history: List[Dict[str, Any]],
    system_prompt: str,
    chat_type: str,
    chat_id: str,
    config: Dict[str, Any],
    max_tokens: int = 32000,
    interrupt_event: threading.Event = None
) -> List[Dict[str, Any]]:
    """
    根据token数量截断聊天历史
    
    Args:
        processor: 处理器
        chat_history: 聊天历史
        system_prompt: 系统提示词
        chat_type: "group" 或 "private"
        chat_id: 群ID或用户ID
        config: 配置字典
        max_tokens: 最大token数
        interrupt_event: 中断事件
    
    Returns:
        截断后的历史
    """
    if chat_history is None:
        return []
    
    if interrupt_event and interrupt_event.is_set():
        return chat_history
    
    if processor is None:
        _log.warning("⚠️ 处理器未初始化，跳过截断")
        return chat_history
    
    # 构建完整消息列表
    full_messages = []
    if system_prompt:
        full_messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })
    full_messages.extend(chat_history)
    
    try:
        # Tokenize检查长度
        inputs = processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
            padding=False
        )
        
        current_length = inputs['input_ids'].shape[1]
        _log.info(f"📏 当前输入tokens长度: {current_length}, 限制: {max_tokens}（{chat_type} {chat_id}）")
        
        if current_length <= max_tokens:
            _log.info(f"✅ 历史长度在限制内，无需截断（{chat_type} {chat_id}）")
            return chat_history
        
        _log.warning(f"⚠️ 历史长度超出限制 ({current_length} > {max_tokens})，开始截断（{chat_type} {chat_id}）")
        
        removed_messages = []
        # 从头部移除消息直到满足长度要求
        while len(chat_history) > 1:
            removed_msg = chat_history.pop(0)  # 移除最旧的消息
            removed_messages.append(removed_msg)
            
            full_messages = []
            if system_prompt:
                full_messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            full_messages.extend(chat_history)
            
            inputs = processor.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False,
                padding=False
            )
            
            current_length = inputs['input_ids'].shape[1]
            
            if current_length <= max_tokens:
                _log.info(f"✅ 截断完成，当前长度：{current_length}")
                break
        
        # 保存被移除的消息
        if removed_messages:
            _log.info(f"💾 保存被截断的 {len(removed_messages)} 条消息到存储")
            try:
                # 异步保存，避免阻塞
                threading.Thread(
                    target=save_chat_history_to_storage,
                    args=(config, chat_type, chat_id, removed_messages),
                    daemon=True
                ).start()
            except Exception as e:
                _log.error(f"❌ 保存截断消息失败: {e}")
        
        return chat_history
        
    except Exception as e:
        _log.warning(f"⚠️ 截断失败: {e}，返回原始历史")
        return chat_history


def generate_reply(
    model,
    processor,
    memory_db,
    recall_token_ids,
    config,
    chat_history: List[Dict[str, Any]],
    max_new_tokens: int = None,
    temperature: float = None,
    chat_type: str = None,
    chat_context: Dict[str, str] = None,
    interrupt_event: threading.Event = None,
    chat_id: str = None,
    response_dict: dict = None,
    log_full_io: bool = True,
    is_training: bool = False,
    training_lock: threading.Lock = None,
    model_lock: threading.Lock = None
) -> Tuple[Optional[str], bool, bool]:
    """
    使用模型生成回复
    
    Args:
        model: 模型实例
        processor: 处理器实例
        memory_db: 记忆数据库
        recall_token_ids: 特殊token IDs
        config: 配置字典
        chat_history: 聊天历史
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        chat_type: "group" 或 "private"
        chat_context: 对话上下文
        interrupt_event: 中断事件
        chat_id: 聊天ID
        response_dict: 响应字典
        log_full_io: 是否记录完整输入输出
        is_training: 是否处于训练模式
        training_lock: 训练锁
    
    Returns:
        (回复文本, 是否需要回复, 是否被中断)
    """
    # 检查训练模式
    if training_lock and is_training:
        _log.warning("⚠️ 当前处于训练模式，拒绝生成回复")
        raise RuntimeError("服务器正在训练中，暂时无法生成回复")
    
    if model is None or processor is None:
        raise RuntimeError("模型未初始化")
    
    # 从配置读取生成参数
    gen_config = config.get("generation", {})
    if max_new_tokens is None:
        max_new_tokens = gen_config.get("max_new_tokens", 1000)
    if temperature is None:
        temperature = gen_config.get("temperature", 1.0)
    
    do_sample = gen_config.get("do_sample", True)
    top_p = gen_config.get("top_p", 0.95)
    top_k = gen_config.get("top_k", 20)
    
    try:
        # 构建系统提示词
        system_prompt = build_system_prompt(config, chat_type, chat_context)
        _log.debug(f"📝 系统提示词长度: {len(system_prompt)}")
        
        # 构建完整消息列表
        full_messages = []
        if system_prompt:
            full_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        full_messages.extend(chat_history)
        
        _log.debug(f"准备推理输入，系统提示词长度: {len(system_prompt)}, 历史消息数: {len(chat_history)}")
        
        # 检查中断
        if interrupt_event and interrupt_event.is_set():
            _log.warning("⚠️ 在apply_chat_template前检测到中断")
            return None, False, True
        
        # 准备输入
        _log.debug("开始apply_chat_template...")
        try:
            inputs = processor.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False,
                padding=False
            )
        except Exception as e:
            # 处理图片相关错误：逐个检查图片有效性
            error_msg = str(e)
            _log.warning(f"⚠️ apply_chat_template失败: {e}，尝试修复图片链接")
            
            # 检查是否是图片错误
            image_errors = ["UnidentifiedImageError", "cannot identify image file", "ConnectionError", "Timeout", "Failed to resolve"]
            if any(err in error_msg for err in image_errors):
                # 逐个检查并移除失效图片
                cleaned_messages = []
                for msg in full_messages:
                    if isinstance(msg.get("content"), list):
                        cleaned_content = []
                        for item in msg["content"]:
                            if item.get("type") == "image":
                                img_url = item.get("image", "")
                                if img_url.startswith("http") and not is_image_url_valid(img_url):
                                    _log.warning(f"⚠️ 移除失效图片: {img_url}")
                                    continue
                                cleaned_content.append(item)
                            else:
                                cleaned_content.append(item)
                        if cleaned_content:
                            msg["content"] = cleaned_content
                            cleaned_messages.append(msg)
                    else:
                        cleaned_messages.append(msg)
                
                # 重试
                inputs = processor.apply_chat_template(
                    cleaned_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    truncation=False,
                    padding=False
                )
            else:
                raise e
        
        # 移到设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        _log.info(f"✅ apply_chat_template成功，输入tokens长度: {inputs['input_ids'].shape[1]}")
        
        # 打印完整的输入（包括特殊token）
        if log_full_io:
            input_ids_text = processor.tokenizer.batch_decode(
                inputs['input_ids'],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            _log.info("=" * 80)
            _log.info("🔤 模型完整输入（包括特殊token）：")
            _log.info(input_ids_text[0])
            _log.info("=" * 80)
        
        # 检查中断
        if interrupt_event and interrupt_event.is_set():
            _log.warning("⚠️ 在打印输入后检测到中断")
            return None, False, True
        
        # 配置停止条件
        stopping_criteria_list = StoppingCriteriaList()
        if interrupt_event:
            stopping_criteria_list.append(InterruptStoppingCriteria(interrupt_event))
        
        _log.info(f"🎯 开始自回归生成，max_new_tokens={max_new_tokens}, temperature={temperature}, do_sample={do_sample}")
        _log.info("开始生成回复...")
        
        # 使用模型锁确保串行推理，并使用torch.no_grad()节省显存
        if model_lock is None:
            import api.server_state as server_state
            model_lock = server_state.model_lock
        
        with model_lock:
            # 在获取锁后再次检查中断
            if interrupt_event and interrupt_event.is_set():
                _log.warning("⚠️ 在获取模型锁后检测到中断")
                return None, False, True
            
            with torch.no_grad():  # 关键：禁用梯度计算，大幅节省显存！
                try:
                    output_ids = custom_generate(
                        model=model,
                        processor=processor,
                        memory_db=memory_db,
                        recall_token_ids=recall_token_ids,
                        config=config,
                        inputs=inputs,
                        max_new_tokens=max_new_tokens,
                        stopping_criteria=stopping_criteria_list,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        interrupt_event=interrupt_event,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    _log.error(f"❌ CUDA显存不足: {e}")
                    # 清理显存
                    torch.cuda.empty_cache()
                    raise
        
        # 处理输出
        if interrupt_event and interrupt_event.is_set():
            _log.warning("⚠️ 生成过程中检测到中断，丢弃未完成的输出")
            return None, False, True

        if isinstance(output_ids, tuple):
            generated_ids = output_ids[0]
        else:
            generated_ids = output_ids
        
        # 提取生成的token（去掉输入部分）
        input_length = inputs['input_ids'].shape[1]
        generated_ids_trimmed = generated_ids[:, input_length:]
        
        _log.info(f"📊 生成完成，输入长度: {input_length}, 输出长度: {generated_ids_trimmed.shape[1]}")
        
        # 解码生成结果（包含特殊token，用于日志）
        output_text_with_special = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )[0]
        
        if log_full_io:
            _log.info("=" * 80)
            _log.info("🔤 模型完整输出（包括特殊token）：")
            _log.info(output_text_with_special)
            _log.info("=" * 80)
        
        # 解码生成结果（跳过特殊token，用于实际回复）
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # 提取最终回复
        final_reply, should_reply, actions = extract_final_reply(output_text)
        
        if log_full_io:
            _log.info("=" * 80)
            _log.info(f"✅ 最终回复: {final_reply}")
            _log.info(f"📌 是否需要回复: {should_reply}")
            if actions:
                _log.info(f"🎬 动作: {actions}")
            _log.info("=" * 80)
        
        return final_reply, should_reply, False
        
    except Exception as e:
        _log.error(f"❌ 生成回复失败: {e}", exc_info=True)
        raise

