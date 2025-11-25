#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试<recall> token的自动回忆机制
在输入中强行放入<recall> token，测试模型能否自动进行记忆向量查找和插入
"""

import os
import sys
from pathlib import Path
import yaml
import logging

# 确保项目根目录在sys.path中
project_root = Path(__file__).resolve().parents[1]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ⚠️ 关键：在导入torch之前设置CUDA_VISIBLE_DEVICES
# 先加载配置，检查是否需要设置CUDA_VISIBLE_DEVICES
try:
    config_path = project_root / "configs" / "config_qwen3vl.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            early_config = yaml.safe_load(f)
        device_config = early_config.get("model", {}).get("device", "cuda:0")
        if isinstance(device_config, list):
            # 多GPU配置，提取GPU索引并设置CUDA_VISIBLE_DEVICES
            gpu_indices = []
            for device in device_config:
                if device.startswith("cuda:"):
                    try:
                        gpu_idx = int(device.split(":")[1])
                        gpu_indices.append(str(gpu_idx))
                    except (ValueError, IndexError):
                        pass
            if gpu_indices:
                cuda_visible_devices = ",".join(gpu_indices)
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                print(f"🔧 在导入torch之前设置CUDA_VISIBLE_DEVICES={cuda_visible_devices}（对应实际GPU {device_config}）")
        elif isinstance(device_config, str) and device_config.startswith("cuda:"):
            # 单GPU配置，也需要设置CUDA_VISIBLE_DEVICES
            try:
                gpu_idx = int(device_config.split(":")[1])
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                print(f"🔧 在导入torch之前设置CUDA_VISIBLE_DEVICES={gpu_idx}（对应实际GPU {device_config}）")
            except (ValueError, IndexError):
                print(f"⚠️ 无法解析单GPU配置: {device_config}")
except Exception as e:
    print(f"⚠️ 预加载配置失败，将在模型初始化时设置CUDA_VISIBLE_DEVICES: {e}")

import torch
from transformers import StoppingCriteriaList, LogitsProcessorList, RepetitionPenaltyLogitsProcessor

# 导入新模块
import api.server_state as server_state
from chat.generate import custom_generate
from memory.vector_db import MemoryVectorDB

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_log = logging.getLogger(__name__)


def load_config():
    """加载配置文件"""
    config_path = project_root / "configs" / "config_qwen3vl.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model_and_processor(config):
    """加载模型和processor（使用server_state的方法）"""
    # 查找最新的训练模型
    training_model_dir = config.get("memory", {}).get("training", {}).get("training_model_dir", "models/trained")
    if not os.path.isabs(training_model_dir):
        training_model_dir = project_root / training_model_dir
    
    model_path = None
    if training_model_dir.exists():
        models = [d for d in os.listdir(training_model_dir) 
                 if (training_model_dir / d).is_dir() and d.startswith("model_")]
        if models:
            model_path = str(training_model_dir / sorted(models)[-1])
            _log.info(f"使用训练模型: {model_path}")
    
    if not model_path:
        # 使用token_added模型
        token_added_dir = config.get("memory", {}).get("training", {}).get("token_added_model_dir", "models/token_added")
        if not os.path.isabs(token_added_dir):
            token_added_dir = project_root / token_added_dir
        if token_added_dir.exists():
            models = [d for d in os.listdir(token_added_dir) 
                     if (token_added_dir / d).is_dir() and d.startswith("model_")]
            if models:
                model_path = str(token_added_dir / sorted(models)[-1])
                _log.info(f"使用token_added模型: {model_path}")
    
    if not model_path:
        # 使用基础模型
        model_path = config.get("model", {}).get("base_model_path", "models/Qwen3-VL-4B-Thinking")
        if not os.path.isabs(model_path):
            model_path = str(project_root / model_path)
        _log.info(f"使用基础模型: {model_path}")
    
    # 获取设备配置
    device = config.get("model", {}).get("device", "cuda:0")
    
    # 使用server_state的方法加载模型
    server_state.load_config()
    server_state.initialize_model(model_path, device)
    
    return server_state.model, server_state.processor


def load_memory_db(config, model, device):
    """加载记忆向量库（使用MemoryVectorDB类）"""
    memory_db_path = config.get("memory", {}).get("memory_db", {}).get("embeddings_path", "models/memory_db/memory_embeddings.pt")
    if not os.path.isabs(memory_db_path):
        memory_db_path = project_root / memory_db_path
    
    # 获取embedding维度（从模型配置中）
    embedding_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
    
    # 创建MemoryVectorDB实例
    memory_db = MemoryVectorDB(embedding_dim=embedding_dim, device=device)
    
    if os.path.exists(memory_db_path):
        # 使用MemoryVectorDB的load_from_pt方法加载数据
        memory_db.load_from_pt(str(memory_db_path))
        _log.info(f"加载记忆向量库: {len(memory_db)} 条记忆")
        if len(memory_db) > 0:
            # 获取第一条记忆的文本预览
            try:
                first_memory = memory_db.get(0)
                if first_memory and 'text' in first_memory:
                    _log.info(f"示例记忆文本: {first_memory['text'][:100]}...")
                else:
                    _log.info(f"示例记忆文本: N/A...")
            except:
                _log.info(f"示例记忆文本: N/A...")
        return memory_db
    else:
        _log.warning(f"记忆向量库不存在: {memory_db_path}")
        return memory_db


def test_recall_mechanism_with_custom_generate(model, processor, memory_db, config):
    """使用custom_generate函数测试<recall> token的自动回忆机制"""
    _log.info("=" * 80)
    _log.info("开始测试<recall> token的自动回忆机制（使用custom_generate）")
    _log.info("=" * 80)
    
    # 获取recall token ID（从server_state）
    recall_token_ids = server_state.recall_token_ids
    if not recall_token_ids:
        # 如果server_state中没有，从tokenizer获取
        recall_token_ids = {
            "<recall>": processor.tokenizer.convert_tokens_to_ids("<recall>"),
            "</recall>": processor.tokenizer.convert_tokens_to_ids("</recall>"),
            "<|memory_pad|>": processor.tokenizer.convert_tokens_to_ids("<|memory_pad|>")
        }
    
    recall_token_id = recall_token_ids.get("<recall>")
    recall_end_token_id = recall_token_ids.get("</recall>")
    memory_pad_token_id = recall_token_ids.get("<|memory_pad|>")

    _log.info(f"<|memory_pad|> token ID: {memory_pad_token_id}")
    _log.info(f"<recall> token ID: {recall_token_id}")
    _log.info(f"</recall> token ID: {recall_end_token_id}")
    
    if recall_token_id is None or recall_token_id == processor.tokenizer.unk_token_id:
        _log.error("❌ <recall> token不存在于tokenizer中！")
        return False
    
    # 直接使用tokenizer编码文本，不使用chat template（与API服务一致）
    # 构建测试文本：在文本末尾添加<recall> token
    test_text = "让我回忆一下用户的生日。<recall>"
    
    _log.info(f"\n测试输入文本: {test_text}")
    
    # 直接使用tokenizer编码（不使用chat template）
    encoded = processor.tokenizer(
        test_text,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False
    )
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    
    # 检查输入中是否包含<recall> token
    recall_positions = (input_ids == recall_token_id).nonzero(as_tuple=True)[1]
    if len(recall_positions) == 0:
        _log.error("❌ 输入中未找到<recall> token！")
        decoded_input = processor.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        _log.info(f"解码后的输入: {decoded_input[:400]}...")
        return False
    
    _log.info(f"✅ 输入中找到 {len(recall_positions)} 个<recall> token，位置: {recall_positions.tolist()}")
    _log.info(f"✅ <recall> token在输入末尾，模型将从<recall>之后开始生成")
    
    # 构建inputs字典（与API服务完全一致）
    device = next(model.parameters()).device
    inputs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device)
    }
    
    # 准备生成参数（测试专用参数）
    gen_config = config.get("generation", {})
    # 测试程序强制限制为500个token，避免运行时间过长
    max_new_tokens = 500
    temperature = gen_config.get("temperature", 1.0)
    top_p = gen_config.get("top_p", 0.95)
    top_k = gen_config.get("top_k", 20)
    do_sample = gen_config.get("do_sample", True)
    repetition_penalty = gen_config.get("repetition_penalty", 1.0)
    
    _log.info(f"\n生成参数: max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}, do_sample={do_sample}")
    _log.info("开始调用custom_generate进行生成...")
    
    # 准备LogitsProcessor（与API服务完全一致）
    logits_processor = LogitsProcessorList()
    if repetition_penalty != 1.0:
        logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    
    # 准备StoppingCriteria（与API服务完全一致）
    stopping_criteria = StoppingCriteriaList()
    
    # 调用custom_generate函数（使用新的函数签名）
    try:
        with torch.no_grad():
            result = custom_generate(
                model=model,
                processor=processor,
                memory_db=memory_db,
                recall_token_ids=recall_token_ids,
                config=config,
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                temperature=temperature,
                top_k=top_k if top_k and top_k > 0 else None,
                top_p=top_p if top_p and top_p < 1.0 else None,
                do_sample=do_sample,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                interrupt_event=None,
                early_stop_on_tool_call=False,
            )
            # 处理返回值：可能是 (input_ids, memory_injection_positions) 或 input_ids
            if isinstance(result, tuple):
                generated_ids, memory_injection_positions = result
            else:
                generated_ids = result
                memory_injection_positions = []
        
        # 解码生成结果
        generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        
        # 注意：记忆向量插入位置现在通过<|memory_pad|> token原生显示，无需额外标注
        _log.info("\n" + "=" * 80)
        _log.info("生成结果（包含特殊token，<|memory_pad|>标记记忆向量插入位置）:")
        _log.info("=" * 80)
        _log.info(generated_text)
        _log.info("=" * 80)
        
        # 检查是否包含<|memory_pad|> token
        if "<|memory_pad|>" in generated_text:
            _log.info("✅ 生成结果中包含<|memory_pad|> token（记忆向量插入位置）")
            # 统计<|memory_pad|>出现的次数
            count = generated_text.count("<|memory_pad|>")
            _log.info(f"📊 <|memory_pad|> token出现次数: {count}")
        else:
            _log.warning("⚠️ 生成结果中不包含<|memory_pad|> token")

        # 检查输入文本中是否包含<|memory_pad|>
        input_text = processor.tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
        if "<|memory_pad|>" in input_text:
            _log.info("✅ 输入文本中包含<|memory_pad|> token")
        else:
            _log.info("ℹ️ 输入文本中不包含<|memory_pad|> token（正常，因为在生成过程中插入）")
        
        # 检查生成结果中是否包含<recall>或</recall>
        if "<recall>" in generated_text:
            _log.info("✅ 生成结果中包含<recall> token")
        else:
            _log.warning("⚠️ 生成结果中不包含<recall> token")
        
        if "</recall>" in generated_text:
            _log.info("✅ 生成结果中包含</recall> token")
        else:
            _log.warning("⚠️ 生成结果中不包含</recall> token")
        
        # 检查是否触发了回忆机制（生成了<recall>后的内容）
        recall_start_idx = generated_text.find("<recall>")
        recall_end_idx = generated_text.find("</recall>")
        
        if recall_start_idx != -1:
            if recall_end_idx != -1 and recall_end_idx > recall_start_idx:
                recall_content = generated_text[recall_start_idx + len("<recall>"):recall_end_idx]
                _log.info(f"\n回忆内容: {recall_content[:400]}...")
                _log.info("✅ 检测到完整的回忆过程（<recall>...</recall>）")
            else:
                _log.warning("⚠️ 检测到<recall>但未找到</recall>")
        
        return True
        
    except Exception as e:
        _log.error(f"❌ 生成过程中出错: {e}", exc_info=True)
        return False


def main():
    """主函数"""
    try:
        config = load_config()
        model, processor = load_model_and_processor(config)
        
        # 获取device
        device = next(model.parameters()).device
        memory_db = load_memory_db(config, model, device)
        
        success = test_recall_mechanism_with_custom_generate(model, processor, memory_db, config)
        
        if success:
            _log.info("\n✅ 测试完成")
        else:
            _log.error("\n❌ 测试失败")
            sys.exit(1)
            
    except Exception as e:
        _log.error(f"测试过程中出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

