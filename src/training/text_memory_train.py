import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import shutil
import random
from tqdm import tqdm
import json
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
from peft import LoraConfig, get_peft_model, TaskType
from modelscope import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from memory.utils import inject_memory_embedding_to_inputs_embeds


def _ensure_prompt_list(prompts, prompt_name: str):
    if isinstance(prompts, list) and len(prompts) > 0:
        return prompts
    raise ValueError(f"{prompt_name}不能为空，请在prompts.yaml中配置")

def enhanced_collate_fn(batch):
    """简化版collate函数 - 支持新的记忆解码训练格式和SFT数据"""

    batch_size = len(batch)
    sample_types = [item.get('sample_type', 'unknown') for item in batch]

    # 检查是否有SFT样本
    has_sft = any(item.get('is_sft', False) for item in batch)
    has_memory = any(not item.get('is_sft', False) for item in batch)

    # 计算最大序列长度（输入+目标）
    max_input_len = max(len(item['sequence_tokens']) for item in batch)
    max_target_len = max(len(item['labels']) - len(item['sequence_tokens']) for item in batch)
    max_total_len = max_input_len + max_target_len

    # 初始化批次张量
    input_ids = torch.zeros(batch_size, max_total_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_total_len, dtype=torch.long)
    labels = torch.full((batch_size, max_total_len), -100, dtype=torch.long)

    # 记录embedding信息（仅用于记忆条目）
    embeddings_to_insert = []
    embedding_positions = []

    for i, item in enumerate(batch):
        input_tokens = item['sequence_tokens']
        item_labels = item['labels']
        input_len = len(input_tokens)
        is_sft = item.get('is_sft', False)

        if is_sft:
            # SFT样本：直接使用input_ids和labels，不需要embedding插入
            total_len = len(input_tokens)
            input_ids[i, :total_len] = input_tokens
            attention_mask[i, :total_len] = 1
            labels[i, :len(item_labels)] = item_labels
            
            # SFT样本使用占位符embedding，position设为-1表示不需要插入
            embeddings_to_insert.append(torch.zeros(1, 4096))  # 占位符
            embedding_positions.append(-1)  # -1表示SFT样本，不需要插入
        else:
            # 记忆条目样本：原有逻辑
            target_labels = item_labels[len(input_tokens):]
            total_tokens = torch.cat([input_tokens, target_labels])
            total_len = len(total_tokens)

            # 填充input_ids和attention_mask
            input_ids[i, :total_len] = total_tokens
            attention_mask[i, :total_len] = 1

            # 直接使用预先计算好的labels（item_labels已经是tensor）
            labels[i, :len(item_labels)] = item_labels

            # 处理embedding插入（位置需要调整，加上输入长度的偏移）
            embeddings_to_insert.append(item['embedding_to_insert'])
            embedding_positions.append(item['embedding_position'])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'embeddings_to_insert': torch.stack(embeddings_to_insert),
        'embedding_positions': torch.tensor(embedding_positions),
        'batch_info': {
            'batch_size': batch_size,
            'max_length': max_total_len,
            'has_sft': has_sft,
            'has_memory': has_memory,
            'sample_types': sample_types,
        }
    }

class EnhancedTextMemoryDataset(Dataset):
    """增强的文本记忆数据集 - 每个embedding对应其自己的记忆文本"""
    
    def _get_tokenizer(self):
        """获取实际的tokenizer对象（处理Qwen3VLProcessor的情况）"""
        if hasattr(self.tokenizer, 'tokenizer'):
            return self.tokenizer.tokenizer
        else:
            return self.tokenizer

    def __init__(
        self,
        texts,
        embeddings,
        tokenizer,
        base_model,
        max_length=3000,
        noise_std=0.01,
        is_main_process_fn=None,
        sft_full_texts=None,
        activation_prompts=None,
        end_prompts=None,
        guide_text=None,
    ):
        self.texts = texts
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.max_length = max_length
        self.noise_std = noise_std
        self._is_main_process_fn = is_main_process_fn
        # SFT完整文本列表，每个元素包含完整文本和思考部分的起止位置
        # 格式: [{"full_text": "...", "thinking_start": int, "thinking_end": int}, ...]
        self.sft_full_texts = sft_full_texts if sft_full_texts is not None else []
        self.activation_prompts = _ensure_prompt_list(activation_prompts, "activation_prompts")
        self.end_prompts = _ensure_prompt_list(end_prompts, "end_prompts")
        self.guide_text = guide_text or ""
        
        # 获取模型参数
        first_param = next(base_model.parameters())
        self.model_dtype = first_param.dtype
        self.model_device = first_param.device
        print(f"🔧 检测到模型数据类型: {self.model_dtype}, 设备: {self.model_device}")

        # 注意：不在__init__中预先移动所有embeddings到GPU，避免显存累积
        # 只在__getitem__中按需移动单个embedding
        print(f"📊 embeddings保持在CPU上，训练时按需移动: {self.embeddings.shape}")
        
        # 获取特殊token ID
        self.recall_start_token = '<recall>'
        self.recall_end_token = '</recall>'
        self.im_start_token = '<|im_start|>'
        # 引导文字（在</recall>之后）
        self.guide_text = self.guide_text or ""
        
        # 获取实际的tokenizer（处理Qwen3VLProcessor的情况）
        actual_tokenizer = self._get_tokenizer()

        self.recall_start_id = actual_tokenizer.convert_tokens_to_ids(self.recall_start_token)
        self.recall_end_id = actual_tokenizer.convert_tokens_to_ids(self.recall_end_token)
        self.im_start_id = actual_tokenizer.convert_tokens_to_ids(self.im_start_token)
        self.memory_pad_id = actual_tokenizer.convert_tokens_to_ids("<|memory_pad|>")
        
        # 验证特殊token（移除<|recall|>的检查）
        if any(token_id == actual_tokenizer.unk_token_id for token_id in
               [self.recall_start_id, self.recall_end_id, self.memory_pad_id]):
            raise ValueError("特殊token不存在！")
        
        # 初始化数据配对 - 会在每个epoch开始时刷新
        self.refresh_epoch_data()
        
        print(f"✅ 增强数据集初始化完成")
        print(f"   特殊token IDs: start={self.recall_start_id}, end={self.recall_end_id}")
        print(f"   引导文字: {self.guide_text}")
        print(f"   原始文本数量: {len(texts)}")
        print(f"   总训练样本数: {self.total_samples}")
    
    def is_main_process(self):
        """判断当前是否为主进程，用于多GPU/分布式场景下的日志控制"""
        # 优先使用外部传入的函数（例如Trainer的is_main_process）
        if callable(self._is_main_process_fn):
            try:
                return bool(self._is_main_process_fn())
            except Exception:
                pass
        # 如果使用torch.distributed，判断rank
        try:
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        except Exception:
            pass
        # 默认认为是主进程
        return True

    def refresh_epoch_data(self):
        """每个epoch开始时刷新数据 - 简化版"""
        num_texts = len(self.texts)

        # 每个embedding只对应自己的记忆文本
        self.text_indices = list(range(num_texts))
        self.total_samples = num_texts

        if self.is_main_process():
            print(f"✅ 数据刷新完成: {self.total_samples} 个训练样本")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """获取训练样本 - 每个embedding对应其自己的记忆文本"""
        text_idx = self.text_indices[idx]
        return self._get_memory_decode_sample(text_idx)
    
    def _split_sft_at_thinking(self, sft_data: dict) -> Tuple[str, str]:
        """
        在SFT完整文本的思考部分内部随机截断，优先在句号后面截断
        返回(截断前文本, 截断后文本)
        """
        import random
        
        full_text = sft_data["full_text"]
        thinking_start = sft_data["thinking_start"]
        thinking_end = sft_data["thinking_end"]
        
        start_tag = "<think>"
        end_tag = "</think>"
        thinking_content = full_text[thinking_start + len(start_tag):thinking_end - len(end_tag)]
        
        if not thinking_content.strip():
            prefix_text = full_text[:thinking_start]
            suffix_text = full_text[len(prefix_text):]
            return prefix_text.strip(), suffix_text.strip()
        
        actual_tokenizer = self._get_tokenizer()
        thinking_tokens = actual_tokenizer(thinking_content, add_special_tokens=False)['input_ids']
        
        if len(thinking_tokens) <= 1:
            prefix_text = full_text[:thinking_start]
            suffix_text = full_text[len(prefix_text):]
            return prefix_text.strip(), suffix_text.strip()
        
        max_truncate_pos = len(thinking_tokens) - 1
        if max_truncate_pos <= 0:
            prefix_text = full_text[:thinking_start]
            suffix_text = full_text[len(prefix_text):]
            return prefix_text.strip(), suffix_text.strip()
        
        sentence_end_tokens = []
        for i, token_id in enumerate(thinking_tokens):
            try:
                token_text = actual_tokenizer.decode([token_id], skip_special_tokens=True)
                if any(punct in token_text for punct in ['。', '.', '！', '!', '？', '?', '；', ';']):
                    sentence_end_tokens.append(i + 1)
            except Exception:
                pass
        
        if sentence_end_tokens:
            truncate_pos = random.choice(sentence_end_tokens)
            truncate_pos = min(truncate_pos, max_truncate_pos)
        else:
            truncate_pos = random.randint(1, max_truncate_pos)
        
        truncated_thinking_tokens = thinking_tokens[:truncate_pos]
        truncated_thinking_text = actual_tokenizer.decode(truncated_thinking_tokens, skip_special_tokens=True)
        
        truncated_text_raw = (
            full_text[:thinking_start + len(start_tag)] +
            truncated_thinking_text
        )
        prefix_text = truncated_text_raw.strip()
        suffix_start = len(truncated_text_raw)
        if suffix_start > len(full_text):
            suffix_start = len(full_text)
        suffix_text = full_text[suffix_start:]
        return prefix_text, suffix_text.strip()
    
    def _get_memory_decode_sample(self, text_idx, context_override: Optional[Dict[str, str]] = None):
        """构造记忆解码训练样本

        正确格式：
        输入：随机上下文 + "<recall>" + [embedding向量]
        目标：-100标签 * 上下文长度 + 记忆文本内容 + "</recall>" + 引导文字

        模型学习：看到<recall> + 特定embedding时，生成对应的记忆内容，忽略上下文干扰
        """
        text = self.texts[text_idx]
        embedding = self.embeddings[text_idx]
        
        # 添加噪声到embedding（可选，用于数据增强）
        if self.noise_std > 0:
            noise = torch.randn_like(embedding) * self.noise_std
            noisy_embedding = embedding + noise
        else:
            noisy_embedding = embedding.clone()

        # 确保数据类型正确，设备分配由Accelerator处理
        noisy_embedding = noisy_embedding.to(self.model_dtype)

        # ===== 添加随机上下文干扰 =====
        # 优先从SFT完整文本中随机选择并在思考部分内部截断，如果没有SFT数据则从记忆条目中选择
        # 每个训练样本都重新随机选择上下文，确保每个epoch的上下文都不同
        import random
        context_tokens = []
        context_idx = None
        context_text = ""
        tail_text = ""
        actual_tokenizer = self._get_tokenizer()
        override = context_override or {}
        if override.get("prefix_text"):
            context_text = override.get("prefix_text", "")
            tail_text = override.get("suffix_text", "")
        if not context_text:
            if len(self.sft_full_texts) > 0:
                sft_data = random.choice(self.sft_full_texts)
                prefix_text, _ = self._split_sft_at_thinking(sft_data)
                context_text = prefix_text
            if not context_text and len(self.texts) > 1:
                other_indices = [i for i in range(len(self.texts)) if i != text_idx]
                context_idx = random.choice(other_indices)
                context_text = self.texts[context_idx]
        if context_text:
            context_tokens = actual_tokenizer(context_text, add_special_tokens=False)['input_ids']
        # 如果只有一个记忆条目且没有SFT数据，则没有上下文（context_tokens已初始化为空列表）

        # ===== 构造核心训练序列 =====
        activation_prompt = random.choice(self.activation_prompts).strip()
        end_prompt = random.choice(self.end_prompts).strip()
        activation_tokens = actual_tokenizer(activation_prompt, add_special_tokens=False)['input_ids'] if activation_prompt else []

        # 构造目标文本：记忆内容 + </recall> + 结束引导
        target_text = f"{text}{self.recall_end_token}{end_prompt}"
        if tail_text:
            tail_text_clean = tail_text.strip()
            if tail_text_clean:
                separator = "" if target_text.endswith("\n") else "\n"
                target_text = f"{target_text}{separator}{tail_text_clean}"
        target_tokens = actual_tokenizer(target_text, add_special_tokens=False)['input_ids']

        # 将<recall>编码为token
        recall_tokens = actual_tokenizer(self.recall_start_token, add_special_tokens=False)['input_ids']
        recall_token_count = len(recall_tokens)

        # 构造核心输入序列：<recall> + <|memory_pad|>
        core_input_tokens = (
            recall_tokens +  # <recall>标签
            [self.memory_pad_id]  # <|memory_pad|> token，将被向量替换
        )

        # ===== 构造完整序列 =====
        base_input_len = len(context_tokens) + len(activation_tokens) + len(core_input_tokens)

        if self.max_length is not None:
            total_length = base_input_len + len(target_tokens)
        if total_length > self.max_length:
                # 预留核心输入（<recall> + <|memory_pad|>）
                min_input_len = len(activation_tokens) + len(core_input_tokens)
                available_input_len = self.max_length - len(target_tokens)
                if available_input_len < min_input_len:
                    # 无法容纳全部目标文本，截断目标文本并保留核心输入
                    available_input_len = min_input_len
                    max_target_len = max(self.max_length - available_input_len, 1)
                    target_tokens = target_tokens[:max_target_len]
                # 只保留下文，确保核心序列存在
                allowed_context_len = max(0, available_input_len - min_input_len)
                if len(context_tokens) > allowed_context_len:
                    context_tokens = context_tokens[:allowed_context_len]
                base_input_len = allowed_context_len + min_input_len

        full_input_tokens = context_tokens + activation_tokens + core_input_tokens
        prefix_len = len(full_input_tokens)
        prefix_labels = [-100] * prefix_len
        recall_start_idx = len(context_tokens) + len(activation_tokens)
        for offset, token_id in enumerate(recall_tokens):
            pos = recall_start_idx + offset
            if 0 <= pos < prefix_len:
                prefix_labels[pos] = token_id
        recall_label_slice = prefix_labels[recall_start_idx:recall_start_idx + recall_token_count]
        if len(recall_label_slice) != recall_token_count or any(label == -100 for label in recall_label_slice):
            raise RuntimeError(
                f"❌ <recall>标签未正确设置，位置[{recall_start_idx}, {recall_start_idx + recall_token_count}) "
                f"labels={recall_label_slice}"
            )
        full_target_tokens = prefix_labels + target_tokens

        # 计算embedding插入位置（上下文 + 激活语 + <recall> 之后）
        embedding_position = len(context_tokens) + len(activation_tokens) + recall_token_count

        # 最终标签
        labels = full_target_tokens

        sample = {
            'sequence_tokens': torch.tensor(full_input_tokens, dtype=torch.long),
            'embedding_to_insert': noisy_embedding,
            'embedding_position': embedding_position,
            'labels': torch.tensor(labels, dtype=torch.long),
            'recall_token_count': recall_token_count,
            'text': text,
            'text_idx': text_idx,
            'context_text': context_text,  # 现在可能是SFT思考文本（截断后）或记忆条目文本
            'context_length': len(context_tokens),
            'activation_prompt': activation_prompt,
            'end_prompt': end_prompt
        }
        if override.get("sample_type"):
            sample['sample_type'] = override["sample_type"]
        else:
            sample['sample_type'] = 'memory'
        return sample

class EnhancedTextMemoryModel(nn.Module):
    """增强的文本记忆模型"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        first_param = next(base_model.parameters())
        self.model_dtype = first_param.dtype
        self.model_device = first_param.device
        print(f"🔧 EnhancedTextMemoryModel 数据类型: {self.model_dtype}, 设备: {self.model_device}")
        
    def forward(self, input_ids, embeddings_to_insert=None, embedding_positions=None, attention_mask=None, labels=None, memory_pad_token_id=None):
        """
        前向传播 - 支持标准前向传播（SFT）和记忆向量插入（记忆条目）
        
        Args:
            input_ids: [batch_size, seq_len] token序列
            embeddings_to_insert: [batch_size, embed_dim] 要插入的表征向量（可选，SFT时为None）
            embedding_positions: [batch_size] 表征向量插入位置（可选，SFT时为None）
            attention_mask: [batch_size, seq_len] 注意力掩码
            labels: [batch_size, seq_len] 标签
            memory_pad_token_id: <|memory_pad|> token ID，用于验证注入位置（可选）
        """
        
        # 确保所有输入在正确设备上
        input_ids = input_ids.to(self.model_device)
        attention_mask = attention_mask.to(self.model_device) if attention_mask is not None else None
        if labels is not None:
            labels = labels.to(self.model_device)
        
        # 判断是否需要插入记忆向量
        # 检查是否有样本需要插入（position >= 0表示需要插入）
        need_memory_injection = (
            embeddings_to_insert is not None and 
            embedding_positions is not None and
            embeddings_to_insert.numel() > 0 and
            (embedding_positions >= 0).any()  # 至少有一个样本的position >= 0
        )
        
        if need_memory_injection:
            # 记忆条目训练或混合batch：需要插入记忆向量
            embeddings_to_insert = embeddings_to_insert.to(self.model_device)
            embedding_positions = embedding_positions.to(self.model_device)
            
            # 通过embedding层获取token embeddings
            embedding_layer = self.base_model.get_input_embeddings()
            token_embeddings = embedding_layer(input_ids)  # [batch_size, seq_len, embed_dim]
            
            # 对于混合batch，只对需要插入的样本进行插入（position >= 0）
            # 对于SFT样本（position < 0），跳过插入
            valid_mask = embedding_positions >= 0
            if valid_mask.all():
                # 所有样本都需要插入
                token_embeddings = inject_memory_embedding_to_inputs_embeds(
                    token_embeddings, embedding_positions, embeddings_to_insert,
                    input_ids=input_ids, memory_pad_token_id=memory_pad_token_id
                )
            else:
                # 混合batch：只对有效样本插入
                for i in range(len(embedding_positions)):
                    if embedding_positions[i] >= 0:
                        pos = embedding_positions[i].item()
                        token_embeddings[i, pos] = embeddings_to_insert[i]
            
            # 使用修改后的embeddings进行前向传播
            outputs = self.base_model(
                inputs_embeds=token_embeddings,
                attention_mask=attention_mask,
                return_dict=True
            )
        else:
            # 纯SFT batch：标准前向传播
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        logits = outputs.logits
        
        # 计算损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': logits
        }

class MixedMemorySFTDataset(Dataset):
    """混合数据集：包含记忆条目和SFT数据，每个epoch重新抽取"""
    
    def __init__(
        self,
        memory_texts,
        memory_embeddings,
        sft_messages_list,  # SFT数据列表，每个元素是标准化的messages
        tokenizer,
        base_model,
        max_length=3000,
        noise_std=0.01,
        is_main_process_fn=None,
        sft_full_texts=None,
        activation_prompts=None,
        end_prompts=None,
        memory_ratio=0.5,  # 记忆条目在混合数据中的比例
        guide_text=None,
        sft_message_source_indices=None,
        sft_full_source_indices=None,
    ):
        self.memory_texts = memory_texts
        self.memory_embeddings = memory_embeddings
        self.full_sft_messages_list = sft_messages_list if sft_messages_list is not None else []
        self.sft_message_source_indices = (
            sft_message_source_indices
            if sft_message_source_indices is not None
            else list(range(len(self.full_sft_messages_list)))
        )
        self.sft_messages_list = self.full_sft_messages_list
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.max_length = max_length
        self.noise_std = noise_std
        self._is_main_process_fn = is_main_process_fn
        self.sft_full_texts = sft_full_texts if sft_full_texts is not None else []
        self.sft_full_source_indices = (
            sft_full_source_indices
            if sft_full_source_indices is not None
            else list(range(len(self.sft_full_texts)))
        )
        self.activation_prompts = _ensure_prompt_list(activation_prompts, "activation_prompts")
        self.end_prompts = _ensure_prompt_list(end_prompts, "end_prompts")
        self.memory_ratio = memory_ratio
        self.guide_text = guide_text or ""
        
        # 创建记忆条目数据集（用于生成记忆训练样本）
        self.memory_dataset = EnhancedTextMemoryDataset(
            memory_texts,
            memory_embeddings,
            tokenizer,
            base_model,
            max_length=max_length,
            noise_std=noise_std,
            is_main_process_fn=is_main_process_fn,
            sft_full_texts=self.sft_full_texts,
            activation_prompts=activation_prompts,
            end_prompts=end_prompts,
            guide_text=self.guide_text,
        )
        
        # 初始化混合数据索引
        self.last_sft_only_indices = []
        self.last_sft_full_indices = []
        self.refresh_epoch_data()
        
        if self.is_main_process():
            print(f"✅ 混合数据集初始化完成")
            print(f"   记忆条目数量: {len(memory_texts)}")
            print(f"   SFT数据数量: {len(sft_messages_list)}")
            print(f"   混合后总样本数: {self.total_samples}")
            print(f"   记忆条目比例: {memory_ratio:.1%}")
    
    def is_main_process(self):
        """判断当前是否为主进程"""
        if callable(self._is_main_process_fn):
            try:
                return bool(self._is_main_process_fn())
            except Exception:
                pass
        try:
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        except Exception:
            pass
        return True
    
    def refresh_epoch_data(self):
        """每个epoch开始时重新抽取数据"""
        memory_count = len(self.memory_texts)
        sft_count = len(self.sft_messages_list)
        
        self.mixed_indices = []
        self.last_sft_only_indices = []
        self.last_sft_full_indices = []
        if memory_count > 0:
            memory_indices = list(range(memory_count))
            random.shuffle(memory_indices)
            memory_full_count = memory_count // 2  # 需要在尾部拼接SFT的记忆数量
            memory_front_count = memory_count - memory_full_count
            
            # 记忆类型A：仅前置SFT
            for idx in memory_indices[:memory_front_count]:
                self.mixed_indices.append(('memory_front', idx, None))
            
            # 记忆类型B：前置+后置SFT
            if len(self.memory_dataset.sft_full_texts) > 0 and memory_full_count > 0:
                sft_full_indices = self._sample_indices(len(self.memory_dataset.sft_full_texts), memory_full_count)
                for mem_idx, sft_idx in zip(memory_indices[memory_front_count:], sft_full_indices):
                    self.mixed_indices.append(('memory_full', mem_idx, sft_idx))
                self.last_sft_full_indices = sft_full_indices[:]
            else:
                for idx in memory_indices[memory_front_count:]:
                    self.mixed_indices.append(('memory_front', idx, None))
                self.last_sft_full_indices = []
            
            # 纯SFT样本（数量为记忆条目的一半，向下取整，至少为1）
            sft_only_target = memory_count // 2
            if memory_count == 1:
                sft_only_target = 1
            if sft_count > 0 and sft_only_target > 0:
                sft_only_indices = self._sample_indices(sft_count, sft_only_target)
                for sft_idx in sft_only_indices:
                    self.mixed_indices.append(('sft', sft_idx, None))
                self.last_sft_only_indices = sft_only_indices[:]
            else:
                self.last_sft_only_indices = []
        else:
            # 没有记忆条目，只能返回SFT样本
            sample_sft = min(32, sft_count)
            sft_only_indices = self._sample_indices(sft_count, sample_sft)
            for sft_idx in sft_only_indices:
                self.mixed_indices.append(('sft', sft_idx, None))
            self.last_sft_only_indices = sft_only_indices[:]
        
        random.shuffle(self.mixed_indices)
        self.total_samples = len(self.mixed_indices)
        
        # 刷新记忆数据集的上文（每个epoch重新抽取）
        self.memory_dataset.refresh_epoch_data()
        
        if self.is_main_process():
            type_a = sum(1 for item in self.mixed_indices if item[0] == 'memory_front')
            type_b = sum(1 for item in self.mixed_indices if item[0] == 'memory_full')
            type_c = sum(1 for item in self.mixed_indices if item[0] == 'sft')
            print(f"✅ 混合数据刷新完成: {self.total_samples} 个样本 (记忆-前置: {type_a}, 记忆-前后拼接: {type_b}, 纯SFT: {type_c})")
            if self.last_sft_only_indices:
                preview = min(5, len(self.last_sft_only_indices))
                preview_indices = sorted(self.last_sft_only_indices[:preview])
                mapped = sorted(
                    self.sft_message_source_indices[idx]
                    if idx < len(self.sft_message_source_indices)
                    else idx
                    for idx in preview_indices
                )
                print(f"   📋 纯SFT样本原始索引(前{preview}条): {mapped}")
                if len(self.last_sft_only_indices) > preview:
                    print(f"   ... 共 {len(self.last_sft_only_indices)} 条纯SFT样本")
            if self.last_sft_full_indices:
                preview = min(5, len(self.last_sft_full_indices))
                preview_indices = sorted(self.last_sft_full_indices[:preview])
                mapped = sorted(
                    self.sft_full_source_indices[idx]
                    if idx < len(self.sft_full_source_indices)
                    else idx
                    for idx in preview_indices
                )
                print(f"   📋 夹心SFT样本原始索引(前{preview}条): {mapped}")
                if len(self.last_sft_full_indices) > preview:
                    print(f"   ... 共 {len(self.last_sft_full_indices)} 条夹心SFT样本")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """获取训练样本"""
        entry = self.mixed_indices[idx]
        data_type = entry[0]
        data_idx = entry[1]
        extra = entry[2] if len(entry) > 2 else None
        
        if data_type == 'memory_front':
            sample = self.memory_dataset._get_memory_decode_sample(data_idx)
            sample['sample_type'] = 'memory_front'
            return sample
        elif data_type == 'memory_full':
            context_override = self._build_context_override(extra)
            sample = self.memory_dataset._get_memory_decode_sample(data_idx, context_override=context_override)
            sample['sample_type'] = 'memory_full'
            return sample
        else:
            sample = self._get_sft_sample(data_idx)
            sample['sample_type'] = 'sft_only'
            return sample
    
    def _get_tokenizer(self):
        """获取实际的tokenizer对象"""
        if hasattr(self.tokenizer, 'tokenizer'):
            return self.tokenizer.tokenizer
        else:
            return self.tokenizer
    
    def _get_sft_sample(self, sft_idx):
        """构造SFT训练样本"""
        messages = self.sft_messages_list[sft_idx]
        
        # 使用tokenizer将messages转换为input_ids和labels
        actual_tokenizer = self._get_tokenizer()
        
        # 使用apply_chat_template转换为input_ids
        batch_inputs = actual_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt"
        )
        
        input_ids = batch_inputs["input_ids"][0]  # [seq_len]
        attention_mask = batch_inputs.get("attention_mask", (input_ids != 0).long())[0]
        
        # 默认全部mask，后续只放开assistant段落
        labels_tensor = torch.full_like(input_ids, -100)
        
        # 计算每条message结束时的长度，用于定位assistant内容区间
        prefix_lengths = []
        for end_idx in range(len(messages)):
            prefix_slice = messages[: end_idx + 1]
            prefix_inputs = actual_tokenizer.apply_chat_template(
                prefix_slice,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt"
            )
            prefix_ids = prefix_inputs["input_ids"][0]
            prefix_lengths.append(prefix_ids.shape[0])
        
        total_len = input_ids.shape[0]
        if prefix_lengths and prefix_lengths[-1] != total_len:
            # 理论上应该完全一致，如果不一致则取交集以避免越界
            total_len = min(total_len, prefix_lengths[-1])
        
        prev_len = 0
        for msg_idx, message in enumerate(messages):
            curr_len = prefix_lengths[msg_idx] if msg_idx < len(prefix_lengths) else total_len
            curr_len = min(curr_len, total_len)
            if message.get("role") == "assistant":
                labels_tensor[prev_len:curr_len] = input_ids[prev_len:curr_len]
            prev_len = curr_len
        
        # 构造序列tokens（用于collate_fn）
        sequence_tokens = input_ids.clone()
        
        # 获取embedding维度（从模型配置或默认值）
        try:
            hidden_size = getattr(self.base_model.config, "hidden_size", 4096)
        except:
            hidden_size = 4096
        
        # 返回格式与记忆条目样本一致
        return {
            'sequence_tokens': sequence_tokens,
            'labels': labels_tensor,
            'embedding_to_insert': torch.zeros(1, hidden_size),  # 占位符，SFT不需要embedding
            'embedding_position': -1,  # -1表示SFT样本，不需要插入
            'context_text': '',
            'text': actual_tokenizer.decode(input_ids, skip_special_tokens=True),
            'activation_prompt': '',
            'end_prompt': '',
            'recall_token_count': 0,
            'context_length': 0,
            'is_sft': True,  # 标记为SFT样本
        }
    
    def _build_context_override(self, sft_full_idx):
        if sft_full_idx is None:
            return None
        sft_full_texts = getattr(self.memory_dataset, "sft_full_texts", [])
        if not sft_full_texts:
            return None
        if sft_full_idx < 0 or sft_full_idx >= len(sft_full_texts):
            sft_full_idx = sft_full_idx % len(sft_full_texts)
        try:
            prefix_text, suffix_text = self.memory_dataset._split_sft_at_thinking(sft_full_texts[sft_full_idx])
            return {
                "prefix_text": prefix_text,
                "suffix_text": suffix_text,
                "sample_type": "memory_full"
            }
        except Exception:
            return None
    
    @staticmethod
    def _sample_indices(pool_size: int, sample_count: int) -> List[int]:
        if pool_size <= 0 or sample_count <= 0:
            return []
        if sample_count <= pool_size:
            return random.sample(range(pool_size), sample_count)
        return [random.randrange(pool_size) for _ in range(sample_count)]

class EnhancedTextMemoryTrainer:
    """增强的文本记忆训练器 - 支持多GPU"""
    
    def _get_tokenizer(self):
        """获取真正的tokenizer（如果传入的是processor，则返回processor.tokenizer）"""
        if hasattr(self.tokenizer, 'tokenizer'):
            # 如果传入的是processor，返回其内部的tokenizer
            return self.tokenizer.tokenizer
        else:
            # 如果传入的是tokenizer，直接返回
            return self.tokenizer

    def __init__(
        self,
        model_name,
        device=None,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        original_device=None,
        preloaded_model=None,
        preloaded_tokenizer=None,
        gradient_accumulation_steps=1,
        max_memory=None,
        generation_config=None,
        epoch_end_hook=None,
        lora_target_modules=None,
        dataset_max_length=3000,
        test_sample_count=2,
        test_max_new_tokens=300,
        test_use_cache=False,
        activation_prompts=None,
        end_prompts=None,
        guide_text=None,
    ):

        # 注意：CUDA_VISIBLE_DEVICES 已经在 app.py 中正确设置，这里不需要重复设置
        # 只保存原始环境变量用于cleanup时恢复
        self._original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

        self.model_name = model_name
        self.specified_device = device
        self.original_device = original_device or device  # 保存原始设备信息用于显示
        self.ddp_enabled = False
        self.local_rank = None

        # ⚠️ 注意：下面这些token字符串会在多个方法（特别是_preloaded路径）里立即使用
        # 过去它们是在_check_and_add_special_tokens()里临时赋值，由于现在支持外部传入已加载模型，
        # 需要在构造函数最开始就显式设置，避免“先访问后定义”导致AttributeError。
        self.recall_start_token = '<recall>'
        self.recall_end_token = '</recall>'
        self.im_start_token = '<|im_start|>'
        self.im_end_token = '<|im_end|>'
        # 显示正确的设备信息
        display_device = self.original_device or device
        if isinstance(display_device, str) and display_device.startswith('cuda:'):
            print(f"   使用GPU设备: {display_device}")
        elif display_device == "auto":
            print("   自动选择设备")
        else:
            print(f"   使用设备: {display_device}")
        # 配置梯度累积步数
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # 根据设备配置决定是否启用DDP
        use_ddp = False
        if isinstance(device, list) and len(device) > 1:
            use_ddp = True
            print(f"   多GPU模式: 启用DDP，GPU数量: {len(device)}")
        elif device == "auto":
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                use_ddp = True
                print(f"   多GPU模式: 自动检测多GPU，启用DDP")

        # 对于单GPU配置，在Accelerator初始化前设置当前设备
        if isinstance(device, str) and device.startswith('cuda:'):
            # 检查CUDA_VISIBLE_DEVICES
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible and cuda_visible.strip():
                # CUDA_VISIBLE_DEVICES已设置，使用重新映射后的设备cuda:0
                torch.cuda.set_device(0)
            else:
                # 未设置CUDA_VISIBLE_DEVICES，直接使用物理设备
                device_idx = int(device.split(':')[1])
                torch.cuda.set_device(device_idx)
        
        # 初始化Accelerator，支持多GPU和梯度累积
        self.accelerator = Accelerator(
            mixed_precision='bf16',
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            # 如果是多GPU，启用DDP
            # 注意：DDP需要在torchrun下启动，这里只是配置
        )

        # 根据设备配置设置相关变量
        if device is None:
            self.use_auto_device = False
            self.primary_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.multi_gpu_list = None
        elif isinstance(device, list):
            # 多GPU配置
            if len(device) > 0:
                self.use_auto_device = False
                self.primary_device = torch.device(device[0])
                self.multi_gpu_list = device
                print(f"   使用多GPU列表: {device}，主设备: {device[0]}")
        elif isinstance(device, str) and device.startswith('cuda:'):
            # 单GPU配置
            # 如果设置了CUDA_VISIBLE_DEVICES，需要使用重新映射后的设备
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible and cuda_visible.strip():
                # CUDA_VISIBLE_DEVICES已设置，使用重新映射后的设备
                self.primary_device = torch.device("cuda:0")
                print(f"   CUDA_VISIBLE_DEVICES={cuda_visible}，使用重新映射设备 cuda:0（对应物理GPU {device}）")
            else:
                # 未设置CUDA_VISIBLE_DEVICES，直接使用物理设备
                self.primary_device = torch.device(device)
                print(f"   使用设备 {device}")
            self.use_auto_device = False
            self.multi_gpu_list = None
        elif device == "auto":
            self.use_auto_device = True
            self.primary_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.multi_gpu_list = None
        else:
            # CPU或其他
            self.use_auto_device = False
            self.primary_device = torch.device('cpu')
            self.multi_gpu_list = None

        # LoRA配置参数
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # LoRA目标模块（如果为None，使用默认配置）
        self.lora_target_modules = lora_target_modules
        self.max_memory = max_memory
        self.generation_config = generation_config or {}
        self.epoch_end_hook = epoch_end_hook
        self.dataset_max_length = dataset_max_length
        self.test_sample_count = max(1, int(test_sample_count))
        self.test_max_new_tokens = max(1, int(test_max_new_tokens))
        self.test_use_cache = bool(test_use_cache)
        self.activation_prompts = _ensure_prompt_list(activation_prompts, "activation_prompts")
        self.end_prompts = _ensure_prompt_list(end_prompts, "end_prompts")
        self.guide_text = guide_text or ""

# 设备变量已在前面设置
        
        # 特殊token定义
        self.special_tokens = ['<recall>', '</recall>']
        self.recall_start_token = '<recall>'
        self.recall_end_token = '</recall>'

        print(f"🤖 初始化增强文本记忆训练器...")
        print(f"   模型: {model_name}")
        print(f"   设备配置: {device}")

        # 若由 torchrun 启动，自动启用DDP并固定到单卡
        if 'LOCAL_RANK' in os.environ and not self.accelerator.state.initialized:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            os.environ.setdefault('RANK', os.environ.get('RANK', '0'))
            os.environ.setdefault('WORLD_SIZE', os.environ.get('WORLD_SIZE', '1'))
            torch.cuda.set_device(self.local_rank)
            if not (dist.is_available() and dist.is_initialized()):
                dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
            self.ddp_enabled = True
            # 在DDP下强制单卡加载，覆盖自动分配
            self.use_auto_device = False
            self.multi_gpu_list = None
            self.primary_device = torch.device(f'cuda:{self.local_rank}')
            self.specified_device = f'cuda:{self.local_rank}'
            if self.is_main_process():
                print(f"🧩 DDP已启用，LOCAL_RANK={self.local_rank}")
        
        # 处理预加载模型或加载新模型
        if preloaded_model is not None and preloaded_tokenizer is not None:
            # 使用预加载的模型
            print("   使用预加载的模型和tokenizer")
            
            # 确保预加载模型在正确的设备上
            first_param = next(preloaded_model.parameters())
            current_device = first_param.device
            target_device = self.primary_device
            
            if current_device != target_device:
                print(f"   ⚠️ 预加载模型在 {current_device}，需要移动到 {target_device}")
                preloaded_model = preloaded_model.to(target_device)
                print(f"   ✅ 模型已移动到 {target_device}")
            
            self.base_model = preloaded_model
            self.tokenizer = preloaded_tokenizer
            # 预加载的tokenizer应该已经包含了正确的特殊token，直接设置token IDs
            self._set_special_token_ids()
            self._skip_model_loading = True
        else:
            # 正常加载模型
            self._load_model()
            # 检查特殊token
            self._check_and_add_special_tokens()
            self._skip_model_loading = False

        # 记录原始embedding
        self._save_original_embeddings()

        # 设置LoRA
        self._setup_lora()

        # 创建包装模型
        self.model = EnhancedTextMemoryModel(self.base_model)

        # 降内存：对基础模型启用梯度检查点并关闭use_cache
        try:
            if hasattr(self.base_model, 'gradient_checkpointing_enable'):
                self.base_model.gradient_checkpointing_enable()
            if hasattr(self.base_model, 'config'):
                setattr(self.base_model.config, 'use_cache', False)
        except Exception:
            pass

        # DDP包装
        if self.ddp_enabled:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
        
        # 显示参数统计
        self._print_parameters()

    def is_main_process(self):
        if hasattr(self, 'accelerator'):
            return self.accelerator.is_main_process
        return (not self.ddp_enabled) or (dist.get_rank() == 0)
    
    def _load_model(self):
        """加载模型和分词器 - 支持多GPU配置"""
        # 检查模型路径是否为本地路径，如果是则使用local_files_only
        import os
        model_path = self.model_name
        if not os.path.isabs(model_path):
            # 转换为绝对路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.abspath(os.path.join(project_root, model_path))
        
        # 如果是本地路径，使用local_files_only避免modelscope尝试从网络下载
        is_local_path = os.path.exists(model_path) and os.path.isdir(model_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path if is_local_path else self.model_name,
            trust_remote_code=True,
            local_files_only=is_local_path
        )
        
        try:
            # 根据设备配置选择device_map
            if self.use_auto_device:
                device_map = "auto"
                print("   使用自动设备分配")
            elif hasattr(self, 'multi_gpu_list') and self.multi_gpu_list:
                # 多GPU配置
                device_map = "auto"
                print(f"   使用多GPU自动分配: {self.multi_gpu_list}")

                # 设置环境变量限制可见GPU
                import os
                if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                    gpu_indices = [gpu.split(':')[1] for gpu in self.multi_gpu_list if gpu.startswith('cuda:')]
                    if gpu_indices:
                        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_indices)
                        print(f"   设置CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

                # max_memory会在模型加载时单独传递，不影响device_map
                if hasattr(self, 'max_memory') and self.max_memory:
                    print(f"   将使用max_memory控制GPU分布: {self.max_memory}")
                else:
                    print(f"   使用自动GPU分布 (未设置max_memory)")
            elif isinstance(self.specified_device, str) and self.specified_device.startswith('cuda:'):
                # 单GPU指定 - 如果服务器已设置CUDA_VISIBLE_DEVICES，使用cuda:0
                import os
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    # 服务器已设置环境变量，使用cuda:0
                    device_map = {"": 0}
                    print(f"   服务器已设置CUDA_VISIBLE_DEVICES，使用cuda:0 (原始设备: {self.specified_device})")
                else:
                    # 服务器未设置，使用原始设备索引
                    device_index = int(self.specified_device.split(':')[1])
                    device_map = {"": device_index}
                    print(f"   使用指定单GPU: {self.specified_device}")
            elif self.specified_device == "cpu":
                device_map = {"": "cpu"}
                print(f"   使用CPU设备")
            else:
                # 默认情况
                if hasattr(self, 'primary_device') and self.primary_device.type == 'cuda':
                    device_map = {"": self.primary_device.index}
                else:
                    device_map = "auto"
                print(f"   使用默认设备映射: {device_map}")
            
            print(f"   实际使用设备映射: {device_map}")
            
            # 检查模型类型：如果是Qwen3-VL，需要使用Qwen3VLForConditionalGeneration
            # 检查config.json文件来确定模型类型
            import json
            config_file = os.path.join(model_path if is_local_path else self.model_name, "config.json")
            is_qwen3vl = False
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        model_config = json.load(f)
                        model_type = model_config.get("model_type", "").lower()
                        if "qwen3_vl" in model_type or "qwen3-vl" in model_type:
                            is_qwen3vl = True
                            print(f"   检测到Qwen3-VL模型类型，使用Qwen3VLForConditionalGeneration加载")
                except:
                    pass
            
            # 准备加载参数
            load_kwargs = {
                "torch_dtype": "auto",
                "device_map": device_map,
                "trust_remote_code": True,
                "local_files_only": is_local_path
            }

            # 如果有max_memory配置，添加它
            if hasattr(self, 'max_memory') and self.max_memory and device_map == "auto":
                load_kwargs["max_memory"] = self.max_memory
                print(f"   添加max_memory参数: {self.max_memory}")

            # 根据模型类型选择加载方式
            if is_qwen3vl:
                # 使用Qwen3VLForConditionalGeneration加载Qwen3-VL模型
                from transformers import Qwen3VLForConditionalGeneration
                self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path if is_local_path else self.model_name,
                    **load_kwargs
                )
            else:
                # 使用AutoModelForCausalLM加载普通文本模型
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    model_path if is_local_path else self.model_name,
                    **load_kwargs
                )
            
            # 获取实际设备信息
            first_param = next(self.base_model.parameters())
            model_dtype = first_param.dtype
            model_device = first_param.device
            
            print(f"✅ 模型加载成功")
            print(f"   实际设备: {model_device}")
            print(f"   数据类型: {model_dtype}")
            
            # 显示设备映射信息
            if hasattr(self.base_model, 'hf_device_map'):
                print(f"   设备映射详情: {self.base_model.hf_device_map}")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 回退策略
            print("🔄 尝试回退到单GPU模式...")
            
            try:
                # 确定回退设备
                if hasattr(self, 'multi_gpu_list') and self.multi_gpu_list:
                    fallback_device = self.multi_gpu_list[0]
                elif isinstance(self.specified_device, str) and self.specified_device.startswith('cuda:'):
                    fallback_device = self.specified_device
                else:
                    fallback_device = 'cuda:0'
                
                # 提取设备索引
                if fallback_device.startswith('cuda:'):
                    device_index = int(fallback_device.split(':')[1])
                    device_map = {"": device_index}
                else:
                    device_map = {"": "cpu"}
                
                print(f"   回退设备映射: {device_map}")
                
                # 检查模型类型：如果是Qwen3-VL，需要使用Qwen3VLForConditionalGeneration
                import json
                config_file = os.path.join(model_path if is_local_path else self.model_name, "config.json")
                is_qwen3vl = False
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            model_config = json.load(f)
                            model_type = model_config.get("model_type", "").lower()
                            if "qwen3_vl" in model_type or "qwen3-vl" in model_type:
                                is_qwen3vl = True
                                print(f"   检测到Qwen3-VL模型类型，使用Qwen3VLForConditionalGeneration加载")
                    except:
                        pass
                
                # 根据模型类型选择加载方式
                if is_qwen3vl:
                    # 使用Qwen3VLForConditionalGeneration加载Qwen3-VL模型
                    from transformers import Qwen3VLForConditionalGeneration
                    self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path if is_local_path else self.model_name,
                        torch_dtype="auto",
                        device_map=device_map,
                        trust_remote_code=True,
                        local_files_only=is_local_path
                    )
                else:
                    # 使用AutoModelForCausalLM加载普通文本模型
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        model_path if is_local_path else self.model_name,
                        torch_dtype="auto",
                        device_map=device_map,
                        trust_remote_code=True,
                        local_files_only=is_local_path
                    )
                
                first_param = next(self.base_model.parameters())
                print(f"✅ 使用回退设备加载成功: {first_param.device}")
                
            except Exception as fallback_error:
                print(f"❌ 回退加载也失败: {fallback_error}")
                raise RuntimeError(f"模型加载完全失败: 原错误={e}, 回退错误={fallback_error}")
    
    def _check_and_add_special_tokens(self):
        """检查并添加特殊token（如果不存在）"""
        # 检查recall_start_token和recall_end_token
        recall_start_id = self.tokenizer.convert_tokens_to_ids(self.recall_start_token)
        recall_end_id = self.tokenizer.convert_tokens_to_ids(self.recall_end_token)
        
        tokens_to_add = []
        if recall_start_id == self.tokenizer.unk_token_id:
            tokens_to_add.append(self.recall_start_token)
        if recall_end_id == self.tokenizer.unk_token_id:
            tokens_to_add.append(self.recall_end_token)
        
        # 获取最终的token ID
        final_recall_start_id = self.tokenizer.convert_tokens_to_ids(self.recall_start_token)
        final_recall_end_id = self.tokenizer.convert_tokens_to_ids(self.recall_end_token)

        if tokens_to_add:
            # token不存在，需要添加
            print(f"⚠️ 以下特殊token不存在，正在添加: {tokens_to_add}")
            original_vocab_size = len(self.tokenizer)

            # 添加特殊token
            for token in tokens_to_add:
                self.tokenizer.add_tokens(token)

            new_vocab_size = len(self.tokenizer)
            print(f"   词表大小: {original_vocab_size} -> {new_vocab_size} (+{new_vocab_size - original_vocab_size})")

            # 调整模型embedding层
            print("   调整模型embedding层...")
            self.base_model.resize_token_embeddings(len(self.tokenizer))

            # 获取新添加的token ID
            final_recall_start_id = self.tokenizer.convert_tokens_to_ids(self.recall_start_token)
            final_recall_end_id = self.tokenizer.convert_tokens_to_ids(self.recall_end_token)
            
            # 初始化新token的权重
            print("   初始化新token权重...")
            try:
                embedding_layer = self.base_model.get_input_embeddings()
                # <recall> token: 使用"总结"和"回忆"的嵌入向量之和
                if self.recall_start_token in tokens_to_add:
                    recall_start_id = self.tokenizer.convert_tokens_to_ids(self.recall_start_token)
                    ref_words = ["总结", "回忆"]
                    ref_embeddings = []
                    used_refs = []
                    
                    for word in ref_words:
                        ref_id = self.tokenizer.convert_tokens_to_ids(word)
                        if ref_id != self.tokenizer.unk_token_id:
                            ref_embeddings.append(embedding_layer.weight[ref_id].clone().detach())
                            used_refs.append(word)
                    
                    if len(ref_embeddings) > 0:
                        new_embedding = ref_embeddings[0]
                        for ref_emb in ref_embeddings[1:]:
                            new_embedding = new_embedding + ref_emb
                        
                        # 直接归一化：缩放到第一个参考token的范数，然后添加小的正交扰动以区分
                        if len(ref_embeddings) > 1:
                            target_norm = ref_embeddings[0].norm()
                            current_norm = new_embedding.norm()
                            if current_norm > 0:
                                new_embedding = new_embedding / current_norm * target_norm
                            
                            # 添加小的正交扰动，避免与参考token过于相似
                            ref1_normalized = ref_embeddings[0] / ref_embeddings[0].norm()
                            new_normalized = new_embedding / new_embedding.norm()
                            proj = torch.dot(new_normalized, ref1_normalized) * ref1_normalized
                            orthogonal = new_normalized - proj
                            if orthogonal.norm() > 1e-6:
                                orthogonal = orthogonal / orthogonal.norm()
                                perturbation_scale = 0.1
                                new_embedding = new_embedding + orthogonal * perturbation_scale * target_norm
                                new_embedding = new_embedding / new_embedding.norm() * target_norm
                        
                        embedding_layer.weight.data[recall_start_id] = new_embedding
                        ref_str = " + ".join(used_refs)
                        print(f"   ✅ {self.recall_start_token} (ID: {recall_start_id}) 初始化完成（参考: {ref_str}）")
                    else:
                        print(f"   ⚠️ {self.recall_start_token} 的参考token都不存在，使用随机初始化")
                
                # </recall> token: 使用"回忆"和"结束"的嵌入向量之和
                if self.recall_end_token in tokens_to_add:
                    recall_end_id = self.tokenizer.convert_tokens_to_ids(self.recall_end_token)
                    ref_words = ["回忆", "结束"]
                    ref_embeddings = []
                    used_refs = []
                    
                    for word in ref_words:
                        ref_id = self.tokenizer.convert_tokens_to_ids(word)
                        if ref_id != self.tokenizer.unk_token_id:
                            ref_embeddings.append(embedding_layer.weight[ref_id].clone().detach())
                            used_refs.append(word)
                    
                    if len(ref_embeddings) > 0:
                        new_embedding = ref_embeddings[0]
                        for ref_emb in ref_embeddings[1:]:
                            new_embedding = new_embedding + ref_emb
                        
                        # 直接归一化：缩放到第一个参考token的范数，然后添加小的正交扰动以区分
                        if len(ref_embeddings) > 1:
                            target_norm = ref_embeddings[0].norm()
                            current_norm = new_embedding.norm()
                            if current_norm > 0:
                                new_embedding = new_embedding / current_norm * target_norm
                            
                            # 添加小的正交扰动，避免与参考token过于相似
                            ref1_normalized = ref_embeddings[0] / ref_embeddings[0].norm()
                            new_normalized = new_embedding / new_embedding.norm()
                            proj = torch.dot(new_normalized, ref1_normalized) * ref1_normalized
                            orthogonal = new_normalized - proj
                            if orthogonal.norm() > 1e-6:
                                orthogonal = orthogonal / orthogonal.norm()
                                perturbation_scale = 0.1
                                new_embedding = new_embedding + orthogonal * perturbation_scale * target_norm
                                new_embedding = new_embedding / new_embedding.norm() * target_norm
                        
                        embedding_layer.weight.data[recall_end_id] = new_embedding
                        ref_str = " + ".join(used_refs)
                        print(f"   ✅ {self.recall_end_token} (ID: {recall_end_id}) 初始化完成（参考: {ref_str}）")
                    else:
                        print(f"   ⚠️ {self.recall_end_token} 的参考token都不存在，使用随机初始化")

            except Exception as e:
                print(f"   ⚠️ 初始化token权重时出错: {e}")
            
            print(f"✅ 特殊token添加完成: {self.recall_start_token} (ID: {final_recall_start_id}), {self.recall_end_token} (ID: {final_recall_end_id})")
        else:
            # token已存在
            print(f"✅ 特殊token检查通过: {self.recall_start_token} (ID: {final_recall_start_id}), {self.recall_end_token} (ID: {final_recall_end_id})")

        # 设置special_token_ids供其他方法使用
        self.special_token_ids = {
            self.recall_start_token: final_recall_start_id,
            self.recall_end_token: final_recall_end_id
        }
    
    def _check_special_tokens(self):
        """检查特殊token是否存在（已废弃，使用_check_and_add_special_tokens代替）"""
        # 这个方法保留是为了兼容性，但实际调用的是_check_and_add_special_tokens
        self._check_and_add_special_tokens()
        
        # 设置special_token_ids供其他方法使用
        self.special_token_ids = {
            self.recall_start_token: self.tokenizer.convert_tokens_to_ids(self.recall_start_token),
            self.recall_end_token: self.tokenizer.convert_tokens_to_ids(self.recall_end_token)
        }

    def _set_special_token_ids(self):
        """直接设置特殊token IDs（假设tokenizer已经包含了正确的token）"""
        tokenizer = self._get_tokenizer()
        self.recall_start_id = tokenizer.convert_tokens_to_ids(self.recall_start_token)
        self.recall_end_id = tokenizer.convert_tokens_to_ids(self.recall_end_token)
        self.memory_pad_id = tokenizer.convert_tokens_to_ids("<|memory_pad|>")
        self.im_start_id = tokenizer.convert_tokens_to_ids(self.im_start_token)
        self.im_end_id = tokenizer.convert_tokens_to_ids(self.im_end_token)

        if self.recall_start_id == tokenizer.unk_token_id:
            raise ValueError(f"❌ {self.recall_start_token} token不存在于tokenizer中！")
        if self.recall_end_id == tokenizer.unk_token_id:
            raise ValueError(f"❌ {self.recall_end_token} token不存在于tokenizer中！")
        if self.memory_pad_id == tokenizer.unk_token_id:
            raise ValueError(f"❌ <|memory_pad|> token不存在于tokenizer中！")

        print(f"✅ 特殊token IDs设置完成: {self.recall_start_token}={self.recall_start_id}, {self.recall_end_token}={self.recall_end_id}, <|memory_pad|>={self.memory_pad_id}")

        # 设置special_token_ids供其他方法使用（包含所有特殊token）
        self.special_token_ids = {
            self.recall_start_token: self.recall_start_id,
            self.recall_end_token: self.recall_end_id,
            "<|memory_pad|>": self.memory_pad_id
        }
        
        # 设置可训练的特殊token（不包括<|memory_pad|>，因为它只是占位符）
        self.trainable_special_token_ids = {
            self.recall_start_token: self.recall_start_id,
            self.recall_end_token: self.recall_end_id
        }
    
    def _save_original_embeddings(self):
        """保存原始特殊token的embedding - 保持数据类型"""
        embedding_layer = self.base_model.get_input_embeddings()
        self.original_embeddings = {}
        
        for token, token_id in self.special_token_ids.items():
            # 保持原始数据类型
            self.original_embeddings[token] = embedding_layer.weight[token_id].clone().detach()
        
        print(f"📝 已保存 {len(self.original_embeddings)} 个特殊token的原始embedding")
        print(f"   原始embedding数据类型: {list(self.original_embeddings.values())[0].dtype}")
    
    def _setup_lora(self):
        """设置LoRA配置 - 修改为不保存整个embedding层"""
        print("⚡ 配置LoRA...")
        print(f"   LoRA参数: r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")

        if hasattr(self.base_model, "peft_config"):
            raise RuntimeError(
                "加载的基础模型仍包含LoRA/PEFT配置，请确认上一次训练输出目录已被清理后再重试"
            )
        
        # 确定target_modules
        if self.lora_target_modules is None:
            # 默认配置：所有模块
            target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        else:
            # 使用传入的配置
            target_modules = self.lora_target_modules
        
        print(f"   LoRA目标模块: {target_modules}")
        print(f"   模块数量: {len(target_modules)} (默认7个，当前{len(target_modules)}个)")
        if len(target_modules) < 7:
            reduction = (1 - len(target_modules) / 7) * 100
            print(f"   ⚡ LoRA参数减少约 {reduction:.1f}%，显存占用相应减少")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules
            # 移除 modules_to_save=["embed_tokens"]
        )
        
        self.base_model = get_peft_model(self.base_model, lora_config)
        print(f"✅ LoRA配置完成")
        
        # 再次检查数据类型和设备
        first_param = next(self.base_model.parameters())
        model_dtype = first_param.dtype
        model_device = first_param.device
        print(f"🔧 LoRA后模型数据类型: {model_dtype}, 设备: {model_device}")
        
        # 添加：只允许特殊token的embedding可训练
        self._freeze_embeddings_except_special_tokens()

    def _freeze_embeddings_except_special_tokens(self):
        """冻结除了特殊token以外的所有embedding参数 - 修复版"""
        print("🧊 冻结除特殊token外的所有embedding参数...")
        
        # 获取正确的embedding层 - 使用get_input_embeddings()方法（适用于所有模型类型）
        # 对于Qwen3-VL模型，这会自动找到正确的embedding层
        try:
            embedding_layer = self.base_model.get_input_embeddings()
        except AttributeError:
            # 如果get_input_embeddings()不存在，尝试其他方法
            print("⚠️ 无法通过get_input_embeddings()获取embedding层，尝试直接访问...")
            try:
                embedding_layer = self.base_model.model.model.embed_tokens
            except:
                print("❌ 无法找到embedding层！")
                return
            
        # 建立行级梯度掩码：仅允许特殊token（不含<|memory_pad|>）更新
        vocab_size = embedding_layer.weight.shape[0]
        row_mask = torch.zeros(vocab_size, dtype=torch.bool, device=embedding_layer.weight.device)
        for token_id in self.trainable_special_token_ids.values():
            row_mask[token_id] = True

        # 开启全局requires_grad，使用hook屏蔽非训练token的梯度
        embedding_layer.weight.requires_grad_(True)

        def _mask_grad(grad):
            if grad is None:
                return grad
            mask = row_mask.to(grad.device, dtype=grad.dtype).unsqueeze(-1)
            return grad * mask

        embedding_layer.weight.register_hook(_mask_grad)

        total_embedding_params = embedding_layer.weight.numel()
        trainable_params = sum(embedding_layer.weight[token_id].numel() for token_id in self.trainable_special_token_ids.values())
        is_trainable = all(row_mask[token_id].item() for token_id in self.trainable_special_token_ids.values())
        
        print(f"✅ embedding层设置完成:")
        print(f"   embedding层路径: {embedding_layer.__class__.__name__}")
        print(f"   总embedding参数: {total_embedding_params:,}")
        print(f"   可训练参数: {trainable_params:,} ({len(self.trainable_special_token_ids)} 个特殊token)")
        print(f"   可训练token: {list(self.trainable_special_token_ids.keys())}")
        print(f"   冻结参数: {total_embedding_params - trainable_params:,}")
        print(f"   特殊token embedding是否可训练: {is_trainable}")
        
        # 调试信息
        if not is_trainable:
            print("⚠️ 警告: 特殊token embedding无法设置为可训练！尝试备用方法...")
            # 备用方法
            for token, token_id in self.trainable_special_token_ids.items():
                param_pointer = embedding_layer.weight[token_id]
                param_pointer.requires_grad = True
                print(f"   {token} ID={token_id}: {param_pointer.requires_grad}")
    
    def _print_parameters(self):
        """显示可训练参数统计 - 更新为只统计特殊token embedding"""
        print("📊 参数统计 (仅特殊token embedding可训练):")
        
        # 获取正确的embedding层路径 - 使用get_input_embeddings()方法（适用于所有模型类型）
        try:
            embedding_layer = self.base_model.get_input_embeddings()
            # 只统计可训练的特殊token（不包括<|memory_pad|>）
            trainable_token_embeddings = [embedding_layer.weight[token_id] for token_id in self.trainable_special_token_ids.values()]
            # 所有特殊token用于显示状态
            all_special_token_embeddings = [embedding_layer.weight[token_id] for token_id in self.special_token_ids.values()]
        except AttributeError:
            # 如果get_input_embeddings()不存在，尝试其他方法
            try:
                embedding_layer = self.base_model.model.model.embed_tokens
                trainable_token_embeddings = [embedding_layer.weight[token_id] for token_id in self.trainable_special_token_ids.values()]
                all_special_token_embeddings = [embedding_layer.weight[token_id] for token_id in self.special_token_ids.values()]
            except Exception as e:
                print(f"⚠️ 无法获取embedding层: {e}")
                embedding_layer = None
                trainable_token_embeddings = []
                all_special_token_embeddings = []
        
        # 统计参数
        lora_params = 0
        embedding_params = 0
        other_params = 0
        
        # 检查embedding层是否可训练（只统计可训练的token）
        if embedding_layer is not None:
            is_trainable = all(emb.requires_grad for emb in trainable_token_embeddings)
            if is_trainable:
                embedding_params = sum(emb.numel() for emb in trainable_token_embeddings)
        
        # 统计所有参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    lora_params += param.numel()
                elif 'embed' in name.lower() and 'embed_tokens.weight' in name:
                    # 已在前面计算，不重复计算
                    pass
                else:
                    other_params += param.numel()
        
        total_trainable = lora_params + embedding_params + other_params
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {total_trainable:,}")
        print(f"     - LoRA参数: {lora_params:,}")
        print(f"     - 特殊Token Embedding参数: {embedding_params:,}")
        print(f"     - 其他参数: {other_params:,}")
        print(f"   可训练比例: {100 * total_trainable / total_params:.4f}%")
        
        # 检查特殊token的embedding状态
        if embedding_layer is not None and all_special_token_embeddings:
            print(f"\n🎯 特殊token状态:")
            for token, token_id in self.special_token_ids.items():
                special_token_embedding = embedding_layer.weight[token_id]
                is_trainable = token in self.trainable_special_token_ids
                trainable_mark = " (可训练)" if is_trainable else " (不可训练，占位符)"
                print(f"   {token} (ID={token_id}): requires_grad={special_token_embedding.requires_grad}{trainable_mark}")
                print(f"     范围: [{special_token_embedding.min().item():.6f}, {special_token_embedding.max().item():.6f}]")
    
    def load_data(self, pt_file_path):
        """加载训练数据"""
        print(f"📖 加载数据: {pt_file_path}")
        
        if not os.path.exists(pt_file_path):
            raise FileNotFoundError(f"数据文件不存在: {pt_file_path}")
        
        data = torch.load(pt_file_path, map_location='cpu')
        texts = data['texts']
        embeddings = data['embeddings']
        
        print(f"   文本数量: {len(texts)}")
        print(f"   表征向量形状: {embeddings.shape}")
        print(f"   原始embedding数据类型: {embeddings.dtype}")
        
        return texts, embeddings
    
    def create_dataloader(self, texts, embeddings, batch_size=2, shuffle=True, noise_std=0.01, sft_full_texts=None):
        """创建增强的数据加载器"""
        dataset = EnhancedTextMemoryDataset(
            texts,
            embeddings,
            self.tokenizer,
            self.base_model,
            max_length=self.dataset_max_length,
            noise_std=noise_std,
            is_main_process_fn=self.is_main_process,
            sft_full_texts=sft_full_texts,
            activation_prompts=self.activation_prompts,
            end_prompts=self.end_prompts,
            guide_text=self.guide_text,
        )
        # 让 Accelerator 接管 sampler/loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=enhanced_collate_fn
        )
        return loader, dataset
    
    def create_mixed_dataloader(
        self,
        memory_texts,
        memory_embeddings,
        sft_messages_list,
        batch_size=2,
        shuffle=True,
        noise_std=0.01,
        sft_full_texts=None,
        sft_message_source_indices=None,
        sft_full_source_indices=None
    ):
        """创建混合数据加载器（记忆条目+SFT数据）"""
        dataset = MixedMemorySFTDataset(
            memory_texts,
            memory_embeddings,
            sft_messages_list,
            self.tokenizer,
            self.base_model,
            max_length=self.dataset_max_length,
            noise_std=noise_std,
            is_main_process_fn=self.is_main_process,
            sft_full_texts=sft_full_texts,
            activation_prompts=self.activation_prompts,
            end_prompts=self.end_prompts,
            memory_ratio=0.5,  # 记忆条目占50%
            guide_text=self.guide_text,
            sft_message_source_indices=sft_message_source_indices,
            sft_full_source_indices=sft_full_source_indices
        )
        # 让 Accelerator 接管 sampler/loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=enhanced_collate_fn
        )
        return loader, dataset
    
    def train_epoch(self, dataloader, dataset, optimizer, epoch_idx=0):
        """训练一个epoch - 在开始时刷新数据配对"""
        # 每个epoch开始时刷新数据配对
        if self.is_main_process():
            print(f"\n🔄 Epoch {epoch_idx + 1} 数据刷新中...")
        dataset.refresh_epoch_data()
        # 分布式采样器设置epoch
        if self.ddp_enabled and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch_idx)
        
        self.model.train()
        total_loss = 0
        accumulation_step = 0

        progress_bar = tqdm(dataloader, desc="训练", disable=not self.is_main_process())

        for batch in progress_bar:
            input_ids = batch['input_ids']
            embeddings_to_insert = batch['embeddings_to_insert']
            embedding_positions = batch['embedding_positions']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            batch_info = batch.get('batch_info', {})
            has_sft = batch_info.get('has_sft', False)
            has_memory = batch_info.get('has_memory', False)

            # 前向传播：统一使用forward方法，模型内部会根据embeddings_to_insert是否为占位符自动判断
            # 对于SFT样本，embeddings_to_insert是全零占位符，模型会自动使用标准前向传播
            outputs = self.model(
                input_ids=input_ids,
                embeddings_to_insert=embeddings_to_insert,
                embedding_positions=embedding_positions,
                attention_mask=attention_mask,
                labels=labels,
                memory_pad_token_id=self.memory_pad_id
            )

            loss = outputs['loss']

            # 梯度累积：损失除以累积步数
            loss = loss / self.gradient_accumulation_steps

            # 反向传播
            self.accelerator.backward(loss)

            accumulation_step += 1

            # 每gradient_accumulation_steps步执行一次优化器步骤
            if accumulation_step % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 累积损失（注意：这里累积的是原始损失，不是除以累积步数的损失）
            total_loss += loss.item() * self.gradient_accumulation_steps

            if self.is_main_process():
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.6f}'
                })

        # 处理最后一个epoch中剩余的梯度累积
        if accumulation_step % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(dataloader)
        
        if self.is_main_process():
            print(f"   平均损失: {avg_loss:.6f}")

        return {
            'total_loss': avg_loss
        }
    
    def test_memory_recall(self, texts, embeddings, num_samples=5, max_new_tokens=300, sft_full_texts=None):
        """测试记忆回忆能力 - 使用与训练一致的数据构建形式（SFT完整文本作为上下文）"""
        
        if not self.is_main_process():
            return {"skipped_on_non_main_process": True}

        print(f"\n🧠 测试记忆回忆能力 (检测token ID)...")
        effective_sample_count = min(num_samples, len(texts))
        print(f"   测试样本数: {effective_sample_count}")
        print(f"   最大生成长度: {max_new_tokens}")
        if sft_full_texts:
            print(f"   使用SFT完整文本作为上下文: {len(sft_full_texts)} 条")
        
        import random
        test_indices = random.sample(range(len(texts)), effective_sample_count)
        
        tokenizer = self._get_tokenizer()
        recall_start_id = self.special_token_ids.get('<recall>')
        recall_id = self.special_token_ids.get('<|recall|>')
        recall_end_id = self.special_token_ids.get('</recall>')
        
        if recall_start_id is None:
            recall_start_id = tokenizer.convert_tokens_to_ids('<recall>')
        if recall_end_id is None:
            recall_end_id = tokenizer.convert_tokens_to_ids('</recall>')
        
        print(f"🔍 特殊token ID:")
        print(f"   <recall>: {recall_start_id}")
        if recall_id is not None:
            print(f"   <|recall|>: {recall_id}")
        print(f"   </recall>: {recall_end_id}")
        print(f"   EOS token: {tokenizer.eos_token_id}")
        
        self.merged_model.eval()

        gen_cfg = self.generation_config or {}
        cfg_max_new_tokens = gen_cfg.get("max_new_tokens")
        do_sample = gen_cfg.get("do_sample", True)
        temperature = gen_cfg.get("temperature", 1.0)
        top_p = gen_cfg.get("top_p", 0.95)
        top_k = gen_cfg.get("top_k", 20)
        repetition_penalty = gen_cfg.get("repetition_penalty", 1.0)

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id
        
        # 使用trainer中已设置的memory_pad_id
        memory_pad_id = self.memory_pad_id
        
        # 编码<recall> token
        recall_tokens = tokenizer(self.recall_start_token, add_special_tokens=False)['input_ids']
        recall_token_count = len(recall_tokens)
        
        # 为了测试模型在真实场景下的表现，测试时也应该有上下文
        # 使用和训练时一样的上下文处理方式：从SFT数据中随机选择并截断
        test_context_text = ""
        
        # 如果提供了SFT数据，使用和训练时一样的截断方式
        if sft_full_texts and len(sft_full_texts) > 0:
            import random
            # 取一个安全的embedding样本（tensor/list都可）
            def _pick_one_embedding(embs):
                try:
                    hidden_size = getattr(self.merged_model.config, "hidden_size", 4096)
                except Exception:
                    hidden_size = 4096
                if isinstance(embs, torch.Tensor):
                    if embs.numel() == 0:
                        return torch.zeros((1, hidden_size), device=embs.device)
                    return embs[:1]
                if isinstance(embs, (list, tuple)) and len(embs) > 0:
                    first = embs[0]
                    if isinstance(first, torch.Tensor):
                        if first.dim() == 1:
                            first = first.unsqueeze(0)
                        return first[:1]
                    try:
                        return torch.tensor(first, dtype=torch.float32).unsqueeze(0)
                    except Exception:
                        return torch.zeros((1, hidden_size))
                return torch.zeros((1, hidden_size))

            # 随机选择一个SFT数据
            sft_data = random.choice(sft_full_texts)
            # 使用和训练时一样的截断方法：_split_sft_at_thinking
            # 创建一个临时的dataset对象来使用这个方法
            temp_dataset_for_context = EnhancedTextMemoryDataset(
                texts[:1] if texts else ["dummy"],  # 只需要一个dummy text
                _pick_one_embedding(embeddings),  # 只需要一个dummy embedding
                self.tokenizer,
                self.merged_model,
                max_length=self.dataset_max_length,
                noise_std=0.0,
                is_main_process_fn=self.is_main_process,
                sft_full_texts=sft_full_texts,
                activation_prompts=self.activation_prompts,
                end_prompts=self.end_prompts,
                guide_text=self.guide_text,
            )
            # 使用和训练时一样的截断方法
            test_context_text, _ = temp_dataset_for_context._split_sft_at_thinking(sft_data)
        
        # 测试时使用固定的激活提示语（使用第一个，确保测试一致性）
        test_activation_prompt = self.activation_prompts[0].strip() if self.activation_prompts else ""
        
        # 编码上下文和激活提示语
        context_tokens = tokenizer(test_context_text, add_special_tokens=False)['input_ids'] if test_context_text else []
        activation_tokens = tokenizer(test_activation_prompt, add_special_tokens=False)['input_ids'] if test_activation_prompt else []
        
        # 构造核心输入序列：<recall> + <|memory_pad|>
        core_input_tokens = recall_tokens + [memory_pad_id]
        
        # 构造完整输入序列
        full_input_tokens = context_tokens + activation_tokens + core_input_tokens
        embedding_position = len(context_tokens) + len(activation_tokens) + recall_token_count
        
        print(f"📋 测试配置:")
        print(f"   上下文: {'有 (' + str(len(context_tokens)) + ' tokens)' if test_context_text else '无'}")
        print(f"   激活提示语: {test_activation_prompt if test_activation_prompt else '无'}")
        print(f"   注意: 结束提示语是训练时的目标，不是输入的一部分")
        print(f"   输入序列长度: {len(full_input_tokens)}")
        print(f"   Embedding插入位置: {embedding_position}")
        
        for i, idx in enumerate(test_indices):
            # 每次测试前彻底清理模型状态，确保测试独立
            self.merged_model.eval()
            
            # 清理模型内部状态（如果有DDP包装，需要访问base_model）
            base_model = self.merged_model.module if hasattr(self.merged_model, 'module') else self.merged_model
            if hasattr(base_model, 'reset_cache'):
                base_model.reset_cache()
            if hasattr(base_model, 'base_model') and hasattr(base_model.base_model, 'reset_cache'):
                base_model.base_model.reset_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            original_text = texts[idx]
            # 直接使用原始embedding，不通过dataset获取
            embedding_to_insert = embeddings[idx]
            # 确保样本embedding不共享引用
            if isinstance(embedding_to_insert, torch.Tensor):
                embedding_to_insert = embedding_to_insert.clone()
            
            print(f"\n{'='*80}")
            print(f"🧪 测试样本 {i+1}/{num_samples} (索引: {idx})")
            print(f"📝 原始文本: {original_text}")
            if test_context_text:
                print(f"📋 测试上下文: {test_context_text[:200]}..." if len(test_context_text) > 200 else f"📋 测试上下文: {test_context_text}")
            if test_activation_prompt:
                print(f"📋 激活提示语: {test_activation_prompt}")
            print(f"📋 期望生成: 记忆文本 + </recall> + 结束提示语")
            
            try:
                # 直接使用原始embedding构建测试输入（不通过dataset）
                sequence_tokens = torch.tensor(full_input_tokens, dtype=torch.long)
                
                # 构建输入embeddings（与训练时一致）
                embedding_layer = self.merged_model.get_input_embeddings()
                device = next(self.merged_model.parameters()).device
                
                # 确保embedding在正确的设备和数据类型上
                # 获取模型的数据类型
                model_dtype = next(self.merged_model.parameters()).dtype
                embedding_to_insert = embedding_to_insert.to(device).to(model_dtype)
                
                # 将token序列转换为embeddings
                sequence_tokens = sequence_tokens.to(device)
                token_embeddings = embedding_layer(sequence_tokens.unsqueeze(0))  # [1, seq_len, embed_dim]
                
                # 替换embedding placeholder位置的embedding为实际的记忆向量
                token_embeddings[0, embedding_position] = embedding_to_insert
                
                # 构建attention mask
                attention_mask = torch.ones(1, sequence_tokens.shape[0], device=device, dtype=torch.long)
                
                prefix_embeddings = token_embeddings
                prefix_attention_mask = attention_mask
                
                print(f"🚀 输入序列: [上下文] + <recall> + [记忆向量] (总长度: {sequence_tokens.shape[0]})")

                requested_max_new_tokens = max_new_tokens or self.test_max_new_tokens
                cfg_limit = cfg_max_new_tokens or requested_max_new_tokens
                effective_max_new_tokens = min(requested_max_new_tokens, cfg_limit, self.test_max_new_tokens)
                print(
                    f"🎯 开始生成（do_sample={do_sample}, temperature={temperature}, top_p={top_p}, "
                    f"top_k={top_k}, repetition_penalty={repetition_penalty}, max_new_tokens={effective_max_new_tokens})..."
                )

                with torch.no_grad():
                    # 确保每次生成都是独立的，不传入past_key_values
                    generate_kwargs = {
                        "inputs_embeds": prefix_embeddings,
                        "attention_mask": prefix_attention_mask,
                        "max_new_tokens": effective_max_new_tokens,
                        "pad_token_id": pad_token_id if pad_token_id is not None else eos_token_id,
                        "eos_token_id": eos_token_id,
                        "return_dict_in_generate": True,
                        "do_sample": do_sample,
                        "use_cache": self.test_use_cache,
                        "past_key_values": None,  # 明确设置为None，确保不使用之前的缓存
                    }

                    if do_sample:
                        generate_kwargs["temperature"] = temperature
                        generate_kwargs["top_p"] = top_p
                        if top_k is not None:
                            generate_kwargs["top_k"] = max(int(top_k), 0)
                    # else分支：do_sample=False时不需要设置temperature、top_p、top_k

                    if repetition_penalty and repetition_penalty != 1.0:
                        generate_kwargs["repetition_penalty"] = repetition_penalty

                    generated_output = self.merged_model.generate(**generate_kwargs)

                sequences = generated_output.sequences if hasattr(generated_output, "sequences") else generated_output
                generated_ids_full = sequences[0].tolist()
                prefix_len = prefix_embeddings.shape[1]
                generated_ids = generated_ids_full[prefix_len:] if len(generated_ids_full) > prefix_len else generated_ids_full

                if len(generated_ids) >= effective_max_new_tokens:
                    print(f"   ⚠️ 达到最大生成长度 {effective_max_new_tokens}")

                recall_end_count = generated_ids.count(recall_end_id) if recall_end_id is not None else 0
                recall_start_count = generated_ids.count(recall_start_id) if recall_start_id is not None else 0
                recall_mid_count = generated_ids.count(recall_id) if recall_id is not None else 0
                eos_count = generated_ids.count(eos_token_id) if eos_token_id is not None else 0

                print(f"\n📊 生成统计:")
                print(f"   总生成token数: {len(generated_ids)}")
                print(f"   生成的token ID列表: {generated_ids[:10]}..." if len(generated_ids) > 10 else f"   生成的token ID列表: {generated_ids}")
                print(f"   </recall> 出现次数: {recall_end_count}")
                print(f"   <recall> 出现次数: {recall_start_count}")
                if recall_id is not None:
                    print(f"   <|recall|> 出现次数: {recall_mid_count}")
                print(f"   EOS token 出现次数: {eos_count}")
                
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                generated_text_clean = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                print(f"\n📤 生成的完整文本（包含special tokens）：")
                print(f"     {generated_text}")
                
                print(f"\n🧹 生成的文本（移除special tokens）：")
                print(f"     {generated_text_clean}")
                
                if original_text in generated_text_clean:
                    print(f"🎯 生成文本包含完整原文")
                elif generated_text_clean in original_text:
                    print(f"🎯 生成文本是原文的一部分")
                elif len(generated_text_clean) > 0 and original_text.startswith(generated_text_clean[:50]):
                    print(f"🎯 生成文本与原文开头匹配")
                else:
                    print(f"❓ 生成文本与原文差异较大")
                    
            except Exception as e:
                print(f"❌ 生成过程出错: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # 清理所有可能存在的变量和缓存
                if 'sequence_tokens' in locals():
                    del sequence_tokens
                if 'embedding_to_insert' in locals():
                    del embedding_to_insert
                if 'token_embeddings' in locals():
                    del token_embeddings
                if 'prefix_embeddings' in locals():
                    del prefix_embeddings
                if 'prefix_attention_mask' in locals():
                    del prefix_attention_mask
                if 'generated_output' in locals():
                    del generated_output
                if 'generated_ids' in locals():
                    del generated_ids
                # 清理模型可能保留的内部状态
                if hasattr(self.merged_model, 'module'):
                    base_model = self.merged_model.module
                else:
                    base_model = self.merged_model
                
                if hasattr(base_model, 'reset_cache'):
                    base_model.reset_cache()
                if hasattr(base_model, 'base_model') and hasattr(base_model.base_model, 'reset_cache'):
                    base_model.base_model.reset_cache()
                
                # 清理显存缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # 确保所有CUDA操作完成
                
                # 清理Python变量引用
                import gc
                gc.collect()
        
        print(f"\n{'='*80}")
        print("🔍 观察以上token ID输出，特别注意:")
        print("   1. 是否生成了</recall> token ID")
        print("   2. token ID计数与解码文本是否一致")
        print("   3. 生成序列的完整性")
        
        return {"test_completed": True}
    
    def compare_embeddings(self):
        """比较训练前后特殊token embedding的变化"""
        print("\n🔍 分析特殊token embedding变化...")
        
        current_embedding_layer = self.merged_model.get_input_embeddings()
        results = {}
        
        for token, token_id in self.special_token_ids.items():
            original_emb = self.original_embeddings[token]
            current_emb = current_embedding_layer.weight[token_id]
            
            # 确保数据类型一致再计算
            if original_emb.dtype != current_emb.dtype:
                original_emb = original_emb.to(current_emb.dtype)
            
            change = torch.abs(current_emb - original_emb).mean().item()
            cosine_sim = nn.CosineSimilarity(dim=0)(current_emb, original_emb).item()
            
            results[token] = {
                'change': change,
                'cosine_similarity': cosine_sim,
                'before_range': (original_emb.min().item(), original_emb.max().item()),
                'after_range': (current_emb.min().item(), current_emb.max().item())
            }
            
            print(f"\n   📊 {token} (ID: {token_id}):")
            print(f"      平均变化: {change:.6f}")
            print(f"      余弦相似度: {cosine_sim:.6f}")
            print(f"      训练前范围: [{original_emb.min().item():.4f}, {original_emb.max().item():.4f}]")
            print(f"      训练后范围: [{current_emb.min().item():.4f}, {current_emb.max().item():.4f}]")
        
        return results
    
    def merge_and_save_model(self, save_path):
        """合并LoRA权重并保存模型"""
        if not self.is_main_process():
            return None
        print("🔄 合并LoRA权重...")

        merged_model = self.base_model.merge_and_unload()

        if os.path.isdir(save_path):
            print(f"🧹 清理已有的模型输出目录: {save_path}")
            shutil.rmtree(save_path)

        os.makedirs(save_path, exist_ok=True)

        # 由于PEFT状态可能混乱，尝试不同的保存方式
        try:
            # 首先尝试正常保存
            merged_model.save_pretrained(save_path)
        except Exception as e:
            print(f"⚠️ 正常保存失败: {e}，尝试备用保存方法...")
            # 如果正常保存失败，尝试禁用adapter相关的保存
            try:
                # 创建一个临时模型，移除所有PEFT相关属性
                import copy
                temp_model = copy.deepcopy(merged_model)
                # 移除可能导致问题的PEFT属性
                peft_attrs = ['peft_config', 'active_adapters', 'adapter_config']
                for attr in peft_attrs:
                    if hasattr(temp_model, attr):
                        delattr(temp_model, attr)

                temp_model.save_pretrained(save_path)
                print("✅ 使用备用方法保存成功")
            except Exception as e2:
                print(f"❌ 备用保存也失败: {e2}")
                raise e  # 抛出原始错误

        # 保存tokenizer/processor
        # 如果self.tokenizer是AutoProcessor，save_pretrained会保存所有组件（tokenizer、image_processor、video_processor等）
        # 如果self.tokenizer是AutoTokenizer，只保存tokenizer
        self.tokenizer.save_pretrained(save_path)

        # 如果self.tokenizer是AutoProcessor，已经保存了所有组件
        # 如果self.tokenizer是AutoTokenizer，需要确保processor的其他组件也被保存
        # 但为了安全，我们在training_service.py的_save_processor_to_path中会处理processor的完整保存
        print(f"✅ 合并后的模型已保存到: {save_path}")

        # 确保特殊token在词汇表中（调试用）
        tokenizer = self._get_tokenizer()
        print("🔍 保存时检查特殊token...")
        special_tokens = ["<recall>", "</recall>"]
        for token in special_tokens:
            if token in tokenizer.get_vocab():
                token_id = tokenizer.convert_tokens_to_ids(token)
                print(f"   ✅ {token} 存在 (ID: {token_id})")
            else:
                print(f"   ❌ {token} 不存在于词汇表中！")
                # 重新添加
                num_added = tokenizer.add_tokens([token], special_tokens=True)
                if num_added > 0:
                    print(f"   🔧 重新添加了 {token}")
                    # 重新保存
                    tokenizer.save_pretrained(save_path)

        # 确保特殊token在特殊token列表中
        if hasattr(tokenizer, 'special_tokens_map'):
            additional_special = tokenizer.special_tokens_map.get('additional_special_tokens', [])
            for token in special_tokens:
                if token not in additional_special:
                    print(f"   ⚠️ {token} 不在特殊token列表中，重新添加")
                    if hasattr(tokenizer, 'add_special_tokens'):
                        tokenizer.add_special_tokens({"additional_special_tokens": [token]})
                        # 重新保存
                        tokenizer.save_pretrained(save_path)

        # 保存merged_model引用供compare_embeddings()使用
        self.merged_model = merged_model

        return save_path  # 返回保存路径供后续使用

    def cleanup(self):
        """清理训练器创建的所有模型实例"""
        print("🧹 清理训练器模型实例...")

        try:
            # 清理merged_model（如果存在）
            if hasattr(self, 'merged_model') and self.merged_model is not None:
                try:
                    self.merged_model.cpu()
                except:
                    pass
                del self.merged_model
                self.merged_model = None

            # 清理LoRA包装模型
            if hasattr(self, 'model') and self.model is not None:
                try:
                    self.model.cpu()
                except:
                    pass
                del self.model
                self.model = None

            # 清理base_model（如果不是预加载的）
            if hasattr(self, 'base_model') and self.base_model is not None and not getattr(self, '_skip_model_loading', False):
                try:
                    self.base_model.cpu()
                except:
                    pass
                del self.base_model
                self.base_model = None

            # 清理tokenizer（如果不是预加载的）
            if hasattr(self, 'tokenizer') and self.tokenizer is not None and not getattr(self, '_skip_model_loading', False):
                del self.tokenizer
                self.tokenizer = None

            # 清理accelerator
            if hasattr(self, 'accelerator') and self.accelerator is not None:
                try:
                    self.accelerator.free_memory()
                except:
                    pass
                # 注意：accelerator实例本身通常不需要显式删除

            # 强制垃圾回收和显存清理
            import gc
            for _ in range(3):
                gc.collect()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            # 恢复原始CUDA_VISIBLE_DEVICES（如果在初始化时修改过）
            if hasattr(self, '_original_cuda_visible_devices') and self._original_cuda_visible_devices is not None:
                original_value = self._original_cuda_visible_devices
                os.environ['CUDA_VISIBLE_DEVICES'] = original_value
                print(f"恢复原始CUDA_VISIBLE_DEVICES: {original_value}")
            elif 'CUDA_VISIBLE_DEVICES' in os.environ and hasattr(self, '_original_cuda_visible_devices'):
                # 如果初始化时设置了环境变量，现在删除它
                del os.environ['CUDA_VISIBLE_DEVICES']
                print("删除CUDA_VISIBLE_DEVICES环境变量")

            print("✅ 训练器清理完成")

        except Exception as e:
            print(f"⚠️ 清理训练器时出现警告: {e}")

    def train(
        self,
        pt_file_path,
        num_epochs=20,
        batch_size=4,
        learning_rate=1e-4,
        noise_std=0.01,
        save_path="enhanced_memory_model",
        sft_full_texts=None,
        sft_messages_list=None,
        sft_full_source_indices=None,
        sft_message_source_indices=None
    ):
        """增强的训练流程 - 支持混合训练（记忆条目+SFT数据）"""
        
        if self.is_main_process():
            print(f"\n🚀 开始增强文本记忆训练（混合模式）")
        print(f"   数据文件: {pt_file_path}")
        print(f"   总训练轮数: {num_epochs}")
        print(f"   批次大小: {batch_size}")
        print(f"   学习率: {learning_rate}")
        print(f"   噪声标准差: {noise_std}")
        print(f"   保存路径: {save_path}")
        if sft_full_texts:
            print(f"   SFT完整文本数量: {len(sft_full_texts)}")
        if sft_messages_list:
            print(f"   SFT消息列表数量: {len(sft_messages_list)}")
        
        # 加载数据
        texts, embeddings = self.load_data(pt_file_path)
        
        if self.is_main_process():
            print(f"\n📊 训练数据:")
            print(f"   记忆条目数量: {len(texts)}")
        
        # 创建混合数据加载器（如果提供了SFT数据）
        if sft_messages_list and len(sft_messages_list) > 0:
            train_loader, dataset = self.create_mixed_dataloader(
                texts,
                embeddings,
                sft_messages_list,
                batch_size,
                True,
                noise_std,
                sft_full_texts=sft_full_texts,
                sft_message_source_indices=sft_message_source_indices,
                sft_full_source_indices=sft_full_source_indices
            )
        else:
            # 回退到原有的数据加载器
            train_loader, dataset = self.create_dataloader(
                texts, embeddings, batch_size, True, noise_std, sft_full_texts=sft_full_texts
            )
        
        # 优化器 - 确保包含特殊token embedding
        optimizer_params = []

        # 先加入所有已设置为可训练的参数
        optimizer_params.extend([p for p in self.base_model.parameters() if p.requires_grad])

        # 获取embedding层，确保特殊token包含在优化器中
        # 使用get_input_embeddings()方法（适用于所有模型类型）
        try:
            embedding_layer = self.base_model.get_input_embeddings()
        except AttributeError:
            # 如果get_input_embeddings()不存在，尝试其他方法
            try:
                embedding_layer = self.base_model.model.model.embed_tokens
            except:
                print("⚠️ 优化器创建时无法找到embedding层")
                embedding_layer = None

        if embedding_layer is not None:
            # 检查可训练的特殊token是否已设置为可训练（不包括<|memory_pad|>）
            for token, token_id in self.trainable_special_token_ids.items():
                special_token_embedding = embedding_layer.weight[token_id]
                if not special_token_embedding.requires_grad:
                    print(f"⚠️ {token} embedding未设置为可训练，手动添加到优化器...")
                    special_token_embedding.requires_grad_(True)
                    # 如果不在optimizer_params中，添加它
                    if all(id(special_token_embedding) != id(p) for p in optimizer_params):
                        optimizer_params.append(special_token_embedding)

        optimizer = optim.AdamW(
            optimizer_params,
            lr=learning_rate,
            weight_decay=0.01
        )
        # 让 Accelerator 接管（模型/优化器/数据）
        self.model, optimizer, train_loader = self.accelerator.prepare(
            self.model, optimizer, train_loader
        )
        
        # 训练循环
        training_history = {
            'total_loss': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 每个epoch开始时，打印一个训练样本以供检查
            if self.is_main_process() and len(dataset) > 0:
                try:
                    # 优先找到记忆条目样本（不是SFT样本）
                    sample_idx = 0
                    sample = None
                    for i in range(min(10, len(dataset))):  # 最多检查前10个样本
                        candidate = dataset[i]
                        if not candidate.get('is_sft', False):
                            sample = candidate
                            sample_idx = i
                            break
                    if sample is None:
                        # 如果前10个都是SFT样本，就使用第一个
                        sample_idx = 0
                        sample = dataset[sample_idx]
                    
                    sample_type = sample.get('sample_type', 'memory')
                    type_display = {
                        'memory_front': '记忆条目（前置SFT）',
                        'memory_full': '记忆条目（前后拼接SFT）',
                        'sft_only': 'SFT样本',
                        'memory': '记忆条目'
                    }.get(sample_type, sample_type)
                    context_text = sample.get('context_text', '')
                    memory_text = sample.get('text', '')
                    activation_prompt = sample.get('activation_prompt', '')
                    end_prompt = sample.get('end_prompt', '')
                    is_sft = sample.get('is_sft', False)
                    
                    print(f"\n📋 Epoch {epoch+1} 训练样本示例（索引 {sample_idx}，类型: {type_display}）:")
                    if not is_sft:
                        print(f"   上下文（截断的SFT文本）: {context_text[:200]}..." if len(context_text) > 200 else f"   上下文（截断的SFT文本）: {context_text}")
                        print(f"   ─────────────────────────────────────────────────────────────────")
                        print(f"   记忆激活引导: {activation_prompt if activation_prompt else '(空)'}")
                        print(f"   回忆结束引导: {end_prompt if end_prompt else '(空)'}")
                        print(f"   ─────────────────────────────────────────────────────────────────")
                        print(f"   记忆文本: {memory_text[:200]}..." if len(memory_text) > 200 else f"   记忆文本: {memory_text}")
                    else:
                        print(f"   SFT样本文本: {memory_text[:200]}..." if len(memory_text) > 200 else f"   SFT样本文本: {memory_text}")
                    print(f"   ─────────────────────────────────────────────────────────────────")
                    # 显示完整的训练样本：输入序列 + 目标序列（包含所有特殊token）
                    tokenizer = self._get_tokenizer()

                    # 获取输入序列（上下文 + <recall> + <|memory_pad|>）
                    input_tokens = sample.get('sequence_tokens')
                    # 获取标签序列（-100 * 输入长度 + 目标文本token）
                    labels = sample.get('labels')

                    if input_tokens is not None and labels is not None:
                        # 将tensor转换为list
                        if isinstance(input_tokens, torch.Tensor):
                            input_tokens = input_tokens.cpu().tolist()
                        if isinstance(labels, torch.Tensor):
                            labels = labels.cpu().tolist()

                        recall_token_count = sample.get('recall_token_count', 1)
                        # 计算<recall>在labels中的位置：context + activation_prompt之后
                        context_length = sample.get('context_length', 0)
                        activation_prompt = sample.get('activation_prompt', '')
                        tokenizer = self._get_tokenizer()
                        activation_tokens = tokenizer(activation_prompt, add_special_tokens=False)['input_ids'] if activation_prompt else []
                        recall_label_start = context_length + len(activation_tokens)
                        prefix_len = len(input_tokens)  # 输入序列长度（包括<recall>和<|memory_pad|>）
                        
                        # 显示位置信息
                        embedding_position = sample.get('embedding_position', 0)
                        recall_position_in_input = context_length + len(activation_tokens)
                        print(f"   <recall>在输入序列中的位置: {recall_position_in_input}")
                        print(f"   记忆向量插入位置: {embedding_position}")
                        print(f"   位置关系: {'✅ 记忆向量在<recall>之后' if embedding_position >= recall_position_in_input + recall_token_count else '⚠️ 位置异常'}")

                        # 构造完整序列：正确区分输入序列和目标序列
                        # 数据结构：
                        # - 输入序列 (input_tokens): [context] [activation] <recall> <|memory_pad|>
                        # - 标签序列 (labels): [-100...] [<recall>的ID] [-100] [memory_text] </recall> [end_prompt]
                        # 打印时应该：
                        # - 输入部分（i < prefix_len）：使用input_tokens（包括<recall>和<|memory_pad|>）
                        # - 目标部分（i >= prefix_len）：使用labels（从memory_text开始）
                        full_sequence = []
                        for i in range(len(labels)):
                            if i < prefix_len:
                                # 输入序列部分：始终使用input_tokens（即使labels[i]不是-100，如<recall>位置）
                                full_sequence.append(input_tokens[i])
                            else:
                                # 目标序列部分：使用labels（这些是memory_text + </recall> + end_prompt）
                                full_sequence.append(labels[i])

                        # 解码完整序列
                        decoded_sample = tokenizer.decode(full_sequence, skip_special_tokens=False)

                        # 显示完整样本，但限制总长度以避免输出过长
                        max_display_len = 800
                        if len(decoded_sample) > max_display_len:
                            # 显示开头和结尾各一半
                            half_len = max_display_len // 2
                            preview = decoded_sample[:half_len] + f"\n...[中间省略{len(decoded_sample) - max_display_len}字符]...\n" + decoded_sample[-half_len:]
                        else:
                            preview = decoded_sample
                        print(f"   完整训练样本 ({len(decoded_sample)}字符):")
                        print(f"   {preview}")
                    else:
                        print(f"   ⚠️ 无法获取sequence_tokens或labels，跳过完整样本显示")
                except Exception as e:
                    print(f"⚠️ 打印训练样本失败: {e}")
            
            # 训练一个epoch - 数据集会在train_epoch内刷新
            epoch_results = self.train_epoch(train_loader, dataset, optimizer, epoch_idx=epoch)
            
            # 记录历史
            training_history['total_loss'].append(epoch_results['total_loss'])
            
            # 注意：如果使用混合数据集，SFT数据已经包含在训练中，不需要epoch_end_hook
            # 如果使用传统数据集，仍然可以调用epoch_end_hook（但通常不需要）
            # 这里保留hook调用以保持向后兼容，但混合训练模式下不会使用
            if not isinstance(dataset, MixedMemorySFTDataset):
                try:
                    if callable(self.epoch_end_hook):
                        self.epoch_end_hook(epoch, self)
                except Exception as hook_err:
                    if self.is_main_process():
                        print(f"⚠️ epoch_end_hook 执行失败但已忽略: {hook_err}")
            
            # # 每5个epoch保存一次模型
            # if (epoch + 1) % 5 == 0:
            #     print(f"🔄 已完成 {epoch+1}/{num_epochs} epochs")
                
            #     # 保存检查点
            #     checkpoint_path = f"{save_path}_checkpoint_{epoch+1}"
            #     os.makedirs(checkpoint_path, exist_ok=True)
            #     self.base_model.save_pretrained(checkpoint_path)
            #     self.tokenizer.save_pretrained(checkpoint_path)
            #     print(f"✅ 检查点已保存到: {checkpoint_path}")
        
        # 合并并保存模型
        if self.is_main_process():
            print(f"\n📦 训练完成，保存最终模型...")
            final_model = self.merge_and_save_model(save_path)
        else:
            final_model = None
        
        # 分析embedding变化
        embedding_analysis = self.compare_embeddings() if self.is_main_process() else {}
        
        # 测试记忆回忆能力
        if self.is_main_process():
            print(f"\n🧠 开始测试记忆回忆能力...")
        test_results = self.test_memory_recall(
            texts,
            embeddings,
            num_samples=self.test_sample_count,
            max_new_tokens=self.test_max_new_tokens,
            sft_full_texts=sft_full_texts
        )
        
        # 保存结果
        results = {
            'training_history': training_history,
            'embedding_analysis': embedding_analysis,
            'memory_test_results': test_results,
            'total_epochs': num_epochs,
            'final_loss': epoch_results['total_loss'],
            'training_config': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'noise_std': noise_std
            }
        }
        
        if self.is_main_process():
            with open(f"{save_path}/training_results.json", 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        if self.is_main_process():
            print(f"\n🎉 增强文本记忆训练完成！")
            print(f"   总训练轮数: {num_epochs}")
            print(f"   最终总体损失: {epoch_results['total_loss']:.6f}")
            print(f"   记忆文本总数: {len(texts)}")
            print(f"   模型保存路径: {save_path}")
            print(f"   测试结果已保存到: {save_path}/training_results.json")
        
        return results

    def expose_training_handles(self):
        """暴露训练句柄，供外部SFT复用LoRA模型"""
        return {
            "model": self.base_model,
            "tokenizer": self.tokenizer,
            "accelerator": getattr(self, "accelerator", None)
        }

def main():
    """主函数 - 增强训练"""
    
    # 配置参数
    MODEL_NAME = "./Qwen2.5-7B-Instruct-with-special-tokens-embedding-trained"
    PT_FILE_PATH = "datasets/embeddings/text_embeddings.pt"
    
    # 训练参数
    NUM_EPOCHS = 20      # 总训练轮数
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NOISE_STD = 0.0
    SAVE_PATH = "Qwen2.5-7B-Instruct-with-special-tokens-memory-trained"
    
    # 指定设备
    DEVICE = "cuda:0"
    
    print("🚀 增强文本记忆训练程序")
    print("=" * 70)
    print(f"模型: {MODEL_NAME}")
    print(f"数据: {PT_FILE_PATH}")
    print(f"设备: {DEVICE}")
    print(f"训练方式: 双上下文记忆训练 ({NUM_EPOCHS}轮)")
    print("=" * 70)
    
    # 检查文件
    if not os.path.exists(PT_FILE_PATH):
        print(f"❌ 数据文件不存在: {PT_FILE_PATH}")
        return
    
    if not os.path.exists(MODEL_NAME):
        print(f"❌ 模型路径不存在: {MODEL_NAME}")
        return
    
    try:
        # 初始化训练器，传递设备参数
        trainer = EnhancedTextMemoryTrainer(model_name=MODEL_NAME, device=DEVICE)
        
        # 开始训练
        results = trainer.train(
            pt_file_path=PT_FILE_PATH,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            noise_std=NOISE_STD,
            save_path=SAVE_PATH
        )
        
        if trainer.is_main_process():
            print("\n✅ 训练流程完成！")
        
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'LOCAL_RANK' in os.environ and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
