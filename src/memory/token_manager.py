# -*- coding: utf-8 -*-
"""
记忆相关特殊token管理器
用于添加和管理<recall>和</recall>特殊token
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

_log = logging.getLogger(__name__)


class MemoryTokenManager:
    """记忆相关特殊token管理器"""
    
    def __init__(self, model, tokenizer):
        """
        初始化token管理器
        
        Args:
            model: 已加载的模型
            tokenizer: 已加载的分词器（可能是processor或tokenizer）
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # 获取真正的tokenizer（如果传入的是processor）
        if hasattr(tokenizer, 'tokenizer'):
            self._actual_tokenizer = tokenizer.tokenizer
        else:
            self._actual_tokenizer = tokenizer
        
        # 要添加的特殊token（包含<recall>、</recall>和<|memory_pad|>）
        self.special_tokens = ["<recall>", "</recall>", "<|memory_pad|>"]
        
        # 参考token映射（用于初始化权重）
        # 每个token可以有多个参考token，按优先级顺序尝试，直到找到一个存在的
        # <recall>: 优先使用"回忆"，备选"总结"、"回想"、"记忆"等
        # </recall>: 优先使用"结束"，备选"完成"、"终止"、"完毕"等
        # <|memory_pad|>: 不使用参考token，使用平均初始化并缩小范数（只是占位符，不需要训练）
        self.reference_tokens = {
            "<recall>": ["回忆", "总结", "回想", "记忆", "回顾", "想起"],
            "</recall>": ["结束", "完成", "终止", "完毕", "完结", "停止"],
            "<|memory_pad|>": []  # 空列表表示不使用参考token，使用特殊初始化
        }
    
    def check_and_add_tokens(self, perturbation_std=0.1):
        """
        检查并添加特殊token（如果不存在）
        
        Args:
            perturbation_std: 初始化权重时的扰动标准差（默认0.1，较大的扰动）
        
        Returns:
            dict: token_id映射，如 {"<recall>": 123456, "</recall>": 123457}
        """
        _log.info("检查特殊token...")
        
        # 检查哪些token已存在，哪些需要添加
        tokens_to_add = []
        existing_token_ids = {}
        
        for token in self.special_tokens:
            token_id = self._actual_tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == self._actual_tokenizer.unk_token_id:
                tokens_to_add.append(token)
                _log.info(f"  {token} 不存在，需要添加")
            else:
                existing_token_ids[token] = token_id
                _log.info(f"  {token} 已存在 (ID: {token_id})")
        
        # 如果没有需要添加的token，直接返回
        if not tokens_to_add:
            _log.info("所有特殊token已存在，无需添加")
            return existing_token_ids
        
        # 添加新token
        _log.info(f"添加 {len(tokens_to_add)} 个新token...")
        original_vocab_size = len(self._actual_tokenizer)
        
        # 将新token注册为真正的特殊token，确保保存后可被重新加载
        additional_specials = list(self._actual_tokenizer.special_tokens_map.get("additional_special_tokens", [])) if hasattr(self._actual_tokenizer, "special_tokens_map") else []
        updated_specials = []
        for token in self.special_tokens:
            if token in tokens_to_add or token in additional_specials:
                if token not in updated_specials:
                    updated_specials.append(token)
        if updated_specials:
            self._actual_tokenizer.add_special_tokens({"additional_special_tokens": updated_specials}, replace_additional_special_tokens=True)
        
        new_vocab_size = len(self._actual_tokenizer)
        _log.info(f"词表大小: {original_vocab_size} -> {new_vocab_size} (+{new_vocab_size - original_vocab_size})")
        
        # 调整模型embedding层
        _log.info("调整模型embedding层...")
        self.model.resize_token_embeddings(len(self._actual_tokenizer))
        
        # 验证embedding层和输出层的大小是否正确调整
        input_embeddings = self.model.get_input_embeddings()
        input_emb_size = input_embeddings.weight.shape[0]
        _log.info(f"✅ Input embeddings大小: {input_emb_size} (期望: {len(self._actual_tokenizer)})")
        
        # 检查输出层（lm_head）
        output_embeddings = None
        if hasattr(self.model, 'lm_head'):
            output_embeddings = self.model.lm_head
        elif hasattr(self.model, 'get_output_embeddings'):
            output_embeddings = self.model.get_output_embeddings()
        
        if output_embeddings is not None:
            output_emb_size = output_embeddings.weight.shape[0]
            _log.info(f"✅ Output embeddings (lm_head)大小: {output_emb_size} (期望: {len(self._actual_tokenizer)})")
            
            # 检查input和output embeddings是否绑定（tied）
            if input_embeddings.weight.data_ptr() == output_embeddings.weight.data_ptr():
                _log.info("ℹ️ Input和Output embeddings是绑定的（tied），只需调整一个即可")
            else:
                _log.info("ℹ️ Input和Output embeddings是独立的，两者都已调整")
        else:
            _log.warning("⚠️ 模型没有输出层（lm_head），可能无法生成新token")
        
        # 获取所有token的ID（包括新添加的）
        token_ids = {}
        for token in self.special_tokens:
            token_ids[token] = self._actual_tokenizer.convert_tokens_to_ids(token)
        
        _log.info(f"特殊token IDs: {token_ids}")
        
        # 验证新token的ID是否在有效范围内
        for token, token_id in token_ids.items():
            if token_id is None or token_id == self._actual_tokenizer.unk_token_id:
                _log.error(f"❌ 错误：token '{token}' 的ID无效: {token_id}")
            elif token_id >= input_emb_size:
                _log.error(f"❌ 错误：token '{token}' 的ID ({token_id}) 超出embedding范围 ({input_emb_size})")
            else:
                _log.info(f"✅ Token '{token}' ID验证通过: {token_id} (在范围内: 0-{input_emb_size-1})")
        
        # 初始化新添加token的权重
        if tokens_to_add:
            self._initialize_token_weights(token_ids, tokens_to_add, perturbation_std)
            # 如果添加了token，保存模型
            self._save_model_with_tokens()
        
        return token_ids
    
    def _save_model_with_tokens(self):
        """
        保存添加了token的模型到指定目录
        """
        import os
        import shutil
        from datetime import datetime
        
        # 获取配置中的token_added_model_dir
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config_qwen3vl.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                token_added_model_dir = config.get("memory", {}).get("training", {}).get("token_added_model_dir", "./models/token_added")
            else:
                token_added_model_dir = "./models/token_added"
        except Exception as e:
            _log.warning(f"读取配置失败，使用默认路径: {e}")
            token_added_model_dir = "./models/token_added"
        
        # 转换为绝对路径
        if not os.path.isabs(token_added_model_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            token_added_model_dir = os.path.abspath(os.path.join(project_root, token_added_model_dir))
        
        # 创建目录
        os.makedirs(token_added_model_dir, exist_ok=True)
        
        # 检查是否已存在添加了token的模型
        existing_models = [d for d in os.listdir(token_added_model_dir) 
                          if os.path.isdir(os.path.join(token_added_model_dir, d)) and d.startswith("model_")]
        
        if existing_models:
            # 如果已存在，使用现有的模型路径
            existing_models.sort(reverse=True)
            existing_model_path = os.path.join(token_added_model_dir, existing_models[0])
            _log.info(f"✅ 已存在添加了token的模型: {existing_model_path}")
            return existing_model_path
        
        # 创建新的模型目录（使用时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir_name = f"model_{timestamp}"
        save_path = os.path.join(token_added_model_dir, model_dir_name)
        
        _log.info(f"💾 保存添加了token的模型到: {save_path}")
        
        try:
            # 保存模型
            self.model.save_pretrained(save_path)
            # 保存tokenizer
            self._actual_tokenizer.save_pretrained(save_path)
            
            # 保存完整的processor配置（包含image_processor、video_processor等所有组件）
            # 从基础模型加载完整的processor，然后更新tokenizer为添加了特殊token的版本
            try:
                from transformers import AutoProcessor
                # 获取基础模型路径
                base_model_path = getattr(self.model.config, "_name_or_path", None)
                if not base_model_path:
                    _log.warning("无法确定基础模型路径，跳过processor保存")
                else:
                    if not os.path.isabs(base_model_path):
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        project_root = os.path.dirname(os.path.dirname(script_dir))
                        base_model_path = os.path.abspath(os.path.join(project_root, base_model_path))
                    
                    if os.path.isdir(base_model_path):
                        # 从基础模型加载完整的processor
                        base_processor = AutoProcessor.from_pretrained(
                            base_model_path,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                        # 更新processor的tokenizer为添加了特殊token的版本
                        base_processor.tokenizer = self._actual_tokenizer

                        # 保存完整的processor配置
                        base_processor.save_pretrained(save_path)
                        _log.info(f"✅ 已保存完整Processor配置到: {save_path}")

                        # 确保所有必要的配置文件都被正确保存（在save_pretrained之后，确保不被覆盖）
                        # 这些文件对于Qwen3VLProcessor的正确工作至关重要
                        import shutil
                        essential_files = [
                            "chat_template.json",
                            "preprocessor_config.json",
                            "video_preprocessor_config.json"
                        ]
                        for file_name in essential_files:
                            source_file = os.path.join(base_model_path, file_name)
                            target_file = os.path.join(save_path, file_name)
                            if os.path.exists(source_file):
                                try:
                                    shutil.copy2(source_file, target_file)
                                    _log.info(f"✅ 已复制{file_name}到: {save_path}")
                                except Exception as e:
                                    _log.warning(f"⚠️ 复制{file_name}失败: {e}")
                            else:
                                _log.warning(f"⚠️ 基础模型中不存在{file_name}，跳过复制")
                    else:
                        _log.warning(f"基础模型路径不存在: {base_model_path}，跳过processor保存")
            except Exception as proc_e:
                _log.warning(f"⚠️ 保存Processor配置失败: {proc_e}，将尝试复制文件")
                # 如果保存processor失败，至少复制额外的文件
                self._copy_additional_files(save_path)
            
            _log.info(f"✅ 模型和tokenizer已保存到: {save_path}")
            return save_path
        except Exception as e:
            _log.error(f"❌ 保存模型失败: {e}", exc_info=True)
            return None
    
    def _copy_additional_files(self, target_path: str):
        """
        将基础模型中的额外文件（chat_template.json 等）复制到新目录
        """
        import shutil
        import os
        
        source_path = getattr(self.model.config, "_name_or_path", None)
        if not source_path:
            _log.warning("无法确定基础模型路径，跳过额外文件复制")
            return
        
        if not os.path.isabs(source_path):
            # 尝试将相对路径转为绝对路径（相对于项目根目录）
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            candidate = os.path.abspath(os.path.join(project_root, source_path))
            if os.path.isdir(candidate):
                source_path = candidate
        if not os.path.isdir(source_path):
            _log.warning(f"基础模型路径不存在，跳过额外文件复制: {source_path}")
            return
        
        extra_files = [
            "chat_template.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "README.md"
        ]
        
        for filename in extra_files:
            src_file = os.path.join(source_path, filename)
            dst_file = os.path.join(target_path, filename)
            if os.path.exists(src_file):
                try:
                    shutil.copy2(src_file, dst_file)
                    _log.info(f"  ✅ 已复制 {filename} 到 {target_path}")
                except Exception as e:
                    _log.warning(f"  ⚠️ 复制 {filename} 失败: {e}")
    
    def _initialize_token_weights(self, token_ids, tokens_to_add, perturbation_std):
        """
        使用参考token的embedding初始化权重，并添加较大扰动
        
        策略：使用语义相近的token embedding作为基础，然后添加随机扰动
        - <recall>: 使用"总结"token的embedding + 扰动
        - </recall>: 使用"结束"token的embedding + 扰动
        
        Args:
            token_ids: 所有特殊token的ID映射
            tokens_to_add: 需要初始化权重的token列表
            perturbation_std: 扰动标准差（默认0.1，较大的扰动）
        """
        _log.info(f"初始化token权重（使用参考token embedding + 扰动，扰动标准差={perturbation_std}）...")
        
        # 获取embedding层和输出层
        embedding_layer = self.model.get_input_embeddings()
        
        # 尝试获取输出层（可能是lm_head或通过get_output_embeddings）
        output_layer = None
        if hasattr(self.model, 'lm_head'):
            output_layer = self.model.lm_head
        elif hasattr(self.model, 'get_output_embeddings'):
            output_layer = self.model.get_output_embeddings()
        
        if output_layer is None:
            _log.warning("模型没有输出层（lm_head），只初始化embedding层")
        
        # 获取模型设备
        model_device = next(self.model.parameters()).device
        
        # 计算所有现有token embedding的平均值（用于参考token不存在时的初始化）
        old_vocab_size = len(self._actual_tokenizer) - len(tokens_to_add)
        avg_embedding = None
        avg_output = None
        
        try:
            # 计算旧词汇表的平均embedding
            avg_embedding = embedding_layer.weight.data[:old_vocab_size].mean(dim=0, keepdim=False)
            _log.info(f"计算得到平均embedding，范数={avg_embedding.norm().item():.4f}")
            
            # 计算旧词汇表的平均output embedding
            if output_layer is not None:
                avg_output = output_layer.weight.data[:old_vocab_size].mean(dim=0, keepdim=False)
                _log.info(f"计算得到平均output embedding，范数={avg_output.norm().item():.4f}")
        except Exception as e:
            _log.warning(f"计算平均embedding失败: {e}，将使用随机初始化")
        
        with torch.no_grad():
            for target_token in tokens_to_add:
                target_id = token_ids[target_token]
                
                # 获取参考token列表
                ref_tokens = self.reference_tokens.get(target_token, [])
                
                # 特殊处理：<|memory_pad|>使用平均初始化并缩小范数（只是占位符，不需要训练）
                if target_token == "<|memory_pad|>":
                    _log.info(f"  🔧 {target_token} 使用占位符初始化（平均初始化 + 缩小范数）")
                    if avg_embedding is not None:
                        embedding_dim = avg_embedding.size(0)
                        # 使用平均embedding，但缩小范数到原来的0.1倍（很小的范数）
                        embedding_vec = avg_embedding.clone() * 0.1
                        embedding_layer.weight.data[target_id] = embedding_vec
                        _log.info(f"    ✅ Input embedding初始化完成，范数={embedding_vec.norm().item():.4f} (原始平均范数的10%)")
                    else:
                        # 如果平均embedding计算失败，使用很小的随机初始化
                        embedding_dim = embedding_layer.weight.size(1)
                        embedding_vec = torch.randn(embedding_dim, device=model_device) * 0.01  # 很小的初始化
                        embedding_layer.weight.data[target_id] = embedding_vec
                        _log.info(f"    ✅ Input embedding随机初始化完成，范数={embedding_vec.norm().item():.4f}")
                    
                    if output_layer is not None:
                        if avg_output is not None:
                            out_dim = avg_output.size(0)
                            # 同样缩小范数
                            output_vec = avg_output.clone() * 0.1
                            output_layer.weight.data[target_id] = output_vec
                            _log.info(f"    ✅ Output embedding初始化完成，范数={output_vec.norm().item():.4f} (原始平均范数的10%)")
                        else:
                            out_dim = output_layer.weight.shape[1]
                            output_vec = torch.randn(out_dim, device=model_device) * 0.01
                            output_layer.weight.data[target_id] = output_vec
                            _log.info(f"    ✅ Output embedding随机初始化完成，范数={output_vec.norm().item():.4f}")
                    
                    _log.info(f"  ✅ {target_token} (ID: {target_id}) 占位符初始化完成（不需要训练）")
                    continue
                
                if not ref_tokens:
                    _log.warning(f"  ⚠️ {target_token} 没有参考token，使用平均初始化")
                    # 使用平均初始化
                    if avg_embedding is not None:
                        embedding_dim = avg_embedding.size(0)
                        perturbation = torch.randn(embedding_dim, device=model_device) * perturbation_std
                        embedding_vec = avg_embedding.clone() + perturbation
                        embedding_layer.weight.data[target_id] = embedding_vec
                    else:
                        # 如果平均embedding计算失败，使用随机初始化
                        embedding_dim = embedding_layer.weight.size(1)
                        init_std = getattr(getattr(self.model, "config", None), "initializer_range", 0.02)
                        embedding_vec = torch.randn(embedding_dim, device=model_device) * init_std
                        embedding_layer.weight.data[target_id] = embedding_vec
                    
                    if output_layer is not None:
                        if avg_output is not None:
                            out_dim = avg_output.size(0)
                            output_perturbation = torch.randn(out_dim, device=model_device) * perturbation_std
                            output_vec = avg_output.clone() + output_perturbation
                            output_layer.weight.data[target_id] = output_vec
                        else:
                            out_dim = output_layer.weight.shape[1]
                            init_std = getattr(getattr(self.model, "config", None), "initializer_range", 0.02)
                            output_vec = torch.randn(out_dim, device=model_device) * init_std
                            output_layer.weight.data[target_id] = output_vec
                    
                    init_method = "平均初始化" if avg_embedding is not None else "随机初始化"
                    _log.info(f"  ✅ {target_token} (ID: {target_id}) {init_method}完成")
                    continue
                
                # 尝试多个参考token，收集所有找到的token ID
                found_token_ids = []
                found_tokens = []
                
                for candidate_token in ref_tokens:
                    candidate_id = None
                    
                    # 方法1: 尝试直接convert_tokens_to_ids
                    try:
                        candidate_id = self._actual_tokenizer.convert_tokens_to_ids(candidate_token)
                        if candidate_id is not None and candidate_id != self._actual_tokenizer.unk_token_id:
                            found_token_ids.append(candidate_id)
                            found_tokens.append(candidate_token)
                            _log.info(f"  ✅ 找到参考token: '{candidate_token}' (ID: {candidate_id}) [方法: convert_tokens_to_ids]")
                            continue
                    except Exception as e:
                        _log.debug(f"  convert_tokens_to_ids失败: {e}")
                    
                    # 方法2: 尝试encode然后取第一个token（对于中文，可能被tokenize成多个token）
                    if candidate_id is None or candidate_id == self._actual_tokenizer.unk_token_id:
                        try:
                            encoded = self._actual_tokenizer.encode(candidate_token, add_special_tokens=False)
                            if encoded and len(encoded) > 0:
                                # 对于中文token，可能被tokenize成多个token，我们使用第一个
                                candidate_id = encoded[0]
                                # 验证这个ID不是unk_token_id
                                if candidate_id != self._actual_tokenizer.unk_token_id:
                                    found_token_ids.append(candidate_id)
                                    found_tokens.append(candidate_token)
                                    decoded = self._actual_tokenizer.decode([candidate_id])
                                    _log.info(f"  ✅ 找到参考token: '{candidate_token}' (ID: {candidate_id}, 解码: '{decoded}') [方法: encode]")
                                    continue
                        except Exception as e:
                            _log.debug(f"  encode失败: {e}")
                    
                    # 方法3: 尝试通过vocab直接查找
                    if candidate_id is None or candidate_id == self._actual_tokenizer.unk_token_id:
                        try:
                            vocab = self._actual_tokenizer.get_vocab()
                            if candidate_token in vocab:
                                candidate_id = vocab[candidate_token]
                                if candidate_id != self._actual_tokenizer.unk_token_id:
                                    found_token_ids.append(candidate_id)
                                    found_tokens.append(candidate_token)
                                    _log.info(f"  ✅ 找到参考token: '{candidate_token}' (ID: {candidate_id}) [方法: vocab]")
                                    continue
                        except Exception as e:
                            _log.debug(f"  vocab查找失败: {e}")
                    
                    # 如果所有方法都失败，记录调试信息
                    if candidate_id is None or candidate_id == self._actual_tokenizer.unk_token_id:
                        try:
                            # 尝试tokenize看看实际结果
                            tokenized = self._actual_tokenizer.tokenize(candidate_token)
                            _log.debug(f"  ⚠️ 参考token '{candidate_token}' tokenize结果: {tokenized}")
                        except Exception:
                            pass
                
                if len(found_token_ids) == 0:
                    # 所有参考token都不存在，使用平均初始化
                    ref_tokens_str = "、".join(ref_tokens)
                    _log.warning(f"  ⚠️ 所有参考token都不存在（{ref_tokens_str}），{target_token} 使用平均初始化")
                    # 使用平均初始化
                    if avg_embedding is not None:
                        embedding_dim = avg_embedding.size(0)
                        perturbation = torch.randn(embedding_dim, device=model_device) * perturbation_std
                        embedding_vec = avg_embedding.clone() + perturbation
                        embedding_layer.weight.data[target_id] = embedding_vec
                    else:
                        # 如果平均embedding计算失败，使用随机初始化
                        embedding_dim = embedding_layer.weight.size(1)
                        init_std = getattr(getattr(self.model, "config", None), "initializer_range", 0.02)
                        embedding_vec = torch.randn(embedding_dim, device=model_device) * init_std
                        embedding_layer.weight.data[target_id] = embedding_vec
                    
                    if output_layer is not None:
                        if avg_output is not None:
                            out_dim = avg_output.size(0)
                            output_perturbation = torch.randn(out_dim, device=model_device) * perturbation_std
                            output_vec = avg_output.clone() + output_perturbation
                            output_layer.weight.data[target_id] = output_vec
                        else:
                            out_dim = output_layer.weight.shape[1]
                            init_std = getattr(getattr(self.model, "config", None), "initializer_range", 0.02)
                            output_vec = torch.randn(out_dim, device=model_device) * init_std
                            output_layer.weight.data[target_id] = output_vec
                    
                    init_method = "平均初始化" if avg_embedding is not None else "随机初始化"
                    _log.info(f"  ✅ {target_token} (ID: {target_id}) {init_method}完成（所有参考token不存在）")
                    continue
                
                # 使用所有找到的参考token的embedding的平均值作为基础
                ref_embeddings = []
                ref_outputs = []
                
                for ref_id in found_token_ids:
                    ref_embeddings.append(embedding_layer.weight.data[ref_id].clone())
                    if output_layer is not None:
                        ref_outputs.append(output_layer.weight.data[ref_id].clone())
                
                # 计算平均embedding
                base_embedding = torch.stack(ref_embeddings).mean(dim=0)
                embedding_dim = base_embedding.size(0)
                
                # 添加较大扰动
                perturbation = torch.randn(embedding_dim, device=model_device) * perturbation_std
                embedding_vec = base_embedding + perturbation
                
                # 归一化到与平均embedding相似的范数（可选，保持embedding的尺度）
                base_norm = base_embedding.norm().item()
                new_norm = embedding_vec.norm().item()
                if base_norm > 0 and new_norm > 0:
                    embedding_vec = embedding_vec / new_norm * base_norm
                
                embedding_layer.weight.data[target_id] = embedding_vec
                
                # 同样处理输出层
                if output_layer is not None and len(ref_outputs) > 0:
                    base_output = torch.stack(ref_outputs).mean(dim=0)
                    out_dim = base_output.size(0)
                    
                    # 添加较大扰动
                    output_perturbation = torch.randn(out_dim, device=model_device) * perturbation_std
                    output_vec = base_output + output_perturbation
                    
                    # 归一化到与平均output相似的范数
                    base_out_norm = base_output.norm().item()
                    new_out_norm = output_vec.norm().item()
                    if base_out_norm > 0 and new_out_norm > 0:
                        output_vec = output_vec / new_out_norm * base_out_norm
                    
                    output_layer.weight.data[target_id] = output_vec
                
                found_tokens_str = "、".join([f"'{t}'" for t in found_tokens])
                _log.info(f"  ✅ {target_token} (ID: {target_id}) 初始化完成（参考: {found_tokens_str}，共{len(found_token_ids)}个token的平均值，扰动标准差={perturbation_std}）")
        
        _log.info("token权重初始化完成")


