import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

class SpecialTokensManager:
    """特殊token管理器 - 支持多GPU"""
    
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.specified_device = device
        
        # 设备处理逻辑 - 与其他训练器保持一致
        if device is None:
            self.use_auto_device = False
            self.primary_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.multi_gpu_list = None
        elif isinstance(device, list):
            if len(device) > 0:
                self.use_auto_device = False
                self.primary_device = torch.device(device[0])
                self.multi_gpu_list = device
                print(f"   使用多GPU列表: {device}，主设备: {device[0]}")
            else:
                self.use_auto_device = True
                self.primary_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.multi_gpu_list = None
        elif isinstance(device, str):
            if device == "auto":
                self.use_auto_device = True
                self.primary_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.multi_gpu_list = None
            else:
                self.use_auto_device = False
                self.primary_device = torch.device(device)
                self.multi_gpu_list = None
        else:
            self.use_auto_device = False
            self.primary_device = device
            self.multi_gpu_list = None
            
        self.model = None
        self.tokenizer = None
        
        # 要添加的特殊token
        self.special_tokens = ["<recall>", "<|recall|>", "</recall>"]  # 保留<|recall|>以防仍在使用
        # 参考token映射（用于初始化权重）
        # <recall>: 使用"总结"和"回忆"的嵌入向量之和
        # </recall>: 使用"回忆"和"结束"的嵌入向量之和
        self.reference_tokens = {
            "<recall>": ["总结", "回忆"],
            "</recall>": ["回忆", "结束"]
        }
        
    def load_model(self):
        """加载模型和分词器 - 支持多GPU配置"""
        print(f"🔧 加载模型: {self.model_path}")
        print(f"🎯 指定设备: {self.specified_device}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
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
            elif isinstance(self.specified_device, str) and self.specified_device.startswith('cuda:'):
                # 单GPU指定
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
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=True
            )
            
            # 获取实际设备信息
            first_param = next(self.model.parameters())
            model_dtype = first_param.dtype
            model_device = first_param.device
            
            print(f"✅ 模型加载成功")
            print(f"   实际设备: {model_device}")
            print(f"   数据类型: {model_dtype}")
            
            # 显示设备映射信息
            if hasattr(self.model, 'hf_device_map'):
                print(f"   设备映射详情: {self.model.hf_device_map}")
                
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
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype="auto",
                    device_map=device_map,
                    trust_remote_code=True
                )
                
                first_param = next(self.model.parameters())
                print(f"✅ 使用回退设备加载成功: {first_param.device}")
                
            except Exception as fallback_error:
                print(f"❌ 回退加载也失败: {fallback_error}")
                raise RuntimeError(f"模型加载完全失败: 原错误={e}, 回退错误={fallback_error}")
        
    def add_special_tokens(self, perturbation_std=0.02):
        """添加特殊token并初始化权重"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        # 检查原始词表大小
        original_vocab_size = len(self.tokenizer)
        print(f"📊 原始词表大小: {original_vocab_size}")
        
        # 添加新token
        print("➕ 添加特殊token...")
        for token in self.special_tokens:
            self.tokenizer.add_tokens(token)
            
        new_vocab_size = len(self.tokenizer)
        print(f"📊 新词表大小: {new_vocab_size} (+{new_vocab_size - original_vocab_size})")
        
        # 调整模型embedding层
        print("🔧 调整模型embedding层...")
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 获取新token的ID
        token_ids = {}
        for token in self.special_tokens:
            token_ids[token] = self.tokenizer.convert_tokens_to_ids(token)
            
        print(f"🆔 新token ID: {token_ids}")
        
        # 初始化权重
        self._initialize_token_weights(token_ids, perturbation_std)
        
        return token_ids
        
    def _initialize_token_weights(self, token_ids, perturbation_std):
        """使用中文参考token初始化权重（使用参考token嵌入向量之和）"""
        print(f"🎯 初始化token权重（使用参考token嵌入向量之和）...")
            
        # 获取embedding层和lm_head
        embedding_layer = self.model.get_input_embeddings()
        lm_head = self.model.lm_head
        
        with torch.no_grad():
            for target_token, ref_words in self.reference_tokens.items():
                if target_token not in token_ids:
                    continue

                target_id = token_ids[target_token]
                
                # 获取参考token ID
                ref_token_ids = []
                for word in ref_words:
                    ref_ids = self.tokenizer.encode(word, add_special_tokens=False)
                    if len(ref_ids) != 1:
                        print(f"   ⚠️ 参考词 '{word}' 不是单个token，跳过")
                        continue
                    ref_token_ids.append(ref_ids[0])
                    print(f"   {target_token} 使用参考词 '{word}' (ID: {ref_ids[0]})")

                if len(ref_token_ids) == 0:
                    print(f"   ⚠️ {target_token} 没有有效的参考token，使用随机初始化")
                    continue

                # 计算参考token嵌入向量的和，然后归一化
                base_embedding = None
                base_lm_weight = None
                ref_embeddings_list = []
                ref_lm_weights_list = []

                for ref_id in ref_token_ids:
                    ref_emb = embedding_layer.weight.data[ref_id].clone()
                    ref_lm = lm_head.weight.data[ref_id].clone()
                    ref_embeddings_list.append(ref_emb)
                    ref_lm_weights_list.append(ref_lm)

                    if base_embedding is None:
                        base_embedding = ref_emb
                        base_lm_weight = ref_lm
                    else:
                        base_embedding = base_embedding + ref_emb
                        base_lm_weight = base_lm_weight + ref_lm

                # 直接归一化：缩放到第一个参考token的范数，然后添加小的正交扰动以区分
                if len(ref_embeddings_list) > 1:
                    target_emb_norm = ref_embeddings_list[0].norm()
                    target_lm_norm = ref_lm_weights_list[0].norm()
                    current_emb_norm = base_embedding.norm()
                    current_lm_norm = base_lm_weight.norm()
                    if current_emb_norm > 0:
                        base_embedding = base_embedding / current_emb_norm * target_emb_norm
                    if current_lm_norm > 0:
                        base_lm_weight = base_lm_weight / current_lm_norm * target_lm_norm
                    
                    # 添加小的正交扰动，避免与参考token过于相似
                    import torch
                    ref1_emb_normalized = ref_embeddings_list[0] / ref_embeddings_list[0].norm()
                    base_emb_normalized = base_embedding / base_embedding.norm()
                    proj_emb = torch.dot(base_emb_normalized, ref1_emb_normalized) * ref1_emb_normalized
                    orthogonal_emb = base_emb_normalized - proj_emb
                    if orthogonal_emb.norm() > 1e-6:
                        orthogonal_emb = orthogonal_emb / orthogonal_emb.norm()
                        perturbation_scale = 0.1
                        base_embedding = base_embedding + orthogonal_emb * perturbation_scale * target_emb_norm
                        base_embedding = base_embedding / base_embedding.norm() * target_emb_norm
                    
                    ref1_lm_normalized = ref_lm_weights_list[0] / ref_lm_weights_list[0].norm()
                    base_lm_normalized = base_lm_weight / base_lm_weight.norm()
                    proj_lm = torch.dot(base_lm_normalized, ref1_lm_normalized) * ref1_lm_normalized
                    orthogonal_lm = base_lm_normalized - proj_lm
                    if orthogonal_lm.norm() > 1e-6:
                        orthogonal_lm = orthogonal_lm / orthogonal_lm.norm()
                        perturbation_scale = 0.1
                        base_lm_weight = base_lm_weight + orthogonal_lm * perturbation_scale * target_lm_norm
                        base_lm_weight = base_lm_weight / base_lm_weight.norm() * target_lm_norm

                # 使用归一化后的embedding作为初始化
                embedding_layer.weight.data[target_id] = base_embedding
                lm_head.weight.data[target_id] = base_lm_weight

                ref_str = " + ".join(ref_words)
                print(f"   ✅ {target_token} (ID: {target_id}) 初始化完成（参考: {ref_str}）")
                
    def save_model(self, save_path):
        """保存添加了特殊token的模型"""
        print(f"💾 保存模型到: {save_path}")
        
        self.tokenizer.save_pretrained(save_path)
        self.model.save_pretrained(save_path)
        
        print("✅ 保存完成")
        return save_path
        
    def process(self, save_path, perturbation_std=0.02):
        """完整处理流程"""
        print("🚀 开始添加特殊token...")
        
        self.load_model()
        token_ids = self.add_special_tokens(perturbation_std)
        model_path = self.save_model(save_path)
        
        print("🎉 特殊token添加完成!")
        return model_path, token_ids