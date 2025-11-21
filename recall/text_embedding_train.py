import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from tqdm import tqdm
from datetime import datetime, timedelta
from peft import LoraConfig, get_peft_model, TaskType
from modelscope import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from recall.model_utils import forward_backbone, ensure_last_hidden_state

class RecallDataset(Dataset):
    """训练数据集：构造"原始文本<recall>"格式"""
    
    def __init__(self, texts, target_embeddings, tokenizer, base_model, max_length=None):
        self.texts = texts
        self.target_embeddings = target_embeddings
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.max_length = max_length
        self.recall_token = '<recall>'
        
        # 获取模型的数据类型（不预先移动到设备）
        first_param = next(base_model.parameters())
        self.model_dtype = first_param.dtype
        self.model_device = first_param.device
        print(f"🔧 RecallDataset检测到模型数据类型: {self.model_dtype}, 设备: {self.model_device}")

        # 注意：不在__init__中预先移动所有embeddings到GPU，避免显存累积
        # 确保target_embeddings在CPU上，避免显存波动
        if isinstance(self.target_embeddings, torch.Tensor) and self.target_embeddings.is_cuda:
            print(f"⚠️ target_embeddings在GPU上，移动到CPU以避免显存波动...")
            self.target_embeddings = self.target_embeddings.cpu()
        print(f"📊 target_embeddings保持在CPU上，训练时按需移动: {self.target_embeddings.shape}")
        
        # 检查token是否存在（如果传入的是processor，使用processor.tokenizer）
        self.actual_tokenizer = self.tokenizer.tokenizer if hasattr(self.tokenizer, 'tokenizer') else self.tokenizer
        self.recall_token_id = self.actual_tokenizer.convert_tokens_to_ids(self.recall_token)
        if self.recall_token_id == self.actual_tokenizer.unk_token_id:
            raise ValueError(f"❌ {self.recall_token} token不存在！请先添加此特殊token")
        
        print(f"✅ 找到特殊token: {self.recall_token} (ID: {self.recall_token_id})")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        target_embedding = self.target_embeddings[idx]
        
        # 构造输入："原始文本<recall>"
        input_text = f"{text}{self.recall_token}"
        
        # 编码 - 不进行padding，保留原始长度
        # 注意：只有在设置了max_length时才使用，否则让tokenizer使用原始长度
        encode_kwargs = {
            'return_tensors': 'pt'
        }
        if self.max_length is not None:
            # 当设置了max_length时，必须显式设置truncation=True以避免警告
            encode_kwargs['max_length'] = self.max_length
            encode_kwargs['truncation'] = True
        else:
            # 如果没有设置max_length，不进行截断
            encode_kwargs['truncation'] = False

        encoding = self.actual_tokenizer(input_text, **encode_kwargs)
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 找到<recall>的位置
        recall_positions = (input_ids == self.recall_token_id).nonzero(as_tuple=True)[0]
        if len(recall_positions) > 0:
            recall_position = recall_positions[-1]  # 使用最后一个位置
        else:
            # 如果被截断了，报错
            raise ValueError(f"文本过长，{self.recall_token} token被截断")
        
        # 注意：target_embedding保持在CPU上，只转换数据类型（不移动设备）
        # 设备分配由Accelerator在collate_fn中统一处理，避免在__getitem__中移动导致显存波动
        # 使用clone().detach()避免创建新的计算图，减少显存波动
        if target_embedding.is_cuda:
            # 如果原本在GPU上，先移到CPU（避免显存波动）
            target_embedding = target_embedding.cpu()
        if target_embedding.dtype != self.model_dtype:
            # 只转换数据类型，不移动设备（保持在CPU上）
            target_embedding = target_embedding.to(dtype=self.model_dtype)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'recall_position': recall_position,
            'target_embedding': target_embedding,
            'seq_len': len(input_ids)  # 保存原始序列长度，用于动态padding
        }

class RecallMemoryTrainer:
    """<recall> token训练器 - 支持多GPU设备选择"""
    
    def _get_tokenizer(self):
        """获取真正的tokenizer（如果传入的是processor，则返回processor.tokenizer）"""
        if hasattr(self.tokenizer, 'tokenizer'):
            # 如果传入的是processor，返回其内部的tokenizer
            return self.tokenizer.tokenizer
        else:
            # 如果传入的是tokenizer，直接返回
            return self.tokenizer
    
    def __init__(self, model_name, device=None, lora_r=8, lora_alpha=32, lora_dropout=0.1, original_device=None, preloaded_model=None, preloaded_tokenizer=None, gradient_accumulation_steps=1, max_memory=None, epoch_end_hook=None, max_length=8000, lora_target_modules=None):
        """
        Args:
            model_name: 模型路径或名称
            device: 训练设备，支持：
                   - None: 使用默认设备
                   - "auto": 自动分配多GPU
                   - ['cuda:0', 'cuda:1', ...]: GPU列表
                   - "cuda:0": 指定单GPU
                   - "cpu": CPU设备
            lora_r: LoRA rank（默认8）
            lora_alpha: LoRA alpha（默认32）
            lora_dropout: LoRA dropout（默认0.1）
        """

        # 注意：CUDA_VISIBLE_DEVICES 已经在 app.py 中正确设置，这里不需要重复设置
        # 只保存原始环境变量用于cleanup时恢复
        self._original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

        self.model_name = model_name
        self.specified_device = device
        self.original_device = original_device or device  # 保存原始设备信息用于显示
        self.recall_token = '<recall>'
        self.max_length = max_length  # 最大序列长度
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
        self.epoch_end_hook = epoch_end_hook

        # 根据设备配置决定是否启用DDP
        use_ddp = False
        cuda_visible_devices = None

        if isinstance(device, list) and len(device) > 1:
            use_ddp = True
            print(f"   多GPU模式: 启用DDP，GPU数量: {len(device)}")
        elif device == "auto":
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                use_ddp = True
                print(f"   多GPU模式: 自动检测多GPU，启用DDP")
        # 初始化Accelerator，支持多GPU和梯度累积
        accelerator_kwargs = {
            'mixed_precision': 'bf16',
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
        }

        self.accelerator = Accelerator(**accelerator_kwargs)

        self.accelerate_enabled = True
        self.ddp_enabled = use_ddp
        self.local_rank = None

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
            self.use_auto_device = False
            self.primary_device = torch.device(device)
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
        self._model_prepared = False

        # 设备处理逻辑 - 与get_text_embedding.py保持一致
        if device is None:
            self.use_auto_device = False
            self.primary_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.multi_gpu_list = None
        elif isinstance(device, list):
            # 处理GPU列表
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
        
        print(f"🤖 初始化训练器...")
        print(f"   模型: {model_name}")
        print(f"   设备配置: {device}")

        # 若由 torchrun 启动，自动启用DDP并固定到单卡
        if 'LOCAL_RANK' in os.environ and not self.accelerator.state.initialized:  # Accelerate 已初始化则跳过
            self.local_rank = int(os.environ['LOCAL_RANK'])
            os.environ.setdefault('RANK', os.environ.get('RANK', '0'))
            os.environ.setdefault('WORLD_SIZE', os.environ.get('WORLD_SIZE', '1'))
            torch.cuda.set_device(self.local_rank)
            if not (dist.is_available() and dist.is_initialized()):
                dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
            self.ddp_enabled = True
            self.use_auto_device = False
            self.multi_gpu_list = None
            self.primary_device = torch.device(f'cuda:{self.local_rank}')
            self.specified_device = f'cuda:{self.local_rank}'
            if self.is_main_process():
                print(f"🧩 DDP已启用，LOCAL_RANK={self.local_rank}")
        
        # 设置环境变量以减少显存碎片化（即使使用预加载模型也需要）
        import os as _os
        _os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        
        # 处理预加载模型或加载新模型
        if preloaded_model is not None and preloaded_tokenizer is not None:
            # 使用预加载的模型
            print("   使用预加载的模型和tokenizer")
            
            # 在创建LoRA前，清理显存并确保模型处于干净状态
            preloaded_model.eval()
            with torch.no_grad():
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            
            self.model = preloaded_model
            self.tokenizer = preloaded_tokenizer
            # 检查特殊token（即使是预加载的模型也需要设置token ID）
            self._check_special_token()
            self._skip_model_loading = True
        else:
            # 正常加载模型
            self._load_model()
            # 检查特殊token
            self._check_special_token()
            self._skip_model_loading = False

        # 获取实际设备信息
        first_param = next(self.model.parameters())
        self.actual_device = first_param.device
        print(f"   实际模型设备: {self.actual_device}")

        # 如果不是预加载模型，才进行后续初始化
        if not getattr(self, '_skip_model_loading', False):
            # 保存原始embedding
            self._save_original_embedding()

            # 设置LoRA
            self._setup_lora()
        else:
            # 对于预加载模型，需要重新设置一些属性
            # 记录原始embedding
            self._save_original_embedding()

            # 设置LoRA
            self._setup_lora()
        
        # 显示参数统计
        self._print_trainable_parameters()

    def is_main_process(self):
        if hasattr(self, 'accelerator'):
            return self.accelerator.is_main_process
        return (not self.ddp_enabled) or (dist.get_rank() == 0)
    
    def _prepare_model_once(self):
        """确保Accelerator仅对模型执行一次prepare，避免重复包装导致显存膨胀"""
        if not self._model_prepared:
            if self.is_main_process():
                print("🔄 首次调用Accelerator.prepare，准备模型...")
            self.model = self.accelerator.prepare(self.model)
            self._model_prepared = True

    def _load_model(self):
        """加载模型和分词器 - 支持多GPU配置"""
        # 降碎片：尽量使用可扩展分配段
        import os as _os
        _os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

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

                # 如果有max_memory配置，设置它来控制多GPU分布
                if hasattr(self, 'max_memory') and self.max_memory:
                    device_map = self.max_memory
                    print(f"   使用max_memory控制GPU分布: {device_map}")
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
                    print(f"   使用指定单GPU: {self.specified_device} (device_map: {device_map})")
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
            
            # 根据模型类型选择加载方式
            if is_qwen3vl:
                # 使用Qwen3VLForConditionalGeneration加载Qwen3-VL模型
                from transformers import Qwen3VLForConditionalGeneration
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path if is_local_path else self.model_name,
                    torch_dtype="auto",
                    device_map=device_map,
                    trust_remote_code=True,
                    local_files_only=is_local_path
                )
            else:
                # 使用AutoModelForCausalLM加载普通文本模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path if is_local_path else self.model_name,
                    torch_dtype="auto",
                    device_map=device_map,
                    trust_remote_code=True,
                    local_files_only=is_local_path
                )

            # 降内存：梯度检查点 + 关闭use_cache
            try:
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
                    print("   ✅ 已启用梯度检查点（gradient checkpointing）以减少显存占用")
                if hasattr(self.model, 'config'):
                    setattr(self.model.config, 'use_cache', False)
                    print("   ✅ 已关闭use_cache以减少显存占用")
            except Exception as e:
                print(f"   ⚠️ 启用显存优化功能失败: {e}")
            
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
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path if is_local_path else self.model_name,
                        torch_dtype="auto",
                        device_map=device_map,
                        trust_remote_code=True,
                        local_files_only=is_local_path
                    )
                else:
                    # 使用AutoModelForCausalLM加载普通文本模型
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path if is_local_path else self.model_name,
                        torch_dtype="auto",
                        device_map=device_map,
                        trust_remote_code=True,
                        local_files_only=is_local_path
                    )
                
                first_param = next(self.model.parameters())
                print(f"✅ 使用回退设备加载成功: {first_param.device}")
                
            except Exception as fallback_error:
                print(f"❌ 回退加载也失败: {fallback_error}")
                raise RuntimeError(f"模型加载完全失败: 原错误={e}, 回退错误={fallback_error}")
    
    def _check_and_add_special_token(self):
        """检查并添加特殊token（如果不存在）"""
        tokenizer = self._get_tokenizer()
        self.recall_token_id = tokenizer.convert_tokens_to_ids(self.recall_token)
        
        if self.recall_token_id == tokenizer.unk_token_id:
            # token不存在，需要添加
            print(f"⚠️ {self.recall_token} token不存在，正在添加...")
            original_vocab_size = len(tokenizer)
            
            # 添加特殊token
            tokenizer.add_tokens(self.recall_token)
            
            new_vocab_size = len(tokenizer)
            print(f"   词表大小: {original_vocab_size} -> {new_vocab_size} (+{new_vocab_size - original_vocab_size})")
            
            # 调整模型embedding层
            print("   调整模型embedding层...")
            self.model.resize_token_embeddings(len(tokenizer))
            
            # 获取新添加的token ID
            self.recall_token_id = tokenizer.convert_tokens_to_ids(self.recall_token)
            
            # 初始化新token的权重（使用"总结"和"回忆"的嵌入向量之和）
            print("   初始化新token权重...")
            try:
                embedding_layer = self.model.get_input_embeddings()

                # <recall> token: 使用"总结"和"回忆"的嵌入向量之和
                ref_words = ["总结", "回忆"]
                ref_embeddings = []
                used_references = []

                for word in ref_words:
                    ref_id = tokenizer.convert_tokens_to_ids(word)
                    if ref_id != tokenizer.unk_token_id:
                        ref_embedding = embedding_layer.weight[ref_id].clone().detach()
                        ref_embeddings.append(ref_embedding)
                        used_references.append(word)
                        print(f"   ✅ 使用参考token: '{word}' (ID: {ref_id})")
                    else:
                        print(f"   ⚠️ 参考token '{word}' 不存在，跳过")

                if len(ref_embeddings) > 0:
                    # 计算参考token嵌入向量的和，然后直接归一化
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
                    
                    embedding_layer.weight.data[self.recall_token_id] = new_embedding
                    ref_str = " + ".join(used_references)
                    print(f"   ✅ {self.recall_token} (ID: {self.recall_token_id}) 初始化完成（参考: {ref_str}）")
                else:
                    print(f"   ⚠️ 所有参考token都不存在，使用随机初始化")

            except Exception as e:
                print(f"   ⚠️ 初始化token权重时出错: {e}")
            
            print(f"✅ 特殊token添加完成: {self.recall_token} (ID: {self.recall_token_id})")
        else:
            # token已存在
            print(f"✅ 特殊token检查通过: {self.recall_token} (ID: {self.recall_token_id})")
    
    def _check_special_token(self):
        """检查特殊token是否存在（已废弃，使用_check_and_add_special_token代替）"""
        # 这个方法保留是为了兼容性，但实际调用的是_check_and_add_special_token
        self._check_and_add_special_token()
    
    def _save_original_embedding(self):
        """保存原始embedding参数（用于训练后对比）"""
        # 使用get_input_embeddings()方法获取embedding层（适用于所有模型类型）
        embedding_layer = self.model.get_input_embeddings()
        self.original_recall_embedding = embedding_layer.weight[self.recall_token_id].clone().detach()
        print(f"📝 已保存原始embedding参数")
        print(f"   原始embedding范围: [{self.original_recall_embedding.min().item():.6f}, {self.original_recall_embedding.max().item():.6f}]")
    
    def _setup_lora(self):
        """设置LoRA配置"""
        print("⚡ 配置LoRA...")
        print(f"   LoRA参数: r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
        
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
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"✅ LoRA配置完成")
        
        # 创建LoRA后清理显存（LoRA包装可能产生临时状态）
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # 再次检查数据类型和设备
        first_param = next(self.model.parameters())
        model_dtype = first_param.dtype
        model_device = first_param.device
        print(f"🔧 LoRA后模型数据类型: {model_dtype}, 设备: {model_device}")
        
        # 重要：在LoRA设置后执行解冻操作
        self._freeze_embeddings_except_special_tokens()

    def _freeze_embeddings_except_special_tokens(self):
        """冻结除了特殊token以外的所有embedding参数 - 修复版"""
        print("🧊 冻结除特殊token外的所有embedding参数...")
        
        # 获取正确的embedding层 - 使用get_input_embeddings()方法（适用于所有模型类型）
        # 对于Qwen3-VL模型，这会自动找到正确的embedding层
        embedding_layer = self.model.get_input_embeddings()
        
        # 首先冻结所有embedding参数
        embedding_layer.weight.requires_grad_(False)
        
        # 然后只解冻特殊token的embedding参数
        embedding_layer.weight[self.recall_token_id].requires_grad_(True)
        
        # 强制设置requires_grad标志
        embedding_layer.weight.requires_grad = True
        
        # 验证设置成功
        total_embedding_params = embedding_layer.weight.numel()
        trainable_params = embedding_layer.weight[self.recall_token_id].numel()
        
        # 验证确实可训练
        is_trainable = embedding_layer.weight[self.recall_token_id].requires_grad
        
        print(f"✅ embedding层设置完成:")
        print(f"   总embedding参数: {total_embedding_params:,}")
        print(f"   可训练参数: {trainable_params:,} ({self.recall_token} token only)")
        print(f"   冻结参数: {total_embedding_params - trainable_params:,}")
        print(f"   特殊token embedding是否可训练: {is_trainable}")
        
        # 调试信息
        if not is_trainable:
            print("⚠️ 警告: 特殊token embedding无法设置为可训练！")
            print("尝试使用以下备用方法...")
            
            # 备用方法：直接修改参数的requires_grad属性
            param_pointer = embedding_layer.weight[self.recall_token_id]
            param_pointer.requires_grad = True
            print(f"   再次检查: {param_pointer.requires_grad}")
    
    def _print_trainable_parameters(self):
        """显示可训练参数统计 - 修复版"""
        print("📊 参数统计 (仅特殊token embedding可训练):")
        
        # 获取正确的embedding层路径 - 使用get_input_embeddings()方法
        try:
            embedding_layer = self.model.get_input_embeddings()
            special_token_embedding = embedding_layer.weight[self.recall_token_id]
        except:
            try:
                # 尝试另一种可能的路径
                embedding_layer = self.model.get_input_embeddings()
                special_token_embedding = embedding_layer.weight[self.recall_token_id]
            except Exception as e:
                print(f"⚠️ 无法获取embedding层: {e}")
                embedding_layer = None
        
        # 统计参数
        lora_params = 0
        embedding_params = 0
        other_params = 0
        
        # 检查embedding层是否可训练
        if embedding_layer is not None:
            is_trainable = special_token_embedding.requires_grad
            if is_trainable:
                embedding_params = special_token_embedding.numel()
        
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
        if embedding_layer is not None:
            print(f"\n🎯 {self.recall_token} token状态:")
            print(f"   Token ID: {self.recall_token_id}")
            print(f"   Requires grad: {special_token_embedding.requires_grad}")
            print(f"   当前值范围: [{special_token_embedding.min().item():.6f}, {special_token_embedding.max().item():.6f}]")
    
    def load_data(self, pt_file_path):
        """加载训练数据"""
        print(f"📖 加载数据: {pt_file_path}")
        
        if not os.path.exists(pt_file_path):
            raise FileNotFoundError(f"数据文件不存在: {pt_file_path}")
        
        data = torch.load(pt_file_path, map_location='cpu')
        texts = data['texts']
        embeddings = data['embeddings']

        # 保存数据元信息（用于数据集划分）
        self.data_info = {
            'memory_count': data.get('memory_count', len(texts)),
            'sft_count': data.get('sft_count', 0)
        }
        
        print(f"   文本数量: {len(texts)}")
        print(f"   嵌入形状: {embeddings.shape}")
        print(f"   原始embedding数据类型: {embeddings.dtype}")
        print(f"   数据组成: {self.data_info['memory_count']} 条记忆条目 + {self.data_info['sft_count']} 条SFT向量")
        
        return texts, embeddings
    
    def create_dataloader(self, texts, embeddings, batch_size=2, shuffle=True):
        """创建数据加载器 - 使用动态padding以节省显存"""

        def collate_fn(batch):
            """动态padding：根据batch内最长序列进行padding"""
            if not batch:
                return {}

            # 找出batch内最长序列的长度
            max_len = max(item['seq_len'] for item in batch)

            # 对所有序列进行padding到max_len
            padded_input_ids = []
            padded_attention_masks = []
            recall_positions = []
            target_embeddings = []

            for item in batch:
                input_ids = item['input_ids']
                attention_mask = item['attention_mask']
                recall_pos = item['recall_position']
                target_emb = item['target_embedding']

                # padding 到 max_len
                pad_len = max_len - len(input_ids)
                if pad_len > 0:
                    # 使用tokenizer的pad_token_id进行padding
                    tokenizer = self._get_tokenizer()
                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                    padded_input_ids.append(torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)]))
                    padded_attention_masks.append(torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)]))
                else:
                    padded_input_ids.append(input_ids)
                    padded_attention_masks.append(attention_mask)

                recall_positions.append(recall_pos)
                target_embeddings.append(target_emb)

            # 堆叠成batch
            # 确保所有tensor在CPU上，设备分配由Accelerator统一处理
            # 注意：recall_positions中的元素是tensor（标量tensor），所以直接stack即可
            batch_dict = {
                'input_ids': torch.stack(padded_input_ids),
                'attention_mask': torch.stack(padded_attention_masks),
                'recall_position': torch.stack(recall_positions),  # recall_positions是标量tensor列表，直接stack
                'target_embedding': torch.stack(target_embeddings)
            }
            
            # 确保所有tensor在CPU上（Accelerator会自动移动到正确设备）
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    batch_dict[key] = value.cpu()
            
            return batch_dict

        dataset = RecallDataset(texts, embeddings, self.tokenizer, self.model, max_length=self.max_length)
        # 使用标准的数据加载优化：pin_memory和num_workers
        # pin_memory=True: 将数据固定在CPU内存中，加速GPU传输
        # num_workers=0: 在主进程中加载数据，避免多进程导致的显存问题
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=collate_fn,
            pin_memory=True,  # 固定内存，加速GPU传输
            num_workers=0,    # 在主进程中加载，避免多进程显存问题
            persistent_workers=False  # 不持久化worker，节省内存
        )

    def compute_loss(self, last_hidden_states, recall_positions, target_embeddings):
        """计算损失：recall token嵌入与目标嵌入的MSE损失
        
        关键优化：使用批量索引直接提取需要的token位置，最小化计算图
        """
        # 关键优化：使用批量索引一次性提取所有需要的token位置
        # 这比循环提取更高效，且计算图更小
        batch_size = last_hidden_states.size(0)
        batch_indices = torch.arange(batch_size, device=last_hidden_states.device)
        
        # 批量提取：直接索引，只保留这些位置的计算图
        recall_embeddings = last_hidden_states[batch_indices, recall_positions, :]  # [batch_size, hidden_dim]

        # 确保数据类型匹配
        target_embeddings = target_embeddings.to(recall_embeddings.dtype)

        # 计算MSE损失
        loss = nn.MSELoss()(recall_embeddings, target_embeddings)
        return loss
    
    def train_epoch(self, dataloader, optimizer, epoch_idx=0):
        """训练一个epoch - 确保设备一致性"""
        self.model.train()
        total_loss = 0
        
        # 获取模型当前设备
        model_device = next(self.model.parameters()).device
        
        progress_bar = tqdm(dataloader, desc="训练", disable=not self.is_main_process())
        
        for batch in progress_bar:
            # 确保所有数据在正确设备上
            input_ids = batch['input_ids'].to(model_device)
            attention_mask = batch['attention_mask'].to(model_device)
            recall_positions = batch['recall_position']
            target_embeddings = batch['target_embedding'].to(model_device)
            
            # 前向传播（直接走backbone以获取last_hidden_state）
            backbone_outputs = forward_backbone(
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            last_hidden_states = ensure_last_hidden_state(backbone_outputs)
            
            # 使用优化的compute_loss方法
            loss = self.compute_loss(last_hidden_states, recall_positions, target_embeddings)
            
            # 立即清理backbone outputs，释放显存
            del backbone_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 反向传播
            optimizer.zero_grad()
            # 使用 Accelerator 进行反传
            self.accelerator.backward(loss)
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            if self.is_main_process():
                progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """评估模型 - 确保设备一致性"""
        self.model.eval()
        total_loss = 0
        total_cosine_sim = 0
        
        # 获取模型当前设备
        model_device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(model_device)
                attention_mask = batch['attention_mask'].to(model_device)
                recall_positions = batch['recall_position']
                target_embeddings = batch['target_embedding'].to(model_device)
                
                backbone_outputs = forward_backbone(
                    self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                last_hidden_states = ensure_last_hidden_state(backbone_outputs)
                
                # 使用优化的compute_loss方法
                loss = self.compute_loss(last_hidden_states, recall_positions, target_embeddings)
                
                # 计算余弦相似度（需要重新提取，但验证阶段显存压力较小）
                batch_size = last_hidden_states.size(0)
                batch_indices = torch.arange(batch_size, device=last_hidden_states.device)
                recall_embeddings = last_hidden_states[batch_indices, recall_positions, :]
                target_embeddings = target_embeddings.to(recall_embeddings.dtype)
                cosine_sim = nn.CosineSimilarity(dim=-1)(recall_embeddings, target_embeddings).mean()
                
                # 立即清理outputs，释放显存
                del backbone_outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                total_loss += loss.item()
                total_cosine_sim += cosine_sim.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_cosine_sim = total_cosine_sim / len(dataloader)
        
        return avg_loss, avg_cosine_sim
    
    def compare_embeddings(self):
        """比较训练前后embedding的变化"""
        print("\n🔍 分析embedding变化...")
        
        # 从合并后的模型获取embedding
        current_embedding = self.merged_model.get_input_embeddings().weight[self.recall_token_id]
        
        # 确保数据类型一致
        if self.original_recall_embedding.dtype != current_embedding.dtype:
            original_embedding = self.original_recall_embedding.to(current_embedding.dtype)
        else:
            original_embedding = self.original_recall_embedding
        
        # 计算变化
        change = torch.abs(current_embedding - original_embedding).mean().item()
        
        # 计算余弦相似度
        cosine_sim = nn.CosineSimilarity(dim=0)(
            current_embedding, 
            original_embedding
        ).item()
        
        print(f"   {self.recall_token} embedding平均变化: {change:.6f}")
        print(f"   训练前后余弦相似度: {cosine_sim:.6f}")
        print(f"   训练前范围: [{original_embedding.min().item():.4f}, {original_embedding.max().item():.4f}]")
        print(f"   训练后范围: [{current_embedding.min().item():.4f}, {current_embedding.max().item():.4f}]")
        
        return {
            'change': change,
            'cosine_similarity': cosine_sim,
            'before_range': (original_embedding.min().item(), original_embedding.max().item()),
            'after_range': (current_embedding.min().item(), current_embedding.max().item())
        }
    
    def merge_and_save_model(self, save_path):
        """合并LoRA权重并保存完整模型"""
        if not self.is_main_process():
            return None
        print("🔄 合并LoRA权重...")

        # 合并权重
        base_model = self.accelerator.unwrap_model(self.model) if hasattr(self, 'accelerator') else self.model
        merged_model = base_model.merge_and_unload()

        if os.path.isdir(save_path):
            print(f"🧹 清理已有的模型输出目录: {save_path}")
            import shutil
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

        tokenizer = self._get_tokenizer()

        # 确保特殊token在词汇表中（调试用）
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

        # 确保特殊token在特殊token列表中
        if hasattr(tokenizer, 'special_tokens_map'):
            additional_special = tokenizer.special_tokens_map.get('additional_special_tokens', [])
            for token in special_tokens:
                if token not in additional_special:
                    print(f"   ⚠️ {token} 不在特殊token列表中，重新添加")
                    if hasattr(tokenizer, 'add_special_tokens'):
                        tokenizer.add_special_tokens({"additional_special_tokens": [token]})

        tokenizer.save_pretrained(save_path)

        print(f"✅ 合并后的完整模型已保存到: {save_path}")

        # 保存完成后立即清理merged_model引用，避免内存泄漏
        try:
            merged_model.cpu()
        except:
            pass
        del merged_model

        return save_path  # 返回保存路径供后续训练使用

    def cleanup(self):
        """清理训练器创建的所有模型实例（支持多GPU）"""
        print("🧹 清理训练器模型实例...")

        try:
            # 清理merged_model（如果存在）
            if hasattr(self, 'merged_model') and self.merged_model is not None:
                try:
                    # 对于多GPU模型，需要更彻底的清理
                    if hasattr(self.merged_model, 'hf_device_map') and self.merged_model.hf_device_map:
                        print("检测到多GPU模型（merged_model），执行彻底清理...")
                    self.merged_model.cpu()
                except:
                    pass
                del self.merged_model
                self.merged_model = None

            # 清理LoRA包装模型
            if hasattr(self, 'model') and self.model is not None:
                try:
                    # 对于多GPU模型，需要更彻底的清理
                    if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                        print("检测到多GPU模型（LoRA model），执行彻底清理...")
                    self.model.cpu()
                except:
                    pass
                del self.model
                self.model = None

            # 清理tokenizer（如果不是预加载的）
            if hasattr(self, 'tokenizer') and self.tokenizer is not None and not getattr(self, '_skip_model_loading', False):
                del self.tokenizer
                self.tokenizer = None

            # 清理accelerator（重要：accelerator可能持有模型引用）
            if hasattr(self, 'accelerator') and self.accelerator is not None:
                try:
                    # 先尝试释放accelerator管理的显存
                    self.accelerator.free_memory()
                    # 如果accelerator有模型引用，也需要清理
                    if hasattr(self.accelerator, 'device'):
                        print(f"清理accelerator管理的设备: {self.accelerator.device}")
                except Exception as e:
                    print(f"清理accelerator时出现警告: {e}")
                # 注意：accelerator实例本身通常不需要显式删除

            # 强制垃圾回收和显存清理（多次清理确保彻底）
            import gc
            for _ in range(5):  # 增加清理次数
                gc.collect()

            import torch
            if torch.cuda.is_available():
                # 根据CUDA_VISIBLE_DEVICES设置决定清理策略
                current_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
                if current_cuda_visible and ',' not in current_cuda_visible:
                    # 单GPU模式，只清理GPU 0（因为CUDA_VISIBLE_DEVICES重新映射了）
                    print(f"单GPU模式: 只清理可见GPU 0")
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        print(f"✅ 已清理可见GPU的显存")
                    except Exception as e:
                        print(f"清理GPU显存时出现警告: {e}")
                else:
                    # 多GPU或未设置环境变量，清理所有GPU
                    gpu_count = torch.cuda.device_count()
                    print(f"清理 {gpu_count} 张GPU的显存...")
                    
                    # 同步并清理所有GPU
                    for i in range(gpu_count):
                        try:
                            with torch.cuda.device(i):
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats()
                        except Exception as e:
                            print(f"清理GPU {i} 时出现警告: {e}")
                    
                    # 再次清理所有GPU
                    for i in range(gpu_count):
                        try:
                            with torch.cuda.device(i):
                                torch.cuda.empty_cache()
                        except Exception as e:
                            print(f"清理GPU {i} 时出现警告: {e}")
                    
                    print(f"✅ 已清理所有 {gpu_count} 张GPU的显存")

            print("✅ 训练器清理完成")

            # 恢复原始CUDA_VISIBLE_DEVICES（如果在初始化时修改过）
            if hasattr(self, '_original_cuda_visible_devices') and self._original_cuda_visible_devices is not None:
                original_value = self._original_cuda_visible_devices
                os.environ['CUDA_VISIBLE_DEVICES'] = original_value
                print(f"恢复原始CUDA_VISIBLE_DEVICES: {original_value}")
            elif 'CUDA_VISIBLE_DEVICES' in os.environ and hasattr(self, '_original_cuda_visible_devices'):
                # 如果初始化时设置了环境变量，现在删除它
                del os.environ['CUDA_VISIBLE_DEVICES']
                print("删除CUDA_VISIBLE_DEVICES环境变量")

        except Exception as e:
            print(f"⚠️ 清理训练器时出现警告: {e}")

    def train(self, pt_file_path, num_epochs=10, batch_size=2, learning_rate=1e-4, save_path="recall_model"):
        """完整训练流程 - 训练/验证集模式"""
        if self.is_main_process():
            print(f"\n🚀 开始训练 {self.recall_token} token")
        print(f"   数据文件: {pt_file_path}")
        print(f"   训练轮数: {num_epochs}")
        print(f"   批次大小: {batch_size}")
        print(f"   学习率: {learning_rate}")
        print(f"   保存路径: {save_path}")

        # 加载数据
        texts, embeddings = self.load_data(pt_file_path)

        if self.is_main_process():
            print(f"\n📊 数据集信息:")
            print(f"   总样本数: {len(texts)}")

        # 分离训练集和验证集
        # 数据格式：记忆条目向量在前，SFT向量在后
        # 验证集只包含从SFT向量中随机抽取的部分

        # 从数据中提取元信息
        data_info = self.data_info if hasattr(self, 'data_info') else {}
        memory_count = data_info.get('memory_count', len(texts))  # 如果没有元信息，假设都是记忆条目
        sft_count = data_info.get('sft_count', 0)

        if sft_count > 0:
            # 有SFT向量：按照要求划分
            # 如果有x条记忆条目，应该有1.5x条SFT向量，其中0.5x用于验证，1.0x用于训练
            memory_indices = list(range(memory_count))  # 所有记忆条目
            sft_indices = list(range(memory_count, memory_count + sft_count))  # SFT向量索引

            # 计算验证集和训练集的SFT数量
            # 理想情况：验证集0.5倍记忆条目数量，训练集1.0倍记忆条目数量
            ideal_val_sft_size = int(memory_count * 0.5)
            ideal_train_sft_size = memory_count
            ideal_total_sft = ideal_val_sft_size + ideal_train_sft_size
            
            # 如果SFT数量不足，按比例分配
            if sft_count < ideal_total_sft:
                # 按比例分配：验证集占1/3，训练集占2/3
                val_sft_size = max(1, int(sft_count / 3))
                train_sft_size = sft_count - val_sft_size
            else:
                # SFT数量充足，使用理想分配
                val_sft_size = ideal_val_sft_size
                train_sft_size = ideal_train_sft_size
                # 如果还有剩余，优先分配给训练集
                if sft_count > ideal_total_sft:
                    train_sft_size += (sft_count - ideal_total_sft)

            # 划分SFT向量：前train_sft_size用于训练，后val_sft_size用于验证
            train_sft_indices = sft_indices[:train_sft_size]
            val_sft_indices = sft_indices[train_sft_size:train_sft_size + val_sft_size]

            # 训练集：所有记忆条目 + 训练用的SFT向量
            train_indices = memory_indices + train_sft_indices
            # 验证集：验证用的SFT向量
            val_indices = val_sft_indices
        else:
            # 没有SFT向量：使用简单的分割
            total_samples = len(texts)
            val_size = max(1, total_samples // 5)  # 20%作为验证集
            train_indices = list(range(total_samples - val_size))
            val_indices = list(range(total_samples - val_size, total_samples))

        train_texts = [texts[i] for i in train_indices]
        train_embeddings = embeddings[train_indices]
        val_texts = [texts[i] for i in val_indices]
        val_embeddings = embeddings[val_indices]

        if self.is_main_process():
            if sft_count > 0:
                print(f"   数据划分详情:")
                print(f"     - 记忆条目: {memory_count} 条（全部用于训练）")
                print(f"     - SFT向量总数: {sft_count} 条")
                print(f"     - 训练集SFT: {len(train_sft_indices)} 条")
                print(f"     - 验证集SFT: {len(val_sft_indices)} 条")
                print(f"   训练集: {len(train_texts)} 样本（{memory_count} 条记忆 + {len(train_sft_indices)} 条SFT）")
                print(f"   验证集: {len(val_texts)} 样本（{len(val_sft_indices)} 条SFT）")
            else:
                print(f"   训练集: {len(train_texts)} 样本")
                print(f"   验证集: {len(val_texts)} 样本")

        # 训练模型
        best_loss = self._train_model(
            train_texts, train_embeddings,
            val_texts, val_embeddings,
            num_epochs, batch_size, learning_rate,
            save_path
        )

        # 保存最终模型
        if self.is_main_process():
            final_model_path = save_path
            print(f"✅ 训练完成，模型已保存到: {final_model_path}")

        return save_path

    def _train_model(self, train_texts, train_embeddings, val_texts, val_embeddings,
                    num_epochs, batch_size, learning_rate, save_path):
        """训练单个折"""
        # 清理上一折可能残留的显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 创建数据加载器
        train_loader = self.create_dataloader(train_texts, train_embeddings, batch_size, True)
        val_loader = self.create_dataloader(val_texts, val_embeddings, batch_size, False)

        # 确保模型只被Accelerator包装一次，避免多次prepare导致显存翻倍
        self._prepare_model_once()

        # 优化器
        optimizer_params = [p for p in self.model.parameters() if p.requires_grad]

        # 确保特殊token embedding被包含
        embedding_layer = self.model.get_input_embeddings()
        special_token_embedding = embedding_layer.weight[self.recall_token_id]
        if special_token_embedding.requires_grad == False:
            print("⚠️ 特殊token embedding未设置为可训练，手动添加到优化器...")
            special_token_embedding.requires_grad_(True)
            optimizer_params.append(special_token_embedding)

        optimizer = optim.AdamW(
            optimizer_params,
            lr=learning_rate,
            weight_decay=0.01
        )

        # 训练循环
        best_val_loss = float('inf')
        model_save_path = save_path

        # 让 Accelerator 接管优化器与数据加载器（模型已在首次折中包装过）
        optimizer, train_loader, val_loader = self.accelerator.prepare(
            optimizer, train_loader, val_loader
        )

        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            accumulation_step = 0

            for batch in train_loader:
                # batch 是一个字典，包含所有输入
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                recall_positions = batch['recall_position']
                target_embeddings = batch['target_embedding']

                # 前向传播
                backbone_outputs = forward_backbone(
                    self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                last_hidden_states = ensure_last_hidden_state(backbone_outputs)
                loss = self.compute_loss(last_hidden_states, recall_positions, target_embeddings)

                # 梯度累积：损失除以累积步数
                loss = loss / self.gradient_accumulation_steps

                # 反向传播
                self.accelerator.backward(loss)
                
                # 释放backbone输出引用，立即释放显存
                del backbone_outputs
                # 每隔几个batch清理一次显存缓存（避免频繁调用影响性能，但确保显存安全）
                if accumulation_step % max(1, self.gradient_accumulation_steps) == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                accumulation_step += 1

                # 每accumulation_steps步执行一次优化器步骤
                if accumulation_step % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    train_steps += 1
                    
                    # 优化器步骤后清理显存（关键位置，确保梯度更新后释放显存）
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 累积损失（注意：这里累积的是原始损失，不是除以累积步数的损失）
                train_loss += loss.item() * self.gradient_accumulation_steps
                
                # 释放loss的引用（虽然已经计算了item()，但可以提前释放）
                del loss

            # 处理最后一个epoch中剩余的梯度累积
            if accumulation_step % self.gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
                train_steps += 1
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_train_loss = train_loss / train_steps

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch in val_loader:
                    # batch 是一个字典，包含所有输入
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    recall_positions = batch['recall_position']
                    target_embeddings = batch['target_embedding']

                    # 前向传播
                    backbone_outputs = forward_backbone(
                        self.model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    last_hidden_states = ensure_last_hidden_state(backbone_outputs)
                    loss = self.compute_loss(last_hidden_states, recall_positions, target_embeddings)

                    # 验证阶段：立即清理backbone输出引用，释放显存
                    del backbone_outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    val_loss += loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / val_steps

            if self.is_main_process():
                print(f"   Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # 保存最好的模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if self.is_main_process():
                    # 保存当前最好的模型
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    unwrapped.save_pretrained(model_save_path)
                    tokenizer = self._get_tokenizer()
                    tokenizer.save_pretrained(model_save_path)
            # 每个epoch结束后的hook（用于插入SFT）
            try:
                if callable(self.epoch_end_hook):
                    self.epoch_end_hook(epoch, self)
            except Exception as hook_err:
                if self.is_main_process():
                    print(f"⚠️ epoch_end_hook 执行失败但已忽略: {hook_err}")

        # 清理当前折的资源，避免K折过程中显存逐步攀升
        try:
            self.accelerator.wait_for_everyone()
            self.accelerator.free_memory()
        except Exception:
            pass

        # 主动释放数据加载器与优化器引用
        del train_loader
        del val_loader
        del optimizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return best_val_loss

    def expose_training_handles(self):
        """暴露训练句柄，供外部SFT复用LoRA模型"""
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "accelerator": getattr(self, "accelerator", None)
        }

def main():
    """主函数 - 支持设备选择"""
    
    # 🔧 配置参数
    MODEL_NAME = "./Qwen2.5-7B-Instruct-with-special-tokens"
    PT_FILE_PATH = "datasets/embeddings/text_embeddings.pt"
    
    # 训练参数
    NUM_EPOCHS = 30
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    SAVE_PATH = "Qwen2.5-7B-Instruct-with-special-tokens-embedding-trained"
    
    # 设备选择 - 支持多种模式
    DEVICE = "cuda:5"  # 可以是 "auto", "cuda:2", "cpu", "cuda:0" 等
    
    print("🚀 记忆token训练程序")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"数据: {PT_FILE_PATH}")
    print(f"设备: {DEVICE}")
    print("=" * 60)
    
    # 检查文件
    if not os.path.exists(PT_FILE_PATH):
        print(f"❌ 数据文件不存在: {PT_FILE_PATH}")
        return
    
    if not os.path.exists(MODEL_NAME):
        print(f"❌ 模型路径不存在: {MODEL_NAME}")
        return
    
    try:
        # 初始化训练器，传递设备参数
        trainer = RecallMemoryTrainer(model_name=MODEL_NAME, device=DEVICE)
        
        # 开始训练
        embedding_analysis = trainer.train(
            pt_file_path=PT_FILE_PATH,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
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