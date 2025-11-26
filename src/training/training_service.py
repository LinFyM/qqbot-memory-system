#! -*- coding: utf-8 -*-
"""
记忆训练服务
负责整合聊天记录、提取记忆条目、训练模型并保存
（从 server/memory_training_service.py 迁移）
"""

import os
import json
import shutil
import torch
import torch.nn as nn
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from transformers import AutoProcessor
import random
import logging as _logging
import sys

# 统一使用 src 包路径，不再依赖顶层 recall 目录
project_root = Path(__file__).resolve().parents[2]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from training import memory_extraction
from chat.history_manager import (
    group_chat_histories,
    private_chat_histories,
    save_chat_history_to_storage,
    chat_history_lock,
)
from api import server_state

# 延迟导入训练器（避免循环导入）
TRAINING_MODULES_AVAILABLE = False



def _ensure_training_modules_loaded() -> bool:
    """确保训练依赖可导入。若可导入则返回True。"""
    global TRAINING_MODULES_AVAILABLE
    if TRAINING_MODULES_AVAILABLE:
        return True
    logger = _log if "_log" in globals() else logging.getLogger(__name__)
    try:
        global RecallMemoryTrainer, EnhancedTextMemoryTrainer  # type: ignore
        from training.text_embedding_train import RecallMemoryTrainer  # type: ignore
        from training.text_memory_train import EnhancedTextMemoryTrainer  # type: ignore
        TRAINING_MODULES_AVAILABLE = True
        logger.info("✅ 训练模块导入成功（qqbot_memory.training.*）")
        return True
    except Exception as e:
        logger.error(f"训练依赖不可用：{e}。请检查 src/qqbot_memory/training 是否完整。", exc_info=True)
        TRAINING_MODULES_AVAILABLE = False
        return False
from memory.vector_db import MemoryVectorDB

if 'TRAINING_MODULES_AVAILABLE' not in locals():
    TRAINING_MODULES_AVAILABLE = False

if '_log' not in locals():
    _log = logging.getLogger(__name__)


def _optimize_multi_gpu_allocation(device_list: List[str], max_memory_config: Dict[int, str] = None, cuda_visible_set: bool = False) -> Dict[str, Any]:
    """
    优化多GPU分配策略，确保模型和数据更均匀地分布在多张GPU上
    
    Args:
        device_list: GPU设备列表，如 ["cuda:0", "cuda:1"] 或 ["cuda:6", "cuda:7"]
        max_memory_config: 用户配置的max_memory，格式如 {0: "20GB", 1: "20GB"}（索引是可见GPU的索引，不是物理索引）
        cuda_visible_set: 是否已经设置了CUDA_VISIBLE_DEVICES（如果已设置，需要使用重新映射后的索引）
    
    Returns:
        包含优化后的max_memory和device_map的字典
    """
    if not torch.cuda.is_available():
        return {"device_map": "cpu", "max_memory": None}
    
    num_gpus = len(device_list)
    if num_gpus == 0:
        return {"device_map": "cpu", "max_memory": None}
    
    # 检测每张GPU的可用显存
    gpu_memories = {}
    for i, device in enumerate(device_list):
        if device.startswith("cuda:"):
            try:
                physical_gpu_idx = int(device.split(":")[1])
                
                # 如果CUDA_VISIBLE_DEVICES已经设置，torch只能看到重新映射后的索引
                # 此时需要使用可见GPU的索引（0, 1, 2...），而不是物理索引
                if cuda_visible_set:
                    # 使用可见GPU的索引（i就是重新映射后的索引）
                    visible_gpu_idx = i
                    # 获取GPU总显存（MB）- 使用可见索引
                    total_memory_mb = torch.cuda.get_device_properties(visible_gpu_idx).total_memory // (1024 * 1024)
                    # 获取当前已用显存（MB）
                    torch.cuda.set_device(visible_gpu_idx)
                    allocated_mb = torch.cuda.memory_allocated(visible_gpu_idx) // (1024 * 1024)
                    reserved_mb = torch.cuda.memory_reserved(visible_gpu_idx) // (1024 * 1024)
                    available_mb = total_memory_mb - reserved_mb
                    _log.info(f"🔍 训练模型 GPU {i} (物理索引 {physical_gpu_idx}, 可见索引 {visible_gpu_idx}): 总显存={total_memory_mb}MB, 可用={available_mb}MB, 已保留={reserved_mb}MB")
                else:
                    # CUDA_VISIBLE_DEVICES未设置，使用物理索引
                    # 获取GPU总显存（MB）
                    total_memory_mb = torch.cuda.get_device_properties(physical_gpu_idx).total_memory // (1024 * 1024)
                    # 获取当前已用显存（MB）
                    torch.cuda.set_device(physical_gpu_idx)
                    allocated_mb = torch.cuda.memory_allocated(physical_gpu_idx) // (1024 * 1024)
                    reserved_mb = torch.cuda.memory_reserved(physical_gpu_idx) // (1024 * 1024)
                    available_mb = total_memory_mb - reserved_mb
                    _log.info(f"🔍 训练模型 GPU {i} (物理索引 {physical_gpu_idx}): 总显存={total_memory_mb}MB, 可用={available_mb}MB, 已保留={reserved_mb}MB")
                
                gpu_memories[i] = {
                    "total_mb": total_memory_mb,
                    "available_mb": available_mb,
                    "reserved_mb": reserved_mb,
                    "allocated_mb": allocated_mb
                }
            except Exception as e:
                _log.warning(f"⚠️ 无法检测GPU {i}的显存: {e}")
                # 使用默认值
                gpu_memories[i] = {"total_mb": 24000, "available_mb": 20000, "reserved_mb": 0, "allocated_mb": 0}
    
    # 计算优化的max_memory配置
    optimized_max_memory = {}
    if max_memory_config:
        # 如果用户提供了配置，使用用户配置，但确保所有GPU都有配置
        for i in range(num_gpus):
            if i in max_memory_config:
                optimized_max_memory[i] = max_memory_config[i]
            else:
                # 如果没有配置，使用可用显存的90%（留10%给系统和其他操作）
                if i in gpu_memories:
                    available_gb = gpu_memories[i]["available_mb"] / 1024
                    optimized_max_memory[i] = f"{int(available_gb * 0.9)}GB"
                else:
                    optimized_max_memory[i] = "20GB"  # 默认值
    else:
        # 如果没有用户配置，自动计算：使用每张GPU可用显存的90%
        for i in range(num_gpus):
            if i in gpu_memories:
                available_gb = gpu_memories[i]["available_mb"] / 1024
                optimized_max_memory[i] = f"{int(available_gb * 0.9)}GB"
            else:
                optimized_max_memory[i] = "20GB"  # 默认值
    
    _log.info(f"✅ 训练模型优化的max_memory配置: {optimized_max_memory}")
    
    # 使用 "balanced" device_map，尽可能均匀地分配模型层到所有GPU
    # 这样可以最大化利用所有GPU的显存，避免单张GPU过载
    # 注意：如果遇到OOM，可以考虑使用 "balanced_low_0" 让cuda:0分配更少
    # 参考：https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.device_map
    if num_gpus > 1:
        device_map_strategy = "balanced"
        _log.info(f"🔧 多GPU模式：使用 device_map='balanced'，均匀分配模型层到所有 {num_gpus} 张GPU")
    else:
        device_map_strategy = "auto"
        _log.info(f"🔧 单GPU模式：使用 device_map='auto'")
    
    return {
        "device_map": device_map_strategy,
        "max_memory": optimized_max_memory
    }


class TrainingModelContext:
    """训练模型上下文管理器 - 管理训练模型的生命周期"""

    def __init__(self, model_path: str, device, multi_gpu_config: Dict[str, Any] = None, add_special_tokens: bool = True):
        """
        初始化训练模型上下文管理器
        """
        self.model_path = model_path
        self.device = device
        self.multi_gpu_config = multi_gpu_config or {}
        self.add_special_tokens = add_special_tokens
        self.model = None
        self.processor = None

    def __enter__(self):
        """进入上下文，加载训练模型"""
        _log.info(f"加载训练模型上下文: {self.model_path}")
        self.model, self.processor = self._load_training_model()
        return self.model, self.processor

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，彻底清理模型（支持多GPU）"""
        _log.info("清理训练模型上下文...")

        try:
            # 清理模型（多GPU情况下需要更彻底的清理）
            if self.model is not None:
                try:
                    # 对于多GPU模型，需要先尝试移动到CPU
                    # 如果模型使用了device_map="auto"，可能需要特殊处理
                    if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                        # 多GPU模型，需要逐个设备清理
                        _log.info("检测到多GPU模型，执行彻底清理...")
                        # 先尝试移动到CPU（可能部分层已经在CPU上）
                        try:
                            self.model.cpu()
                        except Exception as e:
                            _log.warning(f"移动模型到CPU时出现警告: {e}")
                        
                        # 如果模型有accelerator包装，需要先清理accelerator
                        if hasattr(self.model, 'accelerator'):
                            try:
                                self.model.accelerator.free_memory()
                            except:
                                pass
                    else:
                        # 单GPU模型，直接移动到CPU
                        try:
                            self.model.cpu()
                        except:
                            pass
                except Exception as e:
                    _log.warning(f"清理模型时出现警告: {e}")
                
                # 删除模型引用
                del self.model
                self.model = None

            # 清理processor
            if self.processor is not None:
                del self.processor
                self.processor = None

            # 强制垃圾回收和显存清理（多次清理确保彻底）
            import gc
            for _ in range(5):  # 增加清理次数
                gc.collect()

            # 清理所有GPU的显存
            if torch.cuda.is_available():
                active_indices = self._get_active_cuda_indices()
                if not active_indices:
                    active_indices = list(range(torch.cuda.device_count()))

                for i in active_indices:
                    if i >= torch.cuda.device_count():
                        continue
                    with torch.cuda.device(i):
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

                for i in active_indices:
                    if i >= torch.cuda.device_count():
                        continue
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()

                _log.info(f"✅ 已清理 {len(active_indices)} 张GPU的显存: {active_indices}")

            _log.info("✅ 训练模型上下文清理完成")

        except Exception as cleanup_error:
            _log.warning(f"训练模型上下文清理时出现错误: {cleanup_error}")

        return False  # 不抑制异常

    @staticmethod
    def _parse_cuda_index(device_str: Any) -> Optional[int]:
        if isinstance(device_str, str) and device_str.startswith("cuda"):
            parts = device_str.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                return int(parts[1])
        return None

    def _get_active_cuda_indices(self) -> List[int]:
        indices = set()
        try:
            if self.model is not None and hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
                for target in self.model.hf_device_map.values():
                    if isinstance(target, str) and target.startswith("cuda"):
                        parsed = self._parse_cuda_index(target)
                        if parsed is not None:
                            indices.add(parsed)

            if not indices:
                if isinstance(self.device, list):
                    for dev in self.device:
                        parsed = self._parse_cuda_index(dev)
                        if parsed is not None:
                            indices.add(parsed)
                else:
                    parsed = self._parse_cuda_index(self.device)
                    if parsed is not None:
                        indices.add(parsed)

            if not indices:
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if visible:
                    for part in visible.split(","):
                        part = part.strip()
                        if part.isdigit():
                            indices.add(int(part))

            if not indices and torch.cuda.is_available():
                indices = set(range(torch.cuda.device_count()))
        except Exception:
            if torch.cuda.is_available():
                indices = set(range(torch.cuda.device_count()))

        max_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return sorted(idx for idx in indices if idx < max_count)

    def _load_training_model(self):
        """加载训练模型（内部方法）"""
        return self.load_training_model(self.model_path, self.device, self.multi_gpu_config, add_special_tokens=self.add_special_tokens)

    @staticmethod
    def load_training_model(model_path: str, device, multi_gpu_config: Dict[str, Any] = None, add_special_tokens: bool = True):
        """加载统一的训练模型（静态方法）"""
        # 使用与initialize_model相同的加载逻辑
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        multi_gpu_config = multi_gpu_config or {}
        multi_gpu_enabled = multi_gpu_config.get("enabled", True)

        # 将相对路径转换为绝对路径
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.abspath(os.path.join(project_root, model_path))

        # 检查是否为本地路径
        is_local_path = os.path.exists(model_path) and os.path.isdir(model_path)

        try:
            # 加载processor（使用AutoProcessor而不是AutoTokenizer，因为需要处理图片和视频）
            # 正常推理时使用AutoProcessor，训练时也应该使用AutoProcessor
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=is_local_path
            )

            # 处理多GPU设备配置
            if device == "auto" and multi_gpu_enabled:
                # 自动检测所有可用GPU
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    device = [f"cuda:{i}" for i in range(gpu_count)]
                    _log.info(f"🔧 训练模型: 自动检测到 {gpu_count} 张GPU，使用多GPU模式")

            # 加载模型 - 支持多GPU
            from transformers import Qwen3VLForConditionalGeneration
            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "local_files_only": is_local_path
            }

            # 检查CUDA_VISIBLE_DEVICES设置状态（在所有设备配置之前）
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            cuda_visible_set = bool(cuda_visible)
            cuda_visible_devices = cuda_visible

            # 根据设备配置决定device_map
            if isinstance(device, list) and multi_gpu_enabled:
                # 多GPU配置
                # 注意：CUDA_VISIBLE_DEVICES应该在导入torch之前设置（在app.py中已设置）
                # 这里只需要检查是否已经设置，如果没有设置则设置（兼容性处理）
                
                if cuda_visible:
                    _log.info(f"🔧 检测到CUDA_VISIBLE_DEVICES={cuda_visible}（已在导入torch之前设置）")
                else:
                    # 如果未设置，则在这里设置（虽然可能已经太晚了）
                    gpu_indices = []
                    for gpu_device in device:
                        if gpu_device.startswith("cuda:"):
                            try:
                                gpu_idx = int(gpu_device.split(":")[1])
                                gpu_indices.append(str(gpu_idx))
                            except (ValueError, IndexError):
                                _log.warning(f"⚠️ 无效的GPU设备名称: {gpu_device}，跳过")
                                continue
                    if gpu_indices:
                        cuda_visible_devices = ",".join(gpu_indices)
                        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                        _log.warning(f"⚠️ CUDA_VISIBLE_DEVICES未在导入torch之前设置，现在设置={cuda_visible_devices}（可能无效）")
                        # 注意：如果在这里设置，torch可能已经初始化，所以可能无效
                        # 但为了兼容性，我们仍然设置它

                # 使用优化的多GPU分配策略
                # 注意：如果CUDA_VISIBLE_DEVICES已设置，需要使用重新映射后的索引
                max_memory_config = multi_gpu_config.get("max_memory", {})
                allocation = _optimize_multi_gpu_allocation(device, max_memory_config, cuda_visible_set=cuda_visible_set)
                load_kwargs["device_map"] = allocation["device_map"]
                if allocation["max_memory"]:
                    load_kwargs["max_memory"] = allocation["max_memory"]
                _log.info(f"🔧 训练模型: 指定设备{device}，使用优化的分配策略")
            elif isinstance(device, str) and device.startswith("cuda"):
                # 如果设置了CUDA_VISIBLE_DEVICES，需要使用重新映射后的索引
                if cuda_visible_set and cuda_visible_devices:
                    # CUDA_VISIBLE_DEVICES已设置，使用重新映射后的索引
                    device_map_device = "cuda:0"
                    _log.info(f"🔧 训练模型: 单GPU模式，CUDA_VISIBLE_DEVICES={cuda_visible_devices}，使用重新映射设备 {device_map_device}（对应物理GPU {device}）")
                else:
                    # 未设置CUDA_VISIBLE_DEVICES，直接使用物理设备
                    device_map_device = device
                    _log.info(f"🔧 训练模型: 单GPU模式，设备映射到 {device}")
                load_kwargs["device_map"] = {"": device_map_device}
            else:
                load_kwargs["device_map"] = "auto"  # 默认使用auto
                _log.info("🔧 训练模型: 使用自动设备分配")

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs
            )

            if add_special_tokens:
                # 添加特殊token（如果没有的话）
                # 使用MemoryTokenManager，与正常推理时保持一致
                from memory.token_manager import MemoryTokenManager
                token_manager = MemoryTokenManager(model, processor.tokenizer)
                recall_token_ids = token_manager.check_and_add_tokens(perturbation_std=0.02)
                _log.info(f"✅ 特殊token处理完成: {recall_token_ids}")

            _log.info("✅ 训练模型加载成功")
            return model, processor

        except Exception as e:
            _log.error(f"❌ 加载训练模型失败: {e}")
            raise


def _resolve_path(path: Optional[str], project_root: str) -> Optional[str]:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


class MemoryTrainingService:
    """记忆训练服务"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练服务
        
        Args:
            config: 配置字典，包含训练相关参数
        """
        self.config = config
        self.memory_config = config.get("memory", {}).get("training", {})
        
        script_dir = os.path.dirname(os.path.abspath(__file__))  # server/memory
        server_dir = os.path.dirname(script_dir)                 # server
        project_root = os.path.dirname(server_dir)               # 项目根
        self._project_root = project_root
        
        # 路径配置
        self.base_model_path = _resolve_path(self.memory_config.get("base_model_path"), project_root)
        self.trained_model_dir = _resolve_path(self.memory_config.get("trained_model_dir"), project_root)
        self.memory_db_dir = _resolve_path(self.memory_config.get("memory_db_dir"), project_root)
        self.chat_history_storage_dir = _resolve_path(self.memory_config.get("chat_history_storage_dir"), project_root)
        
        # 提示词配置
        self.prompt_library = config.get("prompt_library", {})
        self.memory_training_prompt_cfg = self.prompt_library.get("memory_training", {}) or {}
        self.memory_extraction_prompts = config.get("memory_extraction_prompts", self.prompt_library.get("memory_extraction", {})) or {}
        self.memory_vectorization_prompts = config.get("memory_vectorization_prompts", self.prompt_library.get("memory_vectorization", {})) or {}

        # 训练配置
        self.training_config = self.memory_config.get("training_config", {})
        self.lora_config = self.memory_config.get("lora_config", {})
        self.guides_config = self.memory_config.get("guides") or {}
        default_guides = self.memory_training_prompt_cfg.get("guides", {}) or {}
        if not self.guides_config:
            self.guides_config = default_guides
        else:
            if default_guides.get("activation_prompts") and not self.guides_config.get("activation_prompts"):
                self.guides_config["activation_prompts"] = default_guides["activation_prompts"]
            if default_guides.get("end_prompts") and not self.guides_config.get("end_prompts"):
                self.guides_config["end_prompts"] = default_guides["end_prompts"]
        self.default_guides = default_guides
        self.guide_text = self.training_config.get("guide_text") or self.memory_training_prompt_cfg.get("guide_text", "")
        
        # 设备配置（使用模型配置中的设备）
        model_config = config.get("model", {})
        self.device = model_config.get("device", "cuda:0")
        # 保存原始设备信息（用于训练器日志显示）
        self.original_device = self.device
        self.server_base_url = server_state.server_base_url
        
        # 创建必要的目录
        os.makedirs(self.trained_model_dir, exist_ok=True)
        os.makedirs(self.memory_db_dir, exist_ok=True)
        
        if self.chat_history_storage_dir:
            os.makedirs(self.chat_history_storage_dir, exist_ok=True)
        
        _log.info("记忆训练服务初始化完成")
        _log.info(f"  基础模型路径: {self.base_model_path}")
        _log.info(f"  训练模型目录: {self.trained_model_dir}")
        _log.info(f"  记忆数据库目录: {self.memory_db_dir}")
        _log.info(f"  聊天记录目录: {self.chat_history_storage_dir}")
        
        # SFT相关配置
        sft_cfg = self.memory_config.get("sft", {})
        self.sft_enabled = bool(sft_cfg.get("enabled", False))
        self.sft_path = sft_cfg.get("dataset_path")
        export_cfg = self.memory_config.get("export", {})
        self.export_save_full_vl_assets = bool(export_cfg.get("save_full_vl_assets", True))
        self.export_merge_lora = bool(export_cfg.get("merge_lora", True))
        self._memory_entry_count = None
        self._current_epoch_sample_n = None
        self._saved_history_counts = {
            "group": {},
            "private": {}
        }

    # 下方保留与旧实现一致的大段函数（提取/保存/训练/清理等），为节省篇幅省略重复注释
    # 由于内容较多，这里直接从旧实现完全迁移（逻辑不变）

    def _prepare_output_dir(self, path: str):
        if os.path.isdir(path):
            _log.info(f"清理历史模型目录: {path}")
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def _get_latest_trained_model_path(self) -> Tuple[str, str]:
        """
        获取最新的模型路径
        优先级：训练后的模型 > 添加了token的模型 > 基础模型
        """
        # 获取token_added_model_dir配置
        token_added_model_dir = self.memory_config.get("token_added_model_dir", "./models/token_added")
        
        # 转换为绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_dir = os.path.dirname(script_dir)
        project_root = os.path.dirname(server_dir)
        
        trained_model_dir = self.trained_model_dir
        if not os.path.isabs(trained_model_dir):
            trained_model_dir = os.path.abspath(os.path.join(project_root, trained_model_dir))
        if not os.path.isabs(token_added_model_dir):
            token_added_model_dir = os.path.abspath(os.path.join(project_root, token_added_model_dir))
        
        # 1. 优先查找训练后的模型
        if os.path.exists(trained_model_dir):
            model_dirs = [
                d for d in os.listdir(trained_model_dir)
                if os.path.isdir(os.path.join(trained_model_dir, d)) and d.startswith("model_")
            ]
            if model_dirs:
                model_dirs.sort(reverse=True)
                latest_model = os.path.join(trained_model_dir, model_dirs[0])
                _log.info(f"找到最新训练模型: {latest_model}")
                return latest_model, "trained"
        
        # 2. 如果没有训练模型，查找添加了token的模型
        if os.path.exists(token_added_model_dir):
            model_dirs = [
                d for d in os.listdir(token_added_model_dir)
                if os.path.isdir(os.path.join(token_added_model_dir, d)) and d.startswith("model_")
            ]
            if model_dirs:
                model_dirs.sort(reverse=True)
                latest_model = os.path.join(token_added_model_dir, model_dirs[0])
                _log.info(f"找到添加了token的模型: {latest_model}")
                return latest_model, "token_added"
        
        # 3. 如果都没有，使用基础模型
        _log.info(f"未找到训练模型或添加了token的模型，使用基础模型: {self.base_model_path}")
        return self.base_model_path, "base"

    def _create_trained_model_path(self) -> str:
        """
        创建新的训练模型保存路径，使用时间戳命名格式：model_YYYYMMDD_HHMMSS
        确保与加载逻辑匹配（按时间戳排序选择最新的）
        """
        from datetime import datetime
        if not os.path.isabs(self.trained_model_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))  # memory目录
            server_dir = os.path.dirname(script_dir)  # server目录
            project_root = os.path.dirname(server_dir)  # 项目根目录
            # 路径相对于项目根目录，直接拼接
            trained_model_dir = os.path.abspath(os.path.join(project_root, self.trained_model_dir))
        else:
            trained_model_dir = self.trained_model_dir
        
        os.makedirs(trained_model_dir, exist_ok=True)
        
        # 使用时间戳创建新的模型目录名：model_YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir_name = f"model_{timestamp}"
        model_path = os.path.join(trained_model_dir, model_dir_name)
        
        _log.info(f"创建新的训练模型保存路径: {model_path}")
        return model_path

    def run_training(self, skip_memory_dump: bool = False) -> Optional[str]:
        """
        运行完整的训练流程
        
        Returns:
            最终训练模型的路径
        """
        _log.info("=" * 60)
        _log.info("开始记忆训练流程")
        _log.info("=" * 60)
        
        multi_gpu_config = self.config.get("model", {}).get("multi_gpu", {})

        # 0. 保存内存中的聊天记录到JSON文件（训练前调用）
        if skip_memory_dump:
            _log.info("=" * 60)
            _log.info("步骤0: 已在API中保存聊天记录，跳过此步骤")
            _log.info("=" * 60)
        else:
            _log.info("=" * 60)
            _log.info("步骤0: 保存内存中的聊天记录到JSON文件")
            _log.info("=" * 60)
            self.save_memory_chat_histories_to_storage()

        # 1. 从JSON文件加载聊天记录
        _log.info("=" * 60)
        _log.info("步骤1: 加载JSON文件中的聊天记录")
        _log.info("=" * 60)
        chat_messages = self.load_chat_histories_from_json_only()

        # chat_messages 是一个列表，每个元素是 {"chat_type": ..., "chat_id": ..., "message": ...}
        if len(chat_messages) == 0:
            _log.warning("⚠️ 没有聊天记录可训练，跳过训练")
            _log.info("💡 可能的原因：")
            _log.info("   - 内存中没有聊天记录")
            _log.info("   - chat_history_storage_dir 中没有JSON文件")
            _log.info("   - 请检查聊天记录是否被正确保存")
            return None

        # 统计聊天记录信息（按聊天分组统计）
        chat_groups = {}
        for msg_data in chat_messages:
            chat_type = msg_data.get("chat_type", "unknown")
            chat_id = msg_data.get("chat_id", "unknown")
            key = f"{chat_type}_{chat_id}"
            if key not in chat_groups:
                chat_groups[key] = 0
            chat_groups[key] += 1

        total_messages = len(chat_messages)
        _log.info(f"📊 总共 {total_messages} 条消息，分布在 {len(chat_groups)} 个聊天组中")

        # 2. 使用基础模型提取记忆条目和监督向量
        _log.info("=" * 60)
        _log.info("步骤2: 使用基础模型提取记忆条目和监督向量")
        _log.info("=" * 60)
        _log.info(f"使用基础模型: {self.base_model_path}")

        with TrainingModelContext(self.base_model_path, self.device, multi_gpu_config, add_special_tokens=False) as (base_model, base_processor):
            # 提取记忆条目并保存到临时文件（使用基础模型）
            temp_training_data_path = self.extract_memory_entries(chat_messages, base_model, base_processor)

            if temp_training_data_path is None or not os.path.exists(temp_training_data_path):
                _log.warning("⚠️ 没有提取到记忆条目或生成训练数据文件，跳过训练")
                _log.info("💡 可能的原因：")
                _log.info("   - 模型在提取记忆条目时没有识别到值得记忆的内容")
                _log.info("   - 提取过程中出现错误（请查看上面的日志）")
                _log.info("   - 聊天记录中的内容可能不适合提取为记忆条目")
                return None

            # 加载训练数据以获取统计信息
            training_data = torch.load(temp_training_data_path, map_location='cpu')
            num_entries = len(training_data.get('texts', []))
            _log.info(f"📊 提取到 {num_entries} 个记忆条目，已保存到临时文件")
            # 设置本轮SFT每epoch采样参考数（与记忆条目数量等量）
            try:
                self._memory_entry_count = int(num_entries)
                self._current_epoch_sample_n = int(num_entries)
            except Exception:
                self._memory_entry_count = None
                self._current_epoch_sample_n = None

            # 保存监督向量到MemoryVectorDB（从训练数据文件中提取）
            self.save_memory_embeddings_from_file(temp_training_data_path)

            # 同时提取等量的SFT向量用于第一步训练，防止<recall>token过拟合
            sft_vectors_path = self._extract_sft_vectors_for_recall_training(
                num_entries, base_model, base_processor
            )

        # 基础模型上下文自动清理

        # 3. 使用最新的训练模型进行训练
        _log.info("=" * 60)
        _log.info("步骤3: 使用最新的训练模型进行训练")
        _log.info("=" * 60)
        if getattr(self, "_memory_entry_count", None):
            self._current_epoch_sample_n = self._memory_entry_count
        
        # 查找最新的训练模型路径（如果存在），否则使用基础模型
        training_model_path, training_model_source = self._get_latest_trained_model_path()
        _log.info(f"训练模型路径: {training_model_path}（来源: {training_model_source}）")

        training_context = TrainingModelContext(training_model_path, self.device, multi_gpu_config)
        with training_context as (training_model, training_processor):
            # 3.5. 清理显存，确保模型处于干净状态
            _log.info("=" * 60)
            _log.info("步骤3.5: 清理显存，准备训练")
            _log.info("=" * 60)
            
            # 确保模型处于eval模式，清除梯度
            training_model.eval()
            with torch.no_grad():
                import gc
                for _ in range(5):
                    gc.collect()

                if torch.cuda.is_available():
                    active_indices = training_context._get_active_cuda_indices()
                    if not active_indices:
                        active_indices = list(range(torch.cuda.device_count()))
                    _log.info(f"清理 {len(active_indices)} 张GPU的显存: {active_indices}")

                    for i in active_indices:
                        if i >= torch.cuda.device_count():
                            continue
                        with torch.cuda.device(i):
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()

                    for i in active_indices:
                        if i >= torch.cuda.device_count():
                            continue
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()

                    _log.info(f"✅ 已清理 {len(active_indices)} 张GPU的显存: {active_indices}")
            
            _log.info("✅ 显存清理完成，模型已准备就绪")

            # 4. 第一步训练：<recall> token训练（使用最新的训练模型）
            step1_model_path = self.train_recall_token(temp_training_data_path, training_model, training_processor, sft_vectors_path)

            # 5. 第二步训练：记忆解码训练（重新加载第一阶段训练好的模型）
            final_model_path = self.train_memory_decoding(temp_training_data_path, step1_model_path)

            # 6. 按时间戳保存最终模型
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_save_path = os.path.join(self.trained_model_dir, f"model_{timestamp}")

            import shutil
            if os.path.exists(final_save_path):
                shutil.rmtree(final_save_path)
            shutil.copytree(final_model_path, final_save_path)

            # 保存Processor配置到最终模型目录
            self._save_processor_to_path(final_save_path)
            if self.export_save_full_vl_assets:
                self._ensure_full_vl_assets(final_save_path)

            _log.info(f"最终模型保存在: {final_save_path}")

            # 7. 清理训练数据和缓存（训练模型由上下文管理器自动清理）
            self.cleanup_after_training()

            _log.info("=" * 60)
            _log.info("记忆训练流程完成")
            _log.info("=" * 60)

            # 训练完成后清理上传缓存（图片/视频），避免长期占用磁盘
            try:
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server/
                uploaded_images = os.path.join(script_dir, "uploaded_images")
                uploaded_videos = os.path.join(script_dir, "uploaded_videos")
                cleaned = 0
                for d in [uploaded_images, uploaded_videos]:
                    if os.path.isdir(d):
                        for fname in os.listdir(d):
                            fpath = os.path.join(d, fname)
                            try:
                                os.remove(fpath)
                                cleaned += 1
                            except Exception:
                                pass
                if cleaned:
                    _log.info(f"✅ 训练完成后已清空缓存文件 {cleaned} 个（images/videos）")
            except Exception as ce:
                _log.warning(f"⚠️ 清理上传缓存失败: {ce}")

            return final_save_path

    def _save_processor_to_path(self, target_path: str):
        """
        保存完整的Processor配置到目标路径
        确保使用训练后的tokenizer（包含特殊token），同时保留processor的其他配置
        """
        try:
            base_path = self.base_model_path
            if not os.path.isabs(base_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                base_path = os.path.abspath(os.path.join(project_root, base_path))
            
            # 1. 从基础模型加载processor（包含image_processor、video_processor等配置）
            base_processor = AutoProcessor.from_pretrained(
                base_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 2. 从训练后的模型加载tokenizer（包含特殊token）
            trained_tokenizer = None
            if os.path.exists(target_path):
                try:
                    # 尝试加载训练后的tokenizer
                    from transformers import AutoTokenizer
                    trained_tokenizer = AutoTokenizer.from_pretrained(
                        target_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    _log.info("✅ 已加载训练后的tokenizer（包含特殊token）")
                except Exception as e:
                    _log.warning(f"⚠️ 加载训练后的tokenizer失败: {e}，将使用基础模型的tokenizer")
            
            # 3. 如果训练后的tokenizer存在，更新processor的tokenizer
            if trained_tokenizer is not None:
                base_processor.tokenizer = trained_tokenizer

            # 4. 保存完整的processor配置（包含训练后的tokenizer和其他processor组件）
            base_processor.save_pretrained(target_path)

            # 5. 确保所有必要的配置文件都被正确保存（在save_pretrained之后，确保不被覆盖）
            # 这些文件对于Qwen3VLProcessor的正确工作至关重要
            import shutil
            essential_files = [
                "chat_template.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json"
            ]
            for file_name in essential_files:
                source_file = os.path.join(base_path, file_name)
                target_file = os.path.join(target_path, file_name)
                if os.path.exists(source_file):
                    try:
                        shutil.copy2(source_file, target_file)
                        _log.info(f"✅ 已复制{file_name}到: {target_path}")
                    except Exception as e:
                        _log.warning(f"⚠️ 复制{file_name}失败: {e}")
                else:
                    _log.warning(f"⚠️ 基础模型中不存在{file_name}，跳过复制")
            _log.info(f"✅ 已保存Processor配置到: {target_path}（包含训练后的tokenizer）")
            
        except Exception as e:
            _log.warning(f"⚠️ 保存Processor配置失败: {e}")

    def _ensure_full_vl_assets(self, output_dir: str):
        if not self.export_save_full_vl_assets:
            return
        try:
            base_path = self.base_model_path
            if not os.path.isabs(base_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                base_path = os.path.abspath(os.path.join(project_root, base_path))
            required_files = [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "processor_config.json",
            ]
            required_dirs = [
                "image_processor",
                "processor",
            ]
            os.makedirs(output_dir, exist_ok=True)
            for fname in required_files:
                src = os.path.join(base_path, fname)
                if os.path.exists(src):
                    dst = os.path.join(output_dir, fname)
                    if not os.path.exists(dst):
                        try:
                            shutil.copy2(src, dst)
                            _log.info(f"✅ 复制缺失文件: {fname}")
                        except Exception as ce:
                            _log.warning(f"复制文件失败 {fname}: {ce}")
            for dname in required_dirs:
                srcd = os.path.join(base_path, dname)
                dstd = os.path.join(output_dir, dname)
                if os.path.isdir(srcd) and not os.path.exists(dstd):
                    try:
                        shutil.copytree(srcd, dstd)
                        _log.info(f"✅ 复制目录: {dname}")
                    except Exception as ce:
                        _log.warning(f"复制目录失败 {dname}: {ce}")
        except Exception as e:
            _log.warning(f"⚠️ 确保VL资产时出错: {e}")

    def _resolve_dataset_path(self, path_str: str) -> str:
        if not path_str:
            return None
        if os.path.isabs(path_str):
            return path_str
        project_root = Path(__file__).resolve().parents[2]  # 项目根目录
        abs_path = os.path.abspath(os.path.join(project_root, path_str))
        return abs_path

    def _load_sft_dataset(self) -> List[Dict[str, Any]]:
        return memory_extraction._load_sft_dataset(self)
    
    def _standardize_sft_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        return memory_extraction._standardize_sft_messages(self, sample)
    
    def _is_sft_within_token_limit(
        self,
        processor,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        *,
        add_generation_prompt: bool = False,
        desc: str = "SFT"
    ) -> Tuple[bool, Optional[int]]:
        """
        检查SFT样本的token数量是否在限制内
        """
        if not max_tokens or not messages:
            return True, None
        apply_fn = getattr(processor, "apply_chat_template", None)
        if not callable(apply_fn):
            tokenizer = getattr(processor, "tokenizer", None)
            apply_fn = getattr(tokenizer, "apply_chat_template", None) if tokenizer else None
        if not callable(apply_fn):
            _log.warning(f"⚠️ 无法对{desc}样本执行token长度校验（缺少apply_chat_template），默认放行")
            return True, None
        try:
            encoded = apply_fn(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
                return_tensors="pt"
            )
            seq_len = encoded["input_ids"].shape[1]
            return seq_len <= max_tokens, int(seq_len)
        except Exception as e:
            _log.warning(f"⚠️ 计算{desc}样本token长度失败，跳过该样本: {e}")
            return False, None
    
    @staticmethod
    def _get_base_tokenizer(processor):
        if hasattr(processor, "tokenizer"):
            return processor.tokenizer
        return processor

    def _filter_plain_texts_by_token_limit(
        self,
        tokenizer,
        texts: List[str],
        max_tokens: int,
        desc: str
    ) -> List[int]:
        if not max_tokens or not texts:
            indices = list(range(len(texts)))
            random.shuffle(indices)
            return indices
        eligible = []
        skipped = 0
        indices = list(range(len(texts)))
        random.shuffle(indices)
        for idx in indices:
            text = texts[idx]
            try:
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=False,
                    truncation=False
                )
                seq_len = encoded["input_ids"].shape[1]
                if seq_len <= max_tokens:
                    eligible.append(idx)
                else:
                    skipped += 1
            except Exception as e:
                _log.warning(f"⚠️ 计算{desc}样本token长度失败，跳过该样本: {e}")
                skipped += 1
        if skipped:
            _log.info(f"⚠️ {desc}长度限制：跳过 {skipped}/{len(texts)} 条超过 {max_tokens} tokens 的样本")
        return eligible

    def _sample_sft_for_epoch(
        self,
        sample_records: List[Dict[str, Any]],
        processor,
        max_tokens: int,
        total_target: int
    ) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]], List[int]]:
        """
        从标准化SFT样本中抽取total_target条满足长度限制的样本
        
        Returns:
            (selected_full_texts, selected_messages, selected_sources)
            - selected_full_texts: 包含<think>段的完整文本（用于夹心训练）
            - selected_messages: 标准化的messages（用于纯SFT和前缀训练）
            - selected_sources: 原始数据集索引
        """
        if not sample_records or total_target <= 0:
            return [], [], []
        candidate_indices = list(range(len(sample_records)))
        random.shuffle(candidate_indices)
        selected_messages: List[List[Dict[str, Any]]] = []
        selected_message_sources: List[int] = []
        selected_full_texts: List[Dict[str, Any]] = []
        for idx in candidate_indices:
            record = sample_records[idx]
            messages = record["messages"]
            origin_index = record["index"]
            if max_tokens:
                within_limit, _ = self._is_sft_within_token_limit(
                    processor,
                    messages,
                    max_tokens,
                    add_generation_prompt=False,
                    desc="SFT epoch sampler"
                )
                if not within_limit:
                    continue
            selected_messages.append(messages)
            selected_message_sources.append(origin_index)
            try:
                full_text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                start_tag = "<think>"
                end_tag = "</think>"
                start_idx = full_text.find(start_tag)
                end_idx = full_text.find(end_tag)
                if start_idx != -1 and end_idx != -1:
                    selected_full_texts.append({
                        "full_text": full_text,
                        "thinking_start": start_idx,
                        "thinking_end": end_idx + len(end_tag)
                    })
                else:
                    selected_full_texts.append(None)
            except Exception as e:
                _log.debug(f"处理SFT样本失败: {e}")
                selected_full_texts.append(None)
            if len(selected_messages) >= total_target:
                break
        if len(selected_messages) < total_target:
            raise ValueError(
                f"SFT抽样不足：需要 {total_target} 条满足长度限制的样本，"
                f"但仅收集到 {len(selected_messages)} 条。"
            )
        return selected_full_texts, selected_messages, selected_message_sources
    
    def _build_simple_sft_batch(self, processor, messages: List[List[Dict[str, Any]]]):
        """
        简单SFT批处理：将messages转成input_ids并直接用自回归标签（不区分mask）。
        如果messages中包含字符串格式的文本（用于<recall> token训练），则只让<recall> token参与训练。
        """
        # 检查是否有字符串格式的文本（用于<recall> token训练）
        recall_token_id = None
        try:
            recall_token_id = processor.tokenizer.convert_tokens_to_ids("<recall>")
        except:
            pass
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for msg in messages:
            # 如果是字符串格式（用于<recall> token训练）
            if isinstance(msg, str):
                # 直接tokenize文本
                encoded = processor.tokenizer(
                    msg,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=False,
                    truncation=False
                )
                input_ids = encoded["input_ids"][0]  # [seq_len]
                attention_mask = torch.ones_like(input_ids)
                
                # 创建labels，默认全部mask（-100）
                labels = torch.full_like(input_ids, -100)
                
                # 找到<recall> token的位置，只让这个token参与训练
                if recall_token_id is not None:
                    recall_positions = (input_ids == recall_token_id).nonzero(as_tuple=True)[0]
                    if len(recall_positions) > 0:
                        # 只让最后一个<recall> token参与训练
                        last_recall_pos = recall_positions[-1].item()
                        labels[last_recall_pos] = input_ids[last_recall_pos]
                        _log.debug(f"找到<recall> token位置: {last_recall_pos}, 已设置为参与训练")
                    else:
                        _log.warning(f"⚠️ 文本中未找到<recall> token: {msg}")
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)
            else:
                # 原有的messages格式处理
                batch_inputs = processor.apply_chat_template(
                    [msg], tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_tensors="pt"
                )
                input_ids = batch_inputs["input_ids"][0]  # [seq_len]
                attention_mask = batch_inputs.get("attention_mask", (input_ids != 0).long())[0]
                labels = input_ids.clone()
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)
        
        # 对batch进行padding
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(batch_input_ids)):
            pad_len = max_len - len(batch_input_ids[i])
            padded_input_ids.append(torch.cat([batch_input_ids[i], torch.zeros(pad_len, dtype=batch_input_ids[i].dtype)]))
            padded_attention_mask.append(torch.cat([batch_attention_mask[i], torch.zeros(pad_len, dtype=batch_attention_mask[i].dtype)]))
            padded_labels.append(torch.cat([batch_labels[i], torch.full((pad_len,), -100, dtype=batch_labels[i].dtype)]))
        
        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)
        labels = torch.stack(padded_labels)
        
        return input_ids, attention_mask, labels
    
    def save_memory_chat_histories_to_storage(self):
        """
        将内存中的聊天记录保存到JSON文件（训练前调用）
        这样可以确保训练时使用最新的聊天记录，同时清理内存
        
        注意：直接读取 chat.history_manager 中的运行时聊天记录并写入配置目录。
        """
        _log.info("开始保存内存中的聊天记录到存储...")
        
        # 读取当前内存中的聊天记录快照
        with chat_history_lock:
            group_histories_snapshot = {
                chat_id: list(history)
                for chat_id, history in group_chat_histories.items()
            }
            private_histories_snapshot = {
                chat_id: list(history)
                for chat_id, history in private_chat_histories.items()
            }
        
        if len(group_histories_snapshot) == 0 and len(private_histories_snapshot) == 0:
            _log.warning("⚠️ 当前内存中的聊天记录为空，跳过保存")
            return
        
        _log.info(f"📊 内存中的聊天记录统计:")
        _log.info(f"   群聊数量: {len(group_histories_snapshot)}")
        _log.info(f"   私聊数量: {len(private_histories_snapshot)}")
        
        # 详细统计每个聊天的消息数
        for chat_id, history in group_histories_snapshot.items():
            _log.info(f"   群聊 {chat_id}: {len(history)} 条消息")
        for chat_id, history in private_histories_snapshot.items():
            _log.info(f"   私聊 {chat_id}: {len(history)} 条消息")
        
        chat_history_storage_dir = self.chat_history_storage_dir
        if not chat_history_storage_dir:
            _log.warning("⚠️ 未配置聊天记录存储目录，无法保存")
            return
        
        os.makedirs(chat_history_storage_dir, exist_ok=True)
        _log.info(f"✅ 确保聊天记录存储目录存在: {chat_history_storage_dir}")
        
        saved_count = 0

        def _get_pending_messages(chat_type_key: str, chat_id: str, history: List[Dict[str, Any]]):
            last_saved = self._saved_history_counts[chat_type_key].get(chat_id, 0)
            if last_saved < 0 or last_saved > len(history):
                last_saved = 0
            if last_saved == len(history):
                return [], len(history)
            return history[last_saved:], len(history)
        
        # 保存群聊记录
        for chat_id, history in group_histories_snapshot.items():
            if not history:
                continue
            pending_messages, final_len = _get_pending_messages("group", chat_id, history)
            if not pending_messages:
                continue
            try:
                save_chat_history_to_storage(self.config, "group", chat_id, pending_messages)
                saved_count += len(pending_messages)
                self._saved_history_counts["group"][chat_id] = final_len
                _log.info(f"✅ 保存群聊 {chat_id} 的 {len(pending_messages)} 条新消息到 {chat_history_storage_dir}")
            except Exception as e:
                _log.warning(f"保存群聊 {chat_id} 失败: {e}", exc_info=True)
                try:
                    self._save_chat_history_directly("group", chat_id, pending_messages)
                    saved_count += len(pending_messages)
                    self._saved_history_counts["group"][chat_id] = final_len
                    _log.info(f"✅ 使用直接保存方式成功保存群聊 {chat_id} 的 {len(pending_messages)} 条新消息")
                except Exception as e2:
                    _log.error(f"直接保存也失败: {e2}", exc_info=True)
        
        # 保存私聊记录
        for chat_id, history in private_histories_snapshot.items():
            if not history:
                continue
            pending_messages, final_len = _get_pending_messages("private", chat_id, history)
            if not pending_messages:
                continue
            try:
                save_chat_history_to_storage(self.config, "private", chat_id, pending_messages)
                saved_count += len(pending_messages)
                self._saved_history_counts["private"][chat_id] = final_len
                _log.info(f"✅ 保存私聊 {chat_id} 的 {len(pending_messages)} 条新消息到 {chat_history_storage_dir}")
            except Exception as e:
                _log.warning(f"保存私聊 {chat_id} 失败: {e}", exc_info=True)
                try:
                    self._save_chat_history_directly("private", chat_id, pending_messages)
                    saved_count += len(pending_messages)
                    self._saved_history_counts["private"][chat_id] = final_len
                    _log.info(f"✅ 使用直接保存方式成功保存私聊 {chat_id} 的 {len(pending_messages)} 条新消息")
                except Exception as e2:
                    _log.error(f"直接保存也失败: {e2}", exc_info=True)
        
        _log.info(f"✅ 共保存 {saved_count} 条内存中的聊天记录到存储")
    
    def load_chat_histories_from_json_only(self) -> List[Dict[str, Any]]:
        """
        只从JSON文件加载聊天记录（训练时使用，不从内存加载）
        
        Returns:
            所有聊天记录的列表
        """
        all_messages = []
        json_count = 0
        
        chat_history_storage_dir = self.chat_history_storage_dir
        if not chat_history_storage_dir:
            _log.warning("⚠️ 未配置聊天记录存储目录，返回空列表")
            return all_messages
        
        # 加载JSON文件
        _log.info(f"检查聊天记录存储目录: {chat_history_storage_dir}")
        if os.path.exists(chat_history_storage_dir):
            json_files = list(Path(chat_history_storage_dir).glob("*.json"))
            _log.info(f"找到 {len(json_files)} 个JSON文件")
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # 如果data是列表，直接使用
                            json_count += len(data)
                            all_messages.extend(data)
                            _log.info(f"从 {json_file.name} 加载 {len(data)} 条消息（列表格式）")
                        elif isinstance(data, dict) and "messages" in data:
                            # 如果data是字典且包含messages字段
                            messages = data.get("messages", [])
                            json_count += len(messages)
                            # 需要将消息转换为统一格式
                            chat_type = data.get("chat_type", "unknown")
                            chat_id = data.get("chat_id", "unknown")
                            for msg in messages:
                                all_messages.append({
                                    "chat_type": chat_type,
                                    "chat_id": chat_id,
                                    "message": msg
                                })
                            _log.info(f"从 {json_file.name} 加载 {len(messages)} 条消息（字典格式，chat_type={chat_type}, chat_id={chat_id}）")
                        else:
                            _log.warning(f"JSON文件 {json_file.name} 格式不正确，跳过")
                except Exception as e:
                    _log.warning(f"加载 {json_file} 失败: {e}", exc_info=True)
        else:
            _log.warning(f"聊天记录存储目录不存在: {chat_history_storage_dir}")
        
        _log.info(f"总共从JSON文件加载 {len(all_messages)} 条消息")
        return all_messages
    
    def load_chat_histories(self) -> List[Dict[str, Any]]:
        """
        加载所有聊天记录（包括内存中的和历史JSON文件）
        注意：调用此函数前应该先调用save_memory_chat_histories_to_storage()保存内存中的记录
        
        Returns:
            所有聊天记录的列表
        """
        all_messages = []
        
        # 1. 加载内存中的聊天记录（最新的30条）
        # 注意：这些记录应该在训练前已经保存到JSON文件了
        memory_count = 0
        with chat_history_lock:
            group_histories_snapshot = {
                chat_id: list(history)
                for chat_id, history in group_chat_histories.items()
            }
            private_histories_snapshot = {
                chat_id: list(history)
                for chat_id, history in private_chat_histories.items()
            }
        
        if group_histories_snapshot or private_histories_snapshot:
            _log.info(f"📊 内存中的聊天记录统计（加载时）:")
            _log.info(f"   群聊数量: {len(group_histories_snapshot)}")
            _log.info(f"   私聊数量: {len(private_histories_snapshot)}")
            
            for chat_id, history in group_histories_snapshot.items():
                history_len = len(history)
                memory_count += history_len
                _log.info(f"   群聊 {chat_id}: {history_len} 条消息")
                all_messages.extend([
                    {
                        "chat_type": "group",
                        "chat_id": chat_id,
                        "message": msg
                    }
                    for msg in history
                ])
            
            for chat_id, history in private_histories_snapshot.items():
                history_len = len(history)
                memory_count += history_len
                _log.info(f"   私聊 {chat_id}: {history_len} 条消息")
                all_messages.extend([
                    {
                        "chat_type": "private",
                        "chat_id": chat_id,
                        "message": msg
                    }
                    for msg in history
                ])
        else:
            _log.info("📊 内存中暂无聊天记录")
        
        _log.info(f"从内存加载 {memory_count} 条消息")
        
        # 2. 加载历史JSON文件
        json_count = 0
        _log.info(f"检查聊天记录存储目录: {self.chat_history_storage_dir}")
        if os.path.exists(self.chat_history_storage_dir):
            json_files = list(Path(self.chat_history_storage_dir).glob("*.json"))
            _log.info(f"找到 {len(json_files)} 个JSON文件")
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # 如果data是列表，直接使用
                            json_count += len(data)
                            all_messages.extend(data)
                            _log.info(f"从 {json_file.name} 加载 {len(data)} 条消息（列表格式）")
                        elif isinstance(data, dict) and "messages" in data:
                            # 如果data是字典且包含messages字段
                            messages = data.get("messages", [])
                            json_count += len(messages)
                            # 需要将消息转换为统一格式
                            chat_type = data.get("chat_type", "unknown")
                            chat_id = data.get("chat_id", "unknown")
                            for msg in messages:
                                all_messages.append({
                                    "chat_type": chat_type,
                                    "chat_id": chat_id,
                                    "message": msg
                                })
                            _log.info(f"从 {json_file.name} 加载 {len(messages)} 条消息（字典格式，chat_type={chat_type}, chat_id={chat_id}）")
                        else:
                            _log.warning(f"JSON文件 {json_file.name} 格式不正确，跳过")
                except Exception as e:
                    _log.warning(f"加载 {json_file} 失败: {e}", exc_info=True)
        else:
            _log.warning(f"聊天记录存储目录不存在: {self.chat_history_storage_dir}")
        
        _log.info(f"总共加载 {len(all_messages)} 条消息（内存: {memory_count}, JSON: {json_count}）")
        return all_messages
    
    def _save_chat_history_directly(self, chat_type: str, chat_id: str, messages: List[Dict[str, Any]]):
        """
        直接保存聊天记录到JSON文件（当无法使用api_server的函数时）
        
        Args:
            chat_type: "group" 或 "private"
            chat_id: 群ID或用户ID
            messages: 要保存的消息列表
        """
        if not self.chat_history_storage_dir:
            raise ValueError("聊天记录存储目录未配置")
        
        # 确保目录存在
        os.makedirs(self.chat_history_storage_dir, exist_ok=True)
        
        # 创建存储文件路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{chat_type}_{chat_id}_{timestamp}.json"
        filepath = os.path.join(self.chat_history_storage_dir, filename)
        
        # 保存到JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "chat_type": chat_type,
                "chat_id": chat_id,
                "timestamp": timestamp,
                "messages": messages
            }, f, ensure_ascii=False, indent=2)
        
        _log.info(f"✅ 直接保存 {len(messages)} 条聊天记录到: {filename}")
    
    def extract_memory_entries(self, chat_messages: List[Dict[str, Any]], model=None, processor=None) -> Optional[str]:
        return memory_extraction.extract_memory_entries(self, chat_messages, model, processor)

    def _batch_extract_embeddings(self, memory_texts: List[str], model, processor, max_tokens: int):
        return memory_extraction._batch_extract_embeddings(self, memory_texts, model, processor, max_tokens)

    def _append_memory_text_to_file(self, memory_text: str, file_path: str):
        return memory_extraction._append_memory_text_to_file(self, memory_text, file_path)

    def _extract_sft_vectors_for_recall_training(self, num_memory_entries: int, model, processor) -> Optional[str]:
        return memory_extraction.extract_sft_vectors_for_recall_training(self, num_memory_entries, model, processor)

    def _load_memory_texts_from_file(self, file_path: str) -> List[str]:
        return memory_extraction._load_memory_texts_from_file(self, file_path)

    def _save_training_data_batch(self, texts: List[str], embeddings: List[torch.Tensor]):
        return memory_extraction._save_training_data_batch(self, texts, embeddings)

    def save_memory_embeddings_from_file(self, training_data_path: str):
        """
        从训练数据文件读取监督向量并保存到MemoryVectorDB

        Args:
            training_data_path: 训练数据文件路径
        """
        _log.info("从训练数据文件保存监督向量到MemoryVectorDB...")

        try:
            # 加载训练数据
            training_data = torch.load(training_data_path, map_location='cpu')
            embeddings = training_data.get('embeddings')

            if embeddings is None or len(embeddings) == 0:
                _log.warning("⚠️ 训练数据文件中没有向量数据")
                return

            # 记忆数据库文件路径
            memory_db_path = os.path.join(self.memory_db_dir, "memory_embeddings.pt")

            # 创建MemoryVectorDB并加载现有数据（如果存在）
            embedding_dim = embeddings.shape[-1]
            storage_device = "cpu"
            memory_db = MemoryVectorDB(embedding_dim=embedding_dim, device=storage_device)
            _log.info(f"MemoryVectorDB将在 {storage_device} 上执行保存操作，以避免GPU设备编号不一致问题")

            # 如果文件已存在，先加载现有数据
            if os.path.exists(memory_db_path):
                try:
                    memory_db.load_from_pt(memory_db_path)
                    _log.info(f"加载现有记忆数据库，已有 {memory_db.embeddings.shape[0]} 个向量")
                except Exception as e:
                    _log.warning(f"加载现有记忆数据库失败: {e}，将创建新的数据库")

            # 追加新的向量
            memory_db.add_vectors(embeddings)

            # 保存到文件
            memory_db.save_to_pt(memory_db_path)

            _log.info(f"✅ 成功保存 {len(embeddings)} 个新的监督向量到 {memory_db_path}（总计 {memory_db.embeddings.shape[0]} 个向量）")

        except Exception as e:
            _log.error(f"从文件保存记忆向量失败: {e}")
            raise

    def _parse_memory_entries(self, generated_text: str) -> List[str]:
        return memory_extraction._parse_memory_entries(self, generated_text)
    
    def save_memory_embeddings(self, memory_entries: List[Tuple[str, torch.Tensor]]):
        """
        保存监督向量到MemoryVectorDB（追加模式）
        
        Args:
            memory_entries: (记忆文本, 监督向量) 的列表
        """
        _log.info("保存监督向量到MemoryVectorDB...")
        
        # 提取所有监督向量
        embeddings = torch.stack([entry[1] for entry in memory_entries])
        
        # 记忆数据库文件路径
        memory_db_path = os.path.join(self.memory_db_dir, "memory_embeddings.pt")
        
        # 创建MemoryVectorDB并加载现有数据（如果存在）
        # 注意：MemoryVectorDB主要用于存储，应该使用CPU以避免GPU设备问题
        embedding_dim = embeddings.shape[-1]
        storage_device = "cpu"
        memory_db = MemoryVectorDB(embedding_dim=embedding_dim, device=storage_device)
        _log.debug(f"MemoryVectorDB将在 {storage_device} 上执行保存操作")
        
        # 如果文件已存在，先加载现有数据
        if os.path.exists(memory_db_path):
            try:
                memory_db.load_from_pt(memory_db_path)
                _log.info(f"加载现有记忆数据库，已有 {memory_db.embeddings.shape[0]} 个向量")
            except Exception as e:
                _log.warning(f"加载现有记忆数据库失败: {e}，将创建新的数据库")
        
        # 追加新的向量
        memory_db.add_vectors(embeddings)
        
        # 保存到文件
        memory_db.save_to_pt(memory_db_path)

        _log.info(f"✅ 成功保存 {len(memory_entries)} 个新的监督向量到 {memory_db_path}（总计 {memory_db.embeddings.shape[0]} 个向量）")

        # 注意：memory_entries暂时保留在内存中，用于后续训练
        # 训练完成后在cleanup_after_training中统一清理

    def load_training_model(self):
        """加载统一的训练模型（用于记忆提取和训练）"""
        _log.info(f"加载训练模型: {self.base_model_path}")

        # 使用与initialize_model相同的加载逻辑
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        # 将相对路径转换为绝对路径
        model_path = self.base_model_path
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.abspath(os.path.join(project_root, model_path))

        # 检查是否为本地路径
        is_local_path = os.path.exists(model_path) and os.path.isdir(model_path)

        try:
            # 加载processor（使用AutoProcessor而不是AutoTokenizer，因为需要处理图片和视频）
            # 正常推理时使用AutoProcessor，训练时也应该使用AutoProcessor
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=is_local_path
            )

            # 准备加载参数
            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "local_files_only": is_local_path
            }
            
            # 根据设备配置决定device_map（使用与TrainingModelContext相同的逻辑）
            multi_gpu_config = self.config.get("model", {}).get("multi_gpu", {})
            multi_gpu_enabled = multi_gpu_config.get("enabled", False)
            
            if isinstance(self.device, list) and multi_gpu_enabled:
                # 多GPU配置：使用优化的分配策略
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                cuda_visible_set = bool(cuda_visible)
                max_memory_config = multi_gpu_config.get("max_memory", {})
                allocation = _optimize_multi_gpu_allocation(self.device, max_memory_config, cuda_visible_set=cuda_visible_set)
                load_kwargs["device_map"] = allocation["device_map"]
                if allocation["max_memory"]:
                    load_kwargs["max_memory"] = allocation["max_memory"]
                _log.info(f"🔧 训练模型: 指定设备{self.device}，使用优化的分配策略")
            elif isinstance(self.device, str) and self.device.startswith("cuda"):
                load_kwargs["device_map"] = {"": self.device}
                _log.info(f"🔧 训练模型: 单GPU模式，设备映射到 {self.device}")
            else:
                load_kwargs["device_map"] = "auto"
                _log.info("🔧 训练模型: 使用自动设备分配")

            # 加载模型 - 注意这里使用Qwen3VLForConditionalGeneration，不是AutoModelForCausalLM
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs
            )

            # 添加特殊token（如果没有的话）
            # 使用MemoryTokenManager，与正常推理时保持一致
            from memory.token_manager import MemoryTokenManager
            token_manager = MemoryTokenManager(model, processor.tokenizer)
            recall_token_ids = token_manager.check_and_add_tokens(perturbation_std=0.02)
            _log.info(f"✅ 特殊token处理完成: {recall_token_ids}")

            _log.info("✅ 训练模型加载成功")
            return model, processor

        except Exception as e:
            _log.error(f"❌ 加载训练模型失败: {e}")
            raise
    
    def train_recall_token(self, training_data_path: str, model=None, processor=None, sft_vectors_path: Optional[str] = None) -> str:
        """
        第一步训练：训练<recall> token的embedding

        Args:
            training_data_path: 训练数据文件路径

        Returns:
            训练后的模型路径
        """
        # 尝试确保训练模块已加载
        if not _ensure_training_modules_loaded():
            raise ImportError("训练模块不可用，无法执行训练。请检查 src/qqbot_memory/training 是否完整。")

        _log.info("开始第一步训练：<recall> token embedding训练...")

        trainer = None
        try:
            # 从文件加载训练数据
            training_data = torch.load(training_data_path, map_location='cpu')
            texts = training_data.get('texts', [])
            embeddings = training_data.get('embeddings')

            if not texts or embeddings is None:
                raise ValueError("训练数据文件无效或为空")

            # 创建训练器（传入预加载的模型）
            lora_r = self.lora_config.get("r", 8)
            lora_alpha = self.lora_config.get("lora_alpha", 32)
            lora_dropout = self.lora_config.get("lora_dropout", 0.1)
            # 获取第一步训练的LoRA目标模块（如果配置了，只使用Q和V以减少显存）
            step1_lora_target_modules = self.lora_config.get("step1_lora_target_modules", None)
            # 获取梯度累积步数
            gradient_accumulation_steps = self.config.get("model", {}).get("multi_gpu", {}).get("gradient_accumulation_steps", 1)
            # 获取max_memory配置
            max_memory = self.config.get("model", {}).get("multi_gpu", {}).get("max_memory")

            # 获取第一步训练的最大序列长度（None表示不限制）
            max_length_recall_training = self.config.get("model", {}).get("training", {}).get("training_config", {}).get("max_length_recall_training")
            if max_length_recall_training is None:
                max_length_recall_training = None  # 明确设置为None

            trainer = RecallMemoryTrainer(
                self.base_model_path,
                device=self.device,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                original_device=self.original_device,
                preloaded_model=model,
                preloaded_tokenizer=processor,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_memory=max_memory,
                max_length=max_length_recall_training,
                lora_target_modules=step1_lora_target_modules
                # 第一步训练不设置epoch_end_hook，不插入SFT
            )

            # 准备训练数据：分别加载记忆条目和SFT向量
            memory_data = None
            sft_data = None

            # 加载记忆条目数据
            if os.path.exists(training_data_path):
                memory_data = torch.load(training_data_path, map_location='cpu')
                memory_count = len(memory_data.get('texts', []))
                _log.info(f"📖 加载记忆条目数据: {memory_count} 条")

            # 加载SFT向量数据
            if sft_vectors_path and os.path.exists(sft_vectors_path):
                sft_data = torch.load(sft_vectors_path, map_location='cpu')
                sft_count = len(sft_data.get('texts', []))
                _log.info(f"📖 加载SFT向量数据: {sft_count} 条")

            # 创建训练数据：记忆条目 + 随机抽取的SFT向量
            if memory_data and sft_data:
                memory_texts = memory_data.get('texts', [])
                memory_embeddings = memory_data.get('embeddings', torch.empty(0))
                sft_texts = sft_data.get('texts', [])
                sft_embeddings = sft_data.get('embeddings', torch.empty(0))

                memory_count = len(memory_texts)
                sft_total_count = len(sft_texts)

                # 计算需要的SFT向量数量：1.5倍于记忆条目数量
                required_sft_count = int(memory_count * 1.5)
                sft_max_tokens = int(self.training_config.get("sft_max_tokens") or 0)
                tokenizer_for_sft = self._get_base_tokenizer(processor)
                eligible_indices = self._filter_plain_texts_by_token_limit(
                    tokenizer_for_sft,
                    sft_texts,
                    sft_max_tokens,
                    desc="Recall阶段SFT向量"
                )
                if len(eligible_indices) < required_sft_count:
                    raise ValueError(
                        f"有效的SFT向量不足：需要 {required_sft_count} 条，"
                        f"但仅有 {len(eligible_indices)} 条满足 <= {sft_max_tokens} tokens 的限制。"
                    )
                import random
                random.seed(42)
                selected_indices = random.sample(eligible_indices, required_sft_count)
                selected_sft_texts = [sft_texts[i] for i in selected_indices]
                selected_sft_embeddings = sft_embeddings[selected_indices]
                actual_sft_count = len(selected_sft_texts)

                # 合并数据
                combined_texts = memory_texts + selected_sft_texts
                combined_embeddings = torch.cat([memory_embeddings, selected_sft_embeddings], dim=0)

                # 创建包含元信息的训练数据
                training_data = {
                    'texts': combined_texts,
                    'embeddings': combined_embeddings,
                    'memory_count': memory_count,  # 记忆条目数量
                    'sft_count': actual_sft_count  # SFT向量数量
                }

                temp_data_path = os.path.join(self.memory_db_dir, "temp_recall_training_data.pt")
                torch.save(training_data, temp_data_path)
                _log.info(f"✅ 已准备训练数据: {memory_count} 条记忆条目 + {actual_sft_count} 条SFT向量")

                # 删除SFT向量文件
                try:
                    os.remove(sft_vectors_path)
                    _log.info("🗑️ 已删除临时SFT向量文件")
                except Exception as e:
                    _log.warning(f"⚠️ 删除SFT向量文件失败: {e}")

            elif memory_data:
                # 只有记忆条目数据
                temp_data_path = training_data_path
                _log.info("ℹ️ 只有记忆条目数据，将直接使用")
            else:
                raise ValueError("❌ 没有找到有效的训练数据")

            # 训练
            embedding_epochs = self.training_config.get("embedding_epochs", 10)
            batch_size = self.training_config.get("batch_size", 4)
            learning_rate = float(self.training_config.get("learning_rate", 1e-4))

            step1_save_path = os.path.join(self.trained_model_dir, "step1_recall_token_trained")
            self._prepare_output_dir(step1_save_path)

            # Step1 只训练特殊token，此阶段不插入SFT
            self._current_epoch_sample_n = None
            res = trainer.train(
                pt_file_path=temp_data_path,
                num_epochs=embedding_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_path=step1_save_path
            )
            _ = res

            # 合并LoRA并保存
            model_path = trainer.merge_and_save_model(step1_save_path)
            if self.export_save_full_vl_assets:
                self._ensure_full_vl_assets(model_path)

            # 保存Processor配置
            self._save_processor_to_path(model_path)

            _log.info(f"第一步训练完成，模型保存在: {model_path}")
            return model_path

        except Exception as e:
            _log.error(f"第一步训练失败: {e}")
            raise
        finally:
            # 清理训练器创建的所有模型实例
            if trainer is not None:
                trainer.cleanup()
                del trainer

            # 删除临时合并的训练数据文件（如果存在）
            try:
                temp_merge_path = os.path.join(self.memory_db_dir, "temp_recall_training_data.pt")
                if os.path.exists(temp_merge_path):
                    os.remove(temp_merge_path)
                    _log.info("🗑️ 已删除临时合并的训练数据文件")
            except Exception as e:
                _log.warning(f"⚠️ 删除临时训练数据文件失败: {e}")
    
    def train_memory_decoding(self, training_data_path: str, model_path: str) -> str:
        """
        第二步训练：训练记忆解码能力

        Args:
            training_data_path: 训练数据文件路径
            model_path: 第一步训练后的模型路径

        Returns:
            训练后的模型路径
        """
        # 尝试确保训练模块已加载
        if not _ensure_training_modules_loaded():
            raise ImportError("训练模块不可用，无法执行训练。请检查 src/qqbot_memory/training 是否完整。")

        _log.info("开始第二步训练：记忆解码训练...")

        trainer = None
        try:
            # 从文件加载训练数据
            training_data = torch.load(training_data_path, map_location='cpu')
            texts = training_data.get('texts', [])
            embeddings = training_data.get('embeddings')

            if not texts or embeddings is None:
                raise ValueError("训练数据文件无效或为空")

            # 直接使用传入的训练数据文件路径
            temp_data_path = training_data_path

            # 在创建训练器之前，先确保模型中的特殊token存在
            # 使用MemoryTokenManager加载并检查token，然后将处理过的模型传递给训练器
            _log.info(f"🔧 预处理模型token: {model_path}")
            preloaded_model, preloaded_processor = TrainingModelContext.load_training_model(
                model_path, self.device, self.config.get("model", {}).get("multi_gpu", {})
            )
            _log.info("✅ 模型token预处理完成，已添加特殊token")

            # 创建训练器（传入预处理过的模型和tokenizer）
            lora_r = self.lora_config.get("r", 8)
            lora_alpha = self.lora_config.get("lora_alpha", 32)
            lora_dropout = self.lora_config.get("lora_dropout", 0.1)
            # 获取第二步训练的LoRA目标模块（如果配置了，使用完整配置）
            step2_lora_target_modules = self.lora_config.get("step2_lora_target_modules", None)
            # 获取梯度累积步数
            gradient_accumulation_steps = self.config.get("model", {}).get("multi_gpu", {}).get("gradient_accumulation_steps", 1)
            # 获取max_memory配置
            max_memory = self.config.get("model", {}).get("multi_gpu", {}).get("max_memory")

            dataset_max_length = int(self.training_config.get("memory_dataset_max_length", 3000) or 3000)
            test_sample_count = int(self.training_config.get("memory_test_sample_count", 2) or 2)
            test_max_new_tokens = int(self.training_config.get("memory_test_max_new_tokens", 300) or 300)
            test_use_cache = bool(self.training_config.get("memory_test_use_cache", False))
            activation_prompts = self.guides_config.get("activation_prompts")
            end_prompts = self.guides_config.get("end_prompts")
            
            if not activation_prompts:
                activation_prompts = self.default_guides.get("activation_prompts")
            if not end_prompts:
                end_prompts = self.default_guides.get("end_prompts")
            
            if not activation_prompts:
                raise ValueError("activation_prompts 未配置，请在 prompts.yaml 的 memory_training.guides.activation_prompts 中设置")
            if not end_prompts:
                raise ValueError("end_prompts 未配置，请在 prompts.yaml 的 memory_training.guides.end_prompts 中设置")
            
            _log.info(f"📝 激活语: {activation_prompts}")
            _log.info(f"📝 结束语: {end_prompts}")

            trainer = EnhancedTextMemoryTrainer(
                model_path,
                device=self.device,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                original_device=self.original_device,
                preloaded_model=preloaded_model,  # 传入预处理过的模型
                preloaded_tokenizer=preloaded_processor,  # 传入预处理过的tokenizer
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_memory=max_memory,
                generation_config=self.config.get("generation", {}),
                lora_target_modules=step2_lora_target_modules,
                dataset_max_length=dataset_max_length,
                test_sample_count=test_sample_count,
                test_max_new_tokens=test_max_new_tokens,
                test_use_cache=test_use_cache,
                activation_prompts=activation_prompts,
                end_prompts=end_prompts,
                guide_text=self.guide_text,
            )

            # 训练
            memory_epochs = self.training_config.get("memory_epochs", 20)
            batch_size = self.training_config.get("batch_size", 4)
            learning_rate = float(self.training_config.get("learning_rate", 1e-4))

            step2_save_path = os.path.join(self.trained_model_dir, "step2_memory_decoding_trained")
            self._prepare_output_dir(step2_save_path)

            # 设置SFT每epoch采样参考数（与记忆条目数量相同）
            training_data = torch.load(temp_data_path, map_location='cpu')
            memory_texts = training_data.get('texts', [])
            self._current_epoch_sample_n = len(memory_texts)
            sft_full_texts: List[Dict[str, Any]] = []
            sft_messages_list: List[List[Dict[str, Any]]] = []
            sft_epoch_sampler = None
            sft_epoch_sampler_total = 0
            sft_max_tokens = int(self.training_config.get("sft_max_tokens") or 0)
            if self.sft_enabled and self.sft_path:
                try:
                    raw_sft_samples = self._load_sft_dataset()
                    standardized_sft_samples = []
                    for idx, sample in enumerate(raw_sft_samples):
                        messages = self._standardize_sft_messages(sample)
                        if messages:
                            standardized_sft_samples.append({
                                "messages": messages,
                                "index": idx
                            })
                    sft_epoch_sampler_total = len(standardized_sft_samples)
                    if standardized_sft_samples:
                        def _epoch_sampler(total_target: int):
                            return self._sample_sft_for_epoch(
                                standardized_sft_samples,
                                preloaded_processor,
                                sft_max_tokens,
                                total_target
                            )
                        sft_epoch_sampler = _epoch_sampler
                except Exception as sampler_error:
                    _log.warning(f"⚠️ 初始化SFT采样器失败: {sampler_error}")
            # 注意：sft_messages_list 和 sft_full_texts 现在是空的，因为SFT数据改为每个epoch动态采样
            # 长度检查会在每个epoch的 refresh_epoch_data 中通过采样器内部进行
            # 这里只需要确保采样器创建成功即可
            if self.sft_enabled and self.sft_path and memory_texts:
                if sft_epoch_sampler is None:
                    _log.warning("⚠️ SFT采样器未创建，第二步训练将无法使用SFT数据")
            res2 = trainer.train(
                pt_file_path=temp_data_path,
                num_epochs=memory_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                noise_std=0.01,
                save_path=step2_save_path,
                sft_full_texts=sft_full_texts if sft_full_texts else None,
                sft_messages_list=sft_messages_list if sft_messages_list else None,
                sft_epoch_sampler=sft_epoch_sampler,
                sft_epoch_sampler_total=sft_epoch_sampler_total,
            )
            _ = res2

            # 合并LoRA并保存
            final_model_path = trainer.merge_and_save_model(step2_save_path)
            if self.export_save_full_vl_assets:
                self._ensure_full_vl_assets(final_model_path)

            # 保存Processor配置
            self._save_processor_to_path(final_model_path)

            _log.info(f"第二步训练完成，模型保存在: {final_model_path}")
            return final_model_path

        except Exception as e:
            _log.error(f"第二步训练失败: {e}")
            raise
        finally:
            # 清理训练器创建的所有模型实例
            if trainer is not None:
                trainer.cleanup()
                del trainer

    def cleanup_after_training(self):
        """
        训练完成后清理临时文件
        
        注意：
        - JSON聊天记录文件会被删除（已用于训练，不再需要）
        - 临时训练数据文件会被删除（训练完成后不再需要）
        - 内存中的聊天缓存会被清空（训练完成后不再需要）
        - 记忆向量数据库（memory_embeddings.pt）会被保留（这是训练好的记忆，需要保留）
        """
        _log.info("清理训练后的临时文件和缓存...")
        
        # 1. 清空JSON聊天记录文件（训练完成后不再需要）
        if os.path.exists(self.chat_history_storage_dir):
            json_files = list(Path(self.chat_history_storage_dir).glob("*.json"))
            deleted_count = 0
            for json_file in json_files:
                try:
                    os.remove(json_file)
                    deleted_count += 1
                    _log.info(f"删除JSON文件: {json_file.name}")
                except Exception as e:
                    _log.warning(f"删除JSON文件失败 {json_file}: {e}")
            if deleted_count > 0:
                _log.info(f"✅ 共删除 {deleted_count} 个JSON聊天记录文件")
        
        # 2. 删除临时训练数据文件
        temp_data_path = os.path.join(self.memory_db_dir, "temp_training_data.pt")
        if os.path.exists(temp_data_path):
            try:
                os.remove(temp_data_path)
                _log.info(f"✅ 删除临时训练数据文件: temp_training_data.pt")
            except Exception as e:
                _log.warning(f"删除临时训练数据文件失败: {e}")
        
        # 3. 清空内存中的聊天缓存
        try:
            with chat_history_lock:
                group_count = len(group_chat_histories)
                private_count = len(private_chat_histories)
                group_chat_histories.clear()
                private_chat_histories.clear()
            _log.info(f"✅ 清空内存中的聊天缓存（群聊: {group_count}, 私聊: {private_count}）")
        except Exception as e:
            _log.warning(f"清空内存缓存失败: {e}")
        
        # 4. 记忆向量数据库（memory_embeddings.pt）会被保留，不删除
        memory_db_path = os.path.join(self.memory_db_dir, "memory_embeddings.pt")
        if os.path.exists(memory_db_path):
            _log.info(f"📌 记忆向量数据库已保留: {memory_db_path}（这是训练好的记忆，不会被删除）")
        
        # 5. 清理上传目录中的临时文件
        uploads_dir = Path(self._project_root) / "uploads"
        if uploads_dir.exists():
            sub_dirs = ["images", "videos", "audios", "files"]
            for sub in sub_dirs:
                target_dir = uploads_dir / sub
                if not target_dir.exists():
                    continue
                deleted = 0
                for item in target_dir.iterdir():
                    try:
                        if item.is_file() or item.is_symlink():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                        deleted += 1
                    except Exception as e:
                        _log.warning(f"删除上传文件失败: {item} -> {e}")
                if deleted > 0:
                    _log.info(f"✅ 清空上传目录: {target_dir}（删除 {deleted} 个文件/目录）")
        
        _log.info("✅ 训练后的清理完成")

__all__ = ["MemoryTrainingService", "TrainingModelContext"]


