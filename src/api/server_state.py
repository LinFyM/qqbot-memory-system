# -*- coding: utf-8 -*-
"""
服务器全局状态管理
包含模型、配置、记忆库等全局对象
"""
import logging
import os
import sys
import yaml
import torch
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# 导入记忆相关模块
from memory.vector_db import MemoryVectorDB
from memory.token_manager import MemoryTokenManager
from utils.common import get_project_root, resolve_path

_log = logging.getLogger(__name__)

# =============================================================================
# 全局状态变量
# =============================================================================
model: Optional[Qwen3VLForConditionalGeneration] = None
processor: Optional[AutoProcessor] = None
device: Optional[str] = None
config: Dict[str, Any] = {}
memory_db: Optional[MemoryVectorDB] = None
recall_token_ids: Dict[str, int] = {}  # 特殊token ID映射
token_manager: Optional[MemoryTokenManager] = None

# 训练相关全局状态
is_training: bool = False
training_lock = threading.Lock()
model_lock = threading.Lock()  # 模型推理锁，确保串行
training_scheduler = None
# 记录服务器入口脚本及参数，便于训练后重启
server_script_path: Optional[str] = None
server_script_args: Optional[List[str]] = None

# 服务器基础URL
server_base_url: str = "http://127.0.0.1:9999"

# 上传目录配置
IMAGE_UPLOAD_DIR: str = ""
VIDEO_UPLOAD_DIR: str = ""
AUDIO_UPLOAD_DIR: str = ""
FILE_UPLOAD_DIR: str = ""


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认使用 configs/config_qwen3vl.yaml
    
    Returns:
        配置字典
    """
    global config
    
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "configs" / "config_qwen3vl.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载统一的提示词配置
    prompt_library = {}
    prompt_config_path = config.get("prompt_config_path")
    if not prompt_config_path:
        prompt_config_path = "configs/prompts.yaml"
    try:
        prompt_path = resolve_path(prompt_config_path)
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as pf:
                prompt_library = yaml.safe_load(pf) or {}
            _log.info(f"✅ 已加载提示词配置文件: {prompt_path}")
        else:
            _log.warning(f"⚠️ 提示词配置文件不存在: {prompt_path}")
    except Exception as e:
        _log.warning(f"⚠️ 加载提示词配置失败: {e}")
        prompt_library = {}
    
    config["prompt_library"] = prompt_library
    
    # 向后兼容：保持 config['prompt'] 可用
    if prompt_library.get("chat"):
        config["prompt"] = prompt_library["chat"]
    else:
        config.setdefault("prompt", {})
    
    memory_cfg = config.setdefault("memory", {}).setdefault("training", {})
    memory_training_prompts = prompt_library.get("memory_training", {})
    if "guides" not in memory_cfg and memory_training_prompts.get("guides"):
        memory_cfg["guides"] = memory_training_prompts["guides"]
    if "guide_text" not in memory_cfg and memory_training_prompts.get("guide_text"):
        memory_cfg["guide_text"] = memory_training_prompts["guide_text"]
    
    config["memory_extraction_prompts"] = prompt_library.get("memory_extraction", {})
    config["memory_vectorization_prompts"] = prompt_library.get("memory_vectorization", {})
    
    _log.info(f"✅ 已加载配置文件: {config_path}")
    return config


def initialize_model(model_path: str, target_device: str):
    """
    初始化模型和处理器
    
    Args:
        model_path: 模型路径
        target_device: 目标设备
    """
    global model, processor, device, memory_db, recall_token_ids, token_manager
    
    # 判断是否为训练模型
    is_trained_model = "trained" in model_path or "token_added" in model_path
    model_type = "训练模型" if is_trained_model else "基础模型"
    
    _log.info("=" * 60)
    _log.info("🚀 开始初始化Qwen3-VL模型...")
    _log.info(f"📦 模型类型: {model_type}")
    _log.info(f"📁 模型路径: {model_path}")
    _log.info(f"🖥️  设备: {target_device}")
    if is_trained_model:
        model_name = os.path.basename(model_path)
        _log.info(f"📅 模型时间戳: {model_name}")
    _log.info("=" * 60)
    
    # 解析模型路径（支持相对路径）
    model_path_resolved = resolve_path(model_path)
    
    if not model_path_resolved.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path_resolved}")
    
    # 检查CUDA信息
    _log.info("检查CUDA环境...")
    cuda_available = torch.cuda.is_available()
    _log.info(f"🔧 CUDA可用: {cuda_available}")
    if cuda_available:
        cuda_device_count = torch.cuda.device_count()
        _log.info(f"🔧 CUDA设备数量: {cuda_device_count}")
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            _log.info(f"🔧 CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # 加载处理器
    _log.info("加载AutoProcessor...")
    processor = AutoProcessor.from_pretrained(
        str(model_path_resolved),
        trust_remote_code=True,
        local_files_only=True
    )
    _log.info("✅ Processor加载成功")
    
    # 确保chat_template被正确加载
    if processor.chat_template is None:
        import json
        chat_template_path = model_path_resolved / "chat_template.json"
        if chat_template_path.exists():
            try:
                with open(chat_template_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    processor.chat_template = data["chat_template"]
                _log.info("✅ 手动加载chat_template成功")
            except Exception as e:
                _log.warning(f"⚠️ 手动加载chat_template失败: {e}")
    
    # 配置加载参数
    _log.info("加载Qwen3VLForConditionalGeneration...")
    load_kwargs = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
        "local_files_only": True
    }
    
    # 配置设备映射
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_visible_set = bool(cuda_visible)
    
    if isinstance(target_device, list):
        # 多GPU配置
        if cuda_visible:
            _log.info(f"🔧 检测到CUDA_VISIBLE_DEVICES={cuda_visible}")
        _log.info(f"🔧 多GPU模式: 指定设备{target_device}")
        load_kwargs["device_map"] = "auto"
    elif target_device.startswith("cuda"):
        # 单GPU配置
        if cuda_visible_set and cuda_visible:
            device_map_device = "cuda:0"
            _log.info(f"🔧 单GPU模式: CUDA_VISIBLE_DEVICES={cuda_visible}，使用重新映射设备 {device_map_device}（对应物理GPU {target_device}）")
        else:
            device_map_device = target_device
        _log.info(f"🔧 单GPU模式: 设备映射到 {target_device}")
        load_kwargs["device_map"] = {"": device_map_device}
    else:
        # CPU配置
        load_kwargs["device_map"] = "cpu"
        _log.info("🔧 CPU模式: 加载到CPU")
    
    # 加载模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path_resolved),
        **load_kwargs
    )
    
    # 获取实际设备
    actual_device = next(model.parameters()).device
    device = target_device
    _log.info(f"✅ 模型加载成功，实际设备: {actual_device}")
    model.eval()
    
    # 检查并添加特殊token
    _log.info("检查并添加记忆相关特殊token...")
    token_manager = MemoryTokenManager(model, processor)
    recall_token_ids = token_manager.check_and_add_tokens(perturbation_std=0.02)
    _log.info(f"✅ 特殊token处理完成: {recall_token_ids}")
    
    # 初始化记忆向量库
    memory_config = config.get("memory", {})
    memory_enabled = memory_config.get("enabled", False)
    
    if memory_enabled:
        _log.info("初始化MemoryVectorDB...")
        memory_db_config = memory_config.get("memory_db", {})
        max_size = memory_db_config.get("max_size", 100000)
        enable_eviction = memory_db_config.get("enable_eviction", True)
        
        # 获取 embedding 维度
        try:
            input_embeddings = model.get_input_embeddings()
            embedding_dim = input_embeddings.weight.shape[1]
            _log.info(f"📊 从模型 input_embeddings 获取维度: {embedding_dim}")
        except Exception as e:
            embedding_dim = 4096
            _log.warning(f"⚠️ 无法从模型获取 embedding 维度，使用默认值: {embedding_dim}")
        
        memory_device = actual_device
        if hasattr(memory_device, "type"):
            memory_device = str(memory_device)
        memory_db = MemoryVectorDB(
            embedding_dim=embedding_dim,
            device=memory_device,
            max_size=max_size,
            enable_eviction=enable_eviction
        )
        
        # 尝试加载已有的记忆数据库
        memory_db_path = memory_config.get("memory_db_path")
        if memory_db_path:
            memory_db_path_resolved = resolve_path(memory_db_path)
            if memory_db_path_resolved.exists():
                try:
                    memory_db.load_from_pt(str(memory_db_path_resolved))
                    _log.info(f"✅ 已加载记忆库: {memory_db_path_resolved} (条目数: {len(memory_db)})")
                except Exception as e:
                    _log.warning(f"⚠️ 加载记忆库失败: {e}")
            else:
                _log.info(f"ℹ️ 记忆库文件不存在，将创建新的空库: {memory_db_path_resolved}")
    else:
        _log.info("记忆功能未启用")
    
    _log.info("=" * 60)
    _log.info("✅ 模型初始化完成")
    _log.info("=" * 60)


def get_model_and_processor():
    """
    获取当前的模型和处理器
    
    Returns:
        (model, processor) 元组
    """
    if model is None or processor is None:
        raise RuntimeError("模型未初始化，请先调用 initialize_model()")
    return model, processor


def setup_upload_directories(base_dir: Optional[Path] = None):
    """
    设置上传目录
    
    Args:
        base_dir: 基础目录，默认为项目根目录
    """
    global IMAGE_UPLOAD_DIR, VIDEO_UPLOAD_DIR, AUDIO_UPLOAD_DIR, FILE_UPLOAD_DIR
    
    if base_dir is None:
        base_dir = get_project_root()
    
    IMAGE_UPLOAD_DIR = str(base_dir / "uploads" / "images")
    VIDEO_UPLOAD_DIR = str(base_dir / "uploads" / "videos")
    AUDIO_UPLOAD_DIR = str(base_dir / "uploads" / "audios")
    FILE_UPLOAD_DIR = str(base_dir / "uploads" / "files")
    
    # 创建目录
    for dir_path in [IMAGE_UPLOAD_DIR, VIDEO_UPLOAD_DIR, AUDIO_UPLOAD_DIR, FILE_UPLOAD_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    _log.info(f"✅ 上传目录已设置: {base_dir / 'uploads'}")


def find_latest_model(config_override: Optional[dict] = None) -> str:
    """
    查找最新的模型路径（优先训练模型 -> token添加模型 -> 基础模型）
    """
    cfg = config_override or config
    if not cfg:
        cfg = load_config()
    
    memory_cfg = cfg.get("memory", {}).get("training", {})
    trained_model_dir = resolve_path(memory_cfg.get("trained_model_dir", "./models/trained"))
    token_added_model_dir = resolve_path(memory_cfg.get("token_added_model_dir", "./models/token_added"))
    
    _log.info("=" * 60)
    _log.info("🔍 查找最新模型路径")
    _log.info(f"📁 训练模型目录: {trained_model_dir}")
    _log.info(f"📁 Token模型目录: {token_added_model_dir}")
    _log.info("=" * 60)
    
    if trained_model_dir.exists():
        model_dirs = [
            d for d in os.listdir(trained_model_dir)
            if (trained_model_dir / d).is_dir() and d.startswith("model_")
        ]
        if model_dirs:
            model_dirs.sort(reverse=True)
            latest = trained_model_dir / model_dirs[0]
            _log.info(f"✅ 使用最新训练模型: {latest}")
            return str(latest)
        _log.warning("⚠️ 训练模型目录存在但为空")
    else:
        _log.warning(f"⚠️ 训练模型目录不存在: {trained_model_dir}")
    
    if token_added_model_dir.exists():
        model_dirs = [
            d for d in os.listdir(token_added_model_dir)
            if (token_added_model_dir / d).is_dir() and d.startswith("model_")
        ]
        if model_dirs:
            model_dirs.sort(reverse=True)
            latest = token_added_model_dir / model_dirs[0]
            _log.info(f"✅ 使用最新token添加模型: {latest}")
            return str(latest)
        _log.warning("⚠️ Token模型目录存在但为空")
    else:
        _log.warning(f"⚠️ Token模型目录不存在: {token_added_model_dir}")
    
    model_path = cfg.get("model", {}).get("path")
    if model_path:
        _log.info(f"ℹ️ 使用配置中的基础模型路径: {model_path}")
        return model_path
    
    default_path = memory_cfg.get("base_model_path", "./models/Qwen3-VL-4B-Thinking")
    _log.info(f"ℹ️ 使用默认基础模型路径: {default_path}")
    return default_path


def reload_latest_model(config_override: Optional[dict] = None, device_override: Optional[str] = None) -> str:
    """
    重新加载最新模型，并返回实际使用的模型路径
    """
    cfg = config_override or config
    if not cfg:
        cfg = load_config()
    
    target_device = device_override or cfg.get("model", {}).get("device", "cuda:0")
    model_path = find_latest_model(cfg)
    
    _log.info("=" * 60)
    _log.info("🔄 重新加载最新模型")
    _log.info(f"📁 模型路径: {model_path}")
    _log.info(f"🖥️  目标设备: {target_device}")
    _log.info("=" * 60)
    
    initialize_model(model_path, target_device)
    _log.info("✅ 模型重新加载完成")
    return model_path
