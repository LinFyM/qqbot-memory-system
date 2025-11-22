# -*- coding: utf-8 -*-
"""
API服务器 - Qwen3-VL多模态支持版本
专门用于测试和验证文字+图片信息传递到大模型
"""

from flask import request, jsonify, url_for
import logging
import os
import sys
import time
import yaml
import json
import threading
import queue
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
import base64
from uuid import uuid4
import requests
import mimetypes
from urllib.parse import unquote
import shutil
import re
from utils.cq import (
    extract_cq_image_urls as _u_extract_cq_image_urls,
    extract_cq_video_urls as _u_extract_cq_video_urls,
    extract_cq_audio_urls as _u_extract_cq_audio_urls,
    extract_cq_file_urls as _u_extract_cq_file_urls,
    extract_http_urls as _u_extract_http_urls,
    extract_cq_appshare_cards as _u_extract_cq_appshare_cards,
)
from services.media import (
    download_image_to_storage as svc_download_image_to_storage,
    download_video_to_storage as svc_download_video_to_storage,
    download_audio_to_storage as svc_download_audio_to_storage,
    download_file_to_storage as svc_download_file_to_storage,
)
from services.extractors import (
    extract_text_from_file as svc_extract_text_from_file,
    extract_text_and_images_from_file as svc_extract_text_and_images_from_file,
    download_and_extract_webpage as svc_download_and_extract_webpage,
)
from services.asr import (
    transcribe_audio as svc_transcribe_audio,
)
from services.generation import (
    InterruptStoppingCriteria as SvcInterruptStoppingCriteria,
)
from services import history as svc_history
from services import queueing as svc_queueing
from services import handler as svc_handler
from services.fetch import fetch_url_content as svc_fetch_url_content, fetch_file_content as svc_fetch_file_content

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入Qwen3-VL相关组件
# 强制优先使用 torchvision/PyAV 作为视频解码后端，避免torchcodec走BytesIO路径
os.environ.setdefault("TRANSFORMERS_VIDEO_BACKEND", "torchvision")
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
)
import torch

# 导入记忆框架相关组件
from memory.token_manager import MemoryTokenManager
from memory.vector_db import MemoryVectorDB
from memory.utils import inject_memory_embedding_to_inputs_embeds
from recall.model_utils import forward_backbone, ensure_last_hidden_state, build_causal_lm_output

_log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 全局模型和处理器
model = None
processor = None
device = None

# 记忆框架相关全局变量
memory_db = None  # MemoryVectorDB实例
recall_token_ids = {}  # 特殊token ID映射，如 {"<recall>": 123456, "</recall>": 123457}

# 群聊历史记录（每个群维护30条）
group_chat_histories: Dict[str, list] = {}
private_chat_histories: Dict[str, list] = {}

# 全局配置
config = {}


def get_chat_history_token_limit() -> int:
    """
    获取聊天历史token长度限制
    
    Returns:
        最大token数量，如果配置读取失败则返回默认值35000
    """
    try:
        chat_history_config = config.get("chat_history", {})
        if not isinstance(chat_history_config, dict):
            _log.warning(f"⚠️ chat_history配置不是字典类型: {type(chat_history_config)}，使用默认值35000")
            return 35000
        
        max_tokens = chat_history_config.get("max_input_tokens", 35000)
        result = int(max_tokens)
        
        if result <= 0:
            _log.warning(f"⚠️ max_input_tokens配置值无效: {max_tokens}，使用默认值35000")
            return 35000
        
        return result
    except Exception as e:
        _log.error(f"❌ 读取chat_history_token_limit配置失败: {e}，使用默认值35000", exc_info=True)
        return 35000

# 训练调度器（在main中初始化）
training_scheduler = None

# 训练模式标志（用于阻止API请求和模型生成）
is_training = False
training_lock = threading.Lock()  # 用于保护训练模式标志

# 线程锁，用于保护聊天记录的并发访问
chat_history_lock = threading.Lock()

# 模型锁，用于确保同一时刻只有一个线程使用模型（串行推理）
model_lock = threading.Lock()

# 消息处理队列（用于不同聊天之间的消息排队）
message_queue = queue.Queue()

# 当前正在处理的聊天（用于中断同一聊天内的旧消息）
# {chat_id: {"interrupt_event": Event, "response_dict": dict, "lock": Lock}}
processing_chats: Dict[str, Dict[str, Any]] = {}

# 处理队列的线程锁
queue_lock = threading.Lock()

# 工作线程是否已启动
worker_thread_started = False

# 对外访问的基础URL（在主函数中设置）
server_base_url: Optional[str] = None

# 图片上传目录（确保存在）
IMAGE_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_images")
os.makedirs(IMAGE_UPLOAD_DIR, exist_ok=True)
# 视频上传目录（确保存在）
VIDEO_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_videos")
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
# 音频上传目录（确保存在）
AUDIO_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_audios")
os.makedirs(AUDIO_UPLOAD_DIR, exist_ok=True)
# 文件上传目录（确保存在）
FILE_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_files")
os.makedirs(FILE_UPLOAD_DIR, exist_ok=True)

# 运行期指标
metrics_lock = threading.Lock()
metrics = {
    "requests_total": 0,
    "group_requests": 0,
    "private_requests": 0,
    "replies_sent": 0,
    "no_reply": 0,
    "interruptions": 0,
    "actions_total": 0,
    "image_cached": 0,
    "image_cache_fail": 0,
    "video_cached": 0,
    "video_cache_fail": 0,
    "audio_cached": 0,
    "audio_cache_fail": 0,
    "asr_success": 0,
    "asr_fail": 0,
    "file_cached": 0,
    "file_cache_fail": 0,
    "file_extract_success": 0,
    "file_extract_fail": 0,
    "web_extract_success": 0,
    "web_extract_fail": 0,
    "latency_ms": [],
}
MAX_LATENCY_BUCKET = 200

def _metrics_add(key: str, value: int = 1):
    with metrics_lock:
        if key in metrics and isinstance(metrics[key], int):
            metrics[key] += value

def _metrics_add_latency(ms: float):
    with metrics_lock:
        lst = metrics.get("latency_ms")
        if isinstance(lst, list):
            lst.append(float(ms))
            if len(lst) > MAX_LATENCY_BUCKET:
                del lst[: len(lst) - MAX_LATENCY_BUCKET]

# 路径工具：将相对路径统一解析到项目根目录
def _resolve_project_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))  # server 目录
    project_root = os.path.dirname(script_dir)              # 项目根目录
    normalized = path[2:] if path.startswith("./") else path
    return os.path.abspath(os.path.join(project_root, normalized))

# 多GPU分配优化工具
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
    import torch
    
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
                    _log.info(f"🔍 GPU {i} (物理索引 {physical_gpu_idx}, 可见索引 {visible_gpu_idx}): 总显存={total_memory_mb}MB, 可用={available_mb}MB, 已保留={reserved_mb}MB")
                else:
                    # CUDA_VISIBLE_DEVICES未设置，使用物理索引
                    # 获取GPU总显存（MB）
                    total_memory_mb = torch.cuda.get_device_properties(physical_gpu_idx).total_memory // (1024 * 1024)
                    # 获取当前已用显存（MB）
                    torch.cuda.set_device(physical_gpu_idx)
                    allocated_mb = torch.cuda.memory_allocated(physical_gpu_idx) // (1024 * 1024)
                    reserved_mb = torch.cuda.memory_reserved(physical_gpu_idx) // (1024 * 1024)
                    available_mb = total_memory_mb - reserved_mb
                    _log.info(f"🔍 GPU {i} (物理索引 {physical_gpu_idx}): 总显存={total_memory_mb}MB, 可用={available_mb}MB, 已保留={reserved_mb}MB")
                
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
    
    _log.info(f"✅ 优化的max_memory配置: {optimized_max_memory}")
    
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

# 绑定到 services 薄封装，供后续分离实现时可无感切换
svc_history.bind_backing_stores(group_chat_histories, private_chat_histories, chat_history_lock)
svc_queueing.bind_queue(message_queue)


# 静态文件路由已迁移至 app.py（蓝图模式），本文件不再提供静态端点


def _ensure_processor_files(model_path: str):
    """
    确保模型目录内包含处理器配置；若缺失则从基础模型复制
    """
    try:
        AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        return
    except Exception as missing_error:
        _log.warning(f"⚠️ 模型目录缺少处理器配置，尝试补全: {missing_error}")

    fallback_base = (
        config.get("memory", {}).get("base_model_path")
        or config.get("model", {}).get("path")
        or "./models/Qwen3-VL-4B-Thinking"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if not os.path.isabs(fallback_base):
        fallback_base = os.path.abspath(os.path.join(project_root, fallback_base))
    if not os.path.exists(fallback_base):
        raise FileNotFoundError(f"基础模型路径不存在，无法补全处理器配置: {fallback_base}")
    try:
        processor = AutoProcessor.from_pretrained(
            fallback_base,
            trust_remote_code=True,
            local_files_only=True
        )
        processor.save_pretrained(model_path)
        _log.info(f"✅ 已从基础模型补全处理器配置到: {model_path}")

        # 确保所有必要的配置文件都被正确保存（在save_pretrained之后）
        import shutil
        essential_files = [
            "chat_template.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json"
        ]
        for file_name in essential_files:
            source_file = os.path.join(fallback_base, file_name)
            target_file = os.path.join(model_path, file_name)
            if os.path.exists(source_file) and not os.path.exists(target_file):
                try:
                    shutil.copy2(source_file, target_file)
                    _log.info(f"✅ 已复制{file_name}到: {model_path}")
                except Exception as e:
                    _log.warning(f"⚠️ 复制{file_name}失败: {e}")
    except Exception as fallback_error:
        _log.error(
            f"❌ 无法补全处理器配置，请手动检查模型目录: {model_path} "
            f"(基础模型路径: {fallback_base})，错误: {fallback_error}"
        )
        raise


def _is_image_url_valid(image_url: str) -> bool:
    """
    检查图片URL是否有效（不下载完整内容，只检查是否能访问）
    """
    try:
        # 只获取头部信息，不下载完整图片
        resp = requests.head(image_url, timeout=5, allow_redirects=True)
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "").lower()
            # 检查是否是图片类型
            if any(img_type in content_type for img_type in ["image/", "application/octet-stream"]):
                return True
        return False
    except Exception:
        return False


# 本文件不再实现本地下载/正文抽取/ASR，均已迁移到 services 并通过 svc_* 委托

_asr_backend = None  # 仅保留占位符，避免历史引用；实际ASR委托至 services.asr


@dataclass
class MessageTask:
    """消息处理任务"""
    chat_type: str  # "group" 或 "private"
    chat_id: str  # group_id 或 user_id
    data: Dict[str, Any]  # 原始请求数据
    response_dict: Dict[str, Any]  # 用于返回响应的字典（线程间通信）

# 聊天记录存储目录（用于保存训练数据）
CHAT_HISTORY_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_history_storage')
os.makedirs(CHAT_HISTORY_STORAGE_DIR, exist_ok=True)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
    
    Returns:
        配置字典
    """
    if config_path is None:
        # 默认配置文件路径（相对于当前文件）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config_qwen3vl.yaml")
    
    # 如果配置文件不存在，使用默认配置
    if not os.path.exists(config_path):
        _log.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        _log.info(f"✅ 配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        _log.error(f"❌ 加载配置文件失败: {e}，使用默认配置")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 9999
        },
        "model": {
            "path": "./models/Qwen3-VL-4B-Thinking",
            "device": "cuda:0"
        },
        "generation": {
            "max_new_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True
        },
        "chat_history": {
            "max_history_length": 30
        },
        "logging": {
            "level": "INFO"
        }
    }


def initialize_model(model_path: str = "./models/Qwen3-VL-4B-Thinking", device_id = "cuda:0"):
    """
    初始化Qwen3-VL模型和处理器，并设置记忆框架
    
    Args:
        model_path: 模型路径（相对路径或绝对路径），如果为None则自动选择最新训练模型
        device_id: 设备ID，可以是字符串（如 "cuda:0", "cpu"）或列表（如 ["cuda:0", "cuda:1"]）
    """
    global model, processor, device, memory_db, recall_token_ids, config

    # 获取多GPU配置
    multi_gpu_config = config.get("model", {}).get("multi_gpu", {})

    # 如果已有模型，先卸载以释放显存
    if model is not None:
        _log.info("检测到已有模型，先卸载旧模型以释放显存...")
        try:
            # 将模型移到CPU，然后删除
            model = model.cpu()
        except:
            pass
        del model
        model = None
    if processor is not None:
        del processor
        processor = None
    
    # 强制垃圾回收和显存清理
    import gc
    import os
    gc.collect()
    torch.cuda.empty_cache()
    # 再次同步和清理
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    _log.info("✅ 旧模型已卸载，显存已释放")
    
    # 如果model_path为None，尝试查找最新训练模型
    if model_path is None:
        _log.info("=" * 60)
        _log.info("🔍 initialize_model: model_path为None，开始查找最新训练模型")
        _log.info("=" * 60)
        memory_config = config.get("memory", {}).get("training", {})
        trained_model_dir = memory_config.get("trained_model_dir", "./server/models/trained")
        
        # 转换为绝对路径（相对于项目根目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))  # server目录
        project_root = os.path.dirname(script_dir)  # 项目根目录
        if not os.path.isabs(trained_model_dir):
            # 路径相对于项目根目录，直接拼接
            trained_model_dir = os.path.abspath(os.path.join(project_root, trained_model_dir))
        
        _log.info(f"📁 训练模型目录: {trained_model_dir}")
        
        # 获取token_added_model_dir和base_model_path
        token_added_model_dir = memory_config.get("token_added_model_dir", "./server/models/token_added")
        base_model_path = memory_config.get("base_model_path", "./models/Qwen3-VL-4B-Thinking")
        
        # 转换为绝对路径
        if not os.path.isabs(token_added_model_dir):
            token_added_model_dir = os.path.abspath(os.path.join(project_root, token_added_model_dir))
        if not os.path.isabs(base_model_path):
            base_model_path = os.path.abspath(os.path.join(project_root, base_model_path))
        
        # 优先级：训练后的模型 > 添加了token的模型 > 基础模型
        model_path = None
        
        # 1. 优先查找训练后的模型
        if os.path.exists(trained_model_dir):
            model_dirs = [
                d for d in os.listdir(trained_model_dir)
                if os.path.isdir(os.path.join(trained_model_dir, d)) and d.startswith("model_")
            ]
            if model_dirs:
                model_dirs.sort(reverse=True)
                model_path = os.path.join(trained_model_dir, model_dirs[0])
                _log.info("=" * 60)
                _log.info(f"✅ 找到最新训练模型: {model_path}")
                _log.info(f"📅 模型时间戳: {model_dirs[0]}")
                _log.info("=" * 60)
        
        # 2. 如果没有训练模型，查找添加了token的模型
        if model_path is None and os.path.exists(token_added_model_dir):
            model_dirs = [
                d for d in os.listdir(token_added_model_dir)
                if os.path.isdir(os.path.join(token_added_model_dir, d)) and d.startswith("model_")
            ]
            if model_dirs:
                model_dirs.sort(reverse=True)
                model_path = os.path.join(token_added_model_dir, model_dirs[0])
                _log.info("=" * 60)
                _log.info(f"✅ 找到添加了token的模型: {model_path}")
                _log.info(f"📅 模型时间戳: {model_dirs[0]}")
                _log.info("=" * 60)

        # 3. 如果都没有，使用基础模型
        if model_path is None:
            model_path = base_model_path
            _log.warning(f"⚠️ 未找到训练模型或添加了token的模型，使用基础模型: {model_path}")
        else:
            # 如果model_path不为None，说明已经在app.py中找到了训练模型
            _log.info("=" * 60)
            _log.info("✅ initialize_model: 使用传入的模型路径（已在app.py中查找）")
            _log.info(f"📦 模型路径: {model_path}")
            _log.info("=" * 60)
    
    # 将相对路径转换为绝对路径（相对于项目根目录）
    if not os.path.isabs(model_path):
        # 获取项目根目录（server目录的父目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.abspath(os.path.join(project_root, model_path))
    
    # 判断是训练模型还是基础模型
    is_trained_model = "trained" in model_path and "model_" in os.path.basename(model_path)
    model_type = "训练模型" if is_trained_model else "基础模型"
    
    _log.info("=" * 60)
    _log.info("🚀 开始初始化Qwen3-VL模型...")
    _log.info(f"📦 模型类型: {model_type}")
    _log.info(f"📁 模型路径: {model_path}")
    _log.info(f"🖥️  设备: {device_id}")
    if is_trained_model:
        model_name = os.path.basename(model_path)
        _log.info(f"📅 训练时间戳: {model_name}")
    _log.info("=" * 60)
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    # 确保处理器文件存在
    _ensure_processor_files(model_path)
    
    try:
        # 加载处理器
        _log.info("加载AutoProcessor...")
        # 如果是绝对路径且存在，使用local_files_only=True
        processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=os.path.isabs(model_path) and os.path.exists(model_path)
        )
        _log.info("✅ Processor加载成功")
        
        # 确保chat_template被正确加载（Qwen-VL的特殊处理）
        if processor.chat_template is None:
            import json
            chat_template_path = os.path.join(model_path, "chat_template.json")
            if os.path.exists(chat_template_path):
                try:
                    with open(chat_template_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        processor.chat_template = data["chat_template"]
                except Exception as e:
                    _log.warning(f"⚠️ 手动加载chat_template失败: {e}")
        
        # 加载模型 - 支持多GPU
        _log.info("加载Qwen3VLForConditionalGeneration...")
        load_kwargs = {
            "torch_dtype": "auto",
            "trust_remote_code": True,
            "local_files_only": os.path.isabs(model_path) and os.path.exists(model_path)
        }

        # 检查CUDA_VISIBLE_DEVICES设置状态（在所有设备配置之前）
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        cuda_visible_set = bool(cuda_visible)
        cuda_visible_devices = cuda_visible

        # 根据设备配置决定device_map
        if isinstance(device_id, list):
            # 多GPU配置
            # 注意：CUDA_VISIBLE_DEVICES应该在导入torch之前设置（在app.py中已设置）
            # 这里只需要检查是否已经设置，如果没有设置则设置（兼容性处理）
            
            if cuda_visible:
                _log.info(f"🔧 检测到CUDA_VISIBLE_DEVICES={cuda_visible}（已在导入torch之前设置）")
            else:
                # 如果未设置，则在这里设置（虽然可能已经太晚了）
                gpu_indices = []
                for device in device_id:
                    if device.startswith("cuda:"):
                        try:
                            gpu_idx = int(device.split(":")[1])
                            gpu_indices.append(str(gpu_idx))
                        except (ValueError, IndexError):
                            _log.warning(f"⚠️ 无效的GPU设备名称: {device}，跳过")
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
            allocation = _optimize_multi_gpu_allocation(device_id, max_memory_config, cuda_visible_set=cuda_visible_set)
            load_kwargs["device_map"] = allocation["device_map"]
            if allocation["max_memory"]:
                load_kwargs["max_memory"] = allocation["max_memory"]
            _log.info(f"🔧 多GPU模式: 指定设备{device_id}，使用优化的分配策略")
        elif device_id.startswith("cuda"):
            # 单GPU配置
            # 如果设置了CUDA_VISIBLE_DEVICES，需要使用重新映射后的索引
            if cuda_visible_set and cuda_visible_devices:
                # CUDA_VISIBLE_DEVICES已设置，使用重新映射后的索引
                device_map_device = "cuda:0"
                _log.info(f"🔧 单GPU模式: CUDA_VISIBLE_DEVICES={cuda_visible_devices}，使用重新映射设备 {device_map_device}（对应物理GPU {device_id}）")
            else:
                # 未设置CUDA_VISIBLE_DEVICES，直接使用物理设备
                device_map_device = device_id
            _log.info(f"🔧 单GPU模式: 设备映射到 {device_id}")
            load_kwargs["device_map"] = {"": device_map_device}
        else:
            # CPU配置
            load_kwargs["device_map"] = "cpu"
            _log.info("🔧 CPU模式: 加载到CPU")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            **load_kwargs
        )
        
        # 获取实际设备
        device = next(model.parameters()).device
        _log.info(f"✅ 模型加载成功，实际设备: {device}")
        
        # 检查并添加特殊token
        _log.info("检查并添加记忆相关特殊token...")
        token_manager = MemoryTokenManager(model, processor.tokenizer)
        recall_token_ids = token_manager.check_and_add_tokens(perturbation_std=0.02)
        _log.info(f"✅ 特殊token处理完成: {recall_token_ids}")
        
        # 初始化MemoryVectorDB
        memory_config = config.get("memory", {})
        memory_enabled = memory_config.get("enabled", False)
        if memory_enabled:
            _log.info("初始化MemoryVectorDB...")
            # 获取embedding维度（从模型配置中）
            embedding_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
            memory_db = MemoryVectorDB(embedding_dim=embedding_dim, device=device)
            
            # 加载记忆数据（如果存在）
            memory_db_path = memory_config.get("memory_db_path")
            if memory_db_path:
                resolved_path = _resolve_project_path(memory_db_path)
                if resolved_path and os.path.exists(resolved_path):
                    memory_db.load_from_pt(resolved_path)
                else:
                    _log.warning(f"记忆数据库文件不存在: {resolved_path or memory_db_path}")
            else:
                _log.info("未配置记忆数据库路径，使用空数据库")
        else:
            _log.info("记忆功能未启用")
        
        _log.info("=" * 60)
        
    except Exception as e:
        _log.error(f"❌ 模型加载失败: {e}", exc_info=True)
        raise


def format_multimodal_message(content: str, image_urls: List[str], video_urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    格式化多模态消息为Qwen3-VL格式（使用URL格式）
    
    Args:
        content: 文本内容
        image_urls: 图片URL列表，格式：["https://multimedia.nt.qq.com.cn/download?...", ...]
    
    Returns:
        Qwen3-VL格式的消息内容列表
    """
    message_content = []
    
    # 添加文本部分
    if content:
        message_content.append({"type": "text", "text": content})
    
    # 添加图片部分（使用URL格式，参考样例）
    for img_url in image_urls:
        message_content.append({"type": "image", "image": img_url})
    # 添加视频部分
    if video_urls:
        for v_url in video_urls:
            message_content.append({"type": "video", "video": v_url})
    
    return message_content


def _parse_action_commands(output_text: str) -> List[Dict[str, Any]]:
    """
    从模型输出中解析动作指令，支持以下格式的任意一种：
    1) <action>...</action> 内的 JSON（对象或数组）
    2) ```json ... ``` 代码块中带有type字段的对象或数组
    3) 行内 ACTION: { ... } 或 ACTIONS: [ ... ]
    返回标准化后的列表：[{"type": "...", ...}, ...]
    """
    import re
    import json
    candidates: List[str] = []
    # 1) <action>...</action>
    for m in re.finditer(r'<action>([\s\S]+?)</action>', output_text, flags=re.IGNORECASE):
        candidates.append(m.group(1))
    # 2) ```json ... ```
    for m in re.finditer(r'```json\s*([\s\S]+?)\s*```', output_text, flags=re.IGNORECASE):
        candidates.append(m.group(1))
    # 3) ACTION(S): { ... } / [ ... ]
    for m in re.finditer(r'ACTI(?:ON|ONS)\s*:\s*([\s\S]+)$', output_text, flags=re.IGNORECASE | re.MULTILINE):
        candidates.append(m.group(1))
    # 4) 标准MCP风格：<tool_call name="FETCH_URL">{...}</tool_call> 或 <toolcall name="...">...</toolcall>
    mcp_calls: List[Dict[str, Any]] = []
    for m in re.finditer(r'<tool_?call\s+name\s*=\s*"([^"]+)"\s*>\s*([\s\S]+?)\s*</tool_?call\s*>', output_text, flags=re.IGNORECASE):
        tool_name = m.group(1).strip().upper()
        payload = m.group(2).strip()
        try:
            obj = json.loads(payload)
            if isinstance(obj, dict):
                obj["type"] = tool_name  # 归一到现有动作type
                mcp_calls.append(obj)
            elif isinstance(obj, list):
                for it in obj:
                    if isinstance(it, dict):
                        it["type"] = tool_name
                        mcp_calls.append(it)
        except Exception:
            # 尝试从片段中提取JSON
            try:
                start = min((i for i in [payload.find("{"), payload.find("[")] if i != -1), default=-1)
                end = max(payload.rfind("}"), payload.rfind("]"))
                if start != -1 and end != -1 and end > start:
                    parsed = json.loads(payload[start:end+1])
                    if isinstance(parsed, dict):
                        parsed["type"] = tool_name
                        mcp_calls.append(parsed)
                    elif isinstance(parsed, list):
                        for it in parsed:
                            if isinstance(it, dict):
                                it["type"] = tool_name
                                mcp_calls.append(it)
            except Exception:
                continue
    actions: List[Dict[str, Any]] = []
    def normalize_one(obj: Any):
        if isinstance(obj, dict):
            item = obj
            t = str(item.get("type", "")).upper().strip()
            # 只支持 EMOJI_LIKE 和 POKE（已移除 IMAGE 和 FORWARD）
            if t in {"EMOJI_LIKE", "POKE"}:
                actions.append(item)
        elif isinstance(obj, list):
            for it in obj:
                normalize_one(it)
    for snippet in candidates:
        try:
            parsed = json.loads(snippet)
            normalize_one(parsed)
        except Exception:
            # 尝试提取最外层JSON对象/数组
            try:
                start = min((i for i in [snippet.find("{"), snippet.find("[")] if i != -1), default=-1)
                end = max(snippet.rfind("}"), snippet.rfind("]"))
                if start != -1 and end != -1 and end > start:
                    parsed = json.loads(snippet[start:end+1])
                    normalize_one(parsed)
            except Exception:
                continue
    # 5) 裸JSON容错：扫描可能的JSON对象/数组，解析含有type字段的动作
    json_like_matches = re.findall(r'(\{[\s\S]*?\}|\[[\s\S]*?\])', output_text, flags=re.IGNORECASE)
    for jtxt in json_like_matches:
        try:
            parsed = json.loads(jtxt)
            normalize_one(parsed)
        except Exception:
            continue
    # 合并MCP解析出的调用
    for call in mcp_calls:
        normalize_one(call)
    return actions

def extract_final_reply(output_text: str) -> Tuple[str, bool, List[Dict[str, Any]]]:
    """
    从thinking模型的输出中提取正式回复（</think>标签后的内容）
    
    Args:
        output_text: 模型的完整输出
    
    Returns:
        (回复内容, 是否需要回复, 动作指令列表)
        - 如果包含<no_reply>标签（仅在think结束后），返回("", False)
        - 如果包含正常回复，返回(回复内容, True)
        - 如果没有找到标签，返回(完整输出, True)
    """
    import re
    
    # 定义no_reply标签模式
    no_reply_patterns = [
        r'<no_reply>',
        r'<no_reply/>',
        r'<no_reply\s*/>',
    ]
    
    # 尝试匹配 </think> 标签（thinking模型使用的标签）
    thinking_patterns = [
        r'</think>\s*',
        r'</thinking>\s*'
    ]
    
    # 查找所有thinking结束标签，选择最后一个（从最后一个标签开始提取正式回复）
    last_match = None
    last_pattern = None
    
    for pattern in thinking_patterns:
        # 查找所有匹配项
        matches = list(re.finditer(pattern, output_text, re.IGNORECASE))
        if matches:
            # 选择最后一个匹配项
            current_match = matches[-1]
            # 如果这个匹配项比之前的更靠后，则更新
            if last_match is None or current_match.end() > last_match.end():
                last_match = current_match
                last_pattern = pattern
    
    if last_match:
        # 提取最后一个标签后的内容（这是正式回复部分）
        final_reply = output_text[last_match.end():].strip()
        
        # 只在think结束后的正式回复部分检查<no_reply>标签
        # 这样可以避免误识别思考过程中提到的no_reply标签
        for no_reply_pattern in no_reply_patterns:
            if re.search(no_reply_pattern, final_reply, re.IGNORECASE):
                _log.info("✅ 模型判断不需要回复（在think结束后的正式回答中包含<no_reply>标签）")
                return "", False, _parse_action_commands(output_text)
        
        # 移除任何遗留的工具调用片段，防止泄漏到最终可见输出
        # 兼容标准MCP格式：<tool_call name="...">{...}</tool_call>
        final_reply = re.sub(r'<tool_call\\b[^>]*>.*?</tool_call>', '', final_reply, flags=re.IGNORECASE | re.DOTALL).strip()
        
        # 清理动作泄漏：去除 <action>…</action>、```json``` 中的JSON，以及裸JSON动作片段
        final_reply = re.sub(r'<action>[\s\S]*?</action>', '', final_reply, flags=re.IGNORECASE)
        final_reply = re.sub(r'```json[\s\S]*?```', '', final_reply, flags=re.IGNORECASE)
        # 粗清理：移除包含 "type" 关键字的顶层对象/数组（防止把动作JSON回给用户）
        final_reply = re.sub(r'\{[^{}]*"type"[^{}]*\}', '', final_reply, flags=re.IGNORECASE)
        final_reply = re.sub(r'\[[^\[\]]*"type"[^\[\]]*\]', '', final_reply, flags=re.IGNORECASE)
        # 如果没有no_reply标签，返回正式回复
        _log.info(f"✅ 提取到正式回复（从最后一个{last_match.group(0).strip()}标签开始）")
        return final_reply.strip(), True, _parse_action_commands(output_text)
    
    # 如果没有找到thinking标签，作为fallback检查整个输出
    # 这种情况应该很少见，因为模型应该输出thinking标签
    _log.warning("⚠️ 未找到thinking标签，检查整个输出中的<no_reply>标签")
    for pattern in no_reply_patterns:
        if re.search(pattern, output_text, re.IGNORECASE):
            _log.info("✅ 模型判断不需要回复（整个输出中包含<no_reply>标签，但未找到thinking标签）")
            return "", False, _parse_action_commands(output_text)
    
    # 如果既没有thinking标签也没有no_reply标签，返回完整输出（同时移除工具/动作调用片段）
    cleaned = re.sub(r'<tool_call\\b[^>]*>.*?</tool_call>', '', output_text, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r'<action>[\s\S]*?</action>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'```json[\s\S]*?```', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\{[^{}]*"type"[^{}]*\}', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\[[^\[\]]*"type"[^\[\]]*\]', '', cleaned, flags=re.IGNORECASE)
    return cleaned, True, _parse_action_commands(output_text)


def extract_cq_image_urls(content: str) -> Tuple[str, List[str]]:
    """委托到 utils.cq.extract_cq_image_urls"""
    return _u_extract_cq_image_urls(content)

def extract_cq_video_urls(content: str) -> Tuple[str, List[str]]:
    """委托到 utils.cq.extract_cq_video_urls"""
    return _u_extract_cq_video_urls(content)

def extract_cq_audio_urls(content: str) -> Tuple[str, List[str]]:
    """委托到 utils.cq.extract_cq_audio_urls"""
    return _u_extract_cq_audio_urls(content)

def extract_cq_file_urls(content: str) -> Tuple[str, List[str]]:
    """委托到 utils.cq.extract_cq_file_urls"""
    return _u_extract_cq_file_urls(content)

def _extract_http_urls(text: str, max_urls: int = 5) -> List[str]:
    """委托到 utils.cq.extract_http_urls"""
    return _u_extract_http_urls(text, max_urls)

def extract_cq_appshare_cards(content: str):
    """委托到 utils.cq.extract_cq_appshare_cards"""
    return _u_extract_cq_appshare_cards(content)
# 网页抓取逻辑已迁移到 services.extractors.download_and_extract_webpage


def build_system_prompt(chat_type: str = None, chat_context: Dict[str, str] = None) -> str:
    """
    构建系统提示词（从config文件中读取并组合）
    
    Args:
        chat_type: "group" 或 "private"，表示对话类型
        chat_context: 对话上下文信息，包含：
            - 对于群聊：{"group_name": "群名称", "user_nickname": "用户昵称"}
            - 对于私聊：{"user_nickname": "用户昵称"}
    
    Returns:
        完整的系统提示词
    """
    global config, recall_token_ids
    
    prompt_config = config.get("prompt", {})
    
    # 获取提示词组合顺序（如果未配置，使用默认顺序）
    prompt_order = prompt_config.get("prompt_order", [
        "context",
        "recall_instruction",
        "output_structure",
        "role_playing"
    ])
    
    # 构建各部分提示词
    prompt_parts = {}
    
    # 1. 对话上下文提示
    context_template = prompt_config.get("context_template", {})
    if chat_type and chat_context:
        if chat_type == "group":
            template = context_template.get("group", "当前，你正在群聊「{group_name}」(群号：{group_id})中进行对话。")
            group_id = chat_context.get("group_id", "")
            group_name = chat_context.get("group_name", "群聊")
            user_id = chat_context.get("user_id", "")
            user_nickname = chat_context.get("user_nickname", "用户")
            try:
                prompt_parts["context"] = template.format(
                    group_id=group_id,
                    group_name=group_name,
                    user_id=user_id,
                    user_nickname=user_nickname
                )
            except KeyError:
                # 如果模板中没有某些变量，使用默认值
                prompt_parts["context"] = template.format(
                    group_id=group_id or "未知",
                    group_name=group_name,
                    user_id=user_id or "未知",
                    user_nickname=user_nickname
                )
        elif chat_type == "private":
            template = context_template.get("private", "当前，你正在与用户「{user_nickname}」(QQ号：{user_id})进行私聊对话。")
            user_id = chat_context.get("user_id", "")
            user_nickname = chat_context.get("user_nickname", "用户")
            try:
                prompt_parts["context"] = template.format(
                    user_id=user_id,
                    user_nickname=user_nickname
                )
            except KeyError:
                # 如果模板中没有某些变量，使用默认值
                prompt_parts["context"] = template.format(
                    user_id=user_id or "未知",
                    user_nickname=user_nickname
                )
    else:
        prompt_parts["context"] = ""
    
    # 2. 回忆机制说明（如果启用且token存在）
    memory_config = config.get("memory", {})
    memory_enabled = memory_config.get("enabled", False)
    if memory_enabled and recall_token_ids:
        prompt_parts["recall_instruction"] = prompt_config.get("recall_instruction", "").strip()
    else:
        prompt_parts["recall_instruction"] = ""
    
    # 3. 输出结构提示词
    prompt_parts["output_structure"] = prompt_config.get("output_structure", "").strip()
    
    # 4. 角色扮演提示词
    prompt_parts["role_playing"] = prompt_config.get("role_playing", "").strip()
    
    # 5. 工具使用提示（从配置读取，可选）
    tool_guidance = prompt_config.get("tool_guidance", "").strip()
    if tool_guidance:
        prompt_parts["tool_guidance"] = tool_guidance
    
    # 6. 多样化回复动作提示（从配置读取，可选）
    reply_actions = prompt_config.get("reply_actions", "").strip()
    if reply_actions:
        prompt_parts["reply_actions"] = reply_actions
    
    # 按照配置的顺序组合提示词
    system_prompt_parts = []
    part_labels = {
        "context": "【对话上下文】",
        "recall_instruction": "【回忆机制说明】",
        "output_structure": "【输出结构要求】",
        "role_playing": "【角色设定】",
        "tool_guidance": "【工具使用说明】",
        "reply_actions": "【多样化互动】"
    }
    
    for part_name in prompt_order:
        if part_name in prompt_parts and prompt_parts[part_name]:
            part_content = prompt_parts[part_name].strip()
            if part_content:
                # 添加分隔标签
                label = part_labels.get(part_name, f"【{part_name}】")
                # 如果内容已经以相同标签开头，则不再重复添加标签
                if part_content.startswith(label):
                    system_prompt_parts.append(part_content)
                else:
                    system_prompt_parts.append(f"{label}\n{part_content}")
    
    # 合并所有部分，使用更清晰的分隔符
    # 每个部分之间用分隔线分隔
    separator = "\n\n" + "="*60 + "\n\n"
    system_prompt = separator.join(system_prompt_parts)
    
    return system_prompt


def save_chat_history_to_storage(chat_type: str, chat_id: str, messages: List[Dict[str, Any]]):
    """
    保存聊天记录到存储文件（供训练用）
    使用固定文件名，增量追加模式
    
    Args:
        chat_type: "group" 或 "private"
        chat_id: 群ID或用户ID
        messages: 要保存的消息列表
    """
    try:
        # 使用固定文件名（不带时间戳），便于增量追加
        filename = f"{chat_type}_{chat_id}.json"
        filepath = os.path.join(CHAT_HISTORY_STORAGE_DIR, filename)
        
        # 如果文件已存在，加载现有消息
        existing_messages = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, dict) and "messages" in existing_data:
                        existing_messages = existing_data.get("messages", [])
                    elif isinstance(existing_data, list):
                        existing_messages = existing_data
                _log.info(f"📂 加载现有文件 {filename}，已有 {len(existing_messages)} 条消息")
            except Exception as e:
                _log.warning(f"加载现有文件 {filename} 失败: {e}，将创建新文件")
        
        # 合并消息（去重：比较消息内容和时间戳）
        # 使用消息的文本内容和时间戳作为唯一标识
        existing_message_keys = set()
        for msg in existing_messages:
            # 生成消息的唯一标识
            msg_key = _generate_message_key(msg)
            existing_message_keys.add(msg_key)
        
        # 只添加不在现有消息中的新消息
        new_messages = []
        for msg in messages:
            msg_key = _generate_message_key(msg)
            if msg_key not in existing_message_keys:
                new_messages.append(msg)
                existing_message_keys.add(msg_key)
        
        if not new_messages:
            _log.info(f"ℹ️ {filename} 没有新消息需要追加")
            return
        
        # 合并所有消息
        all_messages = existing_messages + new_messages
        
        # 保存到JSON文件（使用统一的字典格式）
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "chat_type": chat_type,
                "chat_id": chat_id,
                "last_updated": datetime.now().strftime('%Y%m%d_%H%M%S'),
                "messages": all_messages
            }, f, ensure_ascii=False, indent=2)
        
        _log.info(f"✅ 已保存 {len(new_messages)} 条新消息到 {filename}（总计 {len(all_messages)} 条）")
    except Exception as e:
        _log.error(f"保存聊天记录失败: {e}", exc_info=True)


def _generate_message_key(message: Dict[str, Any]) -> str:
    """
    生成消息的唯一标识（用于去重）
    
    Args:
        message: 消息字典
        
    Returns:
        唯一标识字符串
    """
    # 提取消息内容
    content = message.get("content", "")
    if isinstance(content, list):
        # 多模态内容，提取文本部分
        text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        content = " ".join(text_parts)
    
    # 使用角色和内容的前100个字符作为唯一标识
    role = message.get("role", "unknown")
    content_preview = str(content)[:100] if content else ""
    
    return f"{role}:{content_preview}"


def maintain_chat_history(chat_type: str, chat_id: str, history: List[Dict[str, Any]]):
    """
    维护聊天记录长度，超出部分保存到文件
    
    Args:
        chat_type: "group" 或 "private"
        chat_id: 群ID或用户ID
        history: 聊天历史记录
    """
    global config, recall_token_ids
    
    max_history = config.get("chat_history", {}).get("max_history_length", 30)
    
    if len(history) > max_history:
        # 计算需要移除的消息数量
        removed_count = len(history) - max_history
        removed_messages = history[:removed_count]
        
        # 保存被移除的消息
        if removed_messages:
            save_chat_history_to_storage(chat_type, chat_id, removed_messages)
        
        # 只保留最新的max_history条
        history[:] = history[-max_history:]


def process_message_task(task: MessageTask):
    """
    处理单个消息任务（在队列工作线程中执行）
    
    Args:
        task: 消息处理任务
    """
    global processing_chats, is_training, training_lock
    
    # 检查是否处于训练模式
    with training_lock:
        if is_training:
            _log.warning("⚠️ 当前处于训练模式，拒绝处理消息请求")
            if task.response_dict:
                task.response_dict["reply"] = ""
                task.response_dict["should_reply"] = False
                task.response_dict["error"] = "服务器正在训练中，暂时无法处理消息"
            return
    
    chat_id = task.chat_id
    chat_type = task.chat_type
    data = task.data
    response_dict = task.response_dict
    
    try:
        import time as _t
        _req_t0 = _t.time()
        _metrics_add("requests_total", 1)
        if chat_type == "group":
            _metrics_add("group_requests", 1)
        elif chat_type == "private":
            _metrics_add("private_requests", 1)
        # 检查是否同一聊天有新消息（中断旧消息处理）
        old_task_interrupted = False
        with queue_lock:
            if chat_id in processing_chats:
                # 中断旧消息处理
                old_processing = processing_chats[chat_id]
                old_interrupt = old_processing["interrupt_event"]
                old_interrupt.set()
                old_task_interrupted = True
                _log.info(f"⚠️ 中断聊天 {chat_id} 的旧消息处理（旧任务正在处理中）")
            
            # 创建新的中断事件（新任务使用）
            interrupt_event = threading.Event()
            processing_chats[chat_id] = {
                "interrupt_event": interrupt_event,
                "response_dict": response_dict,
                "lock": threading.Lock()
            }
        
        # 如果中断了旧任务，等待一小段时间让旧任务检测到中断并退出
        # 同时，在将消息加入历史之前，再次检查是否仍然是最新的任务
        if old_task_interrupted:
            time.sleep(0.3)  # 给旧任务一些时间检测中断并退出
            
            # 再次检查是否仍然是最新的任务（可能在这期间又有新消息到达）
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is not response_dict:
                    # 已经有更新的任务了，当前任务应该退出
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在等待期间已被更新的消息替换，退出处理")
                    return
                
                # 如果当前任务仍然是最新的，确保interrupt_event没有被错误设置
                # 因为我们是新任务，interrupt_event应该是未设置的
                if interrupt_event.is_set():
                    _log.warning(f"⚠️ 聊天 {chat_id} 的新任务interrupt_event被错误设置，重置")
                    interrupt_event.clear()
        
        # 提取数据
        if chat_type == "group":
            group_id = str(data.get("group_id", ""))
            group_name = data.get("group_name", f"群{group_id}")
            user_id = str(data.get("user_id", ""))
            user_nickname = data.get("user_nickname", f"用户{user_id}")
            user_card = data.get("user_card", user_nickname)
            content = data.get("content", "")
            timestamp = data.get("timestamp", time.time())
            
            # 调试：查看完整的消息数据结构
            _log.debug(f"📋 群聊消息完整数据: {data}")
            
            # 从content中提取CQ图片/视频/语音URL（用于多模态处理）
            _log.info(f"🔍 消息内容分析: {content}")
            cleaned_content, image_urls = extract_cq_image_urls(content)
            _log.info(f"📷 图片CQ码提取: 找到 {len(image_urls)} 个 - {image_urls}")
            cleaned_content, video_urls = extract_cq_video_urls(cleaned_content)
            _log.info(f"🎥 视频CQ码提取: 找到 {len(video_urls)} 个 - {video_urls}")
            cleaned_content, audio_urls = extract_cq_audio_urls(cleaned_content)
            _log.info(f"🎵 语音CQ码提取: 找到 {len(audio_urls)} 个")
            # 提取文件URL（仅用于日志记录，不修改content，保留原始CQ码）
            _, file_urls = extract_cq_file_urls(content)
            # 注意：不修改content，保留文件、链接、卡片的原始CQ码
            content = cleaned_content
            
            # 本地化图片URL
            if image_urls:
                _log.info(f"✅ 从CQ码中提取到 {len(image_urls)} 个图片URL")
                cached_urls = []
                for original_url in image_urls:
                    cached = svc_download_image_to_storage(original_url, IMAGE_UPLOAD_DIR, server_base_url, _metrics_add, _log)
                    cached_urls.append(cached or original_url)
                image_urls = cached_urls
            # 合并客户端直链视频（若提供）
            req_video_urls = data.get("video_urls") or []
            _log.debug(f"🎥 客户端提供的video_urls字段: {req_video_urls}")
            if req_video_urls:
                video_urls = list(set((video_urls or []) + req_video_urls))
                _log.info(f"✅ 合并客户端直链视频: {len(req_video_urls)} 个")
            # 预过滤无效视频源（保留HTTP/HTTPS URL和本地文件路径）
            def _is_valid_video_source_prefilter(p: str) -> bool:
                try:
                    if not p:
                        return False
                    # HTTP/HTTPS URL
                    if p.startswith(("http://", "https://")):
                        return True
                    # 本地文件路径（Windows或Linux）
                    import os as _os
                    return _os.path.isfile(p)
                except Exception:
                    return False
            if video_urls:
                video_urls = [v for v in video_urls if _is_valid_video_source_prefilter(v)]
                _log.debug(f"预过滤后的视频URLs: {video_urls}")
            # 本地化视频URL（支持中断检查，但视频下载会继续）
            if video_urls:
                _log.info(f"✅ 从CQ码中提取到 {len(video_urls)} 个视频URL")
                cached_videos = []
                for i, v in enumerate(video_urls):
                    # 检查是否被新消息中断
                    if interrupt_event and interrupt_event.is_set():
                        if chat_id and response_dict:
                            with queue_lock:
                                current_processing = processing_chats.get(chat_id)
                                if current_processing and current_processing["response_dict"] is not response_dict:
                                    _log.warning(f"⚠️ 聊天 {chat_id} 的视频下载过程中被新消息中断，退出处理（视频下载会继续）")
                                    return
                    
                    _log.info(f"📥 正在处理视频 {i+1}/{len(video_urls)}: {v[:80]}...")
                    # 检测Windows路径，直接跳过（客户端应该在上传前处理）
                    if re.match(r'^[a-zA-Z]:\\', v) or re.match(r'^\\\\', v):
                        _log.error(f"❌ 跳过Windows本地路径（服务器无法访问）: {v}")
                        _log.error(f"💡 客户端应该在发送消息前将本地文件上传到服务器")
                        continue  # 跳过这个视频
                    cached = svc_download_video_to_storage(v, VIDEO_UPLOAD_DIR, server_base_url, _metrics_add, _log)
                    if cached:
                        cached_videos.append(cached)
                    else:
                        _log.warning(f"⚠️ 视频下载/缓存失败: {v}")
                        # 如果是HTTP URL，仍然保留（可能可以访问）
                        if v.startswith(('http://', 'https://')):
                            cached_videos.append(v)
                        else:
                            _log.warning(f"⚠️ 跳过无效视频URL: {v}")
                    
                    # 再次检查中断（下载完成后）
                    if interrupt_event and interrupt_event.is_set():
                        if chat_id and response_dict:
                            with queue_lock:
                                current_processing = processing_chats.get(chat_id)
                                if current_processing and current_processing["response_dict"] is not response_dict:
                                    _log.warning(f"⚠️ 聊天 {chat_id} 的视频下载完成后被新消息中断，退出处理")
                                    return
                # 仅保留http(s)直链或本机可访问的本地文件路径，过滤掉无效的系统路径（如Windows盘符）
                def _is_valid_video_source(p: str) -> bool:
                    try:
                        if not p:
                            return False
                        if p.lower().startswith(("http://", "https://")):
                            return True
                        import os as _os
                        return _os.path.exists(p)
                    except Exception:
                        return False
                # 将服务器静态URL转换为本地文件路径（transformers库需要本地文件路径，不支持HTTP URL）
                valid_video_paths = []
                base_static = (server_base_url or "http://127.0.0.1:9999").rstrip("/") + "/static/videos/"
                for v in cached_videos:
                    if v and v.startswith(base_static):
                        # 这是服务器URL，转换为对应的本地文件路径
                        filename = v.split("/")[-1]
                        local_path = os.path.join(VIDEO_UPLOAD_DIR, filename)
                        if os.path.exists(local_path):
                            # 本地文件存在，使用本地文件路径
                            valid_video_paths.append(local_path)
                            _log.debug(f"🎥 使用本地视频文件路径: {local_path}")
                        else:
                            # 本地文件不存在，保留服务器URL（虽然可能无法访问）
                            valid_video_paths.append(v)
                            _log.warning(f"⚠️ 视频本地文件不存在，使用URL: {local_path}")
                    elif v and (v.startswith('http://') or v.startswith('https://')):
                        # 外部URL，保留（虽然可能无法访问，但至少格式正确）
                        valid_video_paths.append(v)
                        _log.debug(f"🎥 保留外部视频URL: {v}")
                    else:
                        # 无效URL，跳过
                        _log.warning(f"⚠️ 跳过无效视频URL: {v}")
                video_urls = valid_video_paths
            
            # 本地化语音URL并进行ASR转写
            asr_texts: List[str] = []
            if 'audio_urls' in locals() and audio_urls:
                _log.info(f"✅ 从CQ码中提取到 {len(audio_urls)} 个语音URL")
                cached_audios = []
                for a in audio_urls:
                    cached = svc_download_audio_to_storage(a, AUDIO_UPLOAD_DIR, server_base_url, _metrics_add, _log)
                    cached_audios.append(cached or a)
                # 对本地化后的可访问文件执行ASR（仅对本地缓存的文件执行）
                for ca in cached_audios:
                    if ca and ca.startswith((server_base_url or "http://127.0.0.1:9999").rstrip('/') + "/static/audios/"):
                        # 将URL转成本地文件路径
                        filename = ca.rsplit('/', 1)[-1]
                        local_fp = os.path.join(AUDIO_UPLOAD_DIR, filename)
                        text = svc_transcribe_audio(local_fp, _metrics_add, _log)
                        if text:
                            asr_texts.append(text)
                # 将转写文本注入到content
                if asr_texts:
                    content = (content + "\n" if content else "") + "【语音转写】" + " ".join(asr_texts)
            
            # 处理文件：下载并提取文本和图片内容（支持中断）
            file_texts: List[str] = []
            file_image_paths: List[str] = []
            if 'file_urls' in locals() and file_urls:
                _log.info(f"✅ 检测到 {len(file_urls)} 个文件，开始提取内容")
                for file_url in file_urls:
                    # 检查是否被中断
                    if interrupt_event and interrupt_event.is_set():
                        _log.info(f"⚠️ 文件处理被中断，停止处理剩余文件")
                        break
                    
                    try:
                        # 下载文件到服务器
                        cached_file_url = svc_download_file_to_storage(file_url, FILE_UPLOAD_DIR, server_base_url, _metrics_add, _log)
                        
                        # 再次检查中断（下载可能耗时）
                        if interrupt_event and interrupt_event.is_set():
                            _log.info(f"⚠️ 文件下载后被中断，停止处理")
                            break
                        
                        if cached_file_url and cached_file_url.startswith((server_base_url or "http://127.0.0.1:9999").rstrip('/') + "/static/files/"):
                            # 将URL转成本地文件路径
                            filename = cached_file_url.rsplit('/', 1)[-1]
                            local_fp = os.path.join(FILE_UPLOAD_DIR, filename)
                            if os.path.exists(local_fp):
                                # 提取文本和图片
                                file_text, file_images = svc_extract_text_and_images_from_file(
                                    local_fp, IMAGE_UPLOAD_DIR, _metrics_add, _log
                                )
                                
                                # 再次检查中断（提取可能耗时）
                                if interrupt_event and interrupt_event.is_set():
                                    _log.info(f"⚠️ 文件提取后被中断，停止处理")
                                    break
                                
                                if file_text:
                                    file_texts.append(file_text)
                                # 将提取的图片路径转换为URL并添加到image_urls
                                for img_path in file_images:
                                    if interrupt_event and interrupt_event.is_set():
                                        break
                                    if os.path.exists(img_path):
                                        img_filename = os.path.basename(img_path)
                                        img_url = f"{server_base_url.rstrip('/')}/static/images/{img_filename}"
                                        if image_urls is None:
                                            image_urls = []
                                        if img_url not in image_urls:
                                            image_urls.append(img_url)
                                            file_image_paths.append(img_path)
                                if file_text or file_images:
                                    _log.info(f"✅ 文件处理完成: 文本长度={len(file_text)}, 图片数={len(file_images)}")
                    except Exception as file_err:
                        _log.warning(f"⚠️ 处理文件失败 {file_url}: {file_err}")
                
                # 将提取的文本内容添加到content（如果未被中断）
                if not (interrupt_event and interrupt_event.is_set()) and file_texts:
                    file_content = "\n\n".join([f"【文件内容{i+1}】\n{t}" for i, t in enumerate(file_texts)])
                    content = (content + "\n\n" if content else "") + file_content

            media_info = ""
            if image_urls:
                media_info += f" [包含{len(image_urls)}张图片]"
            if video_urls:
                media_info += f" [包含{len(video_urls)}个视频]"
            if 'audio_urls' in locals() and audio_urls:
                media_info += f" [包含{len(audio_urls)}段语音]"
            if 'file_urls' in locals() and file_urls:
                media_info += f" [包含{len(file_urls)}个文件]"
            _log.info(f"收到群消息 [群:{group_id}({group_name})] [用户:{user_id}({user_card})]: {content[:50] if content else '(仅多媒体)'}{media_info}...")
            
            # 格式化时间戳
            time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            # 格式化用户消息（包含QQ号信息，方便模型在历史消息中识别用户）
            if user_card and user_card != user_nickname:
                formatted_message = f"[{time_str}] {user_card}({user_nickname}, QQ:{user_id}): {content}" if content else f"[{time_str}] {user_card}({user_nickname}, QQ:{user_id})"
            else:
                formatted_message = f"[{time_str}] {user_nickname}(QQ:{user_id}): {content}" if content else f"[{time_str}] {user_nickname}(QQ:{user_id})"
            
            # 构建消息内容
            if image_urls or video_urls:
                message_content = format_multimodal_message(formatted_message, image_urls, video_urls)
            else:
                message_content = [{"type": "text", "text": formatted_message}]
            
            # 不再构建/注入目录，占位逻辑移除，直接保留原始CQ文本
            
            # 更新聊天记录
            with chat_history_lock:
                if group_id not in group_chat_histories:
                    group_chat_histories[group_id] = []
                
                group_chat_histories[group_id].append({
                    "role": "user",
                    "content": message_content
                })
                
                maintain_chat_history("group", group_id, group_chat_histories[group_id])
            
            # 在生成回复之前，再次检查是否仍然是最新的任务（防止在加入历史期间有新消息到达）
            # 同时检查interrupt_event是否被错误设置
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is not response_dict:
                    # 已经有更新的任务了，当前任务应该退出（但消息已经加入历史，这是正确的）
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在生成回复前已被更新的消息替换，退出处理")
                    return
                
                # 如果当前任务仍然是最新的，但interrupt_event被设置了，清除它
                # 这可能是误设置（比如在等待期间被中断，但之后又成为最新任务）
                if interrupt_event.is_set():
                    _log.warning(f"⚠️ 聊天 {chat_id} 的任务在生成回复前检测到interrupt_event被设置，但任务仍是最新的，清除中断信号")
                    interrupt_event.clear()
            
            # 生成回复（支持最多1轮工具调用再生成）
            _log.info(f"🧠 开始生成回复（群 {group_id}）...")
            display_name = user_card if user_card and user_card != user_nickname else user_nickname
            chat_context = {
                "group_id": group_id,
                "group_name": group_name,
                "user_id": user_id,
                "user_nickname": display_name
            }
            
            # 构建系统提示词
            system_prompt = build_system_prompt("group", chat_context)
            
            # 在生成前，先对原始历史进行token长度检查和截断
            with chat_history_lock:
                # 确保group_id存在于group_chat_histories中
                if group_id not in group_chat_histories:
                    _log.warning(f"⚠️ 群 {group_id} 的聊天历史不存在，初始化为空列表")
                    group_chat_histories[group_id] = []
                
                # 对原始历史进行截断（会修改原始历史并保存被删除的消息）
                # 确保group_id存在于group_chat_histories中
                if group_id not in group_chat_histories:
                    _log.warning(f"⚠️ 群 {group_id} 的聊天历史不存在，初始化为空列表")
                    group_chat_histories[group_id] = []

                truncated_history = group_chat_histories[group_id].copy()  # 默认值，使用副本
                _log.debug(f"📊 原始聊天历史长度: {len(group_chat_histories[group_id])}（群 {group_id}）")

                try:
                    max_tokens_limit = get_chat_history_token_limit()
                    _log.debug(f"📊 获取到的max_tokens限制: {max_tokens_limit}（群 {group_id}）")

                    max_tokens_limit = get_chat_history_token_limit()
                    _log.info(f"📊 获取到的max_tokens限制: {max_tokens_limit}, 类型: {type(max_tokens_limit)}（群 {group_id}）")
                    
                    # 验证max_tokens_limit
                    if max_tokens_limit is None:
                        _log.error(f"❌ max_tokens_limit为None，使用默认值35000（群 {group_id}）")
                        max_tokens_limit = 35000
                    elif not isinstance(max_tokens_limit, int) or max_tokens_limit <= 0:
                        _log.error(f"❌ max_tokens_limit无效: {max_tokens_limit}，使用默认值35000（群 {group_id}）")
                        max_tokens_limit = 35000

                    # 检查是否需要截断
                    if len(group_chat_histories[group_id]) == 0:
                        _log.info("📊 聊天历史为空，无需截断")
                        truncated_history = []
                    else:
                        _log.info(f"📊 开始调用truncate_history_by_tokens（群 {group_id}），max_tokens={max_tokens_limit}")
                        result = truncate_history_by_tokens(
                            group_chat_histories[group_id],
                            system_prompt,
                            "group",
                            group_id,
                            max_tokens=max_tokens_limit,
                            interrupt_event=interrupt_event
                        )
                        _log.info(f"📊 truncate_history_by_tokens返回: 类型={type(result)}, 是否为None={result is None}, 长度={len(result) if result is not None else 'N/A'}（群 {group_id}）")
                        truncated_history = result

                    # 确保返回值不为None且是列表类型
                    if truncated_history is None:
                        _log.error(f"❌ 截断历史返回None（群 {group_id}），回退到原始历史")
                        truncated_history = group_chat_histories[group_id].copy()
                    elif not isinstance(truncated_history, list):
                        _log.error(f"❌ 截断历史返回非列表类型: {type(truncated_history)}（群 {group_id}），回退到原始历史")
                        truncated_history = group_chat_histories[group_id].copy()
                    else:
                        _log.info(f"✅ 截断历史成功，长度: {len(truncated_history)}（群 {group_id}）")
                except Exception as e:
                    _log.error(f"❌ 截断历史时发生异常（群 {group_id}）: {e}", exc_info=True)
                    # 异常情况下，使用原始历史
                    truncated_history = group_chat_histories[group_id].copy()
                
                # 使用截断后的历史（复制一份用于生成，避免在生成过程中被修改）
                current_history = truncated_history.copy()
            
            # 关闭FETCH相关循环（按当前需求放弃链接/卡片/文件访问）
            tool_iterations = 0
            action_cmds = []
            for _iter in range(tool_iterations + 1):
                _gen_ret = generate_reply(
                current_history, 
                chat_type="group", 
                chat_context=chat_context,
                interrupt_event=interrupt_event,
                chat_id=chat_id,
                    response_dict=response_dict,
                    log_full_io=True
            )
                if isinstance(_gen_ret, tuple) and len(_gen_ret) == 3:
                    reply, should_reply, was_interrupted = _gen_ret
                    action_cmds = []
                else:
                    reply, should_reply, was_interrupted, action_cmds = _gen_ret
                if was_interrupted:
                    break
                # 已禁用FETCH动作
                pending_fetch = []
                # for act in (action_cmds or []):
                #     t = str(act.get("type", "")).upper().strip()
                #     if t in ("FETCH_URL", "FETCH_FILE"):
                #         pending_fetch.append(act)
                if not pending_fetch or _iter >= tool_iterations:
                    break
                # FETCH功能已禁用，无需处理工具调用
            
            # 检查是否被中断
            if was_interrupted:
                _log.info(f"⚠️ 聊天 {chat_id} 的消息处理被中断，跳过回复")
                _metrics_add("interruptions", 1)
                # 被中断的消息需要更新response_dict，否则客户端会一直等待
                # 但需要确保这是当前任务，避免旧任务覆盖新任务的response_dict
                with queue_lock:
                    current_processing = processing_chats.get(chat_id)
                    if current_processing and current_processing["response_dict"] is response_dict:
                        response_dict.update({
                            "status": "success",
                            "should_reply": False,
                            "reply": ""
                        })
                        _log.info(f"✅ 已更新中断响应（聊天 {chat_id}）")
                return
            
            # 在更新聊天记录之前，再次检查是否仍然是最新的任务
            # 因为可能在生成期间有新消息到达并设置了interrupt_event
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is not response_dict:
                    # 已经有更新的任务了，当前任务应该退出
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在生成完成后被更新的消息替换，跳过更新历史")
                    return
                # 再次检查中断事件（双重保险）
                if interrupt_event.is_set():
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在生成完成后被中断，跳过更新历史")
                    return
            
            # 在更新聊天记录之前，再次检查中断（防止在生成完成后、更新前有新消息到达）
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is not response_dict:
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在更新聊天记录前已被新任务替换，跳过更新")
                    return
                if interrupt_event.is_set():
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在更新聊天记录前被中断，跳过更新")
                    return
            
            # 更新聊天记录（只有在没有被中断的情况下）
            with chat_history_lock:
                # 在持有chat_history_lock期间再次检查中断（双重保险）
                if interrupt_event and interrupt_event.is_set():
                    with queue_lock:
                        current_processing = processing_chats.get(chat_id)
                        if current_processing and current_processing["response_dict"] is not response_dict:
                            _log.info(f"⚠️ 聊天 {chat_id} 的任务在更新聊天记录期间被新任务替换，跳过更新")
                            return
                
                if should_reply:
                    _metrics_add("replies_sent", 1)
                    group_chat_histories[group_id].append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": reply}]
                    })
                    maintain_chat_history("group", group_id, group_chat_histories[group_id])
                    _log.info(f"💬 生成回复（群 {group_id}）：{reply[:100]}...")
                else:
                    _metrics_add("no_reply", 1)
                    group_chat_histories[group_id].append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": "<no_reply>"}]
                    })
                    maintain_chat_history("group", group_id, group_chat_histories[group_id])
                    _log.info(f"💬 模型判断不需要回复（群 {group_id}）")
            
            # 更新响应（只有在没有被中断的情况下）
            # 再次检查是否仍然是最新的任务（防止在更新聊天记录时被新消息中断）
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is response_dict:
                    # 再次检查中断事件（防止在更新聊天记录时被中断）
                    if interrupt_event.is_set():
                        _log.warning(f"⚠️ 任务在更新响应前被中断（群 {group_id}）")
                        return
                    
                    response_dict.update({
                        "status": "success",
                        "should_reply": should_reply,
                        "reply": reply if should_reply else "",
                        "actions": action_cmds if should_reply else []
                    })
                    _log.info(f"✅ 已更新响应（群 {group_id}），should_reply={should_reply}, reply长度={len(reply) if reply else 0}")
                else:
                    _log.warning(f"⚠️ 任务已被新消息中断，跳过响应更新（群 {group_id}）")
            
        elif chat_type == "private":
            user_id = str(data.get("user_id", ""))
            user_nickname = data.get("user_nickname", f"用户{user_id}")
            content = data.get("content", "")
            timestamp = data.get("timestamp", time.time())
            
            # 调试：查看完整的消息数据结构
            _log.debug(f"📋 私聊消息完整数据: {data}")

            # 从content中提取CQ图片/视频/语音URL（用于多模态处理）
            _log.info(f"🔍 私聊消息内容分析: {content}")
            cleaned_content, image_urls = extract_cq_image_urls(content)
            _log.info(f"📷 图片CQ码提取: 找到 {len(image_urls)} 个 - {image_urls}")
            cleaned_content, video_urls = extract_cq_video_urls(cleaned_content)
            _log.info(f"🎥 视频CQ码提取: 找到 {len(video_urls)} 个 - {video_urls}")
            cleaned_content, audio_urls = extract_cq_audio_urls(cleaned_content)
            _log.info(f"🎵 语音CQ码提取: 找到 {len(audio_urls)} 个")
            # 提取文件URL（仅用于日志记录，不修改content，保留原始CQ码）
            _, file_urls = extract_cq_file_urls(content)
            # 注意：不修改content，保留文件、链接、卡片的原始CQ码
            content = cleaned_content
            
            if image_urls:
                _log.info(f"✅ 从CQ码中提取到 {len(image_urls)} 个图片URL")
                cached_urls = []
                for original_url in image_urls:
                    cached = svc_download_image_to_storage(original_url, IMAGE_UPLOAD_DIR, server_base_url, _metrics_add, _log)
                    cached_urls.append(cached or original_url)
                image_urls = cached_urls
            # 合并客户端直链视频（若提供）
            req_video_urls = data.get("video_urls") or []
            _log.debug(f"🎥 客户端提供的video_urls字段: {req_video_urls}")
            if req_video_urls:
                video_urls = list(set((video_urls or []) + req_video_urls))
                _log.info(f"✅ 合并客户端直链视频: {len(req_video_urls)} 个")
            # 预过滤无效视频源（保留HTTP/HTTPS URL和本地文件路径）
            def _is_valid_video_source_prefilter_priv(p: str) -> bool:
                try:
                    if not p:
                        return False
                    # HTTP/HTTPS URL
                    if p.startswith(("http://", "https://")):
                        return True
                    # 本地文件路径（Windows或Linux）
                    import os as _os
                    return _os.path.isfile(p)
                except Exception:
                    return False
            if video_urls:
                video_urls = [v for v in video_urls if _is_valid_video_source_prefilter_priv(v)]
                _log.debug(f"预过滤后的视频URLs: {video_urls}")
            # 处理视频URL本地化（支持中断检查，但视频下载会继续）
            if video_urls:
                _log.info(f"✅ 从CQ码中提取到 {len(video_urls)} 个视频URL")
                _log.debug(f"视频URLs详情: {video_urls}")
                cached_videos = []
                for i, v in enumerate(video_urls):
                    # 检查是否被新消息中断
                    if interrupt_event and interrupt_event.is_set():
                        if chat_id and response_dict:
                            with queue_lock:
                                current_processing = processing_chats.get(chat_id)
                                if current_processing and current_processing["response_dict"] is not response_dict:
                                    _log.warning(f"⚠️ 聊天 {chat_id} 的视频下载过程中被新消息中断，退出处理（视频下载会继续）")
                                    return
                    
                    _log.info(f"📥 正在处理视频 {i+1}/{len(video_urls)}: {v[:80]}...")
                    # 检测Windows路径，直接跳过（客户端应该在上传前处理）
                    if re.match(r'^[a-zA-Z]:\\', v) or re.match(r'^\\\\', v):
                        _log.error(f"❌ 跳过Windows本地路径（服务器无法访问）: {v}")
                        _log.error(f"💡 客户端应该在发送消息前将本地文件上传到服务器")
                        continue  # 跳过这个视频
                    vc = svc_download_video_to_storage(v, VIDEO_UPLOAD_DIR, server_base_url, _metrics_add, _log)
                    if vc:
                        # 下载成功，使用服务器URL
                        cached_videos.append(vc)
                    else:
                        # 下载失败，如果是HTTP URL仍然保留（可能可以访问）
                        if v.startswith(('http://', 'https://')):
                            _log.warning(f"⚠️ 视频下载失败，保留原始HTTP URL: {v}")
                            cached_videos.append(v)
                        else:
                            _log.warning(f"⚠️ 跳过无效视频URL: {v}")
                    
                    # 再次检查中断（下载完成后）
                    if interrupt_event and interrupt_event.is_set():
                        if chat_id and response_dict:
                            with queue_lock:
                                current_processing = processing_chats.get(chat_id)
                                if current_processing and current_processing["response_dict"] is not response_dict:
                                    _log.warning(f"⚠️ 聊天 {chat_id} 的视频下载完成后被新消息中断，退出处理")
                                    return
                # 过滤无效视频路径/协议
                def _is_valid_video_source_priv(p: str) -> bool:
                    try:
                        if not p:
                            return False
                        if p.lower().startswith(("http://", "https://")):
                            return True
                        import os as _os
                        return _os.path.exists(p)
                    except Exception:
                        return False
                # 将服务器静态URL转换为本地文件路径（transformers库需要本地文件路径，不支持HTTP URL）
                valid_video_paths = []
                base_static = (server_base_url or "http://127.0.0.1:9999").rstrip("/") + "/static/videos/"
                for v in cached_videos:
                    if v and v.startswith(base_static):
                        # 这是服务器URL，转换为对应的本地文件路径
                        filename = v.split("/")[-1]
                        local_path = os.path.join(VIDEO_UPLOAD_DIR, filename)
                        if os.path.exists(local_path):
                            # 本地文件存在，使用本地文件路径
                            valid_video_paths.append(local_path)
                            _log.debug(f"🎥 使用本地视频文件路径: {local_path}")
                        else:
                            # 本地文件不存在，保留服务器URL（虽然可能无法访问）
                            valid_video_paths.append(v)
                            _log.warning(f"⚠️ 视频本地文件不存在，使用URL: {local_path}")
                    elif v and (v.startswith('http://') or v.startswith('https://')):
                        # 外部URL，保留（虽然可能无法访问，但至少格式正确）
                        valid_video_paths.append(v)
                        _log.debug(f"🎥 保留外部视频URL: {v}")
                    else:
                        # 无效URL，跳过
                        _log.warning(f"⚠️ 跳过无效视频URL: {v}")
                video_urls = valid_video_paths
            
            # 本地化语音URL并进行ASR转写
            asr_texts: List[str] = []
            if 'audio_urls' in locals() and audio_urls:
                _log.info(f"✅ 从CQ码中提取到 {len(audio_urls)} 个语音URL")
                cached_audios = []
                for a in audio_urls:
                    cached = svc_download_audio_to_storage(a, AUDIO_UPLOAD_DIR, server_base_url, _metrics_add, _log)
                    cached_audios.append(cached or a)
                for ca in cached_audios:
                    if ca and ca.startswith((server_base_url or "http://127.0.0.1:9999").rstrip('/') + "/static/audios/"):
                        filename = ca.rsplit('/', 1)[-1]
                        local_fp = os.path.join(AUDIO_UPLOAD_DIR, filename)
                        text = svc_transcribe_audio(local_fp, _metrics_add, _log)
                        if text:
                            asr_texts.append(text)
                if asr_texts:
                    content = (content + "\n" if content else "") + "【语音转写】" + " ".join(asr_texts)
            
            # 处理文件：下载并提取文本和图片内容（支持中断）
            file_texts: List[str] = []
            file_image_paths: List[str] = []
            if 'file_urls' in locals() and file_urls:
                _log.info(f"✅ 检测到 {len(file_urls)} 个文件，开始提取内容")
                for file_url in file_urls:
                    # 检查是否被中断
                    if interrupt_event and interrupt_event.is_set():
                        _log.info(f"⚠️ 文件处理被中断，停止处理剩余文件")
                        break
                    
                    try:
                        # 下载文件到服务器
                        cached_file_url = svc_download_file_to_storage(file_url, FILE_UPLOAD_DIR, server_base_url, _metrics_add, _log)
                        
                        # 再次检查中断（下载可能耗时）
                        if interrupt_event and interrupt_event.is_set():
                            _log.info(f"⚠️ 文件下载后被中断，停止处理")
                            break
                        
                        if cached_file_url and cached_file_url.startswith((server_base_url or "http://127.0.0.1:9999").rstrip('/') + "/static/files/"):
                            # 将URL转成本地文件路径
                            filename = cached_file_url.rsplit('/', 1)[-1]
                            local_fp = os.path.join(FILE_UPLOAD_DIR, filename)
                            if os.path.exists(local_fp):
                                # 提取文本和图片
                                file_text, file_images = svc_extract_text_and_images_from_file(
                                    local_fp, IMAGE_UPLOAD_DIR, _metrics_add, _log
                                )
                                
                                # 再次检查中断（提取可能耗时）
                                if interrupt_event and interrupt_event.is_set():
                                    _log.info(f"⚠️ 文件提取后被中断，停止处理")
                                    break
                                
                                if file_text:
                                    file_texts.append(file_text)
                                # 将提取的图片路径转换为URL并添加到image_urls
                                for img_path in file_images:
                                    if interrupt_event and interrupt_event.is_set():
                                        break
                                    if os.path.exists(img_path):
                                        img_filename = os.path.basename(img_path)
                                        img_url = f"{server_base_url.rstrip('/')}/static/images/{img_filename}"
                                        if image_urls is None:
                                            image_urls = []
                                        if img_url not in image_urls:
                                            image_urls.append(img_url)
                                            file_image_paths.append(img_path)
                                if file_text or file_images:
                                    _log.info(f"✅ 文件处理完成: 文本长度={len(file_text)}, 图片数={len(file_images)}")
                    except Exception as file_err:
                        _log.warning(f"⚠️ 处理文件失败 {file_url}: {file_err}")
                
                # 将提取的文本内容添加到content（如果未被中断）
                if not (interrupt_event and interrupt_event.is_set()) and file_texts:
                    file_content = "\n\n".join([f"【文件内容{i+1}】\n{t}" for i, t in enumerate(file_texts)])
                    content = (content + "\n\n" if content else "") + file_content

            media_info = ""
            if image_urls:
                media_info += f" [包含{len(image_urls)}张图片]"
            if video_urls:
                media_info += f" [包含{len(video_urls)}个视频]"
            if 'audio_urls' in locals() and audio_urls:
                media_info += f" [包含{len(audio_urls)}段语音]"
            if 'file_urls' in locals() and file_urls:
                media_info += f" [包含{len(file_urls)}个文件]"
            _log.info(f"收到私聊消息 [用户:{user_id}({user_nickname})]: {content[:50] if content else '(仅多媒体)'}{media_info}...")
            
            # 格式化时间戳
            time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            # 格式化用户消息（包含QQ号信息，方便模型在历史消息中识别用户）
            message_prefix = f"[{time_str}] {user_nickname}(QQ:{user_id})"
            formatted_message = f"{message_prefix}: {content}" if content else f"{message_prefix}:"
            
            # 构建消息内容
            if image_urls or video_urls:
                message_content = format_multimodal_message(formatted_message, image_urls, video_urls if 'video_urls' in locals() else [])
            else:
                message_content = [{"type": "text", "text": formatted_message}]
            
            # 链接和卡片信息直接保留原始CQ码在content中，不提取内容，不构建目录
            # （已移除目录构建逻辑，保留原始信息）
            
            # 更新聊天记录
            with chat_history_lock:
                if user_id not in private_chat_histories:
                    private_chat_histories[user_id] = []
                
                private_chat_histories[user_id].append({
                    "role": "user",
                    "content": message_content
                })
                
                maintain_chat_history("private", user_id, private_chat_histories[user_id])
            
            # 在生成回复之前，再次检查是否仍然是最新的任务（防止在加入历史期间有新消息到达）
            # 同时检查interrupt_event是否被错误设置
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is not response_dict:
                    # 已经有更新的任务了，当前任务应该退出（但消息已经加入历史，这是正确的）
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在生成回复前已被更新的消息替换，退出处理")
                    return
                
                # 如果当前任务仍然是最新的，但interrupt_event被设置了，清除它
                # 这可能是误设置（比如在等待期间被中断，但之后又成为最新任务）
                if interrupt_event.is_set():
                    _log.warning(f"⚠️ 聊天 {chat_id} 的任务在生成回复前检测到interrupt_event被设置，但任务仍是最新的，清除中断信号")
                    interrupt_event.clear()
            
            # 生成回复（支持最多1轮工具调用再生成）
            _log.info(f"🧠 开始生成私聊回复（用户 {user_id}）...")
            chat_context = {
                "user_id": user_id,
                "user_nickname": user_nickname
            }
            
            # 构建系统提示词
            system_prompt = build_system_prompt("private", chat_context)
            
            # 在生成前，先对原始历史进行token长度检查和截断
            with chat_history_lock:
                # 确保user_id存在于private_chat_histories中
                if user_id not in private_chat_histories:
                    _log.warning(f"⚠️ 用户 {user_id} 的聊天历史不存在，初始化为空列表")
                    private_chat_histories[user_id] = []
                
                # 对原始历史进行截断（会修改原始历史并保存被删除的消息）
                # 确保user_id存在于private_chat_histories中
                if user_id not in private_chat_histories:
                    _log.warning(f"⚠️ 用户 {user_id} 的聊天历史不存在，初始化为空列表")
                    private_chat_histories[user_id] = []

                truncated_history = private_chat_histories[user_id].copy()  # 默认值，使用副本
                _log.info(f"📊 原始聊天历史长度: {len(private_chat_histories[user_id])}（私聊 {user_id}）")

                try:
                    max_tokens_limit = get_chat_history_token_limit()
                    _log.info(f"📊 获取到的max_tokens限制: {max_tokens_limit}, 类型: {type(max_tokens_limit)}（私聊 {user_id}）")
                    
                    # 验证max_tokens_limit
                    if max_tokens_limit is None:
                        _log.error(f"❌ max_tokens_limit为None，使用默认值35000（私聊 {user_id}）")
                        max_tokens_limit = 35000
                    elif not isinstance(max_tokens_limit, int) or max_tokens_limit <= 0:
                        _log.error(f"❌ max_tokens_limit无效: {max_tokens_limit}，使用默认值35000（私聊 {user_id}）")
                        max_tokens_limit = 35000

                    # 检查是否需要截断
                    if len(private_chat_histories[user_id]) == 0:
                        _log.info("📊 聊天历史为空，无需截断")
                        truncated_history = []
                    else:
                        _log.info(f"📊 开始调用truncate_history_by_tokens（私聊 {user_id}），max_tokens={max_tokens_limit}")
                        result = truncate_history_by_tokens(
                            private_chat_histories[user_id],
                            system_prompt,
                            "private",
                            user_id,
                            max_tokens=max_tokens_limit,
                            interrupt_event=interrupt_event
                        )
                        _log.info(f"📊 truncate_history_by_tokens返回: 类型={type(result)}, 是否为None={result is None}, 长度={len(result) if result is not None else 'N/A'}（私聊 {user_id}）")
                        truncated_history = result

                    # 确保返回值不为None且是列表类型
                    if truncated_history is None:
                        _log.error(f"❌ 截断历史返回None（私聊 {user_id}），回退到原始历史")
                        truncated_history = private_chat_histories[user_id].copy()
                    elif not isinstance(truncated_history, list):
                        _log.error(f"❌ 截断历史返回非列表类型: {type(truncated_history)}（私聊 {user_id}），回退到原始历史")
                        truncated_history = private_chat_histories[user_id].copy()
                    else:
                        _log.info(f"✅ 截断历史成功，长度: {len(truncated_history)}（私聊 {user_id}）")
                except Exception as e:
                    _log.error(f"❌ 截断历史时发生异常（私聊 {user_id}）: {e}", exc_info=True)
                    # 异常情况下，使用原始历史
                    truncated_history = private_chat_histories[user_id].copy()
                
                # 使用截断后的历史（已经是原始历史的引用）
                current_history = truncated_history.copy()
            
            # 关闭FETCH相关循环（按当前需求放弃链接/卡片/文件访问）
            tool_iterations = 0
            action_cmds = []
            for _iter in range(tool_iterations + 1):
                _gen_ret = generate_reply(
                current_history,
                chat_type="private",
                chat_context=chat_context,
                interrupt_event=interrupt_event,
                chat_id=chat_id,
                    response_dict=response_dict,
                    log_full_io=True
            )
                if isinstance(_gen_ret, tuple) and len(_gen_ret) == 3:
                    reply, should_reply, was_interrupted = _gen_ret
                    action_cmds = []
                else:
                    reply, should_reply, was_interrupted, action_cmds = _gen_ret
                if was_interrupted:
                    break
                pending_fetch = []
                if not pending_fetch or _iter >= tool_iterations:
                    break
                # FETCH功能已禁用，无需处理工具调用
            
            # 检查是否被中断
            if was_interrupted:
                _log.info(f"⚠️ 聊天 {chat_id} 的消息处理被中断，跳过回复")
                _metrics_add("interruptions", 1)
                # 被中断的消息需要更新response_dict，否则客户端会一直等待
                # 但需要确保这是当前任务，避免旧任务覆盖新任务的response_dict
                with queue_lock:
                    current_processing = processing_chats.get(chat_id)
                    if current_processing and current_processing["response_dict"] is response_dict:
                        response_dict.update({
                            "status": "success",
                            "should_reply": False,
                            "reply": ""
                        })
                        _log.info(f"✅ 已更新中断响应（聊天 {chat_id}）")
                return
            
            # 在更新聊天记录之前，再次检查是否仍然是最新的任务
            # 因为可能在生成期间有新消息到达并设置了interrupt_event
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is not response_dict:
                    # 已经有更新的任务了，当前任务应该退出
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在生成完成后被更新的消息替换，跳过更新历史")
                    return
                # 再次检查中断事件（双重保险）
                if interrupt_event.is_set():
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在生成完成后被中断，跳过更新历史")
                    return
            
            # 在更新聊天记录之前，再次检查中断（防止在生成完成后、更新前有新消息到达）
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is not response_dict:
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在更新聊天记录前已被新任务替换，跳过更新")
                    return
                if interrupt_event.is_set():
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在更新聊天记录前被中断，跳过更新")
                    return
            
            # 更新聊天记录（只有在没有被中断的情况下）
            with chat_history_lock:
                # 在持有chat_history_lock期间再次检查中断（双重保险）
                if interrupt_event and interrupt_event.is_set():
                    with queue_lock:
                        current_processing = processing_chats.get(chat_id)
                        if current_processing and current_processing["response_dict"] is not response_dict:
                            _log.info(f"⚠️ 聊天 {chat_id} 的任务在更新聊天记录期间被新任务替换，跳过更新")
                            return
                
                if should_reply:
                    _metrics_add("replies_sent", 1)
                    private_chat_histories[user_id].append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": reply}]
                    })
                    maintain_chat_history("private", user_id, private_chat_histories[user_id])
                    _log.info(f"💬 生成回复（私聊 {user_id}）：{reply[:100]}...")
                else:
                    _metrics_add("no_reply", 1)
                    private_chat_histories[user_id].append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": "<no_reply>"}]
                    })
                    maintain_chat_history("private", user_id, private_chat_histories[user_id])
                    _log.info(f"💬 模型判断不需要回复（私聊 {user_id}）")
            
            # 更新响应（只有在没有被中断的情况下）
            # 再次检查是否仍然是最新的任务（防止在更新聊天记录时被新消息中断）
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is response_dict:
                    # 再次检查中断事件（防止在更新聊天记录时被中断）
                    if interrupt_event.is_set():
                        _log.warning(f"⚠️ 任务在更新响应前被中断（私聊 {user_id}）")
                        return
                    
                    response_dict.update({
                        "status": "success",
                        "should_reply": should_reply,
                        "reply": reply if should_reply else "",
                        "actions": action_cmds if should_reply else []
                    })
                    _log.info(f"✅ 已更新响应（私聊 {user_id}），should_reply={should_reply}, reply长度={len(reply) if reply else 0}")
                else:
                    _log.warning(f"⚠️ 任务已被新消息中断，跳过响应更新（私聊 {user_id}）")
        # 记录请求处理端到端时延
        try:
            _metrics_add_latency((_t.time() - _req_t0) * 1000.0)
        except Exception:
            pass
        
    except Exception as e:
        _log.error(f"处理消息任务失败: {e}", exc_info=True)
        response_dict.update({
            "status": "error",
            "message": str(e),
            "status_code": 500
        })
    finally:
        # 清理处理状态（只有在这是当前任务时才清理）
        with queue_lock:
            current_processing = processing_chats.get(chat_id)
            if current_processing and current_processing["response_dict"] is response_dict:
                del processing_chats[chat_id]
                _log.debug(f"清理处理状态（{chat_type} {chat_id}）")
            else:
                _log.debug(f"跳过清理（任务已被新消息替换，{chat_type} {chat_id}）")


def message_queue_worker():
    """消息队列工作线程（处理队列中的消息）"""
    global message_queue
    _log.info("📋 消息队列工作线程已启动")
    
    while True:
        try:
            # 从队列获取消息（阻塞等待）
            task = message_queue.get()
            
            if task is None:  # 退出信号
                break
            
            _log.info(f"🔄 开始处理消息任务: {task.chat_type} {task.chat_id}")
            
            # 在新线程中处理任务，这样同一聊天的多个消息可以并发处理
            # 中断机制会在process_message_task内部处理
            task_thread = threading.Thread(
                target=svc_handler.run_process_message_task,
                args=(task,),
                daemon=True
            )
            task_thread.start()
            
            # 不等待任务完成，继续处理下一个消息
            # 这样同一聊天的多条消息可以并发处理，新消息会中断旧消息
            message_queue.task_done()
            
        except Exception as e:
            _log.error(f"消息队列工作线程出错: {e}", exc_info=True)


# 使用 services.generation 中的实现
InterruptStoppingCriteria = SvcInterruptStoppingCriteria


def custom_generate(
    model,
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
    完全复刻transformers库的model.generate()方法实现的自定义生成函数
    
    参考官方源码实现，完全按照官方逻辑：
    1. _get_initial_cache_position() - 初始化cache_position（官方方法）
    2. prepare_inputs_for_generation() - 准备模型输入（自动处理KV cache和attention_mask）
    3. _update_model_kwargs_for_generation() - 更新model_kwargs（包括past_key_values、attention_mask、cache_position）
    4. LogitsProcessor处理（如repetition_penalty）
    5. LogitsWarper处理（如temperature, top_k, top_p）
    6. StoppingCriteria检查（每个token后检查）
    7. EOS token检查（完全按照官方逻辑）
    8. 支持多模态输入（pixel_values等）
    
    这个实现完全按照官方源码逻辑，可以方便后续进行魔改。
    """
    # 获取输入
    input_ids = inputs.get('input_ids')
    attention_mask = inputs.get('attention_mask', None)
    
    # 初始化生成状态
    batch_size = input_ids.shape[0]
    cur_len = input_ids.shape[-1]
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    
    # 初始化stopping_criteria
    if stopping_criteria is None:
        stopping_criteria = StoppingCriteriaList()
    
    # 初始化logits_processor
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    
    # 构建logits_warper（采样时使用）
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
    
    # 准备model_kwargs（完全按照transformers官方实现）
    # 从inputs中提取所有非input_ids和attention_mask的字段
    model_kwargs = {}
    for key, value in inputs.items():
        if key not in ['input_ids', 'attention_mask']:
            model_kwargs[key] = value
    
    # 如果提供了attention_mask，添加到model_kwargs
    if attention_mask is not None:
        model_kwargs['attention_mask'] = attention_mask
    
    # 确保use_cache存在（默认True）
    if 'use_cache' not in model_kwargs:
        model_kwargs['use_cache'] = True
    
    # 初始化cache_position（完全按照官方_get_initial_cache_position逻辑）
    # 参考官方源码：_get_initial_cache_position方法
    # 这是prefilling阶段，确定输入的长度
    if not model_kwargs.get("use_cache", True):
        model_kwargs["cache_position"] = None
    else:
        past_length = 0
        # 如果输入了 past_key_values，则根据 past_key_values 确定缓存序列的长度
        if "past_key_values" in model_kwargs and model_kwargs["past_key_values"] is not None:
            try:
                from transformers.cache_utils import Cache
                if isinstance(model_kwargs["past_key_values"], Cache):
                    past_length = model_kwargs["past_key_values"].get_seq_length()
                else:
                    past_length = model_kwargs["past_key_values"][0][0].shape[2]
            except (ImportError, AttributeError):
                # 如果不是Cache类型，直接取shape
                past_length = model_kwargs["past_key_values"][0][0].shape[2]
        
        # 如果输入 inputs_embeds 则根据这个确定
        if "inputs_embeds" in model_kwargs:
            input_seq_len = model_kwargs["inputs_embeds"].shape[1]
        else:
            # 都没有就根据 input_ids 确定
            input_seq_len = input_ids.shape[-1]
        
        # 创建输入序列的位置索引（完全按照官方逻辑）
        # cache_position = torch.arange(past_length, input_seq_len, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(past_length, input_seq_len, device=input_ids.device)
    
    # 处理EOS token（转换为列表形式）
    if eos_token_id is not None:
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_ids = torch.tensor(list(eos_token_id), device=input_ids.device)
        else:
            eos_token_ids = torch.tensor([eos_token_id], device=input_ids.device)
    else:
        eos_token_ids = None
    
    # 检查是否有EOS停止条件（用于后续处理）
    has_eos_stopping_criteria = eos_token_ids is not None
    
    # 获取recall token ID（用于检测回忆触发）
    global recall_token_ids, memory_db, processor
    recall_token_id = recall_token_ids.get("<recall>") if recall_token_ids else None
    memory_pad_token_id = recall_token_ids.get("<|memory_pad|>") if recall_token_ids else None
    
    # 记录记忆向量插入位置（用于返回，但不再用于打印标注，因为<|memory_pad|>会原生显示）
    memory_injection_positions = []  # 存储 (token_position, memory_score) 元组
    
    memory_cfg = config.get("memory", {}).get("autoregressive_recall", {})
    autorecall_enabled = bool(memory_cfg.get("enabled", False))
    autorecall_top_k = max(1, int(memory_cfg.get("top_k", 5)))
    autorecall_temperature = float(memory_cfg.get("temperature", 1.0))
    autorecall_top_p = float(memory_cfg.get("top_p", 1.0))
    autorecall_use_sampling = bool(memory_cfg.get("use_sampling", True))  # 默认使用采样
    autorecall_debug = bool(memory_cfg.get("debug", False))
    recall_pending = False

    def _update_model_kwargs_helper(outputs_obj):
        """安全更新model_kwargs，兼容不同版本transformers"""
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
        """
        使用backbone执行一次forward，返回等价于CausalLMOutputWithPast的结果，
        并附带last_hidden_state供回忆机制使用。
        """
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
            
    def _sample_memory_embedding_from_db(query_vec):
        """根据查询向量从记忆库中采样记忆embedding"""
        if memory_db is None or len(memory_db) == 0:
            _log.info("🔍 [向量匹配] 记忆向量库为空，无法进行匹配")
            return None, None

        _log.info(f"🔍 [向量匹配] 开始搜索记忆库，查询向量shape: {query_vec.shape}, top_k={autorecall_top_k}")
        search_results = memory_db.search(
            query_vec.detach().clone(),
            top_k=max(autorecall_top_k, 1),
            debug=autorecall_debug
        )
        if not search_results:
            _log.info("🔍 [向量匹配] 未找到匹配的记忆向量")
            return None, None

        _log.info(f"🔍 [向量匹配] 找到 {len(search_results)} 个候选记忆向量")
        for i, result in enumerate(search_results):
            score = result.get('score', 0.0)
            _log.info(f"  [{i+1}] 相似度={score:.4f}")

        temperature = max(1e-5, autorecall_temperature)
        scores = torch.tensor(
            [item['score'] for item in search_results],
            dtype=torch.float32,
            device=query_vec.device
        )
                
        # 可选 top-p 截断
        if 0 < autorecall_top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            probs_for_p = torch.softmax(sorted_scores / temperature, dim=-1)
            cumulative = torch.cumsum(probs_for_p, dim=-1)
            cutoff_mask = cumulative <= autorecall_top_p
            cutoff_mask[..., 0] = True  # 确保至少保留一个
            valid_indices = sorted_indices[cutoff_mask]
            if len(valid_indices) > 0:
                scores = scores[valid_indices]
                search_results = [search_results[i.item()] for i in valid_indices]
                _log.info(f"🔍 [向量匹配] top_p={autorecall_top_p} 截断后保留 {len(search_results)} 个候选")
            probs = torch.softmax(scores / temperature, dim=-1)
        else:
            probs = torch.softmax(scores / temperature, dim=-1)

        if autorecall_use_sampling:
            choice_idx = torch.multinomial(probs, num_samples=1).item()
            _log.info(f"🔍 [向量匹配] 使用采样方式选择记忆，选择索引: {choice_idx}, 概率: {probs[choice_idx]:.4f}")
        else:
            choice_idx = torch.argmax(scores).item()
            _log.info(f"🔍 [向量匹配] 使用贪婪方式选择记忆，选择索引: {choice_idx}, 最高相似度: {scores[choice_idx]:.4f}")

        selected = search_results[choice_idx]
        embedding_tensor = selected['embedding']
        _log.info(f"✅ [向量匹配] 已选择记忆向量，相似度={selected.get('score', 0.0):.4f}")
        return embedding_tensor, selected

    def _inject_memory_embedding(memory_embedding_tensor):
        """将记忆embedding注入模型，返回新的outputs"""
        nonlocal model_kwargs, input_ids, memory_pad_token_id
        if memory_embedding_tensor is None:
            _log.warning("⚠️ [向量插入] 记忆向量为None，无法注入")
            return None

        actual_device = next(model.parameters()).device
        memory_dtype = next(model.parameters()).dtype

        _log.info(f"💉 [向量插入] 开始注入记忆向量，shape: {memory_embedding_tensor.shape}, device: {actual_device}, dtype: {memory_dtype}")

        # 在input_ids末尾添加<|memory_pad|> token
        if memory_pad_token_id is not None:
            memory_pad_tensor = torch.tensor([[memory_pad_token_id]], dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, memory_pad_tensor], dim=-1)
            _log.info(f"💉 [向量插入] 在input_ids中插入<|memory_pad|> token ID: {memory_pad_token_id}")
        else:
            _log.warning("⚠️ [向量插入] <|memory_pad|> token ID不存在，无法在input_ids中标记记忆向量位置")

        # 更新attention_mask供下一步使用
        if 'attention_mask' in model_kwargs and model_kwargs['attention_mask'] is not None:
            old_mask_len = model_kwargs['attention_mask'].shape[1]
            new_attention_mask = torch.ones(1, 1, device=actual_device, dtype=torch.long)
            model_kwargs['attention_mask'] = torch.cat(
                [model_kwargs['attention_mask'], new_attention_mask],
                dim=1
            )
            _log.info(f"💉 [向量插入] 更新attention_mask: {old_mask_len} -> {model_kwargs['attention_mask'].shape[1]}")

        # 使用统一的注入方法：准备inputs_embeds并注入向量
        # 由于这是推理阶段，我们需要获取当前模型输入对应的embeddings
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # 获取token embeddings（只对新添加的token）
        embedding_layer = model.get_input_embeddings()
        # 只对新添加的<|memory_pad|> token生成embedding，然后替换为记忆向量
        memory_position = -1  # 新添加的token在末尾

        # 创建单token的embeddings用于注入
        single_token_embed = embedding_layer(torch.tensor([[memory_pad_token_id]], device=actual_device))
                
        # 使用统一的注入方法替换为记忆向量（传入验证参数）
        injected_embed = inject_memory_embedding_to_inputs_embeds(
            single_token_embed, 0, memory_embedding_tensor,
            input_ids=torch.tensor([[memory_pad_token_id]], device=actual_device),
            memory_pad_token_id=memory_pad_token_id
        )

        _log.info(f"💉 [向量插入] 使用统一注入方法，替换位置: {memory_position}")
                
        # 使用注入后的embeddings进行前向传播
        with torch.no_grad():
            memory_outputs = _forward_with_last_hidden_state({
                "inputs_embeds": injected_embed,
                "attention_mask": torch.ones(1, 1, device=actual_device, dtype=torch.long),
                "past_key_values": model_kwargs.get('past_key_values'),
                "use_cache": True,
            })
                
        _log.info(f"💉 [向量插入] 记忆向量前向传播完成，outputs.logits.shape: {memory_outputs.logits.shape}")

        _update_model_kwargs_helper(memory_outputs)
        _log.info(f"✅ [向量插入] 记忆向量注入成功，已更新model_kwargs")
        return memory_outputs
    
    # 生成循环：完全按照transformers官方实现
    while cur_len < max_new_tokens:
        # 检查中断信号
        if interrupt_event and interrupt_event.is_set():
            break
                
        # 使用官方方法准备模型输入
        # prepare_inputs_for_generation会自动处理：
        # - KV cache时的input_ids裁剪（只传入未缓存的token）
        # - attention_mask的正确长度和格式
        # - position_ids的处理
        # - cache_position的处理
        # - 其他model_kwargs的传递
        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            **model_kwargs
        )
                
        # 🔄 统一记忆触发机制：检测到最新输入是<recall> token时触发回忆
        # 检查当前要处理的最后一个token是否是<recall> token
        current_input_ids = model_inputs.get('input_ids', input_ids)
        if current_input_ids.shape[-1] > 0:
            last_token_id = current_input_ids[0, -1].item()
            if (
                autorecall_enabled
                and recall_token_id is not None
                and last_token_id == recall_token_id
                and not recall_pending  # 避免重复触发
            ):
                if memory_db is None or len(memory_db) == 0:
                    _log.info("ℹ️ [输入检测] 记忆向量库为空，<recall> token按普通token处理")
                else:
                    _log.info(f"🎯 [输入检测] 检测到最新输入是<recall> token (ID: {recall_token_id})，触发回忆机制")
                    recall_pending = True
                
        # 前向传播（使用backbone提取<recall>向量）
        forward_inputs = dict(model_inputs)
        forward_inputs.setdefault("use_cache", model_kwargs.get("use_cache", True))
        outputs = _forward_with_last_hidden_state(forward_inputs)
        last_hidden_state = outputs.last_hidden_state

        if autorecall_enabled and recall_pending:
            recall_pending = False
            _log.info("🔄 [回忆触发] 检测到recall_pending=True，开始处理回忆机制")
            if last_hidden_state is None:
                _log.warning("⚠️ [回忆触发] 无法获取<recall>隐藏向量，继续普通生成")
            elif memory_db is None or len(memory_db) == 0:
                _log.info("ℹ️ [回忆触发] 记忆向量库为空，<recall> 按普通token处理")
            else:
                query_vector = last_hidden_state[0, -1, :]
                _log.info(f"🔍 [回忆触发] 提取<recall> token的hidden state作为查询向量，shape: {query_vector.shape}")
                memory_embedding, selected_meta = _sample_memory_embedding_from_db(query_vector)
                if memory_embedding is None:
                    _log.info("ℹ️ [回忆触发] 未找到可用记忆，<recall> 按普通token处理")
                else:
                    memory_score = selected_meta.get("score") if selected_meta else None
                    if memory_score is not None:
                        _log.info(f"🎯 [回忆触发] 采样到记忆向量，相似度={memory_score:.4f}")
                    memory_outputs = _inject_memory_embedding(memory_embedding)
                    if memory_outputs is None:
                        _log.warning("⚠️ [回忆触发] 记忆向量注入失败，继续普通生成")
                    else:
                        outputs = memory_outputs
                        last_hidden_state = outputs.last_hidden_state
                        # 记录记忆向量插入位置（用于返回，但不再用于打印标注，因为<|memory_pad|>会原生显示）
                        memory_score = selected_meta.get("score", 0.0) if selected_meta else 0.0
                        injection_pos = input_ids.shape[-1]  # 记忆向量插入在当前位置之后
                        memory_injection_positions.append((injection_pos, memory_score))
                        _log.info(f"✅ [回忆触发] 记忆向量注入成功，相似度={memory_score:.4f}")

        # 获取logits（只取最后一个位置的logits）
        next_token_logits = outputs.logits[:, -1, :]
        
        # 应用LogitsProcessor（如repetition_penalty）
        # 注意：LogitsProcessor接收(input_ids, scores)作为参数
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # 应用LogitsWarper（如temperature, top_k, top_p）
        if do_sample and logits_warper is not None:
            next_token_scores = logits_warper(input_ids, next_token_scores)
                
        # 采样下一个token（完全按照官方实现）
        if do_sample:
            # 转换为概率分布
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # 贪婪解码
            next_tokens = torch.argmax(next_token_scores, dim=-1)
                
        # 注意：记忆触发机制已统一为"检测到最新输入是<recall> token时触发"
        # 因此不再需要在这里检测生成的<recall> token
        # 生成的<recall> token会在下一轮循环的前向传播前被检测到
        # 注意：插入完记忆向量后已立即退出回忆模式，不需要检测</recall> token
        
        # 处理EOS token：完全按照transformers官方实现
        # 如果生成完成了，就将新生成的token替换成pad_token_id
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        # 更新generated ids和model inputs（完全按照官方实现）
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        # 使用官方方法更新model_kwargs（完全按照官方实现）
        # _update_model_kwargs_for_generation会自动处理：
        # - past_key_values的更新
        # - attention_mask的更新
        # - cache_position的更新
        # - 移除只在首次前向传播时需要的字段（如pixel_values）
        try:
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
                standardize_cache_format=True,
            )
        except TypeError:
            # 如果参数不支持，尝试不带standardize_cache_format
            try:
                model_kwargs = model._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=False,
                )
            except TypeError:
                # 如果还是不支持，尝试只传必需参数
                model_kwargs = model._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                )
        
        # 更新未完成序列标记（完全按照transformers官方实现）
        if eos_token_ids is not None:
            # 检查新生成的token是否是EOS token
            # 使用广播操作，检查每个next_token是否等于任何一个eos_token_id
            eos_in_sentence = (next_tokens.unsqueeze(-1) == eos_token_ids.unsqueeze(0)).any(dim=-1)
            unfinished_sequences = unfinished_sequences & ~eos_in_sentence
        
        cur_len += 1
        
        # 检查StoppingCriteria（在每个token生成后检查）
        # 这是transformers标准实现的关键部分
        # 注意：官方使用 & 操作符，不是 mul
        # stopping_criteria返回bool或tensor，表示是否应该停止
        should_stop = stopping_criteria(input_ids, next_token_scores)
        
        # 如果stopping_criteria返回单个bool值，需要转换为tensor
        if isinstance(should_stop, bool):
            # 对于单个bool值，转换为tensor（与batch_size匹配）
            should_stop_tensor = torch.tensor([should_stop], device=unfinished_sequences.device, dtype=torch.bool)
            # 如果batch_size > 1，需要扩展到所有序列
            if batch_size > 1:
                should_stop_tensor = should_stop_tensor.expand(batch_size)
        else:
            # 如果已经是tensor，直接使用
            should_stop_tensor = should_stop.bool() if should_stop.dtype != torch.bool else should_stop
        
        # 更新unfinished_sequences：如果should_stop为True，则标记为已完成
        unfinished_sequences = unfinished_sequences & ~should_stop_tensor
        
        # 如果所有序列都完成了，提前停止
        if unfinished_sequences.max() == 0:
            # 记录停止原因（用于调试）
            if interrupt_event and interrupt_event.is_set():
                _log.info("⚠️ 生成因中断而停止")
            else:
                # StoppingCriteria停止是正常情况（如达到最大长度、遇到停止词等），使用debug级别
                _log.debug("生成因StoppingCriteria而停止（正常停止，如达到最大长度或遇到停止词）")
            break
    
        # 早停：检测到<tool_call>闭合标签即停止（首轮即可触发）
        if early_stop_on_tool_call:
            try:
                # 解码当前全部（包含特殊token），查找工具调用闭合
                decoded_so_far = processor.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                open_idx = decoded_so_far.rfind("<tool_call")
                if open_idx != -1:
                    close_idx = decoded_so_far.rfind("</tool_call>")
                    if close_idx != -1 and close_idx > open_idx:
                        _log.info("🔧 检测到工具调用闭合标签，提前结束首轮生成")
                        break
            except Exception:
                pass
    
    # 返回生成结果和记忆插入位置信息
    # 为了保持向后兼容，如果memory_injection_positions为空，只返回input_ids
    # 否则返回元组 (input_ids, memory_injection_positions)
    if memory_injection_positions:
        return input_ids, memory_injection_positions
    else:
        return input_ids




def truncate_history_by_tokens(chat_history: List[Dict[str, Any]], system_prompt: str, 
                                 chat_type: str, chat_id: str, 
                                 max_tokens: int = 35000,
                                 interrupt_event: threading.Event = None) -> List[Dict[str, Any]]:
    """
    根据token数量截断聊天历史记录
    
    Args:
        chat_history: 聊天历史记录（原始列表，会被修改）
        system_prompt: 系统提示词
        chat_type: "group" 或 "private"
        chat_id: 群ID或用户ID
        max_tokens: 最大token数量（默认35000）
        interrupt_event: 中断事件（如果被设置，则立即返回）
    
    Returns:
        截断后的聊天历史记录（如果被中断，返回原始历史）
    """
    global model, processor
    
    # 防御性检查：确保chat_history不为None
    if chat_history is None:
        _log.error(f"❌ chat_history为None，无法进行截断（{chat_type} {chat_id}）")
        return []
    
    # 在开始前检查中断
    if interrupt_event and interrupt_event.is_set():
        _log.info(f"⚠️ 截断历史消息在开始前被中断（{chat_type} {chat_id}）")
        return chat_history
    
    if model is None or processor is None:
        _log.warning("⚠️ 模型未初始化，无法检查token长度，跳过截断")
        return chat_history
    
    # 构建完整的消息列表用于检查
    full_messages = []
    if system_prompt:
        full_messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })
    full_messages.extend(chat_history)
    
    # 第一次tokenize检查长度
    try:
        # 在apply_chat_template前检查中断（处理图片可能需要很长时间）
        if interrupt_event and interrupt_event.is_set():
            _log.info(f"⚠️ 截断历史消息在第一次tokenize前被中断（{chat_type} {chat_id}）")
            return chat_history
        
        inputs = processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            max_length=None,  # 不限制长度
            truncation=False,  # 不截断
            padding=False
        )
    except Exception as e:
        # 处理图片相关错误的情况 - 对于截断检查，我们跳过图片处理
        error_msg = str(e)
        error_type = type(e).__name__

        image_errors = [
            "multimedia.nt.qq.com.cn", "Failed to resolve", "NameResolutionError",
            "UnidentifiedImageError", "cannot identify image file",
            "HTTPConnectionPool", "ConnectionError", "Timeout"
        ]

        is_image_error = any(img_err in error_msg for img_err in image_errors) or error_type in [
            "UnidentifiedImageError", "ConnectionError", "Timeout"
        ]

        video_errors = [
            "PyAV is not installed", "torchvision.io.video", "read_video", "video_utils.py",
            "Using `torchvision` for video decoding is deprecated"
        ]
        is_video_error = any(ve in error_msg for ve in video_errors)

        if is_image_error or is_video_error:
            _log.warning(f"⚠️ 多媒体处理失败（截断检查），尝试移除图片/视频后重试 (错误类型: {error_type}): {error_msg}")
            # 移除图片/视频项，仅用于长度检查
            cleaned_messages = []
            for msg in full_messages:
                if isinstance(msg.get("content"), list):
                    cleaned_content = []
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            cleaned_content.append(item)
                        # 忽略 image / video
                    if cleaned_content:
                        cleaned_messages.append({"role": msg["role"], "content": cleaned_content})
                else:
                    cleaned_messages.append(msg)
            try:
                inputs = processor.apply_chat_template(
                    cleaned_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    max_length=None,
                    truncation=False,
                    padding=False
                )
            except Exception as e2:
                _log.warning(f"⚠️ 即使移除多媒体也无法进行截断检查，退回原始历史: {e2}")
                return chat_history
        else:
            # 其他类型的错误，直接抛出异常，让调用方处理
            _log.error(f"❌ 截断历史时发生非多媒体错误（{chat_type} {chat_id}）: {error_type}: {error_msg}", exc_info=True)
            raise e
    
    # 在apply_chat_template后检查中断（只有在成功tokenize后才执行到这里）
    if interrupt_event and interrupt_event.is_set():
        _log.info(f"⚠️ 截断历史消息在第一次tokenize后被中断（{chat_type} {chat_id}）")
        return chat_history
    
    # 检查inputs是否已定义（在异常处理后可能未定义）
    if 'inputs' not in locals() or inputs is None:
        _log.warning("⚠️ inputs未定义，跳过截断")
        return chat_history
    
    if 'input_ids' not in inputs or not isinstance(inputs['input_ids'], torch.Tensor):
        _log.warning("⚠️ 无法获取input_ids，跳过截断")
        return chat_history
    
    input_length = inputs['input_ids'].shape[-1]
    _log.info(f"📊 检查输入token长度: {input_length}, 最大限制: {max_tokens}")
    
    if input_length <= max_tokens:
        _log.info(f"✅ 输入token长度在限制内，无需截断")
        return chat_history
    
    _log.warning(f"⚠️ 输入token长度 ({input_length}) 超过最大限制 ({max_tokens})，开始截断历史消息...")
    
    # 逐条删除最早的消息，直到长度在限制内
    removed_messages = []  # 用于保存被删除的消息
    iteration = 0
    max_iterations = 5  # 最多迭代5次
    
    while input_length > max_tokens and len(chat_history) > 0 and iteration < max_iterations:
        # 在每次迭代前检查中断（重要：在删除消息前检查，避免不必要的修改）
        if interrupt_event and interrupt_event.is_set():
            _log.info(f"⚠️ 截断历史消息在迭代 {iteration} 中被中断（{chat_type} {chat_id}），恢复被删除的消息")
            # 恢复被删除的消息
            chat_history[:0] = removed_messages
            return chat_history
        
        iteration += 1
        
        # 删除最早的一条消息
        removed_msg = chat_history.pop(0)
        removed_messages.append(removed_msg)
        
        # 重新构建消息并检查长度
        test_messages = []
        if system_prompt:
            test_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        test_messages.extend(chat_history)
        
        try:
            # 在重新tokenize前检查中断
            if interrupt_event and interrupt_event.is_set():
                _log.info(f"⚠️ 截断历史消息在重新tokenize前被中断（{chat_type} {chat_id}），恢复被删除的消息")
                chat_history[:0] = removed_messages
                return chat_history
            
            test_inputs = processor.apply_chat_template(
                test_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                max_length=None,
                truncation=False,
                padding=False
            )
            
            # 在重新tokenize后检查中断
            if interrupt_event and interrupt_event.is_set():
                _log.info(f"⚠️ 截断历史消息在重新tokenize后被中断（{chat_type} {chat_id}），恢复被删除的消息")
                chat_history[:0] = removed_messages
                return chat_history
            
            input_length = test_inputs['input_ids'].shape[-1]
            _log.info(f"📊 删除 {iteration} 条消息后，输入token长度: {input_length}")
            
            if input_length <= max_tokens:
                # 长度在限制内，保存被删除的消息并返回
                if removed_messages:
                    save_chat_history_to_storage(chat_type, chat_id, removed_messages)
                    _log.info(f"✅ 已截断历史消息: 删除 {len(removed_messages)} 条，当前长度: {input_length}")
                return chat_history
                
        except Exception as e:
            _log.error(f"❌ 截断历史消息时重新tokenize失败: {e}", exc_info=True)
            # 如果出错，恢复被删除的消息
            chat_history[:0] = removed_messages
            return chat_history
    
    # 如果超过5次迭代还没有达到要求，清空一半的聊天记录
    if iteration >= max_iterations and input_length > max_tokens:
        # 在清空一半前检查中断
        if interrupt_event and interrupt_event.is_set():
            _log.info(f"⚠️ 截断历史消息在清空一半前被中断（{chat_type} {chat_id}），恢复被删除的消息")
            chat_history[:0] = removed_messages
            return chat_history
        
        _log.warning(f"⚠️ 超过 {max_iterations} 次迭代仍未达到要求，清空一半的聊天记录...")
        
        # 保存将被清空的消息
        half_count = len(chat_history) // 2
        if half_count > 0:
            removed_messages.extend(chat_history[:half_count])
            chat_history[:] = chat_history[half_count:]
        
        # 重新检查长度
        test_messages = []
        if system_prompt:
            test_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        test_messages.extend(chat_history)
        
        try:
            # 在清空一半后tokenize前检查中断
            if interrupt_event and interrupt_event.is_set():
                _log.info(f"⚠️ 截断历史消息在清空一半后tokenize前被中断（{chat_type} {chat_id}），恢复被删除的消息")
                chat_history[:0] = removed_messages
                return chat_history
            
            test_inputs = processor.apply_chat_template(
                test_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                max_length=None,
                truncation=False,
                padding=False
            )
            
            # 在清空一半后tokenize后检查中断
            if interrupt_event and interrupt_event.is_set():
                _log.info(f"⚠️ 截断历史消息在清空一半后tokenize后被中断（{chat_type} {chat_id}），恢复被删除的消息")
                chat_history[:0] = removed_messages
                return chat_history
            
            input_length = test_inputs['input_ids'].shape[-1]
            _log.info(f"📊 清空一半后，输入token长度: {input_length}")
            
            if input_length <= max_tokens:
                # 长度在限制内，保存被删除的消息并返回
                if removed_messages:
                    save_chat_history_to_storage(chat_type, chat_id, removed_messages)
                    _log.info(f"✅ 已清空一半历史消息: 删除 {len(removed_messages)} 条，当前长度: {input_length}")
                return chat_history
            else:
                # 清空一半后仍然超过限制，清空全部聊天记录
                _log.error(f"❌ 清空一半后仍然超过限制 ({input_length} > {max_tokens})，清空全部聊天记录")
                removed_messages.extend(chat_history)
                chat_history.clear()
                
                # 保存所有被删除的消息
                if removed_messages:
                    save_chat_history_to_storage(chat_type, chat_id, removed_messages)
                    _log.warning(f"⚠️ 已清空全部历史消息: 删除 {len(removed_messages)} 条")
                return chat_history
                
        except Exception as e:
            _log.error(f"❌ 清空一半后重新tokenize失败: {e}", exc_info=True)
            # 如果出错，恢复被删除的消息
            chat_history[:0] = removed_messages
            return chat_history
    
    # 保存被删除的消息
    if removed_messages:
        save_chat_history_to_storage(chat_type, chat_id, removed_messages)
        _log.info(f"✅ 已截断历史消息: 删除 {len(removed_messages)} 条，当前长度: {input_length}")
    
    # 确保总是返回chat_history（防御性检查）
    if chat_history is None:
        _log.error(f"❌ chat_history意外变为None（{chat_type} {chat_id}），返回空列表")
        return []
    
    return chat_history
        


def generate_reply(chat_history: List[Dict[str, Any]], max_new_tokens: int = None, 
                   temperature: float = None, chat_type: str = None, 
                   chat_context: Dict[str, str] = None, 
                   interrupt_event: threading.Event = None,
                   chat_id: str = None, response_dict: dict = None,
                   log_full_io: bool = True) -> Tuple[Optional[str], bool, bool]:
    """
    使用Qwen3-VL模型生成回复
    
    Args:
        chat_history: 聊天历史，格式：[{"role": "user", "content": [...]}, ...]
        max_new_tokens: 最大生成token数（如果为None，使用配置文件中的值）
        temperature: 温度参数（如果为None，使用配置文件中的值）
        chat_type: "group" 或 "private"，表示对话类型
        chat_context: 对话上下文信息，包含群名称或用户昵称等
    
    Returns:
        (回复文本, 是否需要回复, 是否被中断)
        - 如果被中断，返回(None, False, True)
        - 如果模型判断不需要回复，返回("", False, False)
        - 如果需要回复，返回(回复文本, True, False)
    """
    global model, processor, device, config, is_training, training_lock
    
    # 检查是否处于训练模式
    with training_lock:
        if is_training:
            _log.warning("⚠️ 当前处于训练模式，拒绝生成回复")
            raise RuntimeError("服务器正在训练中，暂时无法生成回复")
    
    if model is None or processor is None:
        raise RuntimeError("模型未初始化")
    
    # 从配置文件读取生成参数（如果未提供）
    gen_config = config.get("generation", {})
    if max_new_tokens is None:
        max_new_tokens = gen_config.get("max_new_tokens", 1000)
    if temperature is None:
        temperature = gen_config.get("temperature", 1.0)  # 官方默认1.0
    do_sample = gen_config.get("do_sample", True)
    top_p = gen_config.get("top_p", 0.95)  # 官方默认0.95
    top_k = gen_config.get("top_k", 20)  # 官方默认20
    repetition_penalty = gen_config.get("repetition_penalty", 1.0)  # 官方默认1.0
    presence_penalty = gen_config.get("presence_penalty", 0.0)  # 官方默认0.0
    
    try:
        # 构建系统提示词（包含对话上下文）
        system_prompt = build_system_prompt(chat_type, chat_context)
        
        # 构建完整的消息列表：系统提示词 + 聊天记录
        full_messages = []
        if system_prompt:
            full_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # 添加聊天记录
        full_messages.extend(chat_history)
        
        # 准备推理输入
        _log.info(f"准备推理输入，系统提示词长度: {len(system_prompt)}, 历史消息数: {len(chat_history)}")
        
        # 使用processor.apply_chat_template处理消息
        # 注意：处理图片时可能需要很长时间，需要在此前后检查中断
        if interrupt_event and interrupt_event.is_set():
            if chat_id and response_dict:
                with queue_lock:
                    current_processing = processing_chats.get(chat_id)
                    if current_processing and current_processing["response_dict"] is not response_dict:
                        _log.warning(f"⚠️ 聊天 {chat_id} 的任务在apply_chat_template前已被新任务替换，退出生成")
                        return None, False, True
        
        try:
            inputs = processor.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                max_length=None,  # 不限制长度
                truncation=False,  # 不截断
                padding=False
            )
            _log.debug(
                f"✅ 第一次apply_chat_template成功，inputs类型: {type(inputs)}，keys: {inputs.keys() if isinstance(inputs, dict) else '非字典'}"
            )
        except Exception as e:
            # 处理图片相关错误的情况
            error_msg = str(e)
            error_type = type(e).__name__

            # 检查是否是图片处理相关的错误
            image_errors = [
                "multimedia.nt.qq.com.cn", "Failed to resolve", "NameResolutionError",
                "UnidentifiedImageError", "cannot identify image file",
                "HTTPConnectionPool", "ConnectionError", "Timeout"
            ]

            is_image_error = any(img_err in error_msg for img_err in image_errors) or error_type in [
                "UnidentifiedImageError", "ConnectionError", "Timeout"
            ]

            # 检查是否是视频处理相关的错误（如缺少PyAV/torchcodec）
            video_errors = [
                "PyAV is not installed", "torchvision.io.video", "read_video", "video_utils.py",
                "Using `torchvision` for video decoding is deprecated"
            ]
            is_video_error = any(ve in error_msg for ve in video_errors)

            if is_image_error or is_video_error:
                _log.warning(f"⚠️ 图片处理失败，开始逐个检查图片有效性 (错误类型: {error_type}): {error_msg}")

                # 逐个检查并移除失效的图片，保留有效的图片
                cleaned_messages = []
                for msg in full_messages:
                    if isinstance(msg.get("content"), list):
                        # 多模态消息，检查每张图片
                        cleaned_content = []
                        for item in msg["content"]:
                            if item.get("type") == "text":
                                # 文本直接保留
                                cleaned_content.append(item)
                            elif item.get("type") == "image":
                                # 图片需要检查URL有效性
                                img_url = item.get("image", "")
                                if img_url and _is_image_url_valid(img_url):
                                    # 图片URL有效，保留
                                    cleaned_content.append(item)
                                    _log.debug(f"✅ 保留有效图片: {img_url}")
                                else:
                                    # 图片URL无效，移除
                                    _log.warning(f"⚠️ 移除失效图片: {img_url}")
                            elif item.get("type") == "video":
                                # 视频内容，检查是否为本地服务器URL（这些是永久有效的）
                                video_url = item.get("video") or item.get("url", "")
                                if video_url:
                                    # 检查是否为本地服务器URL
                                    if (
                                        video_url.startswith('http://127.0.0.1:9999/static/videos/')
                                        or video_url.startswith('http://localhost:9999/static/videos/')
                                        or (
                                            server_base_url
                                            and video_url.startswith(f"{server_base_url.rstrip('/')}/static/videos/")
                                        )
                                    ):
                                        # 本地服务器URL，保留（永久有效）
                                        cleaned_content.append(item)
                                        _log.debug(f"✅ 保留本地视频URL: {video_url}")
                                    else:
                                        # 非本地URL，移除（外部URL可能已失效或环境不支持解码）
                                        _log.warning(f"⚠️ 移除非本地视频URL: {video_url}")
                                else:
                                    _log.warning("⚠️ 发现无效的视频项（无URL），跳过")
                        if cleaned_content:  # 只保留有内容的消息
                            cleaned_messages.append({
                                "role": msg["role"],
                                "content": cleaned_content
                            })
                    else:
                        # 纯文本消息，直接保留
                        cleaned_messages.append(msg)

                # 如果清理后没有有效消息，返回错误
                if not cleaned_messages:
                    _log.error("❌ 清理失效图片后没有有效消息内容")
                    return None, False, False

                # 使用清理后的消息重试
                try:
                    inputs = processor.apply_chat_template(
                        cleaned_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                        max_length=None,
                        truncation=False,
                        padding=False
                    )
                    _log.info("✅ 成功使用清理后的消息（移除失效图片）继续处理")
                    _log.debug(
                        f"✅ 重试apply_chat_template成功，inputs类型: {type(inputs)}，keys: {inputs.keys() if isinstance(inputs, dict) else '非字典'}"
                    )
                except Exception as retry_error:
                    _log.error(f"❌ 即使移除失效图片也失败: {retry_error}")
                    return None, False, False
            else:
                # 其他类型的错误，直接抛出
                raise e
        
        # 在apply_chat_template后立即检查中断（处理图片可能耗时很长）
        if interrupt_event and interrupt_event.is_set():
            if chat_id and response_dict:
                with queue_lock:
                    current_processing = processing_chats.get(chat_id)
                    if current_processing and current_processing["response_dict"] is not response_dict:
                        _log.warning(f"⚠️ 聊天 {chat_id} 的任务在apply_chat_template后被新任务替换，退出生成")
                        return None, False, True
                    elif current_processing and current_processing["response_dict"] is response_dict:
                        # 如果任务仍然是最新的，但interrupt_event被设置了，可能是误设置，清除它
                        _log.warning(f"⚠️ 聊天 {chat_id} 的任务在apply_chat_template后检测到interrupt_event被设置，但任务仍是最新的，清除中断信号")
                        interrupt_event.clear()
            else:
                # 如果没有chat_id和response_dict，无法验证，为了安全直接退出
                _log.warning("⚠️ 生成任务在apply_chat_template后检测到interrupt_event被设置，但无法验证任务状态，退出")
                return None, False, True
        
        # 移动到正确设备

        # 先验证input_ids的有效性（防止索引越界）
        vocab_size = model.config.vocab_size if hasattr(model, 'config') and hasattr(model.config, 'vocab_size') else None
        if vocab_size is not None and 'input_ids' in inputs:
            input_ids_check = inputs['input_ids']
            if isinstance(input_ids_check, torch.Tensor):
                invalid_tokens = input_ids_check[input_ids_check >= vocab_size]
                if len(invalid_tokens) > 0:
                    _log.error(f"⚠️ 检测到无效token ID: {invalid_tokens}, vocab_size={vocab_size}")
                    # 将无效的token ID限制在有效范围内
                    inputs['input_ids'] = torch.clamp(input_ids_check, 0, vocab_size - 1)
        
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 在移动到设备后检查中断
        if interrupt_event and interrupt_event.is_set():
            if chat_id and response_dict:
                with queue_lock:
                    current_processing = processing_chats.get(chat_id)
                    if current_processing and current_processing["response_dict"] is not response_dict:
                        _log.warning(f"⚠️ 聊天 {chat_id} 的任务在数据移动到设备后被新任务替换，退出生成")
                        return None, False, True
                    elif current_processing and current_processing["response_dict"] is response_dict:
                        # 如果任务仍然是最新的，但interrupt_event被设置了，可能是误设置，清除它
                        _log.warning(f"⚠️ 聊天 {chat_id} 的任务在数据移动到设备后检测到interrupt_event被设置，但任务仍是最新的，清除中断信号")
                        interrupt_event.clear()
        
        # 打印完整的输入（包括特殊token）
        input_ids_text = processor.tokenizer.batch_decode(
            inputs['input_ids'],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )
        if log_full_io:
            _log.info("=" * 80)
            _log.info("🔤 模型完整输入（包括特殊token）：")
            _log.info(input_ids_text[0])
            _log.info("=" * 80)

        # 在打印输入后检查中断
        if interrupt_event and interrupt_event.is_set():
            if chat_id and response_dict:
                with queue_lock:
                    current_processing = processing_chats.get(chat_id)
                    if current_processing and current_processing["response_dict"] is not response_dict:
                        _log.warning(f"⚠️ 聊天 {chat_id} 的任务在打印输入后被新任务替换，退出生成")
                        return None, False, True
                    elif current_processing and current_processing["response_dict"] is response_dict:
                        # 如果任务仍然是最新的，但interrupt_event被设置了，可能是误设置，清除它
                        _log.warning(f"⚠️ 聊天 {chat_id} 的任务在打印输入后检测到interrupt_event被设置，但任务仍是最新的，清除中断信号")
                        interrupt_event.clear()
        
        # 在生成前检查是否已经被中断，并验证任务是否仍然是最新的
        if interrupt_event and interrupt_event.is_set():
            if chat_id and response_dict:
                # 检查processing_chats，确认当前任务是否仍然是最新任务
                with queue_lock:
                    current_processing = processing_chats.get(chat_id)
                    if current_processing and current_processing["response_dict"] is response_dict:
                        # 当前任务仍然是最新任务，但interrupt_event被设置了
                        # 这可能是误设置（比如不同聊天之间切换时），清除它并继续
                        _log.warning(f"⚠️ 聊天 {chat_id} 的任务在生成开始前检测到interrupt_event被设置，但任务仍是最新的，清除中断信号并继续")
                        interrupt_event.clear()
                    else:
                        # 当前任务已经不是最新任务，应该退出
                        _log.warning(f"⚠️ 聊天 {chat_id} 的任务在生成开始前已被新任务替换，退出生成")
                        return None, False, True
            else:
                # 如果没有chat_id和response_dict，无法验证，为了安全直接退出
                _log.warning("⚠️ 生成任务在开始前检测到interrupt_event被设置，但无法验证任务状态，退出")
                return None, False, True
        
        _log.info("开始生成回复...")
        
        # 准备LogitsProcessor（使用transformers标准实现）
        logits_processor = LogitsProcessorList()
        if repetition_penalty != 1.0:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        
        # 准备StoppingCriteria（支持中断）
        stopping_criteria = StoppingCriteriaList()
        if interrupt_event:
            stopping_criteria.append(InterruptStoppingCriteria(interrupt_event))
        
        # 使用完全复刻transformers官方源码的自定义generate方法
        # 这个实现完全按照transformers的逻辑，方便后续魔改
        # 使用模型锁确保同一时刻只有一个线程使用模型（串行推理）
        # 注意：获取锁后，如果已被中断，在生成循环中会检测到
        with model_lock:
            # 在获取锁之后、开始生成之前，再次检查是否已被中断
            # 这很重要，因为可能在等待获取锁期间有新消息到达
            # 关键修复：检查processing_chats，确认当前任务是否仍然是最新任务
            # 如果interrupt_event被设置，但当前任务仍然是最新的，说明是误设置，应该清除
            if interrupt_event and interrupt_event.is_set():
                if chat_id and response_dict:
                    # 检查processing_chats，确认当前任务是否仍然是最新任务
                    with queue_lock:
                        current_processing = processing_chats.get(chat_id)
                        if current_processing and current_processing["response_dict"] is response_dict:
                            # 当前任务仍然是最新任务，但interrupt_event被设置了
                            # 这可能是误设置（比如不同聊天之间切换时），清除它并继续
                            _log.warning(f"⚠️ 聊天 {chat_id} 的任务在获取模型锁后检测到interrupt_event被设置，但任务仍是最新的，清除中断信号并继续")
                            interrupt_event.clear()
                        else:
                            # 当前任务已经不是最新任务，应该退出
                            _log.warning(f"⚠️ 聊天 {chat_id} 的任务在获取模型锁后已被新任务替换，退出生成")
                            return None, False, True
                else:
                    # 如果没有chat_id和response_dict，无法验证，为了安全直接退出
                    _log.warning("⚠️ 生成任务在获取模型锁后检测到interrupt_event被设置，但无法验证任务状态，退出")
                    return None, False, True
            with torch.no_grad():
                try:
                    result = custom_generate(
                        model=model,
                        inputs=inputs,
                        max_new_tokens=max_new_tokens,
                        stopping_criteria=stopping_criteria,
                        logits_processor=logits_processor,
                        temperature=temperature,
                        top_k=top_k if top_k and top_k > 0 else None,
                        top_p=top_p if top_p and top_p < 1.0 else None,
                        do_sample=do_sample,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        interrupt_event=interrupt_event,
                        early_stop_on_tool_call=False,
                    )
                    # 处理返回值：可能是 (input_ids, memory_injection_positions) 或 input_ids
                    if isinstance(result, tuple):
                        generated_ids, memory_injection_positions = result
                    else:
                        generated_ids = result
                        memory_injection_positions = []
                except Exception as e:
                    # 检查是否因为中断而停止
                    if interrupt_event and interrupt_event.is_set():
                        _log.warning("⚠️ 生成过程被中断")
                        return None, False, True
                    raise e
            
            # 检查是否被中断
            if interrupt_event and interrupt_event.is_set():
                _log.warning("⚠️ 生成过程被中断")
                return None, False, True
        
        # 生成完成后立即检查是否被中断（在释放锁之前检查）
        # 这很重要，因为可能在生成过程中有新消息到达
        if interrupt_event and interrupt_event.is_set():
            _log.warning("⚠️ 生成过程在完成后被中断，丢弃结果")
            return None, False, True
        
        # 在释放模型锁之前，再次检查是否仍然是最新的任务
        # 通过检查interrupt_event来判断（如果被中断，说明有更新的任务）
        # 注意：这里不能使用processing_chats检查，因为还在持有model_lock
        if interrupt_event and interrupt_event.is_set():
            _log.warning("⚠️ 生成过程在释放锁前被中断，丢弃结果")
            return None, False, True
        
        # 提取生成的token（去掉输入部分）
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        # 解码生成结果（包含特殊token的版本）
        output_text_with_special = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,  # 不跳过特殊token
            clean_up_tokenization_spaces=False
        )
        
        # 打印完整的输出（包括特殊token）
        # 注意：记忆向量插入位置现在通过<|memory_pad|> token原生显示，无需额外标注
        if log_full_io:
            _log.info("=" * 80)
            _log.info("🔤 模型完整输出（包括特殊token）：")
            _log.info(output_text_with_special[0])
            _log.info("=" * 80)
        
        # 解码生成结果（正常版本，跳过特殊token）
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        full_output = output_text[0] if output_text else ""
        
        # 提取thinking模型的正式回复与动作指令（</think>标签后的内容）
        reply, should_reply, action_cmds = extract_final_reply(full_output)
        
        if should_reply:
            _log.info(f"✅ 生成完成，完整输出长度: {len(full_output)}, 正式回复长度: {len(reply)}")
        else:
            _log.info(f"✅ 生成完成，模型判断不需要回复")
        
        # 附带返回解析到的动作指令
        try:
            if action_cmds:
                _log.info(f"🎯 解析到动作指令 {len(action_cmds)} 条")
        except Exception:
            pass
        return reply, should_reply, False, (action_cmds or [])
        
    except Exception as e:
        _log.error(f"生成回复失败: {e}", exc_info=True)
        raise
    finally:
        # 推理后尽量回收显存
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        # 记录时延
        try:
            import time as __t
            _metrics_add_latency((__t.time() - _t0) * 1000.0)
        except Exception:
            pass


# 健康检查与指标路由已迁移至 routes/health.py

# 指标路由已迁移至 routes/health.py


def trigger_training():
    """
    手动触发记忆训练（用于调试）
    
    Returns:
        JSON响应，包含训练状态和详细信息
    """
    global training_scheduler
    
    _log.info("收到手动训练触发请求")
    
    if training_scheduler is None:
        # 尝试重新初始化训练调度器（多线程环境下可能丢失）
        try:
            _log.warning("检测到training_scheduler为None，尝试重新初始化...")
            from memory_training_scheduler import MemoryTrainingScheduler

            # 保存脚本路径和参数，用于训练完成后重启
            script_path = os.path.abspath(__file__)
            script_args = sys.argv[1:]  # 保存命令行参数（除了脚本名）

            _log.info("正在重新创建 MemoryTrainingScheduler 实例...")
            training_scheduler = MemoryTrainingScheduler(config, script_path, script_args)
            _log.info("✅ 重新初始化训练调度器成功")

            # 尝试启动（如果没有启动的话）
            if not hasattr(training_scheduler, 'scheduler') or not training_scheduler.scheduler.running:
                training_scheduler.start()
                _log.info("✅ 重新启动训练调度器成功")

        except Exception as init_error:
            _log.error(f"❌ 重新初始化训练调度器失败: {init_error}")
            return jsonify({
                "success": False,
                "error": "训练调度器重新初始化失败",
                "message": f"无法重新初始化训练调度器: {str(init_error)}"
            }), 500
    
    try:
        _log.info("=" * 60)
        _log.info("手动触发训练任务")
        _log.info("=" * 60)
        
        # 检查是否正在训练
        if training_scheduler.is_running:
            return jsonify({
                "success": False,
                "error": "训练任务正在运行",
                "message": "请等待当前训练任务完成"
            }), 409
        
        # 在后台线程中执行训练（避免阻塞HTTP请求）
        import threading
        training_result = {"status": "running", "details": {}}
        
        def run_training_async():
            global group_chat_histories, private_chat_histories, CHAT_HISTORY_STORAGE_DIR, training_scheduler, is_training, model, processor

            # 在使用全局变量前先声明（避免SyntaxError）
            global group_chat_histories, private_chat_histories
            
            # 确保torch在函数作用域内可用（已在文件顶部导入）
            import torch
            
            try:
                # 设置训练模式标志，阻止API请求和模型生成
                with training_lock:
                    is_training = True
                _log.info("🔒 已进入训练模式，API接收信息和模型生成回复功能已停止")
                
                training_result["status"] = "running"
                training_result["details"]["started_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # 步骤0: 强制保存内存中的聊天记录到JSON文件（在主线程中，可以访问全局变量）
                _log.info("=" * 60)
                _log.info("步骤0: 强制保存内存中的聊天记录（在训练线程中）")
                _log.info("=" * 60)
                
                # 统计当前内存中的聊天记录（直接使用全局变量）
                group_count = len(group_chat_histories)
                private_count = len(private_chat_histories)
                total_group_messages = sum(len(history) for history in group_chat_histories.values())
                total_private_messages = sum(len(history) for history in private_chat_histories.values())
                
                _log.info(f"📊 内存中的聊天记录统计:")
                _log.info(f"   群聊数量: {group_count}")
                _log.info(f"   私聊数量: {private_count}")
                _log.info(f"   群聊消息总数: {total_group_messages}")
                _log.info(f"   私聊消息总数: {total_private_messages}")
                
                # 详细输出每个聊天的消息数
                for chat_id, history in group_chat_histories.items():
                    _log.info(f"   群聊 {chat_id}: {len(history)} 条消息")
                for chat_id, history in private_chat_histories.items():
                    _log.info(f"   私聊 {chat_id}: {len(history)} 条消息")
                
                # 保存聊天记录（同步阻塞，等待每个保存完成）
                saved_count = 0
                for chat_id, history in group_chat_histories.items():
                    if history:
                        try:
                            _log.info(f"正在保存群聊 {chat_id} 的 {len(history)} 条消息...")
                            save_chat_history_to_storage("group", chat_id, history)  # 同步阻塞，等待完成
                            saved_count += len(history)
                            _log.info(f"✅ 群聊 {chat_id} 保存完成")
                        except Exception as e:
                            _log.error(f"保存群聊 {chat_id} 失败: {e}", exc_info=True)
                
                for chat_id, history in private_chat_histories.items():
                    if history:
                        try:
                            _log.info(f"正在保存私聊 {chat_id} 的 {len(history)} 条消息...")
                            save_chat_history_to_storage("private", chat_id, history)  # 同步阻塞，等待完成
                            saved_count += len(history)
                            _log.info(f"✅ 私聊 {chat_id} 保存完成")
                        except Exception as e:
                            _log.error(f"保存私聊 {chat_id} 失败: {e}", exc_info=True)
                
                _log.info(f"✅ 步骤0完成：共保存 {saved_count} 条内存中的聊天记录到存储（所有保存操作已完成）")
                _log.info("=" * 60)
                
                # 步骤0.5: 卸载主模型以释放显存（训练时会加载新的模型实例）
                _log.info("=" * 60)
                _log.info("步骤0.5: 彻底清理所有模型和显存")
                _log.info("=" * 60)

                # 清理主模型
                if model is not None:
                    _log.info("正在卸载主模型...")
                    try:
                        model = model.cpu()
                    except:
                        pass
                    del model
                    model = None

                if processor is not None:
                    del processor
                    processor = None

                # 清理全局变量（已在函数开始时声明global）
                group_chat_histories.clear()
                private_chat_histories.clear()

                # 多重垃圾回收和显存清理
                import gc
                for _ in range(3):  # 多次GC确保清理彻底
                    gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()

                _log.info("✅ 所有模型和显存已彻底清理")
                _log.info("=" * 60)
                
                # 确保训练服务已初始化
                if training_scheduler is None:
                    _log.error("❌ training_scheduler 为 None，无法继续训练")
                    raise RuntimeError("training_scheduler 未初始化")
                if not training_scheduler.training_service:
                    training_scheduler._setup_training_service()
                
                # 执行训练（训练服务会从JSON文件加载聊天记录）
                model_path = training_scheduler.training_service.run_training(skip_memory_dump=True)
                
                # 训练完成后，重新加载主模型（无论是否提取到记忆条目）
                # 因为训练开始时卸载了主模型，所以必须重新加载
                memory_config = config.get("memory", {}).get("training", {})
                auto_restart = memory_config.get("auto_restart_after_training", False)
                restart_mode = memory_config.get("restart_mode", "reload_model")
                
                if model_path:
                    training_result["status"] = "completed"
                    training_result["details"]["model_path"] = model_path
                    training_result["details"]["completed_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 训练完成后，总是重新启动整个服务器进程，避免占据端口
                    _log.info("训练完成，重新启动服务器进程...")
                    training_result["details"]["restart_mode"] = "restart_server"
                    training_result["details"]["restart_scheduled"] = True
                    training_scheduler.restart_server()  # 重新启动整个进程
                else:
                    training_result["status"] = "skipped"
                    training_result["details"]["reason"] = "没有数据或没有提取到记忆条目"
                    training_result["details"]["completed_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 训练跳过后，也重新启动服务器进程
                    _log.info("训练跳过，重新启动服务器进程...")
                    training_result["details"]["restart_mode"] = "restart_server"
                    training_result["details"]["restart_scheduled"] = True
                    training_scheduler.restart_server()  # 重新启动整个进程
                    
            except Exception as e:
                training_result["status"] = "failed"
                training_result["error"] = str(e)
                training_result["details"]["failed_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                _log.error(f"手动触发训练失败: {e}", exc_info=True)
                
                # 训练失败时，执行彻底清理并退出进程
                _log.info("=" * 60)
                _log.info("训练失败，执行彻底显存清理并退出进程...")
                _log.info("=" * 60)

                import sys
                
                # 等待一小段时间，确保TrainingModelContext的__exit__完全执行
                import time
                time.sleep(2)
                
                # 强制清理所有GPU的显存
                import gc
                import torch
                
                # 多次垃圾回收
                for _ in range(5):
                    gc.collect()
                
                # 清理所有GPU的显存
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                    
                    # 再次清理
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                    
                    _log.info(f"✅ 已清理所有 {torch.cuda.device_count()} 张GPU的显存")
                
                # 训练失败后直接退出进程，避免占据显存和端口
                _log.info("训练失败，进程即将退出...")
                # 在退出前确保训练模式已解除
                with training_lock:
                    is_training = False
                sys.exit(1)
            finally:
                # 训练完成或失败后，解除训练模式（除非进程已退出）
                try:
                    with training_lock:
                        is_training = False
                    _log.info("🔓 已退出训练模式，API接收信息和模型生成回复功能已恢复")
                except Exception:
                    # 如果进程正在退出，忽略这个错误
                    pass
        
        # 启动训练线程
        training_thread = threading.Thread(target=run_training_async, daemon=True)
        training_thread.start()
        
        # 等待一小段时间，检查训练是否立即失败（例如没有数据）
        import time
        time.sleep(0.5)
        
        if training_result["status"] in ["skipped", "failed"]:
            return jsonify({
                "success": training_result["status"] == "skipped",
                "status": training_result["status"],
                "details": training_result.get("details", {}),
                "error": training_result.get("error")
            }), 200 if training_result["status"] == "skipped" else 500
        else:
            # 训练已开始，返回启动信息
            return jsonify({
                "success": True,
                "status": "started",
                "message": "训练任务已在后台启动",
                "details": training_result.get("details", {})
            }), 200
            
    except Exception as e:
        _log.error(f"触发训练时出错: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "触发训练失败"
        }), 500


def get_training_status():
    """
    获取训练状态
    
    Returns:
        JSON响应，包含当前训练状态
    """
    global training_scheduler
    
    if training_scheduler is None:
        return jsonify({
            "training_enabled": False,
            "message": "训练调度器未初始化"
        }), 200
    
    memory_config = config.get("memory", {}).get("training", {})
    training_enabled = memory_config.get("enabled", False)
    schedule = memory_config.get("schedule", "3-7")
    
    return jsonify({
        "training_enabled": training_enabled,
        "schedule": schedule,
        "is_running": training_scheduler.is_running,
        "scheduler_running": training_scheduler.scheduler.running if hasattr(training_scheduler, 'scheduler') else False
    }), 200


def debug_training_scheduler():
    """
    调试训练调度器状态（用于排查问题）
    
    Returns:
        JSON响应，包含训练调度器的详细状态信息
    """
    global training_scheduler
    
    # 检查模块级别的 training_scheduler
    import api_server_qwen3vl as api_module
    module_training_scheduler = api_module.training_scheduler
    
    debug_info = {
        "global_training_scheduler": {
            "is_none": training_scheduler is None,
            "type": str(type(training_scheduler)),
            "id": id(training_scheduler) if training_scheduler is not None else None,
            "value": str(training_scheduler) if training_scheduler is not None else None
        },
        "module_training_scheduler": {
            "is_none": module_training_scheduler is None,
            "type": str(type(module_training_scheduler)),
            "id": id(module_training_scheduler) if module_training_scheduler is not None else None,
            "value": str(module_training_scheduler) if module_training_scheduler is not None else None
        },
        "are_same_object": training_scheduler is module_training_scheduler,
        "config": {
            "memory_training_enabled": config.get("memory", {}).get("training", {}).get("enabled", False),
            "auto_restart": config.get("memory", {}).get("training", {}).get("auto_restart_after_training", False),
            "restart_mode": config.get("memory", {}).get("training", {}).get("restart_mode", "reload_model")
        }
    }
    
    if training_scheduler is not None:
        try:
            debug_info["training_scheduler_details"] = {
                "is_running": training_scheduler.is_running,
                "has_training_service": training_scheduler.training_service is not None,
                "scheduler_running": training_scheduler.scheduler.running if hasattr(training_scheduler, 'scheduler') else False
            }
        except Exception as e:
            debug_info["training_scheduler_details"] = {
                "error": str(e)
            }
    
    return jsonify(debug_info), 200


def save_chat_history_manually():
    """
    手动保存当前内存中的聊天记录到存储（用于调试）
    
    Returns:
        JSON响应，包含保存状态和统计信息
    """
    global training_scheduler, group_chat_histories, private_chat_histories
    
    try:
        _log.info("=" * 60)
        _log.info("手动触发保存聊天记录")
        _log.info("=" * 60)
        
        # 统计当前内存中的聊天记录
        group_count = len(group_chat_histories)
        private_count = len(private_chat_histories)
        total_group_messages = sum(len(history) for history in group_chat_histories.values())
        total_private_messages = sum(len(history) for history in private_chat_histories.values())
        
        _log.info(f"📊 当前内存中的聊天记录统计:")
        _log.info(f"   群聊数量: {group_count}")
        _log.info(f"   私聊数量: {private_count}")
        _log.info(f"   群聊消息总数: {total_group_messages}")
        _log.info(f"   私聊消息总数: {total_private_messages}")
        
        # 详细输出每个聊天的消息数
        for chat_id, history in group_chat_histories.items():
            _log.info(f"   群聊 {chat_id}: {len(history)} 条消息")
        for chat_id, history in private_chat_histories.items():
            _log.info(f"   私聊 {chat_id}: {len(history)} 条消息")
        
        # 直接使用api_server的保存函数（可以访问运行时的全局变量）
        # 不要使用training_service的保存函数，因为它通过模块导入获取不到运行时的全局变量
        saved_count = 0
        for chat_id, history in group_chat_histories.items():
            if history:
                try:
                    save_chat_history_to_storage("group", chat_id, history)
                    saved_count += len(history)
                    _log.info(f"✅ 保存群聊 {chat_id} 的 {len(history)} 条消息到 {CHAT_HISTORY_STORAGE_DIR}")
                except Exception as e:
                    _log.error(f"保存群聊 {chat_id} 失败: {e}", exc_info=True)
        
        for chat_id, history in private_chat_histories.items():
            if history:
                try:
                    save_chat_history_to_storage("private", chat_id, history)
                    saved_count += len(history)
                    _log.info(f"✅ 保存私聊 {chat_id} 的 {len(history)} 条消息到 {CHAT_HISTORY_STORAGE_DIR}")
                except Exception as e:
                    _log.error(f"保存私聊 {chat_id} 失败: {e}", exc_info=True)
        
        return jsonify({
            "success": True,
            "message": "聊天记录已保存",
            "storage_dir": CHAT_HISTORY_STORAGE_DIR,
            "stats": {
                "group_chats": group_count,
                "private_chats": private_count,
                "total_group_messages": total_group_messages,
                "total_private_messages": total_private_messages,
                "saved_messages": saved_count
            }
        }), 200
            
    except Exception as e:
        _log.error(f"保存聊天记录失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "保存聊天记录失败"
        }), 500


def upload_image():
    """
    接收客户端上传的图片（base64）并保存到本地，返回可访问的URL
    """
    try:
        payload = request.get_json(force=True) or {}
        image_data = payload.get("data")
        image_format = str(payload.get("format", "jpeg")).lower().strip()

        if not image_data:
            return jsonify({"status": "error", "message": "缺少图片数据"}), 400

        # 允许的格式映射
        format_map = {
            "jpg": "jpg",
            "jpeg": "jpg",
            "png": "png",
            "webp": "webp",
            "gif": "gif",
        }
        file_ext = format_map.get(image_format, "jpg")

        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except Exception as decode_err:
            _log.warning(f"图片Base64解码失败: {decode_err}")
            return jsonify({"status": "error", "message": "图片数据无效"}), 400

        # 文件名使用时间戳+uuid，避免冲突
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        filename = f"{timestamp}_{uuid4().hex}.{file_ext}"
        file_path = os.path.join(IMAGE_UPLOAD_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(image_bytes)

        file_url = url_for('serve_uploaded_image', filename=filename, _external=True)
        _log.info(f"✅ 图片已保存: {file_path} -> {file_url}")

        return jsonify({
            "status": "success",
            "url": file_url,
            "filename": filename
        }), 200

    except Exception as e:
        _log.error(f"图片上传失败: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


def handle_group_message():
    """
    处理群消息（支持多模态：文字+图片）
    将消息放入队列，由工作线程处理
    """
    global worker_thread_started
    
    try:
        data = request.json
        
        # 验证请求数据
        group_id = str(data.get("group_id", ""))
        content = data.get("content", "")
        
        # 从content中提取CQ图片码中的URL（用于验证）
        cleaned_content, image_urls = extract_cq_image_urls(content)
        
        if not group_id or (not cleaned_content and not image_urls):
            return jsonify({"status": "error", "message": "缺少必要参数"}), 400
        
        # 确保工作线程已启动
        if not worker_thread_started:
            with queue_lock:
                if not worker_thread_started:
                    worker_thread = threading.Thread(target=message_queue_worker, daemon=True)
                    worker_thread.start()
                    worker_thread_started = True
                    _log.info("✅ 消息队列工作线程已启动")
        
        # 创建响应字典（用于线程间通信）
        response_dict = {}
        
        # 创建消息任务
        task = MessageTask(
            chat_type="group",
            chat_id=group_id,
            data=data,
            response_dict=response_dict
        )
        
        # 将任务放入队列
        message_queue.put(task)
        
        # 等待处理完成（最多等待120秒，与客户端超时时间一致）
        # 注意：如果多条消息排队，可能需要更长时间
        timeout = 120
        start_time = time.time()
        while time.time() - start_time < timeout:
            if "status" in response_dict:
                # 处理完成
                status_code = response_dict.pop("status_code", 200)
                return jsonify(response_dict), status_code
            time.sleep(0.1)  # 等待100ms
        
        # 超时
        return jsonify({
            "status": "error",
            "message": "处理超时"
        }), 500
            
    except Exception as e:
        _log.error(f"处理群消息出错: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


def handle_private_message():
    """
    处理私聊消息（支持多模态：文字+图片）
    将消息放入队列，由工作线程处理
    """
    global worker_thread_started
    
    try:
        data = request.json
        
        # 验证请求数据
        user_id = str(data.get("user_id", ""))
        content = data.get("content", "")
        
        # 从content中提取CQ图片码中的URL（用于验证）
        cleaned_content, image_urls = extract_cq_image_urls(content)
        
        if not user_id or (not cleaned_content and not image_urls):
            return jsonify({"status": "error", "message": "缺少必要参数"}), 400
        
        # 确保工作线程已启动
        if not worker_thread_started:
            with queue_lock:
                if not worker_thread_started:
                    worker_thread = threading.Thread(target=message_queue_worker, daemon=True)
                    worker_thread.start()
                    worker_thread_started = True
                    _log.info("✅ 消息队列工作线程已启动")
        
        # 创建响应字典（用于线程间通信）
        response_dict = {}
        
        # 创建消息任务
        task = MessageTask(
            chat_type="private",
            chat_id=user_id,
            data=data,
            response_dict=response_dict
        )
        
        # 将任务放入队列
        message_queue.put(task)
        
        # 等待处理完成（最多等待120秒，与客户端超时时间一致）
        # 注意：如果多条消息排队，可能需要更长时间
        timeout = 120
        start_time = time.time()
        while time.time() - start_time < timeout:
            if "status" in response_dict:
                # 处理完成
                status_code = response_dict.pop("status_code", 200)
                return jsonify(response_dict), status_code
            time.sleep(0.1)  # 等待100ms
        
        # 超时
        return jsonify({
            "status": "error",
            "message": "处理超时"
        }), 500
            
    except Exception as e:
        _log.error(f"处理私聊消息出错: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("本文件不再作为运行入口。请使用统一入口：python server/app.py")

