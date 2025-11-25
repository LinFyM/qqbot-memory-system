# -*- coding: utf-8 -*-
"""
聊天历史管理模块
处理群聊和私聊的历史记录维护、保存、去重等
"""
import os
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

from utils.common import resolve_path

_log = logging.getLogger(__name__)

# 全局历史记录字典
group_chat_histories: Dict[str, list] = {}
private_chat_histories: Dict[str, list] = {}

# 线程锁，用于保护聊天记录的并发访问
chat_history_lock = threading.Lock()


def get_chat_history(chat_type: str, chat_id: str) -> List[Dict[str, Any]]:
    """
    获取聊天历史
    
    Args:
        chat_type: "group" 或 "private"
        chat_id: 群ID或用户ID
    
    Returns:
        历史消息列表
    """
    if chat_type == "group":
        return group_chat_histories.get(chat_id, [])
    elif chat_type == "private":
        return private_chat_histories.get(chat_id, [])
    return []


def set_chat_history(chat_type: str, chat_id: str, history: List[Dict[str, Any]]):
    """
    设置聊天历史
    
    Args:
        chat_type: "group" 或 "private"
        chat_id: 群ID或用户ID
        history: 历史消息列表
    """
    if chat_type == "group":
        group_chat_histories[chat_id] = history
    elif chat_type == "private":
        private_chat_histories[chat_id] = history


def generate_message_key(message: Dict[str, Any]) -> str:
    """
    生成消息的唯一键，用于去重
    
    Args:
        message: 消息字典
    
    Returns:
        消息键
    """
    role = message.get("role", "")
    content = message.get("content", [])
    
    # 提取文本内容
    text_parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text_parts.append(item.get("text", ""))
    
    text = "".join(text_parts)[:100]  # 只取前100个字符
    return f"{role}:{text}"


def maintain_chat_history(
    chat_type: str,
    chat_id: str,
    history: List[Dict[str, Any]],
    max_length: int = 200
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    维护聊天历史，进行去重和长度控制
    
    Args:
        chat_type: "group" 或 "private"
        chat_id: 群ID或用户ID
        history: 历史消息列表
        max_length: 最大历史长度
    
    Returns:
        维护后的历史消息列表
    """
    if not history:
        return [], []
    
    # 去重：使用消息键去重
    seen_keys = set()
    unique_history = []
    
    for message in history:
        key = generate_message_key(message)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_history.append(message)
    
    # 长度控制：保留最新的N条消息
    removed_messages: List[Dict[str, Any]] = []
    if len(unique_history) > max_length:
        removed_messages = unique_history[:-max_length]
        _log.info(
            f"📊 历史记录超过限制（{len(unique_history)} > {max_length}），"
            f"截断并移除最早 {len(removed_messages)} 条（{chat_type} {chat_id}）"
        )
        unique_history = unique_history[-max_length:]
    
    return unique_history, removed_messages


def save_chat_history_to_storage(config: Dict[str, Any], chat_type: str, chat_id: str, messages: List[Dict[str, Any]]):
    """
    保存聊天历史到存储
    
    Args:
        config: 配置字典
        chat_type: "group" 或 "private"
        chat_id: 群ID或用户ID
        messages: 消息列表
    """
    try:
        # 获取存储目录
        memory_config = config.get("memory", {}).get("training", {})
        storage_dir = memory_config.get("chat_history_storage_dir", "./models/chat_history_storage")
        storage_path = resolve_path(storage_dir)
        
        # 确保目录存在
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # 构建文件路径
        filename = f"{chat_type}_{chat_id}.json"
        file_path = storage_path / filename
        
        # 如果文件已存在，先加载历史消息
        existing_messages: List[Dict[str, Any]] = []
        existing_keys = set()
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_messages = existing_data.get("messages", [])
                    existing_keys = {generate_message_key(msg) for msg in existing_messages}
            except Exception as load_err:
                _log.warning(f"⚠️ 读取历史文件失败（{file_path}），将重新创建: {load_err}")
                existing_messages = []
                existing_keys = set()
        
        appended = 0
        merged_messages = list(existing_messages)
        for message in messages:
            key = generate_message_key(message)
            if key not in existing_keys:
                merged_messages.append(message)
                existing_keys.add(key)
                appended += 1
        
        if appended == 0:
            _log.info(f"ℹ️ 聊天 {chat_type} {chat_id} 无新增消息需要保存")
            return
        
        data = {
            "chat_type": chat_type,
            "chat_id": chat_id,
            "messages": merged_messages,
            "saved_at": datetime.now().isoformat(),
            "message_count": len(merged_messages)
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        _log.info(f"💾 已保存聊天历史：{chat_type} {chat_id}，追加{appended}条，累计{len(merged_messages)}条 → {file_path}")
        
    except Exception as e:
        _log.error(f"❌ 保存聊天历史失败（{chat_type} {chat_id}）: {e}", exc_info=True)


def load_chat_history_from_storage(config: Dict[str, Any], chat_type: str, chat_id: str) -> List[Dict[str, Any]]:
    """
    从存储加载聊天历史
    
    Args:
        config: 配置字典
        chat_type: "group" 或 "private"
        chat_id: 群ID或用户ID
    
    Returns:
        消息列表
    """
    try:
        # 获取存储目录
        memory_config = config.get("memory", {}).get("training", {})
        storage_dir = memory_config.get("chat_history_storage_dir", "./models/chat_history_storage")
        storage_path = resolve_path(storage_dir)
        
        # 构建文件路径
        filename = f"{chat_type}_{chat_id}.json"
        file_path = storage_path / filename
        
        if not file_path.exists():
            return []
        
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = data.get("messages", [])
        _log.info(f"📂 已加载聊天历史：{chat_type} {chat_id}，共{len(messages)}条消息")
        
        return messages
        
    except Exception as e:
        _log.error(f"❌ 加载聊天历史失败（{chat_type} {chat_id}）: {e}", exc_info=True)
        return []


def get_all_chat_histories(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取所有聊天历史（用于训练）
    
    Args:
        config: 配置字典
    
    Returns:
        所有聊天历史的字典
    """
    try:
        # 获取存储目录
        memory_config = config.get("memory", {}).get("training", {})
        storage_dir = memory_config.get("chat_history_storage_dir", "./models/chat_history_storage")
        storage_path = resolve_path(storage_dir)
        
        if not storage_path.exists():
            return {}
        
        all_histories = {}
        
        # 遍历所有JSON文件
        for file_path in storage_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                chat_type = data.get("chat_type")
                chat_id = data.get("chat_id")
                messages = data.get("messages", [])
                
                key = f"{chat_type}_{chat_id}"
                all_histories[key] = {
                    "chat_type": chat_type,
                    "chat_id": chat_id,
                    "messages": messages
                }
                
            except Exception as e:
                _log.warning(f"⚠️ 加载历史文件失败 {file_path}: {e}")
                continue
        
        _log.info(f"📚 已加载所有聊天历史，共{len(all_histories)}个会话")
        return all_histories
        
    except Exception as e:
        _log.error(f"❌ 获取所有聊天历史失败: {e}", exc_info=True)
        return {}

