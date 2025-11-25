# -*- coding: utf-8 -*-
"""
QQ机器人客户端 - 完整版
在个人电脑上运行，通过HTTP API与服务器通信
支持群聊和私聊消息的接收、转发和回复
"""

from ncatbot.core import BotClient, GroupMessage, PrivateMessage  # pyright: ignore[reportMissingImports]
from ncatbot.core.event.message_segment import Image, Face, MessageArray  # pyright: ignore[reportMissingImports]
import requests
import time
import logging
import base64
import os
import tempfile
import re
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
from datetime import datetime
from itertools import count

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_log = logging.getLogger(__name__)

# ========== 配置区域 ==========
# 服务器API地址（通过SSH隧道访问）
SERVER_URL = "http://localhost:9999"  # ← 如果使用SSH隧道，保持localhost（默认端口9999）

# API密钥（如果使用server_secure.py，需要设置）
API_KEY = None  # ← 如果使用安全版，例如: "my-secret-key-123"

# API超时时间（秒）
# 注意：如果模型生成需要较长时间，可能需要增加这个值
# 考虑到多条消息排队的情况，设置为120秒（2分钟）
API_TIMEOUT = 600
# ===============================

# ========== 创建 BotClient ==========
bot = BotClient()

# ========== 消息顺序控制 ==========
_message_token_lock = threading.Lock()
_latest_message_token: Dict[str, int] = {}
_message_token_counter = count(start=1)

# 线程池用于异步处理消息，避免阻塞NcatBot的事件线程
_max_message_workers = max(20, (os.cpu_count() or 4) * 4)
_message_executor = ThreadPoolExecutor(max_workers=_max_message_workers)
_log.info(f"消息处理线程池已初始化，线程数: {_max_message_workers}")

# ========== 动作权限与限流配置 ==========
# 全局允许的动作类型（如需限制可修改）
ACTION_ALLOWED_TYPES = {"EMOJI_LIKE", "POKE"}
# 简单的会话级别节流：同一(chat_id, action_type)最短间隔（秒）
ACTION_RATE_LIMIT_SECONDS = 5
# 记录上次动作执行时间
_action_last_exec_time: Dict[str, float] = {}

def _action_key(scope: str, chat_id: str, action_type: str) -> str:
    return f"{scope}:{chat_id}:{action_type.upper()}"

def _is_action_allowed(action_type: str) -> bool:
    return str(action_type).upper() in ACTION_ALLOWED_TYPES

def _should_rate_limit(scope: str, chat_id: str, action_type: str) -> bool:
    key = _action_key(scope, chat_id, action_type)
    now = time.time()
    last = _action_last_exec_time.get(key, 0.0)
    if now - last < ACTION_RATE_LIMIT_SECONDS:
        return True
    _action_last_exec_time[key] = now
    return False

def _safe_try(callable_desc: str, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        _log.info(f"✅ 执行成功: {callable_desc}")
        return True
    except Exception as e:
        _log.warning(f"⚠️ 执行失败: {callable_desc} -> {e}")
        return False

def _execute_group_actions(group_id: str, actions: List[Dict[str, Any]]) -> None:
    """
    执行服务器返回的动作指令（群聊）
    支持类型：EMOJI_LIKE, POKE
    根据 LLM.md 的 API 规范实现
    """
    if not actions:
        return
    for act in actions:
        act_type = str(act.get("type", "")).upper()
        try:
            if not _is_action_allowed(act_type):
                _log.info(f"跳过未被允许的动作类型: {act_type}")
                continue
            if _should_rate_limit("group", group_id, act_type):
                _log.info(f"跳过限流中的动作: {act_type} (group {group_id})")
                continue
            if act_type == "EMOJI_LIKE":
                # 根据 LLM.md: set_msg_emoji_like(message_id, emoji_id, set=True)
                # 需要 message_id 和 emoji_id
                message_id = act.get("message_id")
                emoji_id = act.get("emoji_id") or act.get("emoji") or 128512  # 默认 👍
                
                if message_id:
                    # 如果有 message_id，尝试使用原生 API
                    set_emoji_like = getattr(bot.api, "set_msg_emoji_like_sync", None)
                    if callable(set_emoji_like):
                        ok = _safe_try(f"group emoji like (msg_id={message_id}, emoji_id={emoji_id})",
                            set_emoji_like, message_id, emoji_id, True)
                        if ok:
                            continue
                
                # 降级方案：发送表情消息（使用 Face 或 Unicode 表情）
                # 优先尝试使用 Face 消息段（QQ 表情）
                try:
                    # 将 emoji_id 转换为 QQ 表情 ID（简单映射，可根据需要扩展）
                    # 128512 是 👍 的 Unicode，对应 QQ 表情可能需要查询
                    # 这里先尝试直接使用，如果失败则使用 Unicode
                    face_id = int(emoji_id) if isinstance(emoji_id, (int, str)) and str(emoji_id).isdigit() else None
                    if face_id and 0 <= face_id <= 255:
                        # QQ 表情 ID 范围通常是 0-255
                        msg_array = MessageArray([Face(face_id)])
                        bot.api.post_group_msg_sync(group_id, rtf=msg_array)
                        _log.info(f"✅ 已发送 QQ 表情 (face_id={face_id})")
                        continue
                except Exception as e:
                    _log.debug(f"尝试发送 QQ 表情失败: {e}")
                
                # 最终降级：发送 Unicode 表情文本
                emoji_map = {
                    128512: "👍",  # thumbs up
                    128513: "😁",  # grinning
                    128514: "😂",  # joy
                    128515: "🤣",  # rofl
                    128516: "😃",  # smile
                    128517: "😄",  # smile
                    128518: "😅",  # sweat smile
                    128519: "😆",  # laughing
                    128520: "😉",  # wink
                    128521: "😊",  # blush
                    128522: "😋",  # yum
                    128523: "😌",  # relieved
                    128524: "😍",  # heart eyes
                    128525: "😎",  # sunglasses
                    128526: "😏",  # smirk
                    128527: "😐",  # neutral
                    128528: "😑",  # expressionless
                    128529: "😒",  # unamused
                    128530: "😓",  # sweat
                    128531: "😔",  # pensive
                    128532: "😕",  # confused
                    128533: "😖",  # confounded
                    128534: "😗",  # kissing
                    128535: "😘",  # kiss
                    128536: "😙",  # kiss
                    128537: "😚",  # kiss
                    128538: "😛",  # stuck out tongue
                    128539: "😜",  # stuck out tongue wink
                    128540: "😝",  # stuck out tongue closed eyes
                    128541: "😞",  # disappointed
                    128542: "😟",  # worried
                    128543: "😠",  # angry
                    128544: "😡",  # rage
                    128545: "😢",  # cry
                    128546: "😣",  # persevere
                    128547: "😤",  # triumph
                    128548: "😥",  # disappointed relieved
                    128549: "😦",  # frowning
                    128550: "😧",  # anguished
                    128551: "😨",  # fearful
                    128552: "😩",  # weary
                    128553: "😪",  # sleepy
                    128554: "😫",  # tired
                    128555: "😬",  # grimacing
                    128556: "😭",  # sob
                    128557: "😮",  # open mouth
                    128558: "😯",  # hushed
                    128559: "😰",  # cold sweat
                    128560: "😱",  # scream
                    128561: "😲",  # astonished
                    128562: "😳",  # flushed
                    128563: "😴",  # sleeping
                    128564: "😵",  # dizzy
                    128565: "😶",  # no mouth
                    128566: "😷",  # mask
                    128567: "😸",  # grin cat
                    128568: "😹",  # joy cat
                    128569: "😺",  # smile cat
                    128570: "😻",  # heart eyes cat
                    128571: "😼",  # smirk cat
                    128572: "😽",  # kissing cat
                    128573: "😾",  # pouting cat
                    128574: "😿",  # cry cat
                    128575: "🙀",  # scream cat
                    128576: "🙁",  # slightly frowning
                    128577: "🙂",  # slightly smiling
                    128578: "🙃",  # upside down
                    128579: "🙄",  # rolling eyes
                    128580: "🙅",  # no good
                    128581: "🙆",  # ok woman
                    128582: "🙇",  # bow
                    128583: "🙈",  # see no evil
                    128584: "🙉",  # hear no evil
                    128585: "🙊",  # speak no evil
                    128586: "🙋",  # raising hand
                    128587: "🙌",  # raised hands
                    128588: "🙍",  # person frowning
                    128589: "🙎",  # person pouting
                    128590: "🙏",  # pray
                }
                emoji_text = emoji_map.get(int(emoji_id) if isinstance(emoji_id, (int, str)) and str(emoji_id).isdigit() else 128512, "👍")
                bot.api.post_group_msg_sync(group_id, text=emoji_text)
                _log.info(f"✅ 已发送 Unicode 表情: {emoji_text}")
            elif act_type == "POKE":
                # 根据 LLM.md: group_poke(group_id, user_id)
                target_id = act.get("user_id") or act.get("target_id")
                if target_id:
                    group_poke = getattr(bot.api, "group_poke_sync", None)
                    if callable(group_poke):
                        ok = _safe_try(f"group poke (group_id={group_id}, user_id={target_id})",
                            group_poke, group_id, target_id)
                        if ok:
                            continue
                    # 降级：使用通用 send_poke
                    send_poke = getattr(bot.api, "send_poke_sync", None)
                    if callable(send_poke):
                        ok = _safe_try(f"group poke via send_poke (group_id={group_id}, user_id={target_id})",
                            send_poke, user_id=target_id, group_id=group_id)
                        if ok:
                            continue
                    _log.warning(f"群戳一戳失败，已尝试所有可用 API")
                else:
                    _log.warning("POKE 动作缺少 user_id 或 target_id")
            else:
                _log.debug(f"忽略未知动作类型: {act_type}")
        except Exception as e:
            _log.warning(f"执行动作失败: {act_type} -> {e}", exc_info=True)

def _execute_private_actions(user_id: str, actions: List[Dict[str, Any]]) -> None:
    """
    执行服务器返回的动作指令（私聊）
    支持类型：EMOJI_LIKE, POKE
    根据 LLM.md 的 API 规范实现
    """
    if not actions:
        return
    for act in actions:
        act_type = str(act.get("type", "")).upper()
        try:
            if not _is_action_allowed(act_type):
                _log.info(f"跳过未被允许的动作类型: {act_type}")
                continue
            if _should_rate_limit("private", user_id, act_type):
                _log.info(f"跳过限流中的动作: {act_type} (private {user_id})")
                continue
            if act_type == "EMOJI_LIKE":
                # 根据 LLM.md: set_msg_emoji_like(message_id, emoji_id, set=True)
                # 需要 message_id 和 emoji_id
                message_id = act.get("message_id")
                emoji_id = act.get("emoji_id") or act.get("emoji") or 128512  # 默认 👍
                
                if message_id:
                    # 如果有 message_id，尝试使用原生 API
                    set_emoji_like = getattr(bot.api, "set_msg_emoji_like_sync", None)
                    if callable(set_emoji_like):
                        ok = _safe_try(f"private emoji like (msg_id={message_id}, emoji_id={emoji_id})",
                            set_emoji_like, message_id, emoji_id, True)
                        if ok:
                            continue
                
                # 降级方案：发送表情消息（使用 Face 或 Unicode 表情）
                # 优先尝试使用 Face 消息段（QQ 表情）
                try:
                    face_id = int(emoji_id) if isinstance(emoji_id, (int, str)) and str(emoji_id).isdigit() else None
                    if face_id and 0 <= face_id <= 255:
                        msg_array = MessageArray([Face(face_id)])
                        bot.api.post_private_msg_sync(user_id, rtf=msg_array)
                        _log.info(f"✅ 已发送 QQ 表情 (face_id={face_id})")
                        continue
                except Exception as e:
                    _log.debug(f"尝试发送 QQ 表情失败: {e}")
                
                # 最终降级：发送 Unicode 表情文本
                emoji_map = {
                    128512: "👍", 128513: "😁", 128514: "😂", 128515: "🤣",
                    128516: "😃", 128517: "😄", 128518: "😅", 128519: "😆",
                    128520: "😉", 128521: "😊", 128522: "😋", 128523: "😌",
                    128524: "😍", 128525: "😎", 128526: "😏", 128527: "😐",
                    128528: "😑", 128529: "😒", 128530: "😓", 128531: "😔",
                    128532: "😕", 128533: "😖", 128534: "😗", 128535: "😘",
                    128536: "😙", 128537: "😚", 128538: "😛", 128539: "😜",
                    128540: "😝", 128541: "😞", 128542: "😟", 128543: "😠",
                    128544: "😡", 128545: "😢", 128546: "😣", 128547: "😤",
                    128548: "😥", 128549: "😦", 128550: "😧", 128551: "😨",
                    128552: "😩", 128553: "😪", 128554: "😫", 128555: "😬",
                    128556: "😭", 128557: "😮", 128558: "😯", 128559: "😰",
                    128560: "😱", 128561: "😲", 128562: "😳", 128563: "😴",
                    128564: "😵", 128565: "😶", 128566: "😷", 128567: "😸",
                    128568: "😹", 128569: "😺", 128570: "😻", 128571: "😼",
                    128572: "😽", 128573: "😾", 128574: "😿", 128575: "🙀",
                    128576: "🙁", 128577: "🙂", 128578: "🙃", 128579: "🙄",
                    128580: "🙅", 128581: "🙆", 128582: "🙇", 128583: "🙈",
                    128584: "🙉", 128585: "🙊", 128586: "🙋", 128587: "🙌",
                    128588: "🙍", 128589: "🙎", 128590: "🙏",
                }
                emoji_text = emoji_map.get(int(emoji_id) if isinstance(emoji_id, (int, str)) and str(emoji_id).isdigit() else 128512, "👍")
                bot.api.post_private_msg_sync(user_id, text=emoji_text)
                _log.info(f"✅ 已发送 Unicode 表情: {emoji_text}")
            elif act_type == "POKE":
                # 根据 LLM.md: friend_poke(user_id)
                friend_poke = getattr(bot.api, "friend_poke_sync", None)
                if callable(friend_poke):
                    ok = _safe_try(f"private poke (user_id={user_id})",
                        friend_poke, user_id)
                    if ok:
                        continue
                # 降级：使用通用 send_poke
                send_poke = getattr(bot.api, "send_poke_sync", None)
                if callable(send_poke):
                    ok = _safe_try(f"private poke via send_poke (user_id={user_id})",
                        send_poke, user_id=user_id)
                    if ok:
                        continue
                _log.warning(f"私聊戳一戳失败，已尝试所有可用 API")
            else:
                _log.debug(f"忽略未知动作类型: {act_type}")
        except Exception as e:
            _log.warning(f"执行动作失败: {act_type} -> {e}", exc_info=True)

def _mark_latest_message(chat_type: str, chat_id: str) -> int:
    """
    记录指定会话的最新消息标识，返回当前标识值
    """
    key = f"{chat_type}:{chat_id}"
    with _message_token_lock:
        token = next(_message_token_counter)
        _latest_message_token[key] = token
    return token


def _is_still_latest_message(chat_type: str, chat_id: str, token: int) -> bool:
    """
    判断指定标识是否仍是该会话最新消息
    """
    key = f"{chat_type}:{chat_id}"
    with _message_token_lock:
        return _latest_message_token.get(key) == token


# ========== 连接状态监控 ==========
# 记录最后收到心跳的时间
last_heartbeat_time = None
heartbeat_lock = threading.Lock()
# 心跳超时时间（秒），如果超过这个时间没收到心跳，认为连接可能有问题
HEARTBEAT_TIMEOUT = 600  # 2分钟
# 连接状态检查间隔（秒）
CONNECTION_CHECK_INTERVAL = 30  # 30秒检查一次


def _call_server_api_sync(endpoint: str, data: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    调用服务器API

    Args:
        endpoint: API端点，如 "/api/chat/group" 或 "/api/chat/private"
        data: 请求数据字典（JSON格式）
        files: 文件字典（multipart/form-data格式）

    Returns:
        API响应字典，如果失败返回None
    """
    try:
        url = f"{SERVER_URL}{endpoint}"

        # 构建请求头
        headers = {}
        if API_KEY:
            headers["X-API-Key"] = API_KEY

        # 发送请求
        if files:
            # 文件上传请求
            response = requests.post(
                url,
                data=data,  # 表单数据
                files=files,  # 文件数据
                headers=headers,
                timeout=API_TIMEOUT
            )
        else:
            # JSON请求
            headers["Content-Type"] = "application/json"
            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=API_TIMEOUT
            )
        
        # 检查响应状态
        if response.status_code == 401:
            _log.error("API密钥验证失败，请检查API_KEY配置")
            return None
        elif response.status_code == 429:
            _log.warning("请求过于频繁，服务器返回429")
            return None
        elif response.status_code == 503:
            _log.warning("服务器正在训练中，无法处理请求")
            return None
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.ConnectionError:
        _log.error(f"无法连接到服务器: {SERVER_URL}")
        _log.error("请检查：1. 服务器是否已启动  2. SSH隧道是否已建立  3. SERVER_URL配置是否正确")
        return None
    except requests.exceptions.Timeout:
        _log.error(f"服务器响应超时（{API_TIMEOUT}秒），请稍后再试")
        return None
    except requests.exceptions.RequestException as e:
        _log.error(f"API请求失败: {e}")
        return None
    except Exception as e:
        _log.error(f"处理API响应出错: {e}", exc_info=True)
        return None


async def call_server_api(endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    异步包装的服务器API调用，避免阻塞事件循环
    """
    return await asyncio.to_thread(_call_server_api_sync, endpoint, data)


def check_server_health() -> bool:
    """
    检查服务器健康状态
    
    Returns:
        True如果服务器正常，False否则
    """
    try:
        url = f"{SERVER_URL}/health"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("status") == "healthy"
    except Exception as e:
        _log.error(f"服务器健康检查失败: {e}")
        return False


# ========== 注册群消息回调函数 ==========
def _submit_message_task(description: str, func: callable, payload: Dict[str, Any]) -> None:
    """
    提交一个任务到线程池，并记录日志
    """
    _log.debug(f"提交任务: {description}")
    _message_executor.submit(func, payload)


def _process_group_message_task(payload: Dict[str, Any]) -> None:
    """在线程池中处理群消息"""
    _log.info(f"🚀 开始处理群消息任务: group_id={payload.get('group_id')}, user_id={payload.get('user_id')}")

    group_id = payload["group_id"]
    user_id = payload["user_id"]
    message_token = payload["message_token"]
    group_name = payload["group_name"]
    user_nickname = payload["user_nickname"]
    user_card = payload["user_card"]
    raw_content = payload["raw_message"]
    timestamp = payload["timestamp"]
    images: List[Image] = payload["images"]

    content = raw_content or ""
    _log.info(f"📝 原始消息内容: {content[:200] if content else '(空)'}")
    image_urls: List[str] = []
    # 预处理视频（从CQ中提取，若为本地路径则上传）
    video_urls: List[str] = []

    try:
        if images:
            content = re.sub(r'\[CQ:image[^\]]*\]', '', content).strip()
            _log.info(f"✅ 已移除CQ图片码，清理后的content: {content}")
        for img in images or []:
            temp_dir = None
            img_path = None
            try:
                temp_dir = tempfile.mkdtemp()
                img_path = img.download_sync(temp_dir)
                if img_path and os.path.exists(img_path):
                    with open(img_path, 'rb') as f:
                        img_bytes = f.read()
                        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    img_ext = os.path.splitext(img_path)[1].lower()
                    img_format = img_ext[1:] if img_ext else 'jpeg'
                    if img_format not in ['jpeg', 'jpg', 'png', 'gif', 'webp']:
                        img_format = 'jpeg'
                    upload_result = _call_server_api_sync("/api/upload/image", {
                        "data": img_base64,
                        "format": img_format
                    })
                    if upload_result and upload_result.get("status") == "success":
                        img_url = upload_result.get("url", "")
                        if img_url:
                            image_urls.append(img_url)
                            _log.info(f"✅ 图片已上传，获取URL: {img_url}")
                        else:
                            _log.warning("图片上传成功但未返回URL")
                    else:
                        error_msg = upload_result.get("message", "未知错误") if upload_result else "无响应"
                        _log.warning(f"图片上传失败: {error_msg}")
                else:
                    _log.warning(f"图片下载失败或路径不存在: {img_path}")
            except Exception as e:  # noqa: BLE001
                _log.warning(f"处理单个图片失败: {e}", exc_info=True)
            finally:
                try:
                    if img_path and os.path.exists(img_path):
                        os.remove(img_path)
                    if temp_dir and os.path.isdir(temp_dir):
                        os.rmdir(temp_dir)
                except Exception:
                    pass
    except Exception as e:  # noqa: BLE001
        _log.warning(f"提取图片信息失败: {e}", exc_info=True)
    
    # 提取并处理视频CQ
    _log.info(f"🔍 开始处理视频CQ码，content长度: {len(content) if content else 0}")
    _log.info(f"🔍 content内容: {content[:200] if content else '(空)'}")
    try:
        # 匹配 [CQ:video,...]，提取所有可能的字段
        video_matches = list(re.finditer(r'\[CQ:video([^\]]*)\]', content or "", flags=re.IGNORECASE))
        _log.info(f"🔍 找到 {len(video_matches)} 个视频CQ码")
        for m in video_matches:
            attrs = m.group(1) or ""
            # 按优先级提取：url > file > file_name
            src = None
            for field in ['url', 'file', 'file_name']:
                # 匹配 = 后面的内容，直到遇到逗号或右方括号（允许反斜杠，因为Windows路径需要）
                field_match = re.search(rf'{field}=([^,\]]+)', attrs)
                if field_match:
                    src = field_match.group(1)
                    break

            if not src:
                _log.warning(f"⚠️ 视频CQ码中没有找到url/file/file_name字段: {m.group(0)}")
                continue

            # 处理路径
            src_norm = src.replace("&amp;", "&")
            _log.info(f"🎥 处理视频路径: {src_norm} (原始字段值: {src})")

            # 检查是否是本地路径或HTTP URL
            if src_norm.lower().startswith(("http://", "https://")):
                # HTTP URL，直接使用
                video_urls.append(src_norm)
                _log.info(f"✅ 添加HTTP视频URL: {src_norm}")
            elif re.match(r'^[a-zA-Z]:\\', src_norm) or re.match(r'^\\\\', src_norm):
                # Windows本地路径，优先处理（在Linux客户端上无法用os.path.exists检查）
                try:
                    _log.info(f"🔍 检测到Windows本地路径，尝试直接上传: {src_norm}")
                    # 检查文件是否存在
                    if not os.path.exists(src_norm):
                        _log.error(f"❌ Windows视频文件不存在: {src_norm}")
                        _log.error(f"💡 请确保NapCat和客户端在同一台机器上运行，且文件路径正确")
                        continue
                    
                    _log.info(f"📥 开始读取视频文件: {src_norm}")
                    # 读取文件内容
                    with open(src_norm, "rb") as vf:
                        vbytes = vf.read()
                    
                    file_size = len(vbytes)
                    _log.info(f"✅ 视频文件读取成功，大小: {file_size} 字节")

                    # 使用multipart/form-data上传文件
                    files = {'file': (os.path.basename(src_norm), vbytes, 'video/mp4')}
                    _log.info(f"📤 开始上传视频到服务器...")
                    up = _call_server_api_sync("/api/upload/video", data=None, files=files)

                    if up and up.get("status") == "success" and up.get("url"):
                        video_urls.append(up["url"])
                        # 用上传后的直链替换原CQ片段
                        content = content.replace(m.group(0), f"[CQ:video,url={up['url']}]")
                        _log.info(f"✅ Windows本地视频已上传并替换为直链: {up['url']}")
                    else:
                        _log.warning(f"⚠️ Windows视频上传失败: {up}")
                except FileNotFoundError as fe:
                    _log.error(f"❌ Windows视频文件未找到: {src_norm}")
                    _log.error(f"💡 错误详情: {fe}")
                    _log.error(f"💡 请确保NapCat和客户端在同一台机器上运行")
                except PermissionError as pe:
                    _log.error(f"❌ Windows视频文件权限不足: {src_norm}")
                    _log.error(f"💡 错误详情: {pe}")
                except Exception as ve:
                    _log.error(f"❌ 处理Windows本地视频失败: {ve}", exc_info=True)
                    _log.error(f"💡 请确保NapCat和客户端在同一台机器上运行")
            elif os.path.exists(src_norm):
                # 本地文件存在，上传到服务器
                try:
                    _log.info(f"发现本地视频文件，开始上传: {src_norm}")
                    with open(src_norm, "rb") as vf:
                        vbytes = vf.read()

                    # 使用multipart/form-data上传文件
                    files = {'file': (os.path.basename(src_norm), vbytes, 'video/mp4')}
                    up = _call_server_api_sync("/api/upload/video", data=None, files=files)

                    if up and up.get("status") == "success" and up.get("url"):
                        video_urls.append(up["url"])
                        # 用上传后的直链替换原CQ片段
                        content = content.replace(m.group(0), f"[CQ:video,url={up['url']}]")
                        _log.info(f"✅ 本地视频已上传并替换为直链: {up['url']}")
                    else:
                        _log.warning(f"视频上传失败: {up}")
                except Exception as ve:
                    _log.warning(f"处理本地视频失败: {ve}", exc_info=True)
            else:
                _log.warning(f"视频路径不存在或不可访问: {src_norm}")
    except Exception as e:
        _log.error(f"❌ 提取视频信息失败: {e}", exc_info=True)
    finally:
        _log.info(f"✅ 视频处理完成，最终video_urls数量: {len(video_urls)}")
    
    preview = content[:50] if content else '(仅图片)'
    if image_urls:
        preview += f" [包含{len(image_urls)}张图片]"
    if video_urls:
        preview += f" [包含{len(video_urls)}个视频]"
    _log.info(f"收到群消息 [群:{group_id}({group_name})] [用户:{user_id}({user_card})]: {preview}")
    
    request_data = {
        "type": "group",
        "group_id": group_id,
        "group_name": group_name,
        "user_id": user_id,
        "user_nickname": user_nickname,
        "user_card": user_card,
        "content": content,
        "image_urls": image_urls,
        "video_urls": video_urls,
        "timestamp": timestamp
    }
    
    result = _call_server_api_sync("/api/chat/group", request_data)
    
    if result and result.get("status") == "success":
        should_reply = result.get("should_reply", False)
        reply = result.get("reply", "")
        actions = result.get("actions") or []

        if should_reply and reply:
            if not _is_still_latest_message("group", group_id, message_token):
                _log.info(f"群 {group_id} 在回复生成期间出现更新消息，跳过过期回复发送")
                return
            try:
                bot.api.post_group_msg_sync(group_id, text=reply)
                _log.info(f"已发送群 {group_id} 的回复（普通消息）")
            except Exception as e:  # noqa: BLE001
                _log.error(f"发送群聊回复失败: {e}", exc_info=True)
        # 即使没有文本回复，也可以执行动作（若仍是最新消息）
        if not _is_still_latest_message("group", group_id, message_token):
            _log.info(f"群 {group_id} 在动作执行期间出现更新消息，跳过过期动作")
            return
        try:
            _execute_group_actions(group_id, actions)
        except Exception as e:
            _log.warning(f"执行群动作失败: {e}")
    else:
        if result and result.get("status") == "error":
            error_msg = result.get("message", "未知错误")
            _log.error(f"服务器返回错误: {error_msg}")
        else:
            _log.debug(f"服务器判断不需要回复群 {group_id} 的消息")
            _log.warning("无法获取服务器响应，跳过回复")
            

def _process_private_message_task(payload: Dict[str, Any]) -> None:
    """在线程池中处理私聊消息"""
    _log.info(f"🚀 开始处理私聊消息任务: user_id={payload.get('user_id')}")

    user_id = payload["user_id"]
    user_nickname = payload["user_nickname"]
    message_token = payload["message_token"]
    raw_content = payload["raw_message"]
    timestamp = payload["timestamp"]
    images: List[Image] = payload["images"]

    content = raw_content or ""
    _log.info(f"📝 原始消息内容: {content[:200] if content else '(空)'}")
    image_urls: List[str] = []
    video_urls: List[str] = []

    try:
        if images:
            content = re.sub(r'\[CQ:image[^\]]*\]', '', content).strip()
            _log.info(f"✅ 已移除CQ图片码，清理后的content: {content}")
        for img in images or []:
            temp_dir = None
            img_path = None
            try:
                temp_dir = tempfile.mkdtemp()
                img_path = img.download_sync(temp_dir)
                if img_path and os.path.exists(img_path):
                    with open(img_path, 'rb') as f:
                        img_bytes = f.read()
                        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    img_ext = os.path.splitext(img_path)[1].lower()
                    img_format = img_ext[1:] if img_ext else 'jpeg'
                    if img_format not in ['jpeg', 'jpg', 'png', 'gif', 'webp']:
                        img_format = 'jpeg'
                    upload_result = _call_server_api_sync("/api/upload/image", {
                        "data": img_base64,
                        "format": img_format
                    })
                    if upload_result and upload_result.get("status") == "success":
                        img_url = upload_result.get("url", "")
                        if img_url:
                            image_urls.append(img_url)
                            _log.info(f"✅ 图片已上传，获取URL: {img_url}")
                        else:
                            _log.warning("图片上传成功但未返回URL")
                    else:
                        error_msg = upload_result.get("message", "未知错误") if upload_result else "无响应"
                        _log.warning(f"图片上传失败: {error_msg}")
                else:
                    _log.warning(f"图片下载失败或路径不存在: {img_path}")
            except Exception as e:  # noqa: BLE001
                _log.warning(f"处理单个图片失败: {e}", exc_info=True)
            finally:
                try:
                    if img_path and os.path.exists(img_path):
                        os.remove(img_path)
                    if temp_dir and os.path.isdir(temp_dir):
                        os.rmdir(temp_dir)
                except Exception:
                    pass
    except Exception as e:  # noqa: BLE001
        _log.warning(f"提取图片信息失败: {e}", exc_info=True)
    
    # 提取并处理视频CQ
    _log.info(f"🔍 开始处理视频CQ码，content长度: {len(content) if content else 0}")
    _log.info(f"🔍 content内容: {content[:200] if content else '(空)'}")
    try:
        # 匹配 [CQ:video,...]，提取所有可能的字段
        video_matches = list(re.finditer(r'\[CQ:video([^\]]*)\]', content or "", flags=re.IGNORECASE))
        _log.info(f"🔍 找到 {len(video_matches)} 个视频CQ码")
        for m in video_matches:
            attrs = m.group(1) or ""
            # 按优先级提取：url > file > file_name
            src = None
            for field in ['url', 'file', 'file_name']:
                # 匹配 = 后面的内容，直到遇到逗号或右方括号（允许反斜杠，因为Windows路径需要）
                field_match = re.search(rf'{field}=([^,\]]+)', attrs)
                if field_match:
                    src = field_match.group(1)
                    break

            if not src:
                _log.warning(f"⚠️ 视频CQ码中没有找到url/file/file_name字段: {m.group(0)}")
                continue

            # 处理路径
            src_norm = src.replace("&amp;", "&")
            _log.info(f"🎥 处理视频路径: {src_norm} (原始字段值: {src})")

            # 检查是否是本地路径或HTTP URL
            if src_norm.lower().startswith(("http://", "https://")):
                # HTTP URL，直接使用
                video_urls.append(src_norm)
                _log.info(f"✅ 添加HTTP视频URL: {src_norm}")
            elif re.match(r'^[a-zA-Z]:\\', src_norm) or re.match(r'^\\\\', src_norm):
                # Windows本地路径，优先处理（在Linux客户端上无法用os.path.exists检查）
                try:
                    _log.info(f"🔍 检测到Windows本地路径，尝试直接上传: {src_norm}")
                    # 检查文件是否存在
                    if not os.path.exists(src_norm):
                        _log.error(f"❌ Windows视频文件不存在: {src_norm}")
                        _log.error(f"💡 请确保NapCat和客户端在同一台机器上运行，且文件路径正确")
                        continue
                    
                    _log.info(f"📥 开始读取视频文件: {src_norm}")
                    # 读取文件内容
                    with open(src_norm, "rb") as vf:
                        vbytes = vf.read()
                    
                    file_size = len(vbytes)
                    _log.info(f"✅ 视频文件读取成功，大小: {file_size} 字节")

                    # 使用multipart/form-data上传文件
                    files = {'file': (os.path.basename(src_norm), vbytes, 'video/mp4')}
                    _log.info(f"📤 开始上传视频到服务器...")
                    up = _call_server_api_sync("/api/upload/video", data=None, files=files)

                    if up and up.get("status") == "success" and up.get("url"):
                        video_urls.append(up["url"])
                        # 用上传后的直链替换原CQ片段
                        content = content.replace(m.group(0), f"[CQ:video,url={up['url']}]")
                        _log.info(f"✅ Windows本地视频已上传并替换为直链: {up['url']}")
                    else:
                        _log.warning(f"⚠️ Windows视频上传失败: {up}")
                except FileNotFoundError as fe:
                    _log.error(f"❌ Windows视频文件未找到: {src_norm}")
                    _log.error(f"💡 错误详情: {fe}")
                    _log.error(f"💡 请确保NapCat和客户端在同一台机器上运行")
                except PermissionError as pe:
                    _log.error(f"❌ Windows视频文件权限不足: {src_norm}")
                    _log.error(f"💡 错误详情: {pe}")
                except Exception as ve:
                    _log.error(f"❌ 处理Windows本地视频失败: {ve}", exc_info=True)
                    _log.error(f"💡 请确保NapCat和客户端在同一台机器上运行")
            elif os.path.exists(src_norm):
                # 本地文件存在，上传到服务器
                try:
                    _log.info(f"发现本地视频文件，开始上传: {src_norm}")
                    with open(src_norm, "rb") as vf:
                        vbytes = vf.read()

                    # 使用multipart/form-data上传文件
                    files = {'file': (os.path.basename(src_norm), vbytes, 'video/mp4')}
                    up = _call_server_api_sync("/api/upload/video", data=None, files=files)

                    if up and up.get("status") == "success" and up.get("url"):
                        video_urls.append(up["url"])
                        # 用上传后的直链替换原CQ片段
                        content = content.replace(m.group(0), f"[CQ:video,url={up['url']}]")
                        _log.info(f"✅ 本地视频已上传并替换为直链: {up['url']}")
                    else:
                        _log.warning(f"视频上传失败: {up}")
                except Exception as ve:
                    _log.warning(f"处理本地视频失败: {ve}", exc_info=True)
            else:
                _log.warning(f"视频路径不存在或不可访问: {src_norm}")
    except Exception as e:
        _log.error(f"❌ 提取视频信息失败: {e}", exc_info=True)
    finally:
        _log.info(f"✅ 视频处理完成，最终video_urls数量: {len(video_urls)}")
    
    preview = content[:50] if content else '(仅图片)'
    if image_urls:
        preview += f" [包含{len(image_urls)}张图片]"
    if video_urls:
        preview += f" [包含{len(video_urls)}个视频]"
    _log.info(f"收到私聊消息 [用户:{user_id}({user_nickname})]: {preview}")
    
    request_data = {
        "type": "private",
        "user_id": user_id,
        "user_nickname": user_nickname,
        "content": content,
        "image_urls": image_urls,
        "video_urls": video_urls,
        "timestamp": timestamp
    }
    
    result = _call_server_api_sync("/api/chat/private", request_data)
    
    if result and result.get("status") == "success":
        reply = result.get("reply", "")
        actions = result.get("actions") or []
        if reply:
            if not _is_still_latest_message("private", user_id, message_token):
                _log.info(f"私聊 {user_id} 在回复生成期间出现更新消息，跳过过期回复发送")
                return
            try:
                bot.api.post_private_msg_sync(user_id, text=reply)
                _log.info(f"已发送私聊 {user_id} 的回复（普通消息）")
            except Exception as e:  # noqa: BLE001
                _log.error(f"发送私聊回复失败: {e}", exc_info=True)
        # 执行动作（即使无文本回复）
        if not _is_still_latest_message("private", user_id, message_token):
            _log.info(f"私聊 {user_id} 在动作执行期间出现更新消息，跳过过期动作")
            return
        try:
            _execute_private_actions(user_id, actions)
        except Exception as e:
            _log.warning(f"执行私聊动作失败: {e}")
    else:
        if result and result.get("status") == "error":
            error_msg = result.get("message", "未知错误")
            _log.error(f"服务器返回错误: {error_msg}")
        else:
            _log.debug("服务器未返回回复内容")
            _log.warning("无法获取服务器响应，跳过回复")
            

@bot.group_event()
async def on_group_message(msg: GroupMessage):
    """接收群消息并提交到线程池处理"""

    group_id = str(msg.group_id)
    user_id = str(msg.user_id)
    message_token = _mark_latest_message("group", group_id)

    group_name = f"群{group_id}"
    try:
        if hasattr(msg, 'group') and msg.group:
            group_name = getattr(msg.group, 'name', None) or group_name

        if group_name == f"群{group_id}":
            try:
                group_info = await bot.api.get_group_info(group_id)
                if group_info:
                    if hasattr(group_info, 'name'):
                        group_name = group_info.name
                    elif hasattr(group_info, 'group_name'):
                        group_name = group_info.group_name
                    elif isinstance(group_info, dict):
                        group_name = group_info.get('name') or group_info.get('group_name') or group_name
                    if group_name != f"群{group_id}":
                        _log.info(f"✅ 通过API获取到群名称: {group_name}")
            except Exception as api_e:  # noqa: BLE001
                _log.warning(f"通过API获取群名称失败: {api_e}")
    except Exception as e:  # noqa: BLE001
        _log.warning(f"提取群名称失败: {e}")
        group_name = f"群{group_id}"

    user_nickname = f"用户{user_id}"
    user_card = user_nickname
    if hasattr(msg, 'sender') and msg.sender:
        if hasattr(msg.sender, 'nickname'):
            user_nickname = msg.sender.nickname or user_nickname
        if hasattr(msg.sender, 'card'):
            user_card = msg.sender.card or user_nickname

    payload = {
        "group_id": group_id,
        "user_id": user_id,
        "group_name": group_name,
        "user_nickname": user_nickname,
        "user_card": user_card,
        "raw_message": msg.raw_message or "",
        "timestamp": time.time(),
        "images": list(msg.filter(Image)) if hasattr(msg, 'filter') else [],
        "message_token": message_token
    }

    _submit_message_task(
        f"group:{group_id}:{message_token}",
        _process_group_message_task,
        payload
    )


@bot.private_event()
async def on_private_message(msg: PrivateMessage):
    """接收私聊消息并提交到线程池处理"""

    user_id = str(msg.user_id)
    message_token = _mark_latest_message("private", user_id)

    user_nickname = f"用户{user_id}"
    if hasattr(msg, 'sender') and msg.sender:
        if hasattr(msg.sender, 'nickname'):
            user_nickname = msg.sender.nickname or user_nickname

    payload = {
        "user_id": user_id,
        "user_nickname": user_nickname,
        "raw_message": msg.raw_message or "",
        "timestamp": time.time(),
        "images": list(msg.filter(Image)) if hasattr(msg, 'filter') else [],
        "message_token": message_token
    }

    _submit_message_task(
        f"private:{user_id}:{message_token}",
        _process_private_message_task,
        payload
    )


# ========== 心跳事件处理器 ==========
@bot.on_heartbeat()
def on_heartbeat(event):
    """
    处理心跳事件，用于监控连接状态
    """
    global last_heartbeat_time
    with heartbeat_lock:
        last_heartbeat_time = datetime.now()
    _log.debug("收到心跳信号，连接正常")


# ========== 启动事件处理器 ==========
@bot.on_startup()
def on_startup(event):
    """
    处理启动事件，初始化心跳时间
    """
    global last_heartbeat_time
    with heartbeat_lock:
        last_heartbeat_time = datetime.now()
    _log.info("✅ Bot已启动，开始监控连接状态")


# ========== 关闭事件处理器 ==========
@bot.on_shutdown()
def on_shutdown(event):
    """
    处理关闭事件
    """
    _log.warning("⚠️ Bot已关闭或连接断开")


# ========== 连接状态检查函数 ==========
def check_connection_status():
    """
    定期检查连接状态
    如果长时间没有收到心跳，记录警告
    """
    global last_heartbeat_time
    
    while True:
        try:
            time.sleep(CONNECTION_CHECK_INTERVAL)
            
            with heartbeat_lock:
                if last_heartbeat_time is None:
                    # 如果还没有收到过心跳，可能是刚启动，跳过这次检查
                    continue
                
                time_since_last_heartbeat = (datetime.now() - last_heartbeat_time).total_seconds()
                
                # 检查是否超时
                if time_since_last_heartbeat > HEARTBEAT_TIMEOUT:
                    _log.warning(
                        f"⚠️ 警告：已超过 {int(time_since_last_heartbeat)} 秒未收到心跳信号！"
                        f"连接可能已断开，请检查："
                    )
                    _log.warning("1. NapCat服务是否正常运行")
                    _log.warning("2. QQ是否还在线")
                    _log.warning("3. 网络连接是否正常")
                    _log.warning("4. 是否被QQ踢下线（查看NapCat日志）")
                    
                    # 尝试检查websocket连接状态（如果bot对象有这个方法）
                    try:
                        if hasattr(bot, 'adapter') and hasattr(bot.adapter, 'is_websocket_online'):
                            is_online = bot.adapter.is_websocket_online()
                            if not is_online:
                                _log.error("❌ WebSocket连接已断开！")
                            else:
                                _log.warning("⚠️ WebSocket连接状态显示在线，但未收到心跳，可能是NapCat端的问题")
                    except Exception as e:
                        _log.debug(f"无法检查WebSocket状态: {e}")
                else:
                    _log.debug(f"连接正常，上次心跳: {int(time_since_last_heartbeat)}秒前")
                    
        except Exception as e:
            _log.error(f"连接状态检查出错: {e}", exc_info=True)


# ========== 启动连接状态监控线程 ==========
def start_connection_monitor():
    """
    启动连接状态监控线程
    """
    monitor_thread = threading.Thread(target=check_connection_status, daemon=True)
    monitor_thread.start()
    _log.info("连接状态监控线程已启动")


# ========== 启动 BotClient ==========
if __name__ == "__main__":
    print("=" * 60)
    print("QQ机器人客户端启动 - 完整版")
    print("=" * 60)
    print(f"服务器地址: {SERVER_URL}")
    if API_KEY:
        print(f"API密钥: {API_KEY[:10]}...")
    print("=" * 60)
    
    # 检查服务器连接
    print("正在检查服务器连接...")
    if not check_server_health():
        print("❌ 无法连接到服务器，请检查：")
        print("1. 服务器是否已启动")
        print("2. SSH隧道是否已建立（如果使用SSH隧道）")
        print("3. SERVER_URL配置是否正确")
        print("4. 网络连接是否正常")
        exit(1)
    
    print("✅ 服务器连接正常")
    print("=" * 60)
    
    # 启动连接状态监控
    start_connection_monitor()
    print("✅ 连接状态监控已启动")
    print("=" * 60)
    print("等待消息...")
    print("- 群聊消息将转发到服务器处理")
    print("- 私聊消息将转发到服务器处理")
    print("- 心跳监控：每30秒检查一次连接状态")
    print("- 如果超过2分钟未收到心跳，将发出警告")
    print("- 按 Ctrl+C 停止")
    print("=" * 60)
    
    try:
        bot.run(enable_webui_interaction=False)
    except KeyboardInterrupt:
        print("\n正在关闭客户端...")
        _log.info("客户端已关闭")
    except Exception as e:
        _log.error(f"客户端运行出错: {e}", exc_info=True)
        print(f"\n❌ 客户端运行出错: {e}")
        print("请检查日志以获取更多信息")

