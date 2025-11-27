# -*- coding: utf-8 -*-
"""
消息队列模块
处理消息队列、任务调度、打断机制等
"""
import queue
import logging
import threading
import time
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional

from chat.history_manager import (
    get_chat_history,
    set_chat_history,
    maintain_chat_history,
    chat_history_lock,
)
from chat.reply_handler import generate_reply, truncate_history_by_tokens
from chat.prompting import format_multimodal_message, parse_action_commands, build_system_prompt
from utils.cq import extract_cq_image_urls, extract_cq_video_urls, extract_cq_audio_urls, extract_cq_file_urls, extract_http_urls
import services.media as media_service
from services.asr import transcribe_audio
from services.fetch import fetch_url_content
from services.extractors import extract_text_and_images_from_file
from utils.metrics import metrics_add

_log = logging.getLogger(__name__)

# 消息队列
message_queue = queue.Queue()
queue_lock = threading.Lock()
worker_thread_started = False

# 正在处理的聊天（用于中断同一聊天内的旧消息）
# {chat_id: {"interrupt_event": Event, "response_dict": dict, "start_time": float, "lock": Lock}}
processing_chats: Dict[str, Dict[str, Any]] = {}


@dataclass
class MessageTask:
    """消息处理任务"""
    chat_type: str  # "group" 或 "private"
    chat_id: str  # 群ID或用户ID
    data: Dict[str, Any]  # 原始请求数据
    response_dict: Dict[str, Any]  # 响应字典


def process_message_task(
    task: MessageTask,
    model,
    processor,
    memory_db,
    recall_token_ids,
    config,
    server_base_url,
    image_upload_dir,
    video_upload_dir,
    audio_upload_dir,
    file_upload_dir,
    is_training,
    training_lock,
    model_lock
):
    """
    处理单个消息任务
    
    Args:
        task: 消息任务
        model: 模型实例
        processor: 处理器实例
        memory_db: 记忆数据库
        recall_token_ids: 特殊token IDs
        config: 配置字典
        server_base_url: 服务器基础URL
        image_upload_dir: 图片上传目录
        video_upload_dir: 视频上传目录
        audio_upload_dir: 音频上传目录
        file_upload_dir: 文件上传目录
        is_training: 是否处于训练模式
        training_lock: 训练锁
        model_lock: 模型锁
    """
    # 检查训练模式
    with training_lock:
        if is_training:
            _log.warning("⚠️ 当前处于训练模式，拒绝处理消息")
            if task.response_dict:
                task.response_dict["reply"] = ""
                task.response_dict["should_reply"] = False
                task.response_dict["error"] = "服务器正在训练中"
                task.response_dict["status"] = "error"
                task.response_dict["status_code"] = 503
            return
    
    chat_id = task.chat_id
    chat_type = task.chat_type
    data = task.data
    response_dict = task.response_dict
    
    try:
        start_time = time.time()
        metrics_add("requests_total", 1)
        
        # 检查打断
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
                "start_time": start_time,
                "lock": threading.Lock()
            }
        
        # 如果中断了旧任务，等待一小段时间让旧任务检测到中断并退出
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
        
        try:
            # 提取内容
            content = data.get("content", "")
            timestamp = data.get("timestamp", time.time())
            time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            if chat_type == "group":
                group_id = str(data.get("group_id") or chat_id)
                group_name = data.get("group_name", f"群{group_id}")
                user_id = str(data.get("user_id") or "")
                user_nickname = data.get("user_nickname", f"用户{user_id}") if user_id else data.get("user_nickname", "未知用户")
                user_card = data.get("user_card", user_nickname)
                display_name = user_card if user_card else user_nickname
                _log.info(f"🔍 群聊消息内容分析({group_id}/{display_name}): {content[:100]}...")
            else:
                group_name = None
                user_id = str(data.get("user_id") or chat_id)
                user_nickname = data.get("user_nickname", f"用户{user_id}")
                display_name = user_nickname
                _log.info(f"🔍 私聊消息内容分析({user_id}): {content[:100]}...")
            
            # 提取CQ码
            cleaned_content, image_urls = extract_cq_image_urls(content)
            img_sample = f"，示例: {image_urls[:3]}" if image_urls else ""
            _log.info(f"📷 图片CQ码提取: 找到 {len(image_urls)} 个{img_sample}")
            
            _, video_urls = extract_cq_video_urls(content)
            video_sample = f"，示例: {video_urls[:3]}" if video_urls else ""
            _log.info(f"🎥 视频CQ码提取: 找到 {len(video_urls)} 个{video_sample}")
            
            _, audio_urls = extract_cq_audio_urls(content)
            if audio_urls:
                _log.info(f"🎵 语音CQ码提取: 找到 {len(audio_urls)} 个")
            
            _, file_urls = extract_cq_file_urls(content)
            if file_urls:
                _log.info(f"📄 文件CQ码提取: 找到 {len(file_urls)} 个")
            
            # 处理媒体URL
            video_urls.extend(data.get("video_urls", []))
            
            # 下载媒体
            cached_image_urls = []
            if image_urls:
                _log.info(f"📥 开始下载 {len(image_urls)} 个图片...")
                for url in image_urls:
                    cached = media_service.download_image_to_storage(
                        url, image_upload_dir, server_base_url,
                        metrics_add, _log
                    )
                    cached_image_urls.append(cached or url)
                image_urls = cached_image_urls
            
            cached_video_urls = []
            if video_urls:
                _log.info(f"📥 开始下载 {len(video_urls)} 个视频...")
                for url in video_urls:
                    cached = media_service.download_video_to_storage(
                        url, video_upload_dir, server_base_url,
                        metrics_add, _log
                    )
                    cached_video_urls.append(cached or url)
                video_urls = cached_video_urls
            
            # 处理语音消息（ASR）
            if audio_urls:
                _log.info(f"📥 开始处理 {len(audio_urls)} 个语音消息...")
                for url in audio_urls:
                    # 下载语音
                    cached_audio = media_service.download_audio_to_storage(
                        url, audio_upload_dir, server_base_url,
                        metrics_add, _log
                    )
                    
                    if cached_audio:
                        # 转换为本地路径
                        if cached_audio.startswith(server_base_url):
                            filename = cached_audio.split("/")[-1]
                            local_path = f"{audio_upload_dir}/{filename}"
                            
                            # 执行ASR
                            try:
                                text = transcribe_audio(local_path, metrics_add, _log)
                                if text:
                                    _log.info(f"✅ 语音转写成功: {text}")
                                    content += f"\n[语音转写]: {text}"
                            except Exception as e:
                                _log.warning(f"⚠️ ASR转写失败: {e}")
            
            # 处理文件消息（下载并提取文本/图片）
            if file_urls:
                _log.info(f"📥 开始处理 {len(file_urls)} 个文件...")
                file_texts = []
                for url in file_urls:
                    # 下载文件
                    cached_file = media_service.download_file_to_storage(
                        url, file_upload_dir, server_base_url,
                        metrics_add, _log
                    )
                    
                    if cached_file:
                        # 转换为本地路径
                        if cached_file.startswith(server_base_url):
                            filename = cached_file.split("/")[-1]
                            local_path = f"{file_upload_dir}/{filename}"
                            
                            try:
                                # 提取文本和图片
                                text, images = extract_text_and_images_from_file(
                                    local_path, image_upload_dir, metrics_add, _log
                                )
                                
                                if text:
                                    file_texts.append(text)
                                    _log.info(f"✅ 文件文本提取成功: {len(text)}字符")
                                
                                # 将提取的图片添加到image_urls
                                for img_path in images:
                                    filename = os.path.basename(img_path)
                                    img_url = f"{server_base_url.rstrip('/')}/static/images/{filename}"
                                    if img_url not in image_urls:
                                        image_urls.append(img_url)
                                        _log.info(f"✅ 文件图片提取成功: {filename}")
                                        
                            except Exception as e:
                                _log.warning(f"⚠️ 文件提取失败: {e}")
                
                # 将文件文本追加到内容
                if file_texts:
                    file_content = "\n\n".join([f"【文件内容{i+1}】\n{t}" for i, t in enumerate(file_texts)])
                    content = (content + "\n\n" if content else "") + file_content
            
            # 处理网页链接（抓取内容）
            http_urls = extract_http_urls(content)
            if http_urls:
                _log.info(f"🌐 检测到 {len(http_urls)} 个网页链接，尝试抓取内容...")
                web_contents = []
                for url in http_urls:
                    try:
                        web_text = fetch_url_content(url, metrics_add, _log)
                        if web_text:
                            web_contents.append(f"【网页内容: {url}】\n{web_text}")
                            _log.info(f"✅ 网页抓取成功: {url} ({len(web_text)}字符)")
                    except Exception as e:
                        _log.warning(f"⚠️ 网页抓取失败 {url}: {e}")
                
                if web_contents:
                    web_content_str = "\n\n".join(web_contents)
                    content = (content + "\n\n" if content else "") + web_content_str
            
            media_info = ""
            if image_urls:
                media_info += f" [包含{len(image_urls)}张图片]"
            if video_urls:
                media_info += f" [包含{len(video_urls)}个视频]"
            if audio_urls:
                media_info += f" [包含{len(audio_urls)}段语音]"
            if file_urls:
                media_info += f" [包含{len(file_urls)}个文件]"
            
            prefix = f"[{time_str}] {display_name}(QQ:{user_id})："
            formatted_message = f"{prefix}{cleaned_content}" if cleaned_content else prefix
            
            _log.info(f"🗨️ 收到{ '群聊' if chat_type == 'group' else '私聊' }消息 {chat_id}{media_info}: {formatted_message[:80]}...")
            
            if image_urls or video_urls:
                user_message = format_multimodal_message(formatted_message, image_urls, video_urls)
            else:
                user_message = [{"type": "text", "text": formatted_message}]
            
            max_history = config.get("chat_history", {}).get("max_history_length", 200)
            with chat_history_lock:
                history = list(get_chat_history(chat_type, chat_id))
                history.append({"role": "user", "content": user_message, "timestamp": timestamp})
                history, removed_messages = maintain_chat_history(chat_type, chat_id, history, max_history)
                set_chat_history(chat_type, chat_id, history)
                chat_history_snapshot = history.copy()
            if removed_messages:
                _log.info(f"💾 历史超长，追加保存 {len(removed_messages)} 条旧消息到存储（{chat_type} {chat_id}）")
                threading.Thread(
                    target=save_chat_history_to_storage,
                    args=(config, chat_type, chat_id, removed_messages),
                    daemon=True
                ).start()
            _log.info(f"📝 维护后历史长度: {len(chat_history_snapshot)}（{chat_type} {chat_id}）")
            
            # 在生成前再次检查是否仍然是最新任务
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is not response_dict:
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在生成前已被新消息替换，退出处理")
                    return
                if interrupt_event.is_set():
                    _log.warning(f"⚠️ 聊天 {chat_id} 的任务在生成前检测到中断信号，清除后继续")
                    interrupt_event.clear()
            
            # 截断历史（基于token数，截断后会更新内存中的历史）
            chat_context = {}
            if chat_type == "group":
                chat_context = {"group_id": chat_id, "group_name": group_name or chat_id}
                if user_id:
                    chat_context["user_id"] = user_id
                if user_nickname:
                    chat_context["user_nickname"] = user_nickname
                if display_name:
                    chat_context["display_name"] = display_name
            elif chat_type == "private":
                chat_context = {"user_id": user_id or chat_id, "user_nickname": user_nickname or chat_id}
            
            system_prompt = build_system_prompt(config, chat_type, chat_context)
            max_tokens = config.get("chat_history", {}).get("max_input_tokens", 32000)
            
            _log.info(f"📊 开始截断历史（{chat_type} {chat_id}），max_tokens={max_tokens}")
            original_history_len = len(chat_history_snapshot)
            generation_history = truncate_history_by_tokens(
                processor,
                chat_history_snapshot.copy(),
                system_prompt,
                chat_type,
                chat_id,
                config,
                max_tokens,
                interrupt_event
            )
            if len(generation_history) < original_history_len:
                _log.info(f"✂️ 历史截断: {original_history_len} -> {len(generation_history)}（{chat_type} {chat_id}）")
                # 更新内存中的历史，移除被截断的消息
                with chat_history_lock:
                    set_chat_history(chat_type, chat_id, generation_history)
                    _log.info(f"💾 已更新内存中的历史，移除 {original_history_len - len(generation_history)} 条消息（{chat_type} {chat_id}）")
            else:
                _log.info(f"📏 历史长度 {len(generation_history)}，未超过上限（{chat_type} {chat_id}）")
            
            # 获取生成参数
            gen_config = config.get("generation", {})
            max_new_tokens = gen_config.get("max_new_tokens", 1000)
            temperature = gen_config.get("temperature", 1.0)
            
            # 生成回复
            _log.info(f"🧠 开始生成回复（{chat_type} {chat_id}）...")
            reply, should_reply, interrupted = generate_reply(
                model,
                processor,
                memory_db,
                recall_token_ids,
                config,
                generation_history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                chat_type=chat_type,
                chat_context=chat_context,
                interrupt_event=interrupt_event,
                chat_id=chat_id,
                response_dict=response_dict,
                log_full_io=True,
                is_training=is_training,
                training_lock=training_lock,
                model_lock=model_lock
            )
            
            if interrupted:
                _log.warning(f"⚠️ 生成被中断（{chat_type} {chat_id}）")
                metrics_add("interruptions", 1)
                with queue_lock:
                    current_processing = processing_chats.get(chat_id)
                    if current_processing and current_processing["response_dict"] is response_dict:
                        response_dict.update({
                            "status": "success",
                            "should_reply": False,
                            "reply": "",
                            "status_code": 200
                        })
                        _log.info(f"✅ 已更新中断响应（{chat_type} {chat_id}）")
                return
            
            _log.info(f"📤 生成结果: should_reply={should_reply}, reply_length={len(reply) if reply else 0}")
            
            # 生成结束后，确认是否仍是最新任务
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is not response_dict:
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在生成完成后被新消息替换，跳过后续步骤")
                    return
                if interrupt_event.is_set():
                    _log.info(f"⚠️ 聊天 {chat_id} 的任务在生成完成后检测到中断，跳过更新历史")
                    return
            
            # 保存回复到历史
            with chat_history_lock:
                latest_history = list(get_chat_history(chat_type, chat_id))
                if should_reply and reply:
                    metrics_add("replies_sent", 1)
                    assistant_text = reply
                else:
                    metrics_add("no_reply", 1)
                    assistant_text = "<no_reply>"
                latest_history.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                    "timestamp": time.time()
                })
                latest_history, removed_messages = maintain_chat_history(chat_type, chat_id, latest_history, max_history)
                set_chat_history(chat_type, chat_id, latest_history)
            if removed_messages:
                _log.info(f"💾 assistant历史截断，保存 {len(removed_messages)} 条旧消息到存储（{chat_type} {chat_id}）")
                threading.Thread(
                    target=save_chat_history_to_storage,
                    args=(config, chat_type, chat_id, removed_messages),
                    daemon=True
                ).start()
            _log.info(f"💾 已更新assistant消息到历史（{chat_type} {chat_id}）")
            
            # 提取动作指令
            actions = parse_action_commands(reply) if reply else []
            if actions:
                _log.info(f"🎬 提取到 {len(actions)} 个动作指令: {[a.get('type') for a in actions]}")
            
            # 更新响应前再次确认
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is response_dict:
                    if interrupt_event.is_set():
                        _log.warning(f"⚠️ 聊天 {chat_id} 的任务在更新响应前被中断，跳过响应更新")
                        return
                    response_dict["reply"] = reply if should_reply else ""
                    response_dict["should_reply"] = should_reply
                    response_dict["actions"] = actions
                    response_dict["status"] = "success"
                    response_dict["status_code"] = 200
                    _log.info(f"✅ 已更新响应（{chat_type} {chat_id}），should_reply={should_reply}")
                else:
                    _log.warning(f"⚠️ 聊天 {chat_id} 的任务在更新响应前已被替换，跳过")
            
            elapsed = time.time() - start_time
            metrics_add("response_time", elapsed)
            _log.info(f"✅ 消息处理完成，耗时: {elapsed:.2f}s （{chat_type} {chat_id}）")
            
        finally:
            with queue_lock:
                current_processing = processing_chats.get(chat_id)
                if current_processing and current_processing["response_dict"] is response_dict:
                    del processing_chats[chat_id]
    
    except Exception as e:
        _log.error(f"❌ 处理消息任务失败: {e}", exc_info=True)
        response_dict["status"] = "error"
        response_dict["error"] = str(e)
        response_dict["status_code"] = 500


def message_queue_worker(model, processor, memory_db, recall_token_ids, config, 
                        server_base_url, image_upload_dir, video_upload_dir, audio_upload_dir, file_upload_dir,
                        is_training_getter, training_lock, model_lock):
    """
    消息队列工作线程
    
    Args:
        model: 模型实例
        processor: 处理器实例
        memory_db: 记忆数据库
        recall_token_ids: 特殊token IDs
        config: 配置字典
        server_base_url: 服务器基础URL
        image_upload_dir: 图片上传目录
        video_upload_dir: 视频上传目录
        audio_upload_dir: 音频上传目录
        file_upload_dir: 文件上传目录
        is_training_getter: 获取训练状态的函数
        training_lock: 训练锁
        model_lock: 模型锁
    """
    _log.info("📋 消息队列工作线程已启动")
    
    while True:
        try:
            task = message_queue.get(timeout=1)
            _log.info(f"🔄 开始处理消息任务: {task.chat_type} {task.chat_id}")
            
            def _run_task(task_obj):
                """在独立线程中执行任务，保持与v1.2一致的并发打断行为"""
                try:
                    current_training = is_training_getter() if callable(is_training_getter) else is_training_getter
                    process_message_task(
                        task_obj,
                        model,
                        processor,
                        memory_db,
                        recall_token_ids,
                        config,
                        server_base_url,
                        image_upload_dir,
                        video_upload_dir,
                        audio_upload_dir,
                        file_upload_dir,
                        current_training,
                        training_lock,
                        model_lock
                    )
                except Exception as task_err:
                    _log.error(f"❌ 处理消息任务失败（{task_obj.chat_type} {task_obj.chat_id}）: {task_err}", exc_info=True)
            
            worker_thread = threading.Thread(target=_run_task, args=(task,), daemon=True)
            worker_thread.start()
            message_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            _log.error(f"❌ 队列工作线程错误: {e}", exc_info=True)

