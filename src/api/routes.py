# -*- coding: utf-8 -*-
"""
API路由模块
定义所有HTTP端点
"""
import time
import threading
import os
import sys
from flask import Blueprint, request, jsonify, send_from_directory

import logging
from api import server_state
from chat.message_queue import MessageTask, message_queue, queue_lock, message_queue_worker
from services.media import download_image_to_storage, download_video_to_storage, download_audio_to_storage, download_file_to_storage
from utils.cq import extract_cq_image_urls
from utils.metrics import metrics_add

_log = logging.getLogger(__name__)

# 创建蓝图
bp_health = Blueprint("health", __name__)
bp_chat = Blueprint("chat", __name__)
bp_upload = Blueprint("upload", __name__)
bp_training = Blueprint("training", __name__)

# 工作线程标志（模块级别）
worker_thread_started = False


def _resolve_script_context():
    script_path = getattr(server_state, "server_script_path", None)
    if not script_path:
        script_path = os.environ.get("SERVER_SCRIPT_PATH")
    if not script_path or not os.path.exists(script_path):
        if hasattr(sys, "argv") and sys.argv and sys.argv[0]:
            candidate = os.path.abspath(sys.argv[0])
            if os.path.exists(candidate):
                script_path = candidate
    if not script_path or not os.path.exists(script_path):
        script_path = os.path.abspath(__file__)
    
    script_args = getattr(server_state, "server_script_args", None)
    if script_args is None:
        script_args = sys.argv[1:] if hasattr(sys, 'argv') else []
    return script_path, script_args


# ==================== 健康检查 ====================
@bp_health.route("/health", methods=["GET"])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "model_loaded": server_state.model is not None,
        "processor_loaded": server_state.processor is not None,
        "device": str(server_state.device) if server_state.device else None
    })


@bp_health.route("/metrics", methods=["GET"])
def metrics_endpoint():
    """指标端点"""
    from utils.metrics import get_metrics
    import statistics
    
    m = get_metrics()
    
    # 计算延迟统计
    response_times = m.get("response_times", [])
    if response_times:
        m["latency_p50_ms"] = statistics.median(response_times) * 1000
        m["latency_avg_ms"] = (sum(response_times) / len(response_times)) * 1000
        m["latency_count"] = len(response_times)
    else:
        m["latency_p50_ms"] = 0
        m["latency_avg_ms"] = 0
        m["latency_count"] = 0
    
    # 移除原始数据
    if "response_times" in m:
        del m["response_times"]
    if "latencies" in m:
        del m["latencies"]
    
    return jsonify(m), 200


# ==================== 聊天端点 ====================
def _handle_chat_request(chat_type: str):
    """处理聊天请求的通用逻辑"""
    global worker_thread_started
    
    try:
        data = request.json
        chat_id = str(data.get("group_id" if chat_type == "group" else "user_id", ""))
        content = data.get("content", "")
        
        # 提取图片和视频
        cleaned_content, image_urls = extract_cq_image_urls(content)
        video_urls = data.get("video_urls") or []
        
        if not chat_id or (not cleaned_content and not image_urls and not video_urls):
            return jsonify({"status": "error", "message": "缺少必要参数"}), 400
        
        # 启动队列工作线程
        if not worker_thread_started:
            with queue_lock:
                if not worker_thread_started:
                    worker_thread = threading.Thread(
                        target=message_queue_worker,
                        args=(
                            server_state.model,
                            server_state.processor,
                            server_state.memory_db,
                            server_state.recall_token_ids,
                            server_state.config,
                            server_state.server_base_url,
                            server_state.IMAGE_UPLOAD_DIR,
                            server_state.VIDEO_UPLOAD_DIR,
                            server_state.AUDIO_UPLOAD_DIR,
                            server_state.FILE_UPLOAD_DIR,
                            lambda: server_state.is_training,
                            server_state.training_lock,
                            server_state.model_lock
                        ),
                        daemon=True
                    )
                    worker_thread.start()
                    worker_thread_started = True
                    _log.info("✅ 消息队列工作线程已启动")
        
        # 创建任务
        response_dict = {}
        task = MessageTask(
            chat_type=chat_type,
            chat_id=chat_id,
            data=data,
            response_dict=response_dict
        )
        message_queue.put(task)
        
        # 等待响应
        timeout = 120
        start_time = time.time()
        while time.time() - start_time < timeout:
            if "status" in response_dict:
                status_code = response_dict.pop("status_code", 200)
                return jsonify(response_dict), status_code
            time.sleep(0.1)
        
        return jsonify({"status": "error", "message": "处理超时"}), 500
        
    except Exception as e:
        _log.error(f"❌ 处理{chat_type}消息出错: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@bp_chat.route("/api/chat/private", methods=["POST"])
def chat_private():
    """私聊端点"""
    return _handle_chat_request("private")


@bp_chat.route("/api/chat/group", methods=["POST"])
def chat_group():
    """群聊端点"""
    return _handle_chat_request("group")


# ==================== 上传端点 ====================
@bp_upload.route("/api/upload/image", methods=["POST"])
def upload_image():
    """上传图片"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "没有文件"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "文件名为空"}), 400
        
        # 保存文件
        url = download_image_to_storage(file, server_state.server_base_url, is_file_obj=True)
        
        return jsonify({"status": "success", "url": url})
    except Exception as e:
        _log.error(f"❌ 上传图片失败: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@bp_upload.route("/api/upload/video", methods=["POST"])
def upload_video():
    """上传视频"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "没有文件"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "文件名为空"}), 400
        
        # 保存文件
        url = download_video_to_storage(file, server_state.server_base_url, is_file_obj=True)
        
        return jsonify({"status": "success", "url": url})
    except Exception as e:
        _log.error(f"❌ 上传视频失败: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# ==================== 训练端点 ====================
@bp_training.route("/api/training/trigger", methods=["POST"])
def training_trigger():
    """手动触发训练"""
    try:
        if server_state.training_scheduler is None:
            from memory.training_scheduler import MemoryTrainingScheduler
            script_path, script_args = _resolve_script_context()
            server_state.training_scheduler = MemoryTrainingScheduler(
                server_state.config, script_path, script_args
            )
            if not hasattr(server_state.training_scheduler, 'scheduler') or not server_state.training_scheduler.scheduler.running:
                server_state.training_scheduler.start()
        
        # 在新线程中执行训练
        def run_training():
            server_state.training_scheduler.train_job()
        
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()
        
        return jsonify({"success": True, "message": "训练任务已提交"})
    
    except Exception as e:
        _log.error(f"❌ 触发训练失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bp_training.route("/api/training/status", methods=["GET"])
def training_status():
    """获取训练状态"""
    try:
        if server_state.training_scheduler is None:
            return jsonify({"enabled": False, "message": "训练调度器未启动"})
        
        is_running = server_state.training_scheduler.is_running
        return jsonify({
            "enabled": True,
            "is_running": is_running,
            "is_training": server_state.is_training
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp_training.route("/api/training/debug", methods=["GET"])
def training_debug():
    """训练调试信息"""
    try:
        if server_state.training_scheduler is None:
            return jsonify({"error": "训练调度器未启动"}), 400
            
        scheduler = server_state.training_scheduler.scheduler
        jobs = []
        for job in scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "next_run_time": str(job.next_run_time) if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
            
        return jsonify({
            "is_running": server_state.training_scheduler.is_running,
            "is_training": server_state.is_training,
            "jobs": jobs,
            "config": server_state.config.get("memory", {}).get("training", {})
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp_training.route("/api/training/save-chat-history", methods=["POST"])
def save_chat_history():
    """手动保存聊天历史"""
    try:
        from chat.history_manager import group_chat_histories, private_chat_histories, save_chat_history_to_storage
        
        saved_count = 0
        
        # 保存群聊历史
        for chat_id, messages in group_chat_histories.items():
            save_chat_history_to_storage(server_state.config, "group", chat_id, messages)
            saved_count += 1
        
        # 保存私聊历史
        for chat_id, messages in private_chat_histories.items():
            save_chat_history_to_storage(server_state.config, "private", chat_id, messages)
            saved_count += 1
        
        return jsonify({
            "success": True,
            "saved_count": saved_count,
            "message": f"已保存{saved_count}个会话的历史记录"
        })
    
    except Exception as e:
        _log.error(f"❌ 保存聊天历史失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# 注册所有蓝图
def register_blueprints(app):
    """
    注册所有蓝图到Flask应用
    
    Args:
        app: Flask应用实例
    """
    app.register_blueprint(bp_health)
    app.register_blueprint(bp_chat)
    app.register_blueprint(bp_upload)
    app.register_blueprint(bp_training)
    _log.info("✅ 所有路由已注册")

