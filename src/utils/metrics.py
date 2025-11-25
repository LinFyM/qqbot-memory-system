# -*- coding: utf-8 -*-
"""
指标统计模块
"""
import threading
from typing import Any, Dict

# 全局指标字典
metrics_lock = threading.Lock()
metrics: Dict[str, Any] = {
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


def metrics_add(key: str, value: float = 1) -> None:
    """
    添加指标
    
    Args:
        key: 指标键
        value: 指标值
    """
    global metrics
    
    with metrics_lock:
        if key == "response_time":
            metrics.setdefault("response_times", []).append(value)
            if len(metrics["response_times"]) > MAX_LATENCY_BUCKET:
                metrics["response_times"] = metrics["response_times"][-MAX_LATENCY_BUCKET:]
        elif key == "latency":
            metrics.setdefault("latencies", []).append(value)
            if len(metrics["latencies"]) > MAX_LATENCY_BUCKET:
                metrics["latencies"] = metrics["latencies"][-MAX_LATENCY_BUCKET:]
        else:
            metrics[key] = metrics.get(key, 0) + value


def get_metrics() -> Dict[str, Any]:
    """
    获取所有指标
    
    Returns:
        指标字典
    """
    with metrics_lock:
        return dict(metrics)


def reset_metrics() -> None:
    """重置所有指标"""
    global metrics
    with metrics_lock:
        metrics = {
            "requests_total": 0,
            "group_requests": 0,
            "private_requests": 0,
            "errors_total": 0,
            "response_times": [],
            "latencies": [],
        }
