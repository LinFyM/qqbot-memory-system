# -*- coding: utf-8 -*-
"""
媒体下载服务
处理图片、视频、音频等媒体文件的下载和存储
"""
from typing import Optional, Callable
import os
import mimetypes
import requests
from datetime import datetime
from uuid import uuid4
import re

Logger = object

def _guess_image_extension(content_type: Optional[str], fallback_url: str) -> str:
    if content_type:
        ct = content_type.lower()
        if 'jpeg' in ct or 'jpg' in ct:
            return 'jpg'
        if 'png' in ct:
            return 'png'
        if 'webp' in ct:
            return 'webp'
        if 'gif' in ct:
            return 'gif'
    guessed_type = mimetypes.guess_type(fallback_url)[0]
    if guessed_type:
        ext = mimetypes.guess_extension(guessed_type)
        if ext:
            return ext.lstrip('.')
    return 'jpg'

def _guess_video_extension(content_type: Optional[str], fallback_url: str) -> str:
    if content_type:
        ct = content_type.lower()
        if 'mp4' in ct:
            return 'mp4'
        if 'webm' in ct:
            return 'webm'
        if 'ogg' in ct or 'ogv' in ct:
            return 'ogg'
        if 'x-matroska' in ct or 'mkv' in ct:
            return 'mkv'
        if 'quicktime' in ct or 'mov' in ct:
            return 'mov'
        if 'x-msvideo' in ct or 'avi' in ct:
            return 'avi'
        if 'm4v' in ct:
            return 'm4v'
    guessed_type = mimetypes.guess_type(fallback_url)[0]
    if guessed_type:
        ext = mimetypes.guess_extension(guessed_type)
        if ext:
            return ext.lstrip('.')
    return 'mp4'

def _guess_audio_extension(content_type: Optional[str], fallback_url: str) -> str:
    if content_type:
        ct = content_type.lower()
        if 'mpeg' in ct or 'mp3' in ct:
            return 'mp3'
        if 'wav' in ct:
            return 'wav'
        if 'ogg' in ct:
            return 'ogg'
        if 'm4a' in ct or 'mp4a' in ct or 'aac' in ct:
            return 'm4a'
        if 'amr' in ct:
            return 'amr'
        if 'webm' in ct:
            return 'webm'
        if 'flac' in ct:
            return 'flac'
    guessed_type = mimetypes.guess_type(fallback_url)[0]
    if guessed_type:
        ext = mimetypes.guess_extension(guessed_type)
        if ext:
            return ext.lstrip('.')
    return 'wav'

def download_image_to_storage(image_url: str, image_dir: str, base_url: str,
                              metrics_add: Optional[Callable[[str, int], None]] = None,
                              logger: Optional[Logger] = None) -> Optional[str]:
    try:
        if logger: logger.info(f"尝试在服务器下载图片: {image_url}")
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        file_ext = _guess_image_extension(resp.headers.get("Content-Type", ""), image_url)
        filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{uuid4().hex}.{file_ext}"
        file_path = os.path.join(image_dir, filename)
        with open(file_path, "wb") as f:
            f.write(resp.content)
        file_url = f"{base_url.rstrip('/')}/static/images/{filename}"
        if logger: logger.info(f"✅ 图片缓存成功: {image_url} -> {file_url}")
        if metrics_add: metrics_add("image_cached", 1)
        return file_url
    except Exception as e:
        if logger: logger.warning(f"⚠️ 下载图片失败，使用原始URL: {image_url} ({e})")
        if metrics_add: metrics_add("image_cache_fail", 1)
        return None

def download_video_to_storage(video_url: str, video_dir: str, base_url: str,
                              metrics_add: Optional[Callable[[str, int], None]] = None,
                              logger: Optional[Logger] = None,
                              client_base_url: Optional[str] = None) -> Optional[str]:
    # 检测Windows本地路径（如 D:\... 或 \\...）
    is_windows_path = False
    if re.match(r'^[a-zA-Z]:\\', video_url) or re.match(r'^\\\\', video_url):
        is_windows_path = True
        if logger: logger.warning(f"⚠️ 检测到Windows本地路径，服务器无法直接访问: {video_url}")
        if logger: logger.info(f"💡 提示：客户端应该在发送消息前将本地文件上传到服务器")
        if metrics_add: metrics_add("video_cache_fail", 1)
        return None
    
    # 原有网络URL处理逻辑
    try:
        if logger: logger.info(f"尝试在服务器下载视频: {video_url}")
        resp = requests.get(video_url, timeout=30, stream=True)
        resp.raise_for_status()
        file_ext = _guess_video_extension(resp.headers.get("Content-Type", ""), video_url)
        filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{uuid4().hex}.{file_ext}"
        file_path = os.path.join(video_dir, filename)
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        file_url = f"{base_url.rstrip('/')}/static/videos/{filename}"
        if logger: logger.info(f"✅ 视频缓存成功: {video_url} -> {file_url}")
        if metrics_add: metrics_add("video_cached", 1)
        return file_url
    except Exception as e:
        if logger: logger.warning(f"⚠️ 下载视频失败，使用原始URL: {video_url} ({e})")
        if metrics_add: metrics_add("video_cache_fail", 1)
        return None

def download_audio_to_storage(audio_url: str, audio_dir: str, base_url: str,
                              metrics_add: Optional[Callable[[str, int], None]] = None,
                              logger: Optional[Logger] = None) -> Optional[str]:
    try:
        if logger: logger.info(f"尝试在服务器下载音频: {audio_url}")
        resp = requests.get(audio_url, timeout=30, stream=True)
        resp.raise_for_status()
        file_ext = _guess_audio_extension(resp.headers.get("Content-Type", ""), audio_url)
        filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{uuid4().hex}.{file_ext}"
        file_path = os.path.join(audio_dir, filename)
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        file_url = f"{base_url.rstrip('/')}/static/audios/{filename}"
        if logger: logger.info(f"✅ 音频缓存成功: {audio_url} -> {file_url}")
        if metrics_add: metrics_add("audio_cached", 1)
        return file_url
    except Exception as e:
        if logger: logger.warning(f"⚠️ 下载音频失败，使用原始URL: {audio_url} ({e})")
        if metrics_add: metrics_add("audio_cache_fail", 1)
        return None

def download_file_to_storage(file_url: str, file_dir: str, base_url: str,
                             metrics_add: Optional[Callable[[str, int], None]] = None,
                             logger: Optional[Logger] = None) -> Optional[str]:
    try:
        if logger: logger.info(f"尝试在服务器下载文件: {file_url}")
        resp = requests.get(file_url, timeout=30, stream=True)
        resp.raise_for_status()
        from urllib.parse import urlparse, unquote
        parsed = urlparse(file_url)
        basename = os.path.basename(parsed.path)
        basename = unquote(basename or "")
        if not basename or "." not in basename:
            ct = (resp.headers.get("Content-Type") or "").lower()
            default_ext = mimetypes.guess_extension(ct) or ".bin"
            basename = f"file{default_ext}"
        filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{uuid4().hex}_{basename}"
        safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', filename)
        file_path = os.path.join(file_dir, safe_name)
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        local_url = f"{base_url.rstrip('/')}/static/files/{safe_name}"
        if logger: logger.info(f"✅ 文件缓存成功: {file_url} -> {local_url}")
        if metrics_add: metrics_add("file_cached", 1)
        return local_url
    except Exception as e:
        if logger: logger.warning(f"⚠️ 下载文件失败，使用原始URL: {file_url} ({e})")
        if metrics_add: metrics_add("file_cache_fail", 1)
        return None


