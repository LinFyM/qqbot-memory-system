# -*- coding: utf-8 -*-
"""
媒体相关工具函数
"""
import requests


def is_image_url_valid(image_url: str) -> bool:
    """
    检查图片URL是否有效（不下载完整内容，只检查是否能访问）
    
    Args:
        image_url: 图片URL
    
    Returns:
        是否有效
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

