# -*- coding: utf-8 -*-
"""通用工具函数"""
from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parents[2]


def resolve_path(path_str: str, base_dir: Path = None) -> Path:
    """
    解析路径，支持相对路径和绝对路径
    
    Args:
        path_str: 路径字符串
        base_dir: 基础目录，默认为项目根目录
    
    Returns:
        解析后的绝对路径
    """
    if base_dir is None:
        base_dir = get_project_root()
    
    path = Path(path_str)
    if path.is_absolute():
        return path
    else:
        return (base_dir / path).resolve()

