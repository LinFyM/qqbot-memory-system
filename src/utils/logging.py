# -*- coding: utf-8 -*-
import logging

def setup_logging(level: str = "INFO") -> None:
    """
    薄封装：使用标准 logging 配置；若已有全局配置则不重复设置
    """
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


