# -*- coding: utf-8 -*-
# 记忆模块聚合导出（惰性/可选导入训练相关模块）
from .token_manager import MemoryTokenManager
from .vector_db import MemoryVectorDB

# 训练相关为可选依赖，避免在包导入时强制加载
MemoryTrainingService = None
MemoryTrainingScheduler = None
try:
    from training.training_service import MemoryTrainingService as _MTS
    MemoryTrainingService = _MTS
except Exception:
    pass
try:
    from .training_scheduler import MemoryTrainingScheduler as _MTSch
    MemoryTrainingScheduler = _MTSch
except Exception:
    pass

__all__ = ["MemoryTokenManager", "MemoryVectorDB", "MemoryTrainingService", "MemoryTrainingScheduler"]

