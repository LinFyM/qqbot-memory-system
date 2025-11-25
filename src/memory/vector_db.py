# -*- coding: utf-8 -*-
"""
记忆向量数据库 - 只存储embedding，不存储文本
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import os

_log = logging.getLogger(__name__)


class MemoryVectorDB:
    """记忆向量数据库，只存储embedding向量（不存储文本）"""
    
    def __init__(self, embedding_dim=4096, device="cpu", max_size: int = 100000, enable_eviction: bool = True):
        """
        初始化向量数据库
        
        Args:
            embedding_dim: 向量维度
            device: 存储设备
            max_size: 允许存储的最大向量条数，超过后按命中次数淘汰
            enable_eviction: 是否启用淘汰策略
        """
        self.embedding_dim = embedding_dim
        self.embeddings = None  # 存储所有向量的tensor
        self.hit_counts = None  # 记录每条向量被命中的次数
        self.device = device
        self.primary_device = self.device[0] if isinstance(self.device, list) else self.device
        self.max_size = max_size
        self.enable_eviction = enable_eviction
        _log.info(f"初始化MemoryVectorDB (维度: {embedding_dim}, 设备: {device}, 上限: {max_size}, 淘汰: {enable_eviction})")
    
    def add_vectors(self, embeddings):
        """
        添加向量到数据库
        
        Args:
            embeddings: 向量tensor，形状为 [num_vectors, embedding_dim]
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        
        # 确保数据类型为bfloat16并移动到正确设备上
        embeddings = embeddings.to(dtype=torch.bfloat16, device=self.primary_device)
        
        # 归一化向量用于余弦相似度
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        new_count = embeddings.shape[0]
        if self.embeddings is None:
            self.embeddings = embeddings
            self.hit_counts = torch.zeros(new_count, device=self.primary_device, dtype=torch.long)
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
            new_hits = torch.zeros(new_count, device=self.primary_device, dtype=torch.long)
            self.hit_counts = torch.cat([self.hit_counts, new_hits], dim=0)

        # 超出上限时按命中次数淘汰
        if self.enable_eviction and len(self.embeddings) > self.max_size:
            self._evict_if_needed()

        _log.info(f"向量数据库现有 {len(self.embeddings)} 条记忆")
    
    def search(self, query_embedding, top_k=5, debug=False):
        """
        搜索最相似的向量
        
        Args:
            query_embedding: 查询向量，形状为 [embedding_dim] 或 [1, embedding_dim]
            top_k: 返回top_k个最相似的结果
            debug: 是否输出调试信息
        
        Returns:
            list: 包含相似度、embedding和索引的字典列表
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            if debug:
                _log.warning("记忆数据库为空")
            return []
        
        # 确保查询向量和存储向量在同一设备上且为相同数据类型
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding)
        
        if debug:
            _log.info(f"查询向量信息: 设备={query_embedding.device}, 形状={query_embedding.shape}, 数据类型={query_embedding.dtype}")
        
        # 移动查询向量到与存储向量相同的设备上
        query_embedding = query_embedding.to(dtype=torch.bfloat16, device=self.primary_device)
        
        # 确保查询向量有正确的维度
        if query_embedding.dim() == 1:
            # 如果是单个向量 [embed_dim]，添加批次维度
            query_embedding = query_embedding.unsqueeze(0)  # [1, embed_dim]
        
        # 归一化查询向量
        query_embedding_normalized = F.normalize(query_embedding, p=2, dim=-1)
        
        # 计算余弦相似度
        similarities = torch.matmul(query_embedding_normalized, self.embeddings.t())
        
        if debug:
            sim_mean = torch.mean(similarities).item()
            sim_std = torch.std(similarities).item()
            sim_max = torch.max(similarities).item()
            sim_min = torch.min(similarities).item()
            _log.info(f"相似度分布: 平均={sim_mean:.4f}, 标准差={sim_std:.4f}, 最大={sim_max:.4f}, 最小={sim_min:.4f}")
        
        # 获取top_k个最相似的结果
        top_k = min(top_k, len(self.embeddings))
        top_scores, top_indices = torch.topk(similarities, top_k, largest=True)
        
        # 处理维度，确保结果可迭代
        if top_scores.dim() == 1:
            top_scores = top_scores.unsqueeze(0)
            top_indices = top_indices.unsqueeze(0)
        
        results = []
        # 命中计数更新
        for i, (score, idx) in enumerate(zip(top_scores[0], top_indices[0])):
            idx_int = int(idx.item())
            result = {
                'embedding': self.embeddings[idx_int].clone(),
                'score': float(score.item()),
                'index': idx_int
            }
            results.append(result)
            if self.hit_counts is not None:
                # 累加命中次数
                self.hit_counts[idx_int] += 1
            if debug:
                _log.info(f"匹配结果 #{i+1}: 相似度={score.item():.4f}, 索引={idx_int}")
        
        return results
    
    def load_from_pt(self, pt_file_path):
        """
        从.pt文件加载向量数据（直接替换，不追加）
        
        Args:
            pt_file_path: .pt文件路径
        """
        if not os.path.exists(pt_file_path):
            _log.warning(f"记忆数据文件不存在: {pt_file_path}")
            return
        
        _log.info(f"从 {pt_file_path} 加载记忆数据...")
        data = torch.load(pt_file_path, map_location='cpu')
        
        if isinstance(data, dict):
            if 'embeddings' in data:
                embeddings = data['embeddings']
                hit_counts = data.get('hit_counts')
            else:
                # 尝试推断键名
                embedding_keys = [k for k in data.keys() if 'embed' in k.lower()]
                if embedding_keys:
                    embeddings = data[embedding_keys[0]]
                    hit_counts = data.get('hit_counts')
                else:
                    raise ValueError(f"无法从数据中识别嵌入向量字段: {list(data.keys())}")
        else:
            # 假设是直接的嵌入向量
            embeddings = data
            hit_counts = None
        
        # 直接替换，而不是追加
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        if hit_counts is not None and isinstance(hit_counts, np.ndarray):
            hit_counts = torch.from_numpy(hit_counts)
        
        embeddings = embeddings.to(dtype=torch.bfloat16, device=self.primary_device)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        self.embeddings = embeddings

        if hit_counts is not None:
            self.hit_counts = hit_counts.to(device=self.primary_device, dtype=torch.long)
            # 防御：长度不匹配则重置
            if len(self.hit_counts) != len(self.embeddings):
                _log.warning("命中计数长度与嵌入不一致，重置计数")
                self.hit_counts = torch.zeros(len(self.embeddings), device=self.primary_device, dtype=torch.long)
        else:
            self.hit_counts = torch.zeros(len(self.embeddings), device=self.primary_device, dtype=torch.long)

        # 如启用淘汰且超过上限，立即裁剪
        if self.enable_eviction and len(self.embeddings) > self.max_size:
            self._evict_if_needed()

        _log.info(f"成功加载 {len(self.embeddings)} 条记忆")
    
    def save_to_pt(self, pt_file_path):
        """
        保存向量数据到.pt文件
        
        Args:
            pt_file_path: 保存路径
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            _log.warning("没有记忆数据可保存")
            return
        
        os.makedirs(os.path.dirname(pt_file_path), exist_ok=True)
        
        save_data = {
            'embeddings': self.embeddings.cpu(),
            'embedding_dim': self.embedding_dim,
            'num_vectors': len(self.embeddings),
            'hit_counts': self.hit_counts.cpu() if self.hit_counts is not None else None,
            'max_size': self.max_size,
            'enable_eviction': self.enable_eviction,
        }
        
        torch.save(save_data, pt_file_path)
        _log.info(f"成功保存 {len(self.embeddings)} 条记忆到 {pt_file_path}")
    
    def __len__(self):
        """返回记忆数量"""
        return len(self.embeddings) if self.embeddings is not None else 0

    def _evict_if_needed(self):
        """如果超过上限，按命中次数淘汰最少的向量"""
        if self.embeddings is None or len(self.embeddings) <= self.max_size:
            return
        total = len(self.embeddings)
        remove_n = total - self.max_size
        if remove_n <= 0:
            return
        if self.hit_counts is None:
            self.hit_counts = torch.zeros(total, device=self.primary_device, dtype=torch.long)
        # 取命中次数最少的若干个
        _, remove_idx = torch.topk(self.hit_counts, k=remove_n, largest=False)
        keep_mask = torch.ones(total, device=self.primary_device, dtype=torch.bool)
        keep_mask[remove_idx] = False
        self.embeddings = self.embeddings[keep_mask]
        self.hit_counts = self.hit_counts[keep_mask]
        _log.info(f"触发淘汰策略: 移除 {remove_n} 条记忆，当前剩余 {len(self.embeddings)} 条")


