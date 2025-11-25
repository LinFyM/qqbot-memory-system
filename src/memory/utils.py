# -*- coding: utf-8 -*-
"""
记忆系统工具函数
"""

import torch

def inject_memory_embedding_to_inputs_embeds(token_embeddings, embedding_positions, embeddings_to_insert, input_ids=None, memory_pad_token_id=None):
    """
    统一的记忆向量注入方法 - 静态函数，可被训练和推理代码共享

    Args:
        token_embeddings: [batch_size, seq_len, embed_dim] token embeddings
        embedding_positions: [batch_size] 或单个int，要注入的位置
        embeddings_to_insert: [batch_size, embed_dim] 或 [embed_dim] 要注入的向量
        input_ids: [batch_size, seq_len] token IDs，用于验证注入位置（可选）
        memory_pad_token_id: <|memory_pad|> token ID，用于验证注入位置（可选）

    Returns:
        修改后的token_embeddings
    """
    # 确保embeddings_to_insert是正确的形状
    if embeddings_to_insert.dim() == 1:
        # 单个向量，扩展为批次维度
        embeddings_to_insert = embeddings_to_insert.unsqueeze(0)

    # 确保embedding_positions是tensor
    if isinstance(embedding_positions, int):
        embedding_positions = torch.tensor([embedding_positions], device=token_embeddings.device)

    # 替换指定位置的embedding为记忆向量
    for i, pos in enumerate(embedding_positions):
        if i < embeddings_to_insert.size(0):
            # 保险检查：如果提供了input_ids和memory_pad_token_id，验证注入位置
            if input_ids is not None and memory_pad_token_id is not None:
                actual_token_id = input_ids[i, pos].item()
                if actual_token_id != memory_pad_token_id:
                    import logging
                    _log = logging.getLogger(__name__)
                    _log.warning(f"⚠️ [向量注入检查] 位置 {pos} 的token ID ({actual_token_id}) 不是预期的<|memory_pad|> token ID ({memory_pad_token_id})，但仍继续注入")

            token_embeddings[i, pos] = embeddings_to_insert[i]

    return token_embeddings
