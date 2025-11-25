# 自回归生成方法详细实现说明

## 概述

`custom_generate` 函数实现了带记忆机制的自回归文本生成。它保持了标准 Transformer 生成流程，同时将记忆检索视为特殊的"head"（Memory Head），将记忆向量视为特殊的嵌入（Memory Embedding）。

**核心设计理念**：
- **Memory Head**：通过 `memory_head()` 函数实现，类似于`lm_head`，用于从记忆库中选择记忆向量（通过softmax计算概率分布后采样）
- **Memory Embedding**：通过 `memory_embedding()` 函数实现，类似于`input_embeddings`，但直接使用记忆向量，跳过embedding层计算
- **对称设计**：记忆机制与标准自回归生成使用完全对称的框架（head + embedding）
- **统一采样**：使用相同的采样策略（softmax + 温度 + top-p），保持行为一致性
- **动态触发**：在生成过程中动态触发记忆检索，无需在输入阶段确定需要哪些记忆

**函数组织**：
- `memory_head()`: 实现记忆检索和采样逻辑，与`lm_head`对应
- `memory_embedding()`: 实现记忆向量准备和注入逻辑，与`input_embeddings`对应
- `custom_generate()`: 主生成函数，调用上述两个函数实现记忆机制

---

## 一、函数签名与参数

```python
def custom_generate(
    model,                    # 模型实例（Qwen3VLForConditionalGeneration）
    processor,                # 处理器实例（AutoProcessor）
    memory_db,                # 记忆向量数据库（MemoryVectorDB）
    recall_token_ids,         # 特殊token ID映射：{"<recall>": id, "</recall>": id, "<|memory_pad|>": id}
    config,                   # 配置字典
    inputs,                   # 输入字典：{"input_ids": tensor, "attention_mask": tensor}
    max_new_tokens: int,      # 最大生成token数
    stopping_criteria,        # 停止条件列表
    logits_processor,         # Logits处理器列表
    temperature: float,      # 温度参数（用于普通token生成采样）
    top_k: int,              # Top-k采样（用于普通token生成）
    top_p: float,            # Top-p采样（用于普通token生成）
    do_sample: bool,         # 是否使用采样（用于普通token生成）
    pad_token_id: int,       # Padding token ID
    eos_token_id: int,       # EOS token ID
    interrupt_event,         # 中断事件（用于消息打断）
    early_stop_on_tool_call: bool,  # 是否在工具调用时提前停止
)
```

**参数说明**：
- **普通token生成参数**：从`config["generation"]`中读取，通过函数参数传入：
  - `temperature`：普通token生成温度（默认从`config["generation"]["temperature"]`读取）
  - `top_k`：普通token生成top-k（默认从`config["generation"]["top_k"]`读取）
  - `top_p`：普通token生成top-p（默认从`config["generation"]["top_p"]`读取）
  - `do_sample`：是否使用采样（默认从`config["generation"]["do_sample"]`读取）
- **记忆向量采样参数**：从`config["memory"]["autoregressive_recall"]`中读取：
  - `autorecall_temperature`：记忆向量采样温度（应用在 logits warper 上）
  - `autorecall_top_p`：记忆向量采样top-p（应用在 logits warper 上）
  - `autorecall_top_k`：记忆向量采样top-k（应用在 logits warper 上，控制候选截断）
  - `autorecall_use_sampling`：是否使用采样选择记忆向量
```

---

## 二、初始化阶段

### 2.1 输入准备

```python
input_ids = inputs.get('input_ids')           # [batch_size, seq_len]
attention_mask = inputs.get('attention_mask') # [batch_size, seq_len]
batch_size = input_ids.shape[0]
cur_len = input_ids.shape[-1]                # 当前序列长度
unfinished_sequences = torch.ones(batch_size, ...)  # 标记哪些序列未完成
```

### 2.2 Logits处理配置

```python
# Logits Processor：用于修改logits（如重复惩罚）
logits_processor = LogitsProcessorList([...])

# Logits Warper：用于采样（温度、top-k、top-p）
if do_sample:
    logits_warper = LogitsProcessorList([
        TemperatureLogitsWarper(temperature),
        TopKLogitsWarper(top_k),
        TopPLogitsWarper(top_p),
    ])
```

### 2.3 模型状态管理

```python
# model_kwargs：存储模型状态（past_key_values、attention_mask等）
model_kwargs = {
    'attention_mask': attention_mask,
    'use_cache': True,
    'cache_position': torch.arange(...),  # 缓存位置索引
}
```

### 2.4 记忆配置

```python
memory_cfg = config.get("memory", {}).get("autoregressive_recall", {})
autorecall_enabled = memory_cfg.get("enabled", False)      # 是否启用自动回忆
autorecall_top_k = memory_cfg.get("top_k", 5)              # 检索top-k个候选
autorecall_temperature = memory_cfg.get("temperature", 1.0) # 记忆采样温度
autorecall_top_p = memory_cfg.get("top_p", 1.0)            # 记忆采样top-p
autorecall_use_sampling = memory_cfg.get("use_sampling", True)  # 是否采样选择记忆

recall_token_id = recall_token_ids.get("<recall>")         # <recall> token ID
memory_pad_token_id = recall_token_ids.get("<|memory_pad|>")  # <|memory_pad|> token ID
```

---

## 三、核心生成循环

### 3.1 循环结构

```python
while cur_len < max_new_tokens:
    # 1. 检查中断
    if interrupt_event and interrupt_event.is_set():
        break
    
    # 2. 准备模型输入
    model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
    
    # 3. 检查是否触发回忆
    recall_triggered = (last_token_id == recall_token_id)
    
    # 4. 前向传播
    outputs = _forward_with_last_hidden_state(forward_inputs)
    
    # 5. 处理回忆触发（如果发生）
    if recall_triggered:
        # 检索记忆向量并准备注入
        ...
    
    # 6. 获取下一个token
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens = sample_or_greedy(next_token_logits)
    
    # 7. 更新序列
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    
    # 8. 更新模型状态
    _update_model_kwargs_helper(outputs)
    
    # 9. 检查停止条件
    if should_stop:
        break
```

---

## 四、关键步骤详解

### 4.1 前向传播（`_forward_with_last_hidden_state`）

```python
def _forward_with_last_hidden_state(forward_inputs):
    # 1. 解包backbone（处理PEFT/Distributed包装）
    backbone = unwrap_backbone(model)
    
    # 2. 调用backbone的forward
    backbone_outputs = backbone(
        **forward_inputs,
        use_cache=True,
        output_hidden_states=True,  # 需要获取hidden states
        return_dict=True,
    )
    
    # 3. 获取最后一层的hidden state
    last_hidden_state = backbone_outputs.last_hidden_state  # [batch, seq, hidden_dim]
    
    # 4. 通过lm_head计算logits
    logits = model.lm_head(last_hidden_state)  # [batch, seq, vocab_size]
    
    # 5. 构造输出对象
    outputs = CausalLMOutputWithPast(
        logits=logits,
        past_key_values=backbone_outputs.past_key_values,
    )
    outputs.last_hidden_state = last_hidden_state  # 额外保存，用于记忆检索
    
    return outputs
```

**关键点**：
- 使用 `forward_backbone` 解包模型包装（PEFT、DDP等）
- 必须获取 `last_hidden_state`，用于记忆检索
- 通过 `lm_head` 计算词汇表logits

---

### 4.2 回忆触发检测

```python
# 在每次前向传播前检查（检查上一次生成的token）
current_input_ids = model_inputs.get('input_ids', input_ids)
if current_input_ids.shape[-1] > 0:
    last_token_id = current_input_ids[0, -1].item()
    if (
        autorecall_enabled
        and recall_token_id is not None
        and last_token_id == recall_token_id
    ):
        recall_triggered = True
```

**触发条件**：
- 自动回忆功能已启用
- 上一次生成的最后一个token是 `<recall>`

**时序说明**：
- 检测发生在**前向传播之前**（检查上一次生成的token）
- 如果检测到 `<recall>`，则在前向传播后立即检索记忆向量
- 记忆向量会在**下一次前向传播**时注入

---

### 4.3 记忆向量检索与采样（Memory Head）

**深度统一设计**：
- `memory_head` 函数完全模拟 `lm_head` 的行为
- **只输出logits**：不进行softmax和采样，与`lm_head`完全一致
- **统一处理流程**：在生成流程中，记忆logits和token logits使用完全相同的处理步骤
  - 应用 logits_processor（可选）
  - 应用 logits_warper（温度、top-k、top-p）
  - 进行 softmax 和采样/贪婪选择

```python
def memory_head(query_vector, memory_db, debug=False):
    """
    Memory Head: 从记忆库中检索记忆向量，输出logits（相似度分数）
    
    设计理念：完全类似于lm_head用于生成token。
    - lm_head: hidden_state -> logits (vocab_size) - 对所有vocab计算logits
    - memory_head: query_vector -> logits (memory_candidates) - 对所有记忆向量计算相似度
    
    与lm_head完全一致：
    - 都计算所有候选的分数（vocab_size 或 所有记忆向量）
    - 都只输出logits，不进行softmax和采样
    - softmax和采样将在生成流程中统一处理
    - top-k截断在logits_warper中进行，与token生成完全一致
    
    Args:
        query_vector: 查询向量 [hidden_dim]，来自<recall>位置的last_hidden_state
        memory_db: 记忆向量数据库
        debug: 是否输出调试信息
    
    Returns:
        memory_logits: 记忆向量的logits（相似度分数）[num_candidates]
        memory_candidates: 候选记忆向量列表
    """
    # 1. 向量相似度搜索（类似attention机制）
    # 注意：检索所有记忆向量（与lm_head对所有vocab计算logits一致）
    # memory_db.search内部会计算所有向量的相似度，然后返回所有结果
    # 真正的top-k截断在logits_warper中进行
    search_results = memory_db.search(
        query_vector.detach().clone(),
        top_k=len(memory_db),  # 检索所有向量，与lm_head对所有vocab计算logits一致
        debug=debug
    )
    # search_results格式：
    # [
    #     {'embedding': tensor, 'score': float, 'index': int},
    #     ...
    # ]
    
    # 2. 提取logits（相似度分数）- 与lm_head输出logits完全一致
    memory_logits = torch.tensor(
        [item['score'] for item in search_results],
        dtype=torch.float32,
        device=query_vector.device
    )
    
    # 3. 返回logits和候选列表（不进行softmax和采样）
    return memory_logits, search_results
```

**关键点**：
- **完全对称设计**：`memory_head` 与 `lm_head` 完全对称
  - `lm_head`: `hidden_state` → `logits` (vocab_size)
  - `memory_head`: `query_vector` → `logits` (memory_candidates)
- **只输出logits**：不进行softmax和采样，与`lm_head`完全一致
- **统一处理流程**：记忆logits和token logits在生成流程中使用完全相同的处理步骤
- **查询向量**：使用 `<recall>` 位置的 `last_hidden_state` 作为查询向量
- **相似度计算**：通过余弦相似度在记忆库中搜索（归一化的向量点积）

**统一处理流程**（在生成循环中）：
```python
# 记忆检索处理（与token生成完全统一）
if memory_logits is not None:
    # 1. 应用logits warper（温度、top-k、top-p）- 与token生成完全一致
    if autorecall_use_sampling:
        memory_warper = LogitsProcessorList([
            TemperatureLogitsWarper(temperature=autorecall_temperature),
            TopKLogitsWarper(top_k=autorecall_top_k),  # top-k在warper中应用
            TopPLogitsWarper(top_p=autorecall_top_p)
        ])
        memory_scores = memory_warper(dummy_input_ids, memory_logits)
    
    # 2. 采样或贪婪选择（与token生成完全一致）
    if autorecall_use_sampling:
        probs = torch.nn.functional.softmax(memory_scores, dim=-1)
        choice_idx = torch.multinomial(probs, num_samples=1).item()
    else:
        choice_idx = torch.argmax(memory_scores).item()
    
    # 3. 获取选中的记忆向量
    selected = memory_candidates[choice_idx]

# Token生成处理（与记忆检索完全统一）
next_token_logits = outputs.logits[:, -1, :]
next_token_scores = logits_processor(input_ids, next_token_logits)
if do_sample and logits_warper is not None:
    next_token_scores = logits_warper(input_ids, next_token_scores)  # 包含TopKLogitsWarper
if do_sample:
    probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
else:
    next_tokens = torch.argmax(next_token_scores, dim=-1)
```

**关键点**：
- **对称流程**：记忆和token都使用相同的logits_warper处理（温度、top-k、top-p）
- **top-k 的位置**：top-k 仅在 `logits_warper` 中应用，用于在采样前截断候选，与 token 生成完全一致

---

### 4.4 记忆向量注入（Memory Embedding）

**实现说明**：
- 代码中通过 `memory_embedding` 函数实现 Memory Embedding 功能
- 该函数准备记忆向量，调整形状和设备，返回可直接作为 `inputs_embeds` 使用的张量
- **关键**：当 `override_next_embed` 不为 None 时，`forward_backbone` 函数会直接将 `inputs_embeds` 传入 backbone，**完全跳过 embedding 层**

```python
def memory_embedding(memory_vector, model, device=None, dtype=None):
    """
    Memory Embedding: 准备记忆向量用于注入，跳过embedding层
    
    设计理念：将记忆向量视为特殊的embedding，类似于input_embeddings用于token ID，
    但这里直接使用记忆向量，不经过embedding层计算。记忆向量是已经计算好的hidden state，
    直接作为下一个位置的输入。
    
    Args:
        memory_vector: 记忆向量 [hidden_dim] 或 [1, hidden_dim] 或 [1, 1, hidden_dim]
        model: 模型实例（用于获取设备和数据类型）
        device: 目标设备（如果为None，则从model获取）
        dtype: 目标数据类型（如果为None，则从model获取）
    
    Returns:
        memory_embedding: 准备好的记忆向量 [1, 1, hidden_dim]，可直接作为inputs_embeds使用
    """
    # 调整形状为 [1, 1, hidden_dim]
    if memory_vector.dim() == 1:
        memory_vector = memory_vector.unsqueeze(0)  # [1, hidden_dim]
    if memory_vector.dim() == 2:
        memory_vector = memory_vector.unsqueeze(0)  # [1, 1, hidden_dim]
    
    # 获取设备和数据类型
    if device is None or dtype is None:
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        if device is None:
            device = model_device
        if dtype is None:
            dtype = model_dtype
    
    # 移动到目标设备和数据类型
    memory_embedding = memory_vector.to(device=device, dtype=dtype)
    return memory_embedding

# 在 custom_generate 中的使用
if recall_triggered:
    # 1. 获取查询向量（<recall>位置的hidden state）
    query_vector = last_hidden_state[0, -1, :]  # [hidden_dim]
    
    # 2. Memory Head：检索并采样记忆向量
    memory_logits, memory_candidates = memory_head(
        query_vector=query_vector,
        memory_db=memory_db,
        debug=autorecall_debug
    )
    
    if memory_logits is not None:
        scores = apply_memory_logits_warper(memory_logits, autorecall_temperature, autorecall_top_k, autorecall_top_p)
        selected_idx = sample_or_greedy(scores, use_sampling=autorecall_use_sampling)
        selected_memory = memory_candidates[selected_idx]
        forced_next_token_id = memory_pad_token_id
        override_next_embed = memory_embedding(selected_memory["embedding"], model=model)
        
        # 5. 更新attention mask（为新的token位置添加mask）
        model_kwargs['attention_mask'] = torch.cat([
            model_kwargs['attention_mask'],
            torch.ones((1, 1), device=..., dtype=...)
        ], dim=1)

# 在前向传播时
forward_inputs = dict(model_inputs)
if override_next_embed is not None:
    forward_inputs["inputs_embeds"] = override_next_embed  # 直接使用记忆向量
outputs = _forward_with_last_hidden_state(forward_inputs)  # 传入backbone，跳过embedding层
```

**Memory Embedding机制**：
1. **跳过embedding层**：当 `override_next_embed` 不为 None 时，`forward_inputs["inputs_embeds"]` 直接设置为记忆向量
2. **直接传入backbone**：`forward_backbone` 函数会将 `inputs_embeds` 直接传入模型的 backbone，**完全跳过 `get_input_embeddings()` 的计算**
3. **验证**：在 `forward_backbone` 中，如果提供了 `inputs_embeds`，模型不会调用 embedding 层，而是直接使用提供的 embeddings
4. **占位符token**：`<|memory_pad|>` token仅作为占位符，标记记忆向量的位置，其embedding会被记忆向量覆盖
5. **与训练一致**：训练时也是直接注入记忆向量，不经过embedding层

**设计理念**：
- **Memory Head**：类似于`lm_head`，用于从记忆库中选择记忆向量（通过softmax采样）
- **Memory Embedding**：类似于`input_embeddings`，但直接使用记忆向量，不通过token ID查找
- **统一框架**：记忆机制与标准自回归生成使用相同的框架（head + embedding），只是数据来源不同

---

### 4.5 下一个Token生成

```python
# 1. 获取logits（最后一个位置的logits）
next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

# 2. 应用logits processor（如重复惩罚）
next_token_scores = logits_processor(input_ids, next_token_logits)

# 3. 应用logits warper（采样相关）
if do_sample and logits_warper is not None:
    next_token_scores = logits_warper(input_ids, next_token_scores)

# 4. 生成token（强制或采样/贪婪）
if forced_next_token_id is not None:
    # 强制生成（用于记忆注入）
    next_tokens = torch.full((batch_size,), forced_next_token_id, ...)
else:
    if do_sample:
        # 采样生成
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        # 贪婪生成
        next_tokens = torch.argmax(next_token_scores, dim=-1)

# 5. 处理EOS
if has_eos_stopping_criteria:
    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

# 6. 添加到序列
input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
```

---

### 4.6 模型状态更新

```python
def _update_model_kwargs_helper(outputs_obj):
    """更新model_kwargs，主要是past_key_values"""
    model_kwargs = model._update_model_kwargs_for_generation(
        outputs_obj,
        model_kwargs,
        is_encoder_decoder=False,
        standardize_cache_format=True,
    )
```

**更新内容**：
- `past_key_values`：KV缓存，避免重复计算
- `cache_position`：缓存位置索引
- `attention_mask`：注意力掩码（已扩展）

---

### 4.7 停止条件检查

```python
# 1. EOS检查
if eos_token_ids is not None:
    eos_in_sentence = (next_tokens.unsqueeze(-1) == eos_token_ids.unsqueeze(0)).any(dim=-1)
    unfinished_sequences = unfinished_sequences & ~eos_in_sentence

# 2. StoppingCriteria检查
should_stop = stopping_criteria(input_ids, next_token_scores)
unfinished_sequences = unfinished_sequences & ~should_stop_tensor

# 3. 所有序列完成
if unfinished_sequences.max() == 0:
    break

# 4. 工具调用提前停止（可选）
if early_stop_on_tool_call:
    decoded_so_far = processor.batch_decode(input_ids, ...)
    if "</tool_call>" in decoded_so_far:
        break
```

---

## 五、记忆注入的完整流程

### 5.1 触发阶段

```
循环N: 生成 <recall> token
         ↓
循环N+1开始:
  检测到: last_token_id == recall_token_id (上一次生成的token)
         ↓
  recall_triggered = True
         ↓
  前向传播: 获取 <recall> 位置的 hidden state
         ↓
  检索记忆向量
         ↓
  设置: override_next_embed = memory_embedding
        forced_next_token_id = memory_pad_token_id
```

**关键时序**：
1. 在循环N中，模型生成了 `<recall>` token
2. 在循环N+1开始时，检测到上一次生成的是 `<recall>`
3. 在循环N+1的前向传播中，获取 `<recall>` 位置的 hidden state
4. 在循环N+1的前向传播后，检索记忆向量并设置注入参数
5. 在循环N+2中，使用 `override_next_embed` 注入记忆向量

### 5.2 检索阶段

```
获取查询向量: query_vector = last_hidden_state[0, -1, :]
         ↓
向量搜索: memory_db.search(query_vector, top_k=5)
         ↓
候选记忆: [
    {embedding: vec1, score: 0.95},
    {embedding: vec2, score: 0.87},
    ...
]
         ↓
温度采样: 选择 embedding (概率分布)
```

### 5.3 注入阶段

```
强制生成: forced_next_token_id = memory_pad_token_id
         ↓
覆盖embedding: override_next_embed = memory_embedding
         ↓
前向传播: 
  - 使用 override_next_embed 替代 <|memory_pad|> 的embedding
  - 模型继续生成，基于记忆向量生成后续内容
```

### 5.4 生成阶段

```
模型基于记忆向量生成: 
  <|memory_pad|> [记忆向量] → 生成记忆内容 → </recall> → 继续生成
```

---

## 六、与标准generate的区别

| 特性 | 标准generate | custom_generate |
|------|-------------|----------------|
| **记忆机制** | ❌ 无 | ✅ 动态记忆检索与注入 |
| **Hidden States** | ❌ 不保存 | ✅ 保存last_hidden_state |
| **Embedding覆盖** | ❌ 不支持 | ✅ 支持override_next_embed |
| **强制Token生成** | ❌ 不支持 | ✅ 支持forced_next_token_id |
| **向量检索** | ❌ 无 | ✅ 集成MemoryVectorDB |
| **采样策略** | 单一 | 双重（token采样 + 记忆采样） |

---

## 七、关键技术细节

### 7.1 为什么使用`override_next_embed`？

**原因**：
- **记忆向量直接作为embedding**：记忆向量是从backbone的`last_hidden_state`提取的，维度与模型的`hidden_size`相同（例如4096维）
- **跳过embedding层**：使用`override_next_embed`可以直接将记忆向量作为`inputs_embeds`传入，跳过`<|memory_pad|>` token通过embedding层的计算
- **与训练一致**：模型在训练时已经学会了如何处理直接注入的记忆向量（通过`memory_embedding`机制）
- **设计理念**：将记忆向量视为特殊的embedding，而非通过token ID查找embedding

**实现**：
```python
if override_next_embed is not None:
    forward_inputs["inputs_embeds"] = override_next_embed
    # 注意：此时不使用input_ids，而是直接使用inputs_embeds
    # 记忆向量直接作为embedding输入，跳过embedding层
```

### 7.2 为什么需要`forced_next_token_id`？

**原因**：
- **形式统一**：在标准自回归生成中，每个位置都有一个对应的token ID。当`<recall>` token触发记忆检索时，下一个位置需要一个占位符token来标记记忆向量的插入位置
- **占位符作用**：`<|memory_pad|>` token作为占位符，标记记忆向量应该插入的位置，但实际的embedding会被记忆向量覆盖（通过`override_next_embed`）
- **序列一致性**：保持序列的token结构完整，便于后续处理和日志记录
- **训练对齐**：与训练时的格式保持一致，训练时也是使用`<|memory_pad|>`标记记忆向量位置

### 7.3 记忆向量的维度

```
查询向量: [hidden_dim]  (来自backbone的last_hidden_state，例如4096维)
         ↓
记忆向量: [hidden_dim]  (存储在MemoryVectorDB中，维度与hidden_size相同)
         ↓
注入方式: 直接作为inputs_embeds传入，跳过embedding层
         ↓
模型处理: 通过backbone处理，输出新的hidden states
```

**维度说明**：
- `input_embeddings`的维度：`[vocab_size, hidden_size]`，其中`hidden_size`是模型的隐藏层维度
- `last_hidden_state`的维度：`[batch, seq_len, hidden_size]`
- **记忆向量的维度**：`[hidden_size]`，与`last_hidden_state`的最后一维相同
- **维度一致性**：记忆向量、`last_hidden_state`、`input_embeddings`的`hidden_size`维度都是相同的

### 7.4 KV缓存的维护

```python
# 每次前向传播后更新
model_kwargs = model._update_model_kwargs_for_generation(
    outputs,
    model_kwargs,
    ...
)
# 更新past_key_values，避免重复计算已生成的token
```

---

## 八、性能优化

### 8.1 缓存机制

- **KV缓存**：使用 `past_key_values` 避免重复计算
- **Cache Position**：精确跟踪缓存位置，支持动态长度

### 8.2 批量处理

- 支持batch_size > 1的批量生成
- 每个样本独立处理记忆检索

### 8.3 中断机制

- 通过 `interrupt_event` 支持实时中断
- 用于消息打断功能

---

## 九、配置示例

```yaml
memory:
  autoregressive_recall:
    enabled: true           # 启用自动回忆
    top_k: 5                # 检索top-5个候选记忆
    temperature: 1.0        # 记忆采样温度
    top_p: 0.9              # 记忆采样top-p
    use_sampling: true      # 使用采样选择记忆
    debug: false            # 是否输出调试信息
```

---

## 十、设计理念总结

### 10.1 Memory Head 与 Memory Embedding

`custom_generate` 的核心设计理念是将记忆机制视为与标准自回归生成**完全对称**的组件：

| 组件 | 标准自回归生成 | 记忆机制 |
|------|--------------|---------|
| **Head** | `lm_head`: hidden_state → logits (vocab_size) | **Memory Head**: hidden_state → logits (memory_candidates) |
| **输入** | 最后一个位置的 hidden_state | `<recall>` 位置的 hidden_state |
| **输出** | logits（未归一化） | logits（未归一化，相似度分数） |
| **处理流程** | 1. logits_processor<br>2. logits_warper (温度、top-k、top-p)<br>3. softmax + 采样/贪婪 | 1. （可选）logits_processor<br>2. logits_warper (温度、top-p)<br>3. softmax + 采样/贪婪 |
| **采样** | softmax + 温度采样，从vocab中选择token | softmax + 温度采样，从记忆库中选择记忆向量 |
| **Embedding** | `input_embeddings`: 通过token ID查找embedding | **Memory Embedding**: 直接使用记忆向量，跳过embedding层 |
| **输入来源** | token ID → embedding → hidden state | 记忆向量 → hidden state（直接） |

### 10.2 核心创新

1. **对称设计**：Memory Head 和 Memory Embedding 与标准生成组件完全对称
2. **统一框架**：使用相同的采样策略（softmax + 温度 + top-p），保持一致性
3. **无缝集成**：记忆机制完全集成在标准生成流程中，无需修改模型结构
4. **动态触发**：通过检测 `<recall>` token动态触发记忆检索
5. **直接注入**：记忆向量直接作为embedding输入，不经过embedding层计算

### 10.3 与标准generate的区别

| 特性 | 标准generate | custom_generate |
|------|-------------|----------------|
| **Head数量** | 1个（lm_head） | 2个（lm_head + Memory Head） |
| **Embedding来源** | 仅token ID | token ID + 记忆向量 |
| **采样对象** | 仅vocab | vocab + 记忆库 |
| **Hidden States** | ❌ 不保存 | ✅ 保存last_hidden_state（用于Memory Head） |
| **Embedding覆盖** | ❌ 不支持 | ✅ 支持Memory Embedding（override_next_embed） |
| **强制Token生成** | ❌ 不支持 | ✅ 支持（用于占位符token） |

这种设计使得模型可以在生成过程中自然地"回忆"相关信息，而无需在输入阶段就确定需要哪些记忆。记忆检索和token生成使用相同的概率采样框架，保证了行为的一致性。

