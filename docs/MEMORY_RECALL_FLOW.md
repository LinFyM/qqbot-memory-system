# 记忆回忆流程详细说明

## 从生成 `<recall>` token 开始的完整流程

### 阶段划分

整个回忆流程可以分为以下几个阶段：
1. **进入回忆流程**：检测到 `<recall>` token
2. **回忆检索阶段**：检索并采样记忆向量
3. **记忆注入阶段**：将记忆向量注入到下一个位置
4. **退出回忆流程**：清除回忆相关标志，恢复正常自回归

---

## 详细流程

### 循环 N：生成 `<recall>` token（进入回忆流程的触发点）

#### 步骤 1：正常生成 `<recall>` token
```
1. 前向传播，获取 logits
2. 应用 logits_processor 和 logits_warper
3. 采样/贪婪选择，生成 token
4. 生成的 token 恰好是 <recall>
5. 将 <recall> token 添加到 input_ids
```

**关键点**：此时还没有进入回忆流程，只是正常生成了一个特殊 token。

---

### 循环 N+1：检测并处理回忆触发（回忆流程的核心）

#### 步骤 1：准备输入并检测回忆触发
```python
# 准备输入
model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

# 检测回忆触发：检查上一个循环生成的最后一个 token
current_input_ids = model_inputs.get('input_ids', input_ids)
last_token_id = current_input_ids[0, -1].item()  # 这是 <recall> token

if last_token_id == recall_token_id:
    recall_triggered = True  # 标记进入回忆流程
```

**关键点**：
- 检测发生在**前向传播之前**
- 通过检查 `input_ids` 的最后一个 token 来判断
- 设置 `recall_triggered = True` 作为进入回忆流程的标志

#### 步骤 2：前向传播（获取 `<recall>` 位置的 hidden state）
```python
# 前向传播
forward_inputs = dict(model_inputs)
outputs = _forward_with_last_hidden_state(forward_inputs)

# 获取关键信息
last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
logits = outputs.logits  # [batch, seq_len, vocab_size]

# 清除上一次循环设置的标志（如果有）
override_next_embed = None
forced_next_token_id = None
```

**关键点**：
- `last_hidden_state[0, -1, :]` 是 `<recall>` 位置的 hidden state
- 这个 hidden state 将作为查询向量用于记忆检索
- 清除上一次循环的标志，确保状态干净

#### 步骤 3：Memory Head - 检索记忆向量（输出 logits）
```python
if recall_triggered:
    # 提取查询向量
    query_vector = last_hidden_state[0, -1, :]  # [hidden_dim]
    
    # Memory Head：检索记忆向量，输出 logits
    # 检索所有记忆向量（与lm_head对所有vocab计算logits一致）
    memory_logits, memory_candidates = memory_head(
        query_vector=query_vector,
        memory_db=memory_db,
        debug=autorecall_debug
    )
```

**关键点**：
- `memory_head` 只输出 logits，不进行 softmax 和采样
- 与 `lm_head` 完全一致：都只输出 logits
- **检索所有向量**：
  - `memory_db.search` 会计算所有记忆向量的相似度（通过矩阵乘法）
  - 返回所有向量的相似度分数作为 logits
  - 与 `lm_head` 对所有 vocab_size 计算 logits 完全一致
  - 真正的 top-k 截断在 `logits_warper` 中进行

#### 步骤 4：统一处理流程 - 应用 logits_warper 和采样
```python
if memory_logits is not None and memory_candidates is not None:
    # 1. 应用 logits_warper（温度、top-k、top-p）
    if autorecall_use_sampling:
        memory_warper = LogitsProcessorList([
            TemperatureLogitsWarper(temperature=autorecall_temperature),
            TopKLogitsWarper(top_k=autorecall_top_k),  # 真正的 top-k 截断在这里
            TopPLogitsWarper(top_p=autorecall_top_p)
        ])
        memory_scores = memory_warper(dummy_input_ids, memory_logits.unsqueeze(0)).squeeze(0)
    
    # 2. 采样或贪婪选择
    if autorecall_use_sampling:
        probs = torch.nn.functional.softmax(memory_scores, dim=-1)
        choice_idx = torch.multinomial(probs, num_samples=1).item()
    else:
        choice_idx = torch.argmax(memory_scores).item()
    
    # 3. 获取选中的记忆向量
    selected = memory_candidates[choice_idx]
    memory_vector = selected['embedding']
```

**关键点**：
- 与 token 生成使用完全相同的处理流程
- top-k 截断在 `logits_warper` 中应用，不是在检索时
- 采样策略与 token 生成完全一致

#### 步骤 5：Memory Embedding - 准备记忆注入
```python
if memory_pad_token_id is not None:
    # 设置强制生成的 token（占位符）
    forced_next_token_id = memory_pad_token_id  # <|memory_pad|>
    
    # Memory Embedding：准备记忆向量
    override_next_embed = memory_embedding(
        memory_vector=memory_vector,
        model=model
    )  # 返回 [1, 1, hidden_dim]
    
    # 更新 attention mask
    model_kwargs['attention_mask'] = torch.cat([
        model_kwargs['attention_mask'],
        torch.ones((1, 1), device=..., dtype=...)
    ], dim=1)
```

**关键点**：
- `forced_next_token_id`：强制生成 `<|memory_pad|>` token（占位符）
- `override_next_embed`：记忆向量，将在下一次前向传播时使用
- 这两个标志将在**下一次循环**中生效

#### 步骤 6：生成下一个 token（此时还是正常流程）
```python
# 获取 token logits
next_token_logits = outputs.logits[:, -1, :]

# 应用 logits_processor 和 logits_warper
next_token_scores = logits_processor(input_ids, next_token_logits)
if do_sample and logits_warper is not None:
    next_token_scores = logits_warper(input_ids, next_token_scores)

# 生成 token（如果设置了 forced_next_token_id，强制生成）
if forced_next_token_id is not None:
    next_tokens = torch.full((batch_size,), forced_next_token_id, ...)  # 强制生成 <|memory_pad|>
else:
    # 正常采样/贪婪生成
    if do_sample:
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_tokens = torch.argmax(next_token_scores, dim=-1)

# 添加到序列
input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
```

**关键点**：
- 由于 `forced_next_token_id` 不为 None，强制生成 `<|memory_pad|>` token
- 这个 token 只是占位符，实际的 embedding 会被记忆向量覆盖

---

### 循环 N+2：记忆向量注入（退出回忆流程）

#### 步骤 1：准备输入（此时 `override_next_embed` 不为 None）
```python
# 准备输入
model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

# 检测回忆触发（此时最后一个 token 是 <|memory_pad|>，不是 <recall>）
last_token_id = current_input_ids[0, -1].item()
if last_token_id == recall_token_id:
    recall_triggered = True  # 这次是 False
else:
    recall_triggered = False  # 退出回忆流程
```

**关键点**：
- 此时最后一个 token 是 `<|memory_pad|>`，不是 `<recall>`
- `recall_triggered = False`，回忆流程结束

#### 步骤 2：前向传播（使用记忆向量，跳过 embedding 层）
```python
# 前向传播
forward_inputs = dict(model_inputs)

# 关键：如果 override_next_embed 不为 None，使用记忆向量
if override_next_embed is not None:
    forward_inputs["inputs_embeds"] = override_next_embed  # 记忆向量
    # 注意：此时不使用 input_ids，而是直接使用 inputs_embeds
    # 完全跳过 embedding 层！

outputs = _forward_with_last_hidden_state(forward_inputs)

# 清除标志（退出回忆流程）
override_next_embed = None  # 清除，恢复正常
forced_next_token_id = None  # 清除，恢复正常
```

**关键点**：
- `override_next_embed` 不为 None，直接使用记忆向量作为 `inputs_embeds`
- 跳过 embedding 层，记忆向量直接进入 backbone
- **清除标志**：`override_next_embed = None`，`forced_next_token_id = None`
- 这是**退出回忆流程的关键**：清除标志后，后续循环恢复正常自回归

#### 步骤 3：正常生成后续内容
```python
# 获取 token logits（基于记忆向量生成的内容）
next_token_logits = outputs.logits[:, -1, :]

# 正常处理（没有 forced_next_token_id）
next_token_scores = logits_processor(input_ids, next_token_logits)
if do_sample and logits_warper is not None:
    next_token_scores = logits_warper(input_ids, next_token_scores)

# 正常采样/贪婪生成
if do_sample:
    probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
else:
    next_tokens = torch.argmax(next_token_scores, dim=-1)

# 添加到序列
input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
```

**关键点**：
- 此时所有标志都已清除，完全恢复正常自回归
- 模型基于记忆向量继续生成后续内容
- 后续循环都是正常的自回归生成

---

## 进入和退出回忆流程的机制

### 进入回忆流程

**触发条件**：
```python
# 在每个循环开始时检测
recall_triggered = False  # 每个循环开始时自动重置为False
last_token_id = current_input_ids[0, -1].item()
if last_token_id == recall_token_id:
    recall_triggered = True  # 进入回忆流程
```

**进入标志**：
- `recall_triggered = True`：标记当前循环需要处理回忆

**自动退出机制**：
- `recall_triggered` 在每个循环开始时自动重置为 `False`
- 如果最后一个 token 不是 `<recall>`，则 `recall_triggered` 保持为 `False`
- 因此不需要显式设置为 `False`，它会自动退出

### 退出回忆流程

**退出机制**：
```python
# 在循环 N+2 的前向传播后
override_next_embed = None  # 清除记忆向量标志
forced_next_token_id = None  # 清除强制生成标志

# 在循环 N+3 开始时
recall_triggered = False  # 自动重置（因为最后一个token不是<recall>）
```

**退出标志**：
- `override_next_embed = None`：不再使用记忆向量（在循环 N+2 中清除）
- `forced_next_token_id = None`：不再强制生成特殊 token（在循环 N+2 中清除）
- `recall_triggered = False`：在循环 N+3 开始时自动重置（因为最后一个 token 不是 `<recall>`）

**自动退出**：
- `recall_triggered` 不需要显式设置为 `False`
- 它在每个循环开始时自动重置为 `False`
- 只有当最后一个 token 是 `<recall>` 时才会被设置为 `True`

### 如何确保不影响正常自回归

1. **标志的局部性**：
   - `override_next_embed` 和 `forced_next_token_id` 在每个循环开始时都会被清除
   - 只在需要时设置，使用后立即清除

2. **时序控制**：
   - 循环 N+1：设置标志（`override_next_embed`、`forced_next_token_id`）
   - 循环 N+2：使用标志（前向传播时使用记忆向量），然后清除标志
   - 循环 N+3 及之后：标志已清除，完全恢复正常

3. **条件检查**：
   - 只有在检测到 `<recall>` token 时才进入回忆流程
   - 只有在 `override_next_embed` 不为 None 时才使用记忆向量
   - 只有在 `forced_next_token_id` 不为 None 时才强制生成 token

4. **状态隔离**：
   - 回忆流程的状态（`memory_logits`、`memory_candidates`）只在当前循环有效
   - 不会影响后续循环的状态

---

## 流程图总结

```
循环 N：
  生成 <recall> token
  └─> 正常生成，添加到 input_ids

循环 N+1（进入回忆流程）：
  检测到 <recall> token
  ├─> 前向传播，获取 <recall> 位置的 hidden_state
  ├─> Memory Head：检索记忆向量，输出 logits
  ├─> 应用 logits_warper（温度、top-k、top-p）
  ├─> 采样/贪婪选择记忆向量
  ├─> Memory Embedding：准备记忆向量
  ├─> 设置 override_next_embed = 记忆向量
  ├─> 设置 forced_next_token_id = <|memory_pad|>
  └─> 强制生成 <|memory_pad|> token

循环 N+2（退出回忆流程）：
  检测到 <|memory_pad|> token（不是 <recall>）
  ├─> recall_triggered = False（退出回忆流程）
  ├─> 前向传播：使用 override_next_embed（记忆向量）
  │   └─> 跳过 embedding 层，记忆向量直接进入 backbone
  ├─> 清除 override_next_embed = None
  ├─> 清除 forced_next_token_id = None
  └─> 正常生成后续内容（基于记忆向量）

循环 N+3 及之后：
  完全恢复正常自回归
  └─> 所有标志已清除，正常生成
```

---

## 关键设计点

1. **两阶段设计**：
   - 循环 N+1：检索和准备记忆向量
   - 循环 N+2：注入记忆向量并清除标志

2. **标志驱动**：
   - 通过 `override_next_embed` 和 `forced_next_token_id` 控制流程
   - 标志的清除确保退出回忆流程

3. **完全兼容**：
   - 回忆流程不影响正常自回归
   - 退出后完全恢复正常生成

4. **统一处理**：
   - 记忆 logits 和 token logits 使用相同的处理流程
   - 确保行为一致性

