# QQ聊天机器人长期记忆系统技术报告

## 1. 项目概述

### 1.1 主要内容

本项目是一个基于 **Qwen3-VL-4B-Thinking** 模型的QQ聊天机器人系统，集成了长期记忆存储与检索功能。系统采用客户端-服务器分离架构，实现了对话的长期记忆管理、自回归生成过程中的动态记忆召回，以及自动化的记忆提取与模型训练流程。

### 1.2 核心目标

1. **长期记忆存储与检索**：自动从对话历史中提取记忆条目，存储为向量表示，支持高效的相似度检索
2. **自回归生成中的动态记忆召回**：在模型生成回复的过程中，当检测到需要回忆时，动态检索并注入相关记忆
3. **自动化的训练流程**：实现两阶段训练流程，自动提取记忆、训练模型，并定期更新

### 1.3 主要功能

- **智能对话回复**：支持群聊和私聊消息的智能回复，能够理解上下文并生成合适的回复
- **记忆条目自动提取**：从聊天记录中自动提取关键记忆信息，存储为向量表示
- **两阶段训练流程**：
  - 第一阶段：训练 `<recall>` token的embedding，使其能够表示"需要回忆"的语义
  - 第二阶段：训练模型在看到 `<recall>` + 记忆向量时，生成对应的记忆文本
- **自回归生成中的记忆召回**：在生成过程中检测 `<recall>` token，动态检索并注入相关记忆
- **消息打断与任务管理**：支持新消息到达时中断当前生成任务，确保及时响应
- **自动化的训练调度**：使用定时任务自动执行训练，训练完成后自动重启服务器

---

## 2. 代码框架介绍

### 2.1 项目结构

```
qqbot_new/
├── server/                    # 服务器端代码
│   ├── api_server_qwen3vl.py  # 主API服务器（3972行）
│   ├── app.py                 # Flask应用入口
│   ├── config_qwen3vl.yaml    # 配置文件
│   ├── memory/                # 记忆系统模块
│   │   ├── training_service.py    # 训练服务（3097行）
│   │   ├── training_scheduler.py  # 训练调度器（254行）
│   │   ├── vector_db.py           # 向量数据库
│   │   ├── token_manager.py       # 特殊token管理
│   │   └── utils.py               # 工具函数
│   ├── routes/                # API路由
│   │   ├── chat.py            # 聊天接口
│   │   ├── training.py        # 训练接口
│   │   └── health.py          # 健康检查
│   ├── services/              # 业务服务
│   │   ├── generation.py      # 生成服务
│   │   ├── history.py         # 历史管理
│   │   └── queueing.py        # 消息队列
│   └── core/                  # 核心模块
│       ├── config.py          # 配置管理
│       ├── logging.py         # 日志管理
│       └── model.py           # 模型管理
├── recall/                    # 训练相关代码
│   ├── text_embedding_train.py  # 第一步训练（1542行）
│   ├── text_memory_train.py     # 第二步训练（2089行）
│   ├── model_utils.py            # 模型工具函数（76行）
│   └── get_text_embedding.py    # 向量提取工具
├── client/                    # QQ客户端代码
│   ├── qqbot_client_full.py  # 完整客户端实现
│   └── README.md             # 客户端说明
└── run.txt                   # 基本使用说明
```

### 2.2 技术栈

- **后端框架**：Flask（Python Web框架）
- **深度学习框架**：PyTorch + Transformers
- **基础模型**：Qwen3-VL-4B-Thinking（多模态大语言模型）
- **参数高效微调**：LoRA（Low-Rank Adaptation）
- **分布式训练**：Accelerate + DDP（可选）
- **QQ协议客户端**：ncatbot
- **向量检索**：基于余弦相似度的top-k检索
- **任务调度**：APScheduler

### 2.3 核心模块说明

#### 2.3.1 API服务器 (`server/api_server_qwen3vl.py`)

主API服务器，负责：
- 模型初始化和管理
- 消息接收和处理
- 自回归生成循环
- 记忆召回机制
- 消息打断处理

#### 2.3.2 训练服务 (`server/memory/training_service.py`)

训练服务模块，负责：
- 聊天记录的加载和处理
- 记忆条目的提取（使用模型生成）
- 记忆向量的提取
- 两阶段训练的协调
- SFT数据的穿插训练

#### 2.3.3 训练调度器 (`server/memory/training_scheduler.py`)

使用APScheduler实现定时训练：
- 定时触发训练任务
- 训练完成后的服务器重启
- 训练状态管理

#### 2.3.4 向量数据库 (`server/memory/vector_db.py`)

轻量级向量数据库：
- 存储记忆条目的embedding向量
- 支持top-k相似度检索
- 向量归一化处理

---

## 3. QQ Bot实现

### 3.1 客户端架构

系统采用**客户端-服务器分离架构**：

- **客户端**（个人电脑）：运行QQ协议客户端，负责接收和发送QQ消息
- **服务器**（GPU服务器）：运行LLM模型和训练服务，负责模型推理和训练

#### 3.1.1 连接方式

客户端通过 **SSH隧道** 连接到服务器：

```bash
# 客户端通过SSH隧道连接
ssh -L 9999:gpu02:9999 ymdai@210.75.240.172 -p 2277

# 客户端配置
SERVER_URL = "http://localhost:9999"  # 通过SSH隧道访问
```

#### 3.1.2 QQ协议客户端

使用 **ncatbot** 作为QQ协议客户端：

```python
from ncatbot.core import BotClient, GroupMessage, PrivateMessage
bot = BotClient()
```

### 3.2 消息处理流程

#### 3.2.1 消息接收

客户端监听QQ消息事件：

```python
@bot.on_group_message
async def handle_group_message(message: GroupMessage):
    # 处理群聊消息
    pass

@bot.on_private_message
async def handle_private_message(message: PrivateMessage):
    # 处理私聊消息
    pass
```

#### 3.2.2 消息转发

消息通过HTTP POST请求转发到服务器：

- **群聊消息**：`POST /api/chat/group`
- **私聊消息**：`POST /api/chat/private`

请求格式：
```json
{
    "group_id": "123456789",
    "user_id": "987654321",
    "user_nickname": "用户名",
    "content": "消息内容",
    "timestamp": 1234567890.0
}
```

#### 3.2.3 消息顺序控制

使用 **token机制** 确保消息按顺序处理：

```python
# 为每条消息分配唯一token
_message_token_lock = threading.Lock()
_latest_message_token: Dict[str, int] = {}
_message_token_counter = count(start=1)

def _get_message_token(chat_id: str) -> int:
    with _message_token_lock:
        token = next(_message_token_counter)
        _latest_message_token[chat_id] = token
        return token
```

#### 3.2.4 异步处理

使用线程池处理消息，避免阻塞NcatBot事件线程：

```python
_message_executor = ThreadPoolExecutor(max_workers=_max_message_workers)
```

### 3.3 消息打断机制

#### 3.3.1 中断信号

使用 `threading.Event` 实现消息打断：

```python
# 在服务器端
interrupt_event = threading.Event()

# 新消息到达时设置中断信号
if chat_id in processing_chats:
    old_interrupt = processing_chats[chat_id]["interrupt_event"]
    old_interrupt.set()  # 中断旧任务
```

#### 3.3.2 生成循环中的检查

在自回归生成循环中检查中断信号：

```python
while cur_len < max_new_tokens:
    # 检查中断信号
    if interrupt_event and interrupt_event.is_set():
        break  # 提前退出生成循环
    # ... 继续生成
```

#### 3.3.3 任务管理

使用 `processing_chats` 字典跟踪每个聊天的处理状态：

```python
processing_chats = {
    chat_id: {
        "interrupt_event": Event,
        "response_dict": dict,
        "lock": Lock
    }
}
```

当新消息到达时：
1. 检查是否有正在处理的任务
2. 如果有，设置中断信号
3. 创建新的处理任务
4. 旧任务检测到中断后自动退出

**参考代码**：
- `server/api_server_qwen3vl.py` 第1234-1240行：新消息到达时的中断处理
- `server/api_server_qwen3vl.py` 第2444行：生成循环中的中断检查
- `client/qqbot_client_full.py`：客户端消息处理逻辑

---

## 4. 模型推理方法

### 4.1 自回归生成流程

系统使用 `custom_generate` 函数实现完整的自回归生成循环，完全遵循transformers官方实现：

#### 4.1.1 生成循环结构

```python
def custom_generate(model, inputs, max_new_tokens, ...):
    # 初始化生成状态
    input_ids = inputs.get('input_ids')
    cur_len = input_ids.shape[-1]
    
    # 生成循环
    while cur_len < max_new_tokens:
        # 1. 准备模型输入（处理KV cache）
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        
        # 2. 前向传播
        outputs = _forward_with_last_hidden_state(model_inputs)
        
        # 3. 检查<recall> token（记忆召回机制）
        if 检测到<recall> token:
            执行记忆召回和注入
        
        # 4. 获取logits并采样下一个token
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = sample(next_token_logits)
        
        # 5. 更新input_ids和model_kwargs
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(...)
        
        cur_len += 1
```

#### 4.1.2 KV Cache支持

使用 `past_key_values` 实现KV cache，避免重复计算：

- 首次前向传播：计算所有token的KV cache
- 后续步骤：只计算新token的KV cache
- 使用 `model.prepare_inputs_for_generation()` 自动处理KV cache的裁剪和更新

### 4.2 记忆召回机制

#### 4.2.1 触发条件

在生成循环中，每生成一个token后检查：

```python
# 检查最新生成的token是否是<recall>
last_token_id = generated_ids[0, -1].item()
if last_token_id == recall_token_id:  # recall_token_id = 151669
    触发记忆召回机制
```

#### 4.2.2 向量提取

从 `<recall>` token的hidden state提取查询向量：

```python
# 使用backbone获取last_hidden_state
backbone_outputs = forward_backbone(
    model,
    inputs_embeds=current_embeddings,
    use_cache=True,
    output_hidden_states=False  # 关键：不输出所有hidden states
)

# 提取<recall> token的hidden state（最后一个token）
query_vector = backbone_outputs.last_hidden_state[0, -1, :]
```

**关键技术**：
- 使用 `forward_backbone()` 直接调用模型的backbone（`model.model`）
- 避免使用 `output_hidden_states=True`，减少显存占用
- 使用 `ensure_last_hidden_state()` 统一处理模型输出

#### 4.2.3 记忆检索

使用向量数据库进行top-k检索：

```python
search_results = memory_db.search(
    query_vector,
    top_k=10,  # 默认top_k=10
    debug=False
)
```

检索过程：
1. 计算查询向量与所有记忆向量的余弦相似度
2. 选择top-k个最相似的记忆向量
3. 返回候选记忆及其相似度分数

#### 4.2.4 记忆采样

对检索到的记忆进行采样选择：

**温度采样模式**（`use_sampling=True`）：
```python
# 计算概率分布
scores = torch.tensor([item['score'] for item in search_results])
probs = torch.softmax(scores / temperature, dim=-1)

# 可选：top-p截断
if 0 < top_p < 1.0:
    # 按概率排序，累积概率直到top_p
    # 只保留累积概率在top_p范围内的候选

# 采样选择
choice_idx = torch.multinomial(probs, num_samples=1).item()
```

**贪婪选择模式**（`use_sampling=False`）：
```python
choice_idx = torch.argmax(scores).item()
```

#### 4.2.5 向量注入

将选中的记忆向量注入到模型输入中：

**步骤1：添加 `<|memory_pad|>` token**

```python
# 在input_ids末尾添加<|memory_pad|> token
memory_pad_token_id = 151671
input_ids = torch.cat([
    input_ids,
    torch.tensor([[memory_pad_token_id]], device=device)
], dim=-1)

# 更新attention_mask
attention_mask = torch.cat([
    attention_mask,
    torch.ones(1, 1, device=device)
], dim=-1)
```

**步骤2：替换embedding**

```python
# 获取token embeddings
embedding_layer = model.get_input_embeddings()
token_embeddings = embedding_layer(input_ids)

# 使用统一注入方法替换<|memory_pad|>位置的embedding
injected_embeddings = inject_memory_embedding_to_inputs_embeds(
    token_embeddings,
    embedding_positions=[-1],  # 最后一个token位置
    embeddings_to_insert=memory_embedding,
    input_ids=input_ids,
    memory_pad_token_id=memory_pad_token_id
)
```

**步骤3：前向传播更新状态**

```python
# 使用注入后的embeddings进行前向传播
backbone_outputs = forward_backbone(
    model,
    inputs_embeds=injected_embeddings,
    attention_mask=attention_mask,
    past_key_values=model_kwargs.get('past_key_values'),
    use_cache=True
)

# 构建CausalLMOutputWithPast
memory_outputs = build_causal_lm_output(model, backbone_outputs)

# 更新model_kwargs（包括past_key_values）
model_kwargs = model._update_model_kwargs_for_generation(
    memory_outputs,
    model_kwargs,
    is_encoder_decoder=False
)
```

#### 4.2.6 继续生成

记忆注入后，模型继续自回归生成，此时：
- `past_key_values` 已包含记忆向量的KV cache
- 模型在生成时会"看到"记忆内容
- 生成的文本会自然地融入记忆信息

### 4.3 技术细节

#### 4.3.1 显存优化

**避免 `output_hidden_states=True`**：
- `output_hidden_states=True` 会返回所有层的hidden states，显存占用巨大
- 本项目只使用最后一层的hidden state，直接调用backbone获取 `last_hidden_state`

**使用backbone直接获取hidden state**：
```python
# 错误方式（显存占用大）
outputs = model(inputs, output_hidden_states=True)
hidden_states = outputs.hidden_states[-1]  # 所有层的hidden states都保存在内存中

# 正确方式（显存占用小）
backbone_outputs = forward_backbone(model, inputs, output_hidden_states=False)
last_hidden_state = backbone_outputs.last_hidden_state  # 只返回最后一层
```

**参考代码**：
- `recall/model_utils.py`：`forward_backbone()` 和 `ensure_last_hidden_state()` 函数
- `server/api_server_qwen3vl.py`：`_forward_with_last_hidden_state()` 辅助函数

#### 4.3.2 KV Cache管理

使用 `model.prepare_inputs_for_generation()` 自动处理：
- KV cache时的input_ids裁剪（只传入未缓存的token）
- attention_mask的正确长度和格式
- position_ids的处理
- cache_position的处理

#### 4.3.3 统一注入方法

`inject_memory_embedding_to_inputs_embeds()` 函数被训练和推理代码共享：
- 训练时：在 `EnhancedTextMemoryModel.forward()` 中使用
- 推理时：在 `_inject_memory_embedding()` 中使用
- 确保训练和推理时向量注入方式一致

**参考代码**：
- `server/memory/utils.py`：`inject_memory_embedding_to_inputs_embeds()` 函数
- `server/api_server_qwen3vl.py`：`_inject_memory_embedding()` 函数（第2375行）
- `recall/text_memory_train.py`：`EnhancedTextMemoryModel.forward()` 方法

---

## 5. 训练流程

### 5.1 训练流程概述

训练分为**两个阶段**：

1. **第一阶段：`<recall>` Token Embedding训练**
   - 目标：让 `<recall>` token的embedding能够表示"需要回忆"的语义
   - 输入：`原始文本 + <recall>`
   - 目标：对应的记忆向量（从记忆文本提取的embedding）
   - 损失：MSE Loss（`<recall>` token的hidden state vs 目标embedding）

2. **第二阶段：记忆文本解码训练**
   - 目标：让模型在看到 `<recall>` + 记忆向量时，生成对应的记忆文本
   - 输入：`[随机上下文] + [激活提示语] + <recall> + <|memory_pad|> + [记忆向量]`
   - 目标：`[-100...] + [<recall>的ID] + [-100] + [记忆文本] + </recall> + [结束提示语]`
   - 损失：CrossEntropy Loss（标准语言模型损失）

### 5.2 第一阶段：`<recall>` Token Embedding训练

#### 5.2.1 训练目标

训练 `<recall>` token的embedding，使其能够表示"需要回忆"的语义。当模型看到某个文本后生成 `<recall>` token时，该token的hidden state应该接近该文本对应的记忆向量。

#### 5.2.2 训练数据构建

**数据格式**：
- **输入序列**：`原始文本 + <recall>`
- **目标向量**：从记忆文本提取的embedding向量

**数据构建过程**：
```python
# 1. 从聊天记录中提取记忆条目（文本）
memory_texts = extract_memory_entries(chat_history)

# 2. 提取记忆文本的embedding向量
memory_embeddings = extract_embeddings(memory_texts)

# 3. 构建训练样本
for text, embedding in zip(memory_texts, memory_embeddings):
    input_text = f"{text}<recall>"
    target_embedding = embedding
```

**参考代码**：
- `recall/text_embedding_train.py`：`RecallDataset` 类（第16行）
- `server/memory/training_service.py`：记忆提取和向量提取逻辑

#### 5.2.3 损失函数

使用MSE Loss计算 `<recall>` token的hidden state与目标embedding的差异：

```python
def compute_loss(self, last_hidden_states, recall_positions, target_embeddings):
    """计算损失：recall token嵌入与目标嵌入的MSE损失"""
    # 批量提取<recall> token位置的hidden state
    recall_embeddings = last_hidden_states[
        torch.arange(last_hidden_states.size(0), device=last_hidden_states.device),
        recall_positions,
        :
    ]
    
    # 计算MSE损失
    loss = nn.MSELoss()(recall_embeddings, target_embeddings)
    return loss
```

#### 5.2.4 技术细节

**LoRA配置**：
- 只训练Q和V投影：`["q_proj", "v_proj"]`
- 减少约71%的LoRA参数，降低显存占用
- LoRA rank: 4, alpha: 8

**显存优化**：
- 使用backbone的 `last_hidden_state`，避免 `output_hidden_states=True`
- 使用梯度累积（`gradient_accumulation_steps`）
- 使用梯度检查点（`gradient_checkpointing_enable()`）
- 关闭训练时的KV cache（`use_cache=False`）

**训练参数**：
- Epochs: 30
- Batch size: 1
- Learning rate: 1e-4
- 梯度累积步数: 根据配置

**参考代码**：
- `recall/text_embedding_train.py`：完整的训练实现
- `recall/text_embedding_train.py`：`compute_loss` 方法
- `recall/text_embedding_train.py`：`_setup_lora` 方法

### 5.3 第二阶段：记忆文本解码训练

#### 5.3.1 训练目标

训练模型在看到 `<recall>` + 记忆向量时，生成对应的记忆文本。模型需要学习：
1. 忽略随机上下文的干扰
2. 根据记忆向量生成准确的记忆文本
3. 使用激活提示语触发回忆
4. 使用结束提示语结束回忆

#### 5.3.2 训练数据格式

**输入序列**：
```
[随机上下文] + [激活提示语] + <recall> + <|memory_pad|> + [记忆向量]
```

**目标序列（labels）**：
```
[-100...] + [<recall>的ID] + [-100] + [记忆文本] + </recall> + [结束提示语]
```

**说明**：
- `-100` 表示该位置的token不参与损失计算（CrossEntropyLoss的ignore_index）
- `<recall>` 的label设置为实际ID，让模型学习生成这个token
- `<|memory_pad|>` 的label为 `-100`，因为它是占位符，不需要生成
- 记忆文本和结束提示语是模型需要生成的内容

#### 5.3.3 关键技术

##### 5.3.3.1 随机上下文干扰

**目的**：训练模型忽略上下文干扰，只根据记忆向量生成文本

**实现方式**：
- 从SFT完整文本中随机选择一条
- 在思考部分内部随机截断（优先在句号后面）
- 返回从开始到截断点的全部内容作为上下文

**截断方法**：
```python
def _truncate_sft_at_thinking(self, sft_data: dict) -> str:
    """在SFT完整文本的思考部分内部随机截断"""
    full_text = sft_data["full_text"]
    thinking_start = sft_data["thinking_start"]
    thinking_end = sft_data["thinking_end"]
    
    # 提取思考部分内容
    thinking_content = full_text[thinking_start + len(start_tag):thinking_end - len(end_tag)]
    
    # 在token级别进行截断，优先在句号后面
    thinking_tokens = tokenizer(thinking_content, add_special_tokens=False)['input_ids']
    
    # 随机选择截断位置（优先在句号后面）
    if sentence_end_tokens:
        truncate_pos = random.choice(sentence_end_tokens)
    else:
        truncate_pos = random.randint(1, max_truncate_pos)
    
    # 返回截断后的文本
    truncated_text = full_text[:thinking_start + len(start_tag)] + truncated_thinking_text
    return truncated_text
```

**参考代码**：
- `recall/text_memory_train.py`：`_truncate_sft_at_thinking` 方法（第187行）

##### 5.3.3.2 激活/结束提示语

**目的**：让模型学习使用自然的提示语来触发和结束回忆

**配置**：
- 12种不同风格的激活提示语（如"【启动记忆检索】"、"我先唤醒记忆系统……"等）
- 12种不同风格的结束提示语（如"——回忆完成。"、"记忆内容同步完毕。"等）

**使用方式**：
- 每个训练样本随机选择一对激活/结束提示语
- 每个epoch重新随机选择，增加训练数据的多样性

**参考配置**：
- `server/config_qwen3vl.yaml`：`memory.training.guides` 部分（第263-289行）

##### 5.3.3.3 Embedding注入

在训练时，记忆向量通过embedding注入的方式加入到输入中：

```python
# 1. 构建输入序列（包含<|memory_pad|>占位符）
input_tokens = context_tokens + activation_tokens + [<recall>] + [<|memory_pad|>]

# 2. 获取token embeddings
token_embeddings = embedding_layer(input_tokens)

# 3. 替换<|memory_pad|>位置的embedding为记忆向量
injected_embeddings = inject_memory_embedding_to_inputs_embeds(
    token_embeddings,
    embedding_positions=[memory_pad_position],
    embeddings_to_insert=memory_embedding
)

# 4. 使用注入后的embeddings进行前向传播
outputs = model(inputs_embeds=injected_embeddings, ...)
```

**参考代码**：
- `recall/text_memory_train.py`：`EnhancedTextMemoryModel.forward()` 方法（第395行）
- `server/memory/utils.py`：`inject_memory_embedding_to_inputs_embeds()` 函数

##### 5.3.3.4 SFT穿插训练

**目的**：在记忆解码训练的同时，保持模型的通用对话能力

**实现方式**：
- 每个epoch的记忆解码训练后，穿插SFT训练
- 使用通用的SFT数据集（Chinese-Qwen3-235B-Thinking-2507-Distill-data-110k-SFT）
- 随机采样SFT样本进行训练

**训练流程**：
```python
for epoch in range(memory_epochs):
    # 1. 记忆解码训练
    train_memory_decoding()
    
    # 2. SFT穿插训练
    if sft_enabled and sft_per_epoch:
        train_sft_one_epoch()
```

**参考代码**：
- `server/memory/training_service.py`：`_run_sft_one_epoch()` 方法

#### 5.3.4 训练参数

- **Epochs**: 30
- **Batch size**: 1
- **Learning rate**: 1e-4
- **LoRA配置**：
  - Rank: 4, Alpha: 8
  - Target modules: `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **最大序列长度**: 3000 tokens
- **梯度累积**: 根据配置

**参考代码**：
- `recall/text_memory_train.py`：完整的训练实现
- `recall/text_memory_train.py`：`EnhancedTextMemoryDataset._get_memory_decode_sample()` 方法（第260行）

### 5.4 训练流程管理

#### 5.4.1 自动调度

使用 **APScheduler** 实现定时训练：

```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
# 每天凌晨3点执行一次训练
scheduler.add_job(
    train_job,
    'cron',
    hour=3,
    minute=0
)
scheduler.start()
```

**参考代码**：
- `server/memory/training_scheduler.py`：训练调度器实现

#### 5.4.2 数据准备流程

**步骤1：加载聊天记录**
```python
chat_history_files = load_chat_history_files(chat_history_storage_dir)
```

**步骤2：提取记忆条目**
```python
# 使用基础模型生成记忆条目
memory_entries = extract_memory_entries(chat_history)
# 格式：["用户LinF称呼我为'萝卜子'", "用户LinF的生日为11月6日", ...]
```

**步骤3：提取记忆向量**
```python
# 批量提取记忆条目的embedding向量
memory_embeddings = batch_extract_embeddings(memory_entries)
# 格式：torch.Tensor([num_entries, embedding_dim])
```

**步骤4：准备SFT数据**
```python
# 加载SFT数据集
sft_samples = load_sft_dataset(sft_dataset_path)
# 提取SFT文本的思考部分
sft_thinking_texts = extract_thinking_texts(sft_samples)
```

**参考代码**：
- `server/memory/training_service.py`：`_extract_memory_entries()` 方法
- `server/memory/training_service.py`：`_batch_extract_embeddings()` 方法

#### 5.4.3 训练执行

**第一步训练**：
```python
# 创建第一步训练器
trainer1 = RecallMemoryTrainer(
    model_name=base_model_path,
    device=device,
    lora_target_modules=["q_proj", "v_proj"],  # 只训练Q和V
    ...
)

# 执行训练
trained_model_path = trainer1.train(
    texts=memory_texts,
    target_embeddings=memory_embeddings,
    num_epochs=30,
    ...
)
```

**第二步训练**：
```python
# 创建第二步训练器
trainer2 = EnhancedTextMemoryTrainer(
    model_name=trained_model_path,  # 使用第一步训练后的模型
    device=device,
    lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", ...],  # 完整配置
    activation_prompts=activation_prompts,
    end_prompts=end_prompts,
    ...
)

# 执行训练
final_model_path = trainer2.train(
    texts=memory_texts,
    embeddings=memory_embeddings,
    num_epochs=30,
    sft_full_texts=sft_full_texts,  # SFT数据用于穿插训练
    ...
)
```

#### 5.4.4 模型保存与重启

**模型保存**：
- 第一步训练后：保存到 `token_added_model_dir`
- 第二步训练后：保存到 `trained_model_dir`
- 合并LoRA权重：`merge_lora=True`
- 保存完整VL资产：`save_full_vl_assets=True`

**自动重启**：
训练完成后，自动重启服务器以加载新模型：

```python
def restart_server():
    # 使用subprocess启动新进程
    new_process = subprocess.Popen(
        [python_exe, script_path],
        start_new_session=True
    )
    # 等待新进程启动
    time.sleep(2.0)
    # 强制退出当前进程，释放端口
    os._exit(0)
```

**参考代码**：
- `server/memory/training_scheduler.py`：`restart_server()` 方法

---

## 6. 引用与致谢

### 6.1 开源项目

本项目使用了以下开源项目和技术：

1. **ncatbot**
   - 用途：QQ协议客户端
   - 说明：用于接收和发送QQ消息，实现QQ机器人的基础功能

2. **Qwen3-VL-4B-Thinking**
   - 来源：ModelScope / Hugging Face
   - 用途：基础多模态大语言模型
   - 说明：本项目基于此模型进行微调，实现对话和记忆功能

3. **Chinese-Qwen3-235B-Thinking-2507-Distill-data-110k-SFT**
   - 用途：SFT（Supervised Fine-Tuning）训练数据集
   - 说明：用于穿插训练，保持模型的通用对话能力

4. **Hugging Face Transformers**
   - 用途：模型框架和工具库
   - 说明：提供模型加载、训练、推理等核心功能

5. **PEFT / LoRA**
   - 用途：参数高效微调
   - 说明：使用LoRA技术进行模型微调，大幅减少训练参数和显存占用

6. **Accelerate**
   - 用途：分布式训练和混合精度训练
   - 说明：支持多GPU训练和显存优化

7. **PyTorch**
   - 用途：深度学习框架
   - 说明：模型训练和推理的基础框架

8. **Flask**
   - 用途：Web框架
   - 说明：实现API服务器

9. **APScheduler**
   - 用途：任务调度
   - 说明：实现定时训练功能

### 6.2 技术参考

本项目在实现过程中参考了以下技术方案和最佳实践：

1. **Transformers官方实现**：自回归生成循环完全遵循transformers库的官方实现
2. **LoRA微调技术**：使用LoRA进行参数高效微调
3. **向量检索技术**：基于余弦相似度的top-k检索
4. **梯度检查点技术**：使用gradient checkpointing优化显存

### 6.3 项目信息

- **项目名称**：QQ聊天机器人长期记忆系统
- **项目版本**：当前版本
- **开发时间**：2024-2025
- **主要开发者**：ymdai

### 6.5 使用声明

**重要声明**：本项目仅供个人学习使用，不开放给研究用途或其他任何用途。未经授权，禁止用于任何商业、研究或其他目的。

### 6.4 致谢

感谢所有开源项目的开发者和维护者，他们的工作为本项目提供了坚实的基础。特别感谢：

- ncatbot项目提供稳定的QQ协议支持
- Qwen团队提供优秀的多模态大语言模型
- Hugging Face团队提供完善的模型框架和工具
- 所有为本项目提供帮助和建议的开发者

---

## 附录

### A. 配置文件说明

主要配置文件：`server/config_qwen3vl.yaml`

关键配置项：
- `model.path`：模型路径
- `model.device`：设备配置
- `memory.enabled`：是否启用记忆功能
- `memory.autoregressive_recall`：自回归回忆配置
- `memory.training`：训练配置

### B. 特殊Token说明

- `<recall>` (ID: 151669)：触发回忆的特殊token
- `</recall>` (ID: 151670)：结束回忆的特殊token
- `<|memory_pad|>` (ID: 151671)：记忆向量占位符token

### C. 关键文件路径

- API服务器：`server/api_server_qwen3vl.py`
- 训练服务：`server/memory/training_service.py`
- 训练调度器：`server/memory/training_scheduler.py`
- 第一步训练：`recall/text_embedding_train.py`
- 第二步训练：`recall/text_memory_train.py`
- 模型工具：`recall/model_utils.py`
- 向量数据库：`server/memory/vector_db.py`
- 客户端：`client/qqbot_client_full.py`

---

**文档版本**：1.0  
**最后更新**：2025年11月

