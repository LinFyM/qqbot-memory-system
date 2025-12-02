# 萝卜子QQ机器人 v2.2

基于Qwen3-VL的多模态QQ机器人，支持记忆机制和持续学习。

## 快速开始

1. 获取代码并安装依赖
   ```bash
   git clone https://github.com/LinFyM/Episodic-Memory-Chatbot.git
   cd Episodic-Memory-Chatbot

   # 推荐使用新的虚拟环境
   conda create -n qqbot python=3.10 -y
   conda activate qqbot
   pip install -r requirements.txt
   ```
2. 准备配置文件（可直接编辑 `configs/config_qwen3vl.yaml` 与 `configs/prompts.yaml`）
3. 启动服务
   ```bash
   python scripts/run_server.py
   ```

默认会监听 `configs/config_qwen3vl.yaml` 中的 `server.host`/`server.port`（初始值 `0.0.0.0:9999`）。

## 项目结构

```
qqbot_new/
├── configs/          # 配置文件
│   └── config_qwen3vl.yaml
├── src/             # 主代码
│   ├── api/         # API层（Flask应用、路由、状态管理）
│   ├── chat/        # 聊天逻辑（生成、历史、队列、prompting）
│   ├── memory/      # 记忆管理（向量库、token、训练调度）
│   ├── training/    # 训练模块（模型训练、向量提取）
│   ├── services/    # 服务层（ASR、媒体、提取器）
│   └── utils/       # 工具函数（CQ码、日志、指标）
├── scripts/         # 启动脚本
├── models/          # 模型和数据
├── data/            # 数据集
└── client/          # QQ客户端
```

## 配置

编辑 `configs/config_qwen3vl.yaml`：

```yaml
server:
  host: "0.0.0.0"
  port: 9999

model:
  path: "./models/Qwen3-VL-4B-Thinking"
  device: "cuda:5"

memory:
  enabled: true
  memory_db_path: "./models/memory_db/memory_embeddings.pt"
  training:
    enabled: true
    schedule: "2"  # 训练时间：凌晨2点
    first_training_delay_days: 2  # 程序启动后，第x天的凌晨开始第一次训练
    training_interval_days: 3  # 训练间隔天数（每3天训练一次）
    embedding_model_path: "./models/Qwen3-Embedding-4B"  # Embedding模型路径（用于提取记忆和SFT向量）
```

### 提示词配置

所有 prompt 文案都集中在 `configs/prompts.yaml`，便于统一查看与修改。该文件分为：

- `chat`：系统提示词、角色扮演、动作指导、拼接顺序等
- `memory_training`：记忆激活语、结束语以及训练时的引导文字
- `memory_extraction`：记忆条目提取的 system prompt、递归子树提示、媒资描述提醒、用户指令
- `memory_vectorization`：批量向量提取时的文本压缩提示

如果需要调整提示词，直接编辑该文件后重启/重新加载配置即可。

## 客户端部署

本项目采用**客户端-服务器分离架构**：

- **服务器端**：运行在GPU服务器上，负责LLM推理、记忆管理、训练等
- **客户端**：运行在个人电脑上，负责连接QQ并转发消息到服务器

### 客户端安装与配置

1. **安装依赖**（在个人电脑上）：
   ```bash
   pip install ncatbot requests flask-cors
   ```

2. **配置客户端**：
   编辑 `client/qqbot_client_full.py`，修改以下配置：
   ```python
   SERVER_URL = "http://your-server-ip:9999"  # 替换为实际服务器IP
   config.set_bot_uin("你的QQ号")
   config.set_ws_uri("ws://localhost:3001")  # napcat的WebSocket地址
   config.set_token("你的token")
   ```

3. **启动napcat**（QQ协议实现，需要在本地运行）

4. **启动客户端**：
   ```bash
   python client/qqbot_client_full.py
   ```

详细说明请参考 [`client/README.md`](client/README.md)。

## API端点

- `GET /api/health` - 健康检查
- `POST /api/chat/private` - 私聊
- `POST /api/chat/group` - 群聊
- `POST /api/upload/image` - 上传图片
- `POST /api/training/trigger` - 手动触发训练
- `GET /api/training/status` - 训练状态
- `POST /api/training/save-chat-history` - 保存历史

## 记忆机制

- **自动记忆**：对话内容自动保存到向量库
- **智能检索**：通过 `<recall>` token触发记忆检索
- **向量注入**：检索到的记忆通过 `<|memory_pad|>` 注入模型
- **持续学习**：定时训练更新模型（可配置首次训练延迟和训练间隔）
- **专用Embedding模型**：使用Qwen3-Embedding模型专门提取记忆条目和SFT数据的向量表征，提高向量质量
- **记忆混合训练**：第二步训练会将记忆条目拆成"前置SFT""前后拼接SFT"两种场景，并额外插入同等数量的纯SFT样本，形成 1:1:1 的混合比例，确保模型在回忆之后仍能输出完整的SFT结构
- **内存优化**：修复token截断后内存历史未更新的问题，避免内存泄漏

## 开发

### 添加新功能
1. 在对应模块中添加代码
2. 在 `src/api/routes.py` 添加端点
3. 测试功能

### 模块说明
- **api/**: Flask应用、路由、全局状态
- **chat/**: 生成、历史、队列、prompting
- **memory/**: 向量库、训练调度
- **training/**: 训练流程
- **services/**: 媒体、ASR、提取器
- **utils/**: 工具函数

## 引用与致谢

本项目基于以下开源项目构建，特此致谢：

### 模型与框架

- **[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)** - 阿里云通义千问团队开发的多模态大语言模型
  - 本项目使用 Qwen3-VL-4B-Thinking 作为基础模型
  - 模型许可：请遵循 [Qwen3-VL 的许可协议](https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE)

- **[Transformers](https://github.com/huggingface/transformers)** - Hugging Face 的模型库
  - 用于模型加载和推理

- **[PEFT](https://github.com/huggingface/peft)** - Hugging Face 的参数高效微调库
  - 用于 LoRA 微调训练

### 数据集

- **Chinese-Qwen3-235B-Thinking-2507-Distill-data-110k-SFT** - SFT 训练数据集
  - 用于模型的有监督微调训练

### QQ 客户端框架

- **[ncatbot](https://github.com/lz1998/ncatbot)** - QQ 机器人 Python SDK
  - 用于客户端与 QQ 的交互

- **[NapCat](https://github.com/NapNeko/NapCat)** - QQ 协议实现
  - 提供 WebSocket 接口，用于客户端连接 QQ

### 其他依赖

- **Flask** - Web 框架
- **APScheduler** - 定时任务调度
- **ModelScope** - 模型下载与管理

---

## License

本项目采用 **MIT License**。

### ⚠️ 重要使用限制

**本项目及其代码、模型、数据集均不允许用于以下用途：**

1. **科学研究**：包括但不限于学术研究、论文发表、实验分析等
2. **商业用途**：包括但不限于商业产品、服务、盈利性应用等
3. **任何违反法律法规的用途**

### 许可范围

- ✅ 个人学习与测试
- ✅ 非商业性的个人项目
- ✅ 符合法律法规的合法用途

### 模型与数据集许可

本项目使用的模型和数据集遵循其原始许可协议：

- **Qwen3-VL 模型**：请遵循 [Qwen3-VL 许可协议](https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE)
- **训练数据集**：请遵循数据集提供方的许可协议

**使用本项目即表示您已阅读、理解并同意遵守上述所有限制和许可条款。**

---

## 更新日志

### v2.2 - Embedding模型集成与训练优化 (2024-12-02)

**主要更新：**

1. **集成Qwen3-Embedding模型**
   - 使用专门的Qwen3-Embedding模型提取记忆条目和SFT数据的向量表征
   - 实现批量向量提取，提高效率
   - 使用L2归一化处理embedding向量，提高检索质量
   - 支持配置embedding模型路径（`embedding_model_path`）

2. **训练调度优化**
   - 支持配置首次训练延迟天数（`first_training_delay_days`）
   - 支持配置训练间隔天数（`training_interval_days`）
   - 默认配置：启动后第2天凌晨开始第一次训练，之后每3天训练一次

3. **Bug修复**
   - 修复token截断后内存历史未更新的问题，避免内存泄漏
   - 修复embedding维度配置（从4096改为2560，适配4B模型）
   - 修复训练过程中的IndentationError和NameError
   - 修复SFT向量提取时processor未传递的问题
   - 改进设备映射逻辑，正确处理CUDA_VISIBLE_DEVICES

4. **代码优化**
   - 移除Instruct prompt，直接使用文本提取embedding
   - 改进模型加载/卸载逻辑，确保正确清理GPU显存
   - 添加详细的调试日志

### v2.1 - 训练调度优化 (2024-11-26)
