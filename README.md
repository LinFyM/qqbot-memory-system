# 萝卜子QQ机器人 v2.0

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

## QQ 客户端运行

服务器只负责推理/训练，要让 QQ 账号真正上线，还需要在本地运行 Napcat + `client/qqbot_client_full.py`：

1. 安装并登录 Napcat（或其他支持 OneBot v11 的 QQ 协议端）
2. 在本地机器上安装依赖
   ```bash
   pip install -r client/requirements.txt
   ```
3. 编辑 `client/qqbot_client_full.py`
   - `SERVER_URL` 改成你的服务器地址，例如 `http://your-server-ip:9999`
   - `config.set_bot_uin` / `config.set_ws_uri` / `config.set_token` 按 Napcat 的配置填写
4. 运行客户端
   ```bash
   python client/qqbot_client_full.py
   ```

客户端会把 QQ 消息转发到服务器的 `/api/chat/*` 接口，并负责执行服务器返回的动作指令（如表情、戳一戳等）。更详细的参数和故障排查可以参考 `client/README.md`。

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
    schedule: "3"  # 凌晨3点自动训练
```

### 提示词配置

所有 prompt 文案都集中在 `configs/prompts.yaml`，便于统一查看与修改。该文件分为：

- `chat`：系统提示词、角色扮演、动作指导、拼接顺序等
- `memory_training`：记忆激活语、结束语以及训练时的引导文字
- `memory_extraction`：记忆条目提取的 system prompt、递归子树提示、媒资描述提醒、用户指令
- `memory_vectorization`：批量向量提取时的文本压缩提示

如果需要调整提示词，直接编辑该文件后重启/重新加载配置即可。

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
- **持续学习**：定时训练更新模型（每两天凌晨3点）
- **记忆混合训练**：第二步训练会将记忆条目拆成“前置SFT”“前后拼接SFT”两种场景，并额外插入同等数量的纯SFT样本，形成 1:1:1 的混合比例，确保模型在回忆之后仍能输出完整的SFT结构。

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

## License

MIT

---
v2.0 - 全新架构 (2025-11-24)
