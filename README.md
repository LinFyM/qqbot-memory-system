# QQ聊天机器人长期记忆系统

基于 Qwen3-VL-4B-Thinking 模型的QQ聊天机器人，集成长期记忆存储与检索功能。

## 项目简介

本项目实现了一个具有长期记忆能力的QQ聊天机器人系统，主要特性包括：

- 🤖 **智能对话**：支持群聊和私聊消息的智能回复
- 🧠 **长期记忆**：自动从对话历史中提取记忆条目，支持高效的相似度检索
- 🔄 **动态召回**：在生成回复过程中，动态检索并注入相关记忆
- 🎓 **自动训练**：两阶段训练流程，自动提取记忆、训练模型
- ⚡ **消息打断**：支持新消息到达时中断当前生成任务

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU训练和推理)
- 至少 16GB GPU 显存（推荐 24GB+）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

1. 复制并编辑配置文件：
```bash
cp server/config_qwen3vl.yaml server/config_qwen3vl.yaml.local
# 编辑配置文件，设置模型路径、设备等
```

2. 配置QQ客户端（如果需要）：
```bash
# 编辑 client/qqbot_client_full.py
# 设置服务器地址和QQ配置
```

### 运行

**启动服务器**：
```bash
python server/app.py
```

**启动客户端**（在另一台机器上）：
```bash
# 先建立SSH隧道
ssh -L 9999:gpu02:9999 ymdai@210.75.240.172 -p 2277

# 启动客户端
python client/qqbot_client_full.py
```

## 项目结构

```
qqbot_new/
├── server/              # 服务器端代码
│   ├── api_server_qwen3vl.py  # 主API服务器
│   ├── app.py          # Flask应用入口
│   ├── memory/         # 记忆系统模块
│   └── routes/         # API路由
├── recall/             # 训练相关代码
│   ├── text_embedding_train.py  # 第一步训练
│   └── text_memory_train.py     # 第二步训练
├── client/             # QQ客户端代码
├── requirements.txt    # 依赖包列表
└── TECHNICAL_REPORT.md  # 技术报告
```

## 文档

- [技术报告](TECHNICAL_REPORT.md) - 详细的技术文档，包括架构设计、训练流程等
- [配置文件说明](server/README_CONFIG.md) - 配置文件详细说明
- [客户端使用说明](client/README.md) - QQ客户端使用指南

## 主要功能

### 1. 记忆系统

- **自动提取**：从聊天记录中自动提取关键记忆信息
- **向量存储**：将记忆存储为向量表示，支持高效检索
- **相似度检索**：基于余弦相似度的top-k检索

### 2. 自回归生成中的记忆召回

- **触发机制**：检测到 `<recall>` token时自动触发回忆
- **动态注入**：检索相关记忆并注入到生成过程中
- **无缝融合**：记忆内容自然融入生成的回复中

### 3. 两阶段训练

- **第一阶段**：训练 `<recall>` token的embedding
- **第二阶段**：训练记忆文本解码能力

## 技术栈

- **后端框架**：Flask
- **深度学习**：PyTorch + Transformers
- **基础模型**：Qwen3-VL-4B-Thinking
- **参数高效微调**：LoRA (Low-Rank Adaptation)
- **分布式训练**：Accelerate
- **QQ协议**：ncatbot

## 许可证

**重要声明**：本项目仅供个人学习使用，不开放给研究用途或其他任何用途。未经授权，禁止用于任何商业、研究或其他目的。

## 致谢

感谢以下开源项目：
- [Qwen3-VL](https://github.com/QwenLM/Qwen2-VL) - 基础模型
- [ncatbot](https://github.com/ncatbot/ncatbot) - QQ协议客户端
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 模型框架

## 联系方式

如有问题或建议，请提交 Issue。

