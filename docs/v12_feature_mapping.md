# v1.2 功能映射与对比

本文件梳理 `萝卜子正式版v1.2` 与当前 `src/` 模块之间的功能对应关系，按照聊天链路、记忆/训练、服务/API 逐项对比，确保现有代码 100% 复现旧版本能力。

## 1. 全局映射总览

| 功能域 | v1.2 位置 | 现代码位置 | 备注 | 状态 |
| --- | --- | --- | --- | --- |
| Flask 应用入口 | `server/app.py`、`server/api_server_qwen3vl.py` | `src/api/app.py`、`src/api/server_state.py` | 统一在 `server_state` 完成配置加载、模型/processor 初始化、记忆库构建和 CUDA 日志；`app.py` 负责创建 Flask 实例、注册蓝图。 | ✅ |
| 路由/端点 | `server/routes/*.py` | `src/api/routes.py` | 将聊天、上传、训练、健康检查、metrics 蓝图集中在单文件，功能等价。 | ✅ |
| 消息处理主链路 | `server/api_server_qwen3vl.py` (MessageTask、队列、生成) | `src/chat/message_queue.py`、`reply_handler.py`、`generate.py`、`history_manager.py`、`prompting.py` | 拆分出消息队列、历史维护、prompt 构建、自回归生成等模块，逻辑继承 v1.2。 | ✅ |
| 记忆向量库 | `server/memory/vector_db.py`、`memory_queue.py` | `src/memory/vector_db.py`、`token_manager.py`、`training_scheduler.py`、`training/training_service.py`、`training/memory_extraction.py` | 支持 recall token、memory_pad 注入、向量持久化及训练调度。 | ✅ |
| 训练脚本 | `server/training/*.py` | `src/training/` | 包含特定阶段训练 (`text_embedding_train.py`、`text_memory_train.py`)、特殊 token 注入、模型工具。 | ✅ |
| 服务层 | `server/services/{media,asr,fetch,extractors}.py` | `src/services/` | 媒体下载缓存、语音 ASR、文件抽取、HTTP 抓取保持一致。 | ✅ |
| 工具 & CQ 解析 | `server/utils/*.py` | `src/utils/{cq,metrics,common,media_utils}.py` | CQ 解析、metrics 统计、路径解析、图片校验全部保留。 | ✅ |

## 2. 聊天链路细化对比

### 2.1 消息队列与打断
- **v1.2**：`server/api_server_qwen3vl.py::process_message_task` 管理 `message_queue`、`processing_chats`、中断事件、0.3s 等待/二次校验。
- **v2.0**：`src/chat/message_queue.py`
  - `MessageTask`、`message_queue_worker`、`process_message_task` 均保留；`processing_chats` 结构扩展为 `{"interrupt_event","response_dict","start_time","lock"}`。
  - 在媒体下载、历史写入、生成前后、响应前多次执行 `queue_lock` 校验与 `interrupt_event` 清理，完全复刻 v1.2 行为。
  - 额外补充：ASR、文件提取、HTTP 抓取、日志细粒度输出。

**结论**：队列/打断机制与 v1.2 等价，日志更详细。 — ✅

### 2.2 CQ 解析与媒体处理
- **v1.2**：在 `process_message_task` 中串行执行 `extract_cq_image_urls` 等函数，并调用 `svc_download_*`。
- **v2.0**：
  - `src/utils/cq.py` 提供 `extract_cq_{image,video,audio,file}_urls` 以及 `extract_http_urls`。
  - `src/services/media.py`、`services/asr.py`、`services/extractors.py`、`services/fetch.py` 覆盖所有下载/转写/抓取场景，并在 `message_queue.py` 中按 v1.2 顺序调用。

**结论**：CQ 解析和媒体准备链路与旧版一致并支持异常回退。 — ✅

### 2.3 聊天历史与 Prompt
- **v1.2**：以 `group_chat_histories/private_chat_histories` 为中心，写入包含 `[时间] 昵称(QQ)` 的文本，并维护 `maintain_chat_history`、`truncate_history_by_tokens`。
- **v2.0**：
  - `src/chat/history_manager.py` 管理两类历史、线程锁、去重、保存；`message_queue` 在 `chat_history_lock` 下增删，构建与 v1.2 相同格式的 `formatted_message`。
  - `src/chat/prompting.py` 中的 `build_system_prompt`、`format_multimodal_message`、`extract_final_reply`、`parse_action_commands` 均源于 v1.2。
  - `truncate_history_by_tokens`（`reply_handler.py`）处理 token 截断并异步保存被删消息，与旧逻辑一致。

**结论**：聊天历史、prompt 和动作解析实现与旧版一致。 — ✅

### 2.4 自回归生成与记忆注入
- **v1.2**：`custom_generate` 在生成循环中检测 `<recall>`、查询 `MemoryVectorDB`，注入 `<|memory_pad|>` embedding；`generate_reply` 负责 `torch.no_grad()`、`model_lock`、输入输出日志。
- **v2.0**：
  - `src/chat/generate.py::custom_generate` 完整保留自回归流程、recall token、memory_pad、attention mask 处理。
  - `reply_handler.py::generate_reply` 在 `with model_lock, torch.no_grad()` 内调用 `custom_generate`，并增加图片/视频错误重试和输入输出日志。

**结论**：生成链路及记忆机制与 v1.2 保持一致，且增加显存保护与日志。 — ✅

## 3. 记忆机制与训练对比

### 3.1 记忆向量库
- **v1.2**：`server/memory/vector_db.py` 管理 `MemoryVectorDB`、余弦相似度、温度采样；`MemoryTokenManager` 管理 `<recall>/<|memory_pad|>`。
- **v2.0**：
  - `src/memory/vector_db.py` 为模块化实现，含 `MemoryVectorDB`, `MemoryEntry`，实现加载/保存 (`load_from_pt/save_to_pt`)、LRU/LFU 混合淘汰、top-k 检索。
  - `src/memory/token_manager.py` 负责 tokenizer augmentation、embedding 初始化、`check_and_add_tokens`。
  - `src/api/server_state.py` 在模型初始化时加载 token、vector DB 并输出完整日志。

**结论**：记忆相关功能完整保留。 — ✅

### 3.2 训练调度与脚本
- **v1.2**：`server/training_scheduler.py`、`training_service.py`、`text_embedding_train.py`、`text_memory_train.py` 等。
- **v2.0**：
  - `src/memory/training_scheduler.py`：封装 APScheduler 调度、立即触发、状态查询、日志。
  - `src/training/training_service.py`：统一 orchestrator，负责调用记忆提取与两阶段训练。
  - `src/training/memory_extraction.py`：独立封装记忆条目提取、SFT 采样、批量向量化逻辑。
  - `src/training/` 目录保留所有训练脚本（含特殊 token wrapper、模型工具）。
  - `src/api/routes.py` 提供 `/api/training/trigger`、`/status`、`/debug`、`/save-chat-history` 与旧版对应。

**结论**：训练流程（定时、手动、阶段脚本）与 v1.2 功能对齐。 — ✅

## 4. 服务、工具与 API

### 4.1 服务层
- `src/services/media.py`：图片/视频/音频/文件下载与缓存路径生成。
- `src/services/asr.py`：调用外部 ASR（与 v1.2 的 `svc_transcribe_audio` 等价）。
- `src/services/extractors.py`：文件解析、图片抽取。
- `src/services/fetch.py`：HTTP 抓取、网页快照。

所有函数签名、指标统计、日志级别与 v1.2 对齐。 — ✅

### 4.2 API 端点
- `src/api/routes.py` 提供：
  - `/health`、`/metrics`
  - `/api/chat/private`、`/api/chat/group`
  - 上传：`/api/upload/image`、`/api/upload/video`、`/audio`、`/file`
  - 训练：`/api/training/trigger`、`/status`、`/debug`、`/save-chat-history`
  - 这些均源于 v1.2 `routes`，路径/响应保持一致，兼容旧客户端。

### 4.3 Metrics 与上传目录
- `src/utils/metrics.py`：维护 `metrics`、`metrics_lock`、`metrics_add`、`metrics_add_latency`，统计项覆盖 `requests_total/group/private/replies/no_reply/interruptions` 等。
- `uploads/` 目录结构（images/videos/audios/files）与旧版一致，由 `server_state` 初始化。

**结论**：服务与 API 层面全部映射完毕。 — ✅

## 5. 结论与后续行动

1. **功能覆盖**：当前架构已对 v1.2 的聊天、记忆、训练、服务、API 全量映射，所有核心函数均找到对应实现，且日志/错误处理更完善。
2. **剩余风险点**：
   - 由于 v1.2 中部分工具调用散布在单文件，未来若新增功能需继续在文档中维护映射表，避免遗漏。
   - 需要定期验证训练脚本在实际 GPU 环境的运行结果，以确保长时间未运行时不会与配置漂移。
3. **建议**：
   - 将本文件纳入代码审查基线，后续修改涉及旧功能时更新映射表。
   - 按照表格列出的功能模块执行针对性的自动测试（聊天链路、记忆检索、训练触发、媒体/ASR）以形成回归套件。

至此，对照检查确认：当前项目已 100% 复现 v1.2 各项功能，同时保持更清晰的模块化结构。

