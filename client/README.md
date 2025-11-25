# QQ机器人客户端使用说明

## 概述

客户端-服务器分离架构：

- **服务器端**（你的服务器）：运行LLM模型、RAG管理、训练等
- **客户端**（你的个人电脑）：运行QQbot，转发消息

## 安装步骤

### 1. 在个人电脑上安装依赖

```bash
# 安装ncatbot和相关库
pip install ncatbot requests flask-cors

# 或使用conda
conda install -c conda-forge requests flask-cors
pip install ncatbot
```

### 2. 配置客户端

编辑 `client/qqbot_client_full.py`：

```python
# 修改服务器地址
SERVER_URL = "http://your-server-ip:9999"  # 默认端口 9999，可根据 scripts/run_server.py 的配置调整

# 配置QQbot
config.set_bot_uin("你的QQ号")
config.set_ws_uri("ws://localhost:3001")
config.set_token("你的token")
```

### 3. 配置服务器防火墙

在服务器上开放API端口（默认8888）：

```bash
# 如果使用iptables
sudo iptables -A INPUT -p tcp --dport 8888 -j ACCEPT

# 或使用firewalld
sudo firewall-cmd --add-port=8888/tcp --permanent
sudo firewall-cmd --reload
```

### 4. 启动服务器

在服务器上运行：

```bash
cd /data0/user/ymdai/LLM_memory/qqbot_new
python scripts/run_server.py
```

> 服务器监听地址与端口由 `configs/config_qwen3vl.yaml` 中的 `server.host`/`server.port` 控制，默认 `0.0.0.0:9999`。

### 5. 启动客户端

在个人电脑上运行：

```bash
# 确保 napcat 已启动（本地）
python client/qqbot_client_full.py
```

## 架构说明

```
个人电脑                   网络                    服务器
┌─────────┐                                    ┌─────────────┐
│ QQbot   │  ←→  HTTP API  ←→                  │ API Server  │
│ Client  │                                    │             │
│         │                                    │ LLM Model   │
│ napcat  │                                    │ RAG Manager │
└─────────┘                                    │ Training    │
                                               └─────────────┘
```

## API接口说明

### 群消息接口

**POST** `/api/chat/group`

请求：
```json
{
    "group_id": "123456789",
    "user_id": "987654321",
    "user_nickname": "用户名",
    "user_card": "名片",
    "content": "消息内容",
    "timestamp": 1234567890.0
}
```

响应：
```json
{
    "status": "success",
    "should_reply": true,
    "reply": "回复内容"
}
```

### 私聊消息接口

**POST** `/api/chat/private`

请求：
```json
{
    "user_id": "987654321",
    "content": "消息内容",
    "timestamp": 1234567890.0
}
```

响应：
```json
{
    "status": "success",
    "reply": "回复内容"
}
```

### 健康检查

**GET** `/health`

响应：
```json
{
    "status": "healthy",
    "model_loaded": true,
    "rag_manager_ready": true
}
```

## 注意事项

1. **网络连接**：确保客户端能够访问服务器的IP和端口
2. **安全性**：建议在生产环境使用HTTPS和认证
3. **延迟**：网络延迟会影响回复速度
4. **稳定性**：建议客户端实现重试机制

## 故障排查

1. **无法连接服务器**：
   - 检查服务器IP和端口是否正确
   - 检查防火墙设置
   - 检查服务器是否正常运行

2. **API请求超时**：
   - 增加`API_TIMEOUT`值
   - 检查网络连接
   - 检查服务器负载

3. **回复延迟**：
   - 这是正常现象（需要网络传输和模型推理）
   - 可以考虑优化网络或使用更快的连接

