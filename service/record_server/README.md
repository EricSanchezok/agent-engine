# Record Memory Server API

Record Memory Server 是一个基于 FastAPI 的 REST API 服务器，提供对 RecordMemory 系统的远程访问功能。该服务器运行在端口 5050 上，支持能力管理、代理管理和任务结果记录等操作。

## 服务器信息

- **服务器地址**: http://10.244.9.104:5050
- **API 文档**: http://10.244.9.104:5050/docs
- **健康检查**: http://10.244.9.104:5050/health

- **代理地址**: http://76358938.r8.cpolar.top/

## API 接口文档

### 1. 获取所有能力 (GET /capabilities)

获取系统中所有已注册的能力。

**请求：**
```http
GET http://10.244.9.104:5050/capabilities
```

**响应：**
```json
[
  {
    "name": "Chat with Conversational AI Assistant",
    "definition": "This service enables users to engage in natural language conversation...",
    "alias": ["chat", "conversation"],
    "agents": [
      {
        "name": "ChatAgent",
        "url": "http://localhost:8001"
      }
    ]
  }
]
```

### 2. 搜索相似能力 (POST /capabilities/search)

根据能力名称和定义搜索相似的能力。

**请求：**
```http
POST http://10.244.9.104:5050/capabilities/search
Content-Type: application/json

{
  "name": "Chat Assistant",
  "definition": "A conversational AI service",
  "top_k": 5,
  "threshold": 0.55
}
```

**响应：**
```json
[
  {
    "name": "Chat with Conversational AI Assistant",
    "definition": "This service enables users to engage in natural language conversation...",
    "similarity_score": 0.85,
    "metadata": {
      "alias": ["chat", "conversation"],
      "agents": [...]
    }
  }
]
```

### 3. 获取能力的代理 (POST /capabilities/agents)

获取能够执行特定能力的所有代理。

**请求：**
```http
POST http://10.244.9.104:5050/capabilities/agents
Content-Type: application/json

{
  "name": "Chat with Conversational AI Assistant",
  "definition": "This service enables users to engage in natural language conversation..."
}
```

**响应：**
```json
[
  {
    "name": "ChatAgent",
    "url": "http://localhost:8001"
  },
  {
    "name": "ConversationAgent",
    "url": "http://localhost:8002"
  }
]
```

### 4. 获取代理的能力 (GET /agents/{agent_name}/capabilities)

获取特定代理能够执行的所有能力。

**请求：**
```http
GET http://10.244.9.104:5050/agents/ChatAgent/capabilities?agent_url=http://localhost:8001
```

**响应：**
```json
[
  {
    "name": "Chat with Conversational AI Assistant",
    "definition": "This service enables users to engage in natural language conversation..."
  },
  {
    "name": "Text Summarization",
    "definition": "Summarize long text content..."
  }
]
```

### 5. 添加任务结果 (POST /task-result)

记录代理执行任务的结果。

**注意：** 系统会验证 `agent_name`、`agent_url`、`capability_name` 和 `capability_definition` 是否在系统中存在。如果不存在，操作将失败。

**请求：**
```http
POST http://10.244.9.104:5050/task-result
Content-Type: application/json

{
  "agent_name": "ChatAgent",
  "agent_url": "http://localhost:8001",
  "capability_name": "Chat with Conversational AI Assistant",
  "capability_definition": "This service enables users to engage in natural language conversation...",
  "success": true,
  "task_content": "User asked: What is the weather today?",
  "task_result": "I cannot provide real-time weather information as I don't have access to current weather data."
}
```

**响应：**
```json
{
  "message": "Task result added successfully",
  "success": true
}
```

**验证失败响应：**
```json
{
  "message": "Failed to add task result. Please verify that the capability and agent exist in the system.",
  "success": false
}
```

### 6. 删除任务结果 (DELETE /task-result)

删除特定的任务执行记录。

**注意：** 系统会验证 `agent_name`、`agent_url`、`capability_name` 和 `capability_definition` 是否在系统中存在。如果不存在，操作将失败。

**请求：**
```http
DELETE http://10.244.9.104:5050/task-result
Content-Type: application/json

{
  "agent_name": "ChatAgent",
  "agent_url": "http://localhost:8001",
  "capability_name": "Chat with Conversational AI Assistant",
  "capability_definition": "This service enables users to engage in natural language conversation...",
  "task_content": "User asked: What is the weather today?",
  "task_result": "I cannot provide real-time weather information as I don't have access to current weather data.",
  "timestamp": "2024-01-15T10:30:00"
}
```

**响应：**
```json
{
  "message": "Task result deleted successfully",
  "success": true
}
```

**验证失败响应：**
```json
{
  "message": "Failed to delete task result. Please verify that the capability and agent exist in the system.",
  "success": false
}
```

### 7. 获取能力表现 (POST /capabilities/performance)

获取执行特定能力的所有代理的表现统计。

**请求：**
```http
POST http://10.244.9.104:5050/capabilities/performance
Content-Type: application/json

{
  "name": "Chat with Conversational AI Assistant",
  "definition": "This service enables users to engage in natural language conversation..."
}
```

**响应：**
```json
[
  {
    "name": "ChatAgent",
    "url": "http://localhost:8001",
    "success_count": 15,
    "total_count": 20,
    "success_rate": 0.75
  },
  {
    "name": "ConversationAgent",
    "url": "http://localhost:8002",
    "success_count": 8,
    "total_count": 12,
    "success_rate": 0.67
  }
]
```

### 8. 获取能力历史 (POST /capabilities/history)

获取特定能力的所有相关代理的历史记录。

**请求：**
```http
POST http://10.244.9.104:5050/capabilities/history
Content-Type: application/json

{
  "name": "Chat with Conversational AI Assistant",
  "definition": "This service enables users to engage in natural language conversation..."
}
```

**响应：**
```json
{
  "ChatAgent_http://localhost:8001": {
    "agent_name": "ChatAgent",
    "agent_url": "http://localhost:8001",
    "success_count": 15,
    "total_count": 20,
    "tasks": [
      {
        "task_content": "User asked: What is the weather today?",
        "task_result": "I cannot provide real-time weather information...",
        "success": true,
        "timestamp": "2024-01-15T10:30:00"
      }
    ]
  }
}
```

### 9. 获取所有代理 (GET /agents)

获取系统中所有唯一的代理。

**请求：**
```http
GET http://10.244.9.104:5050/agents
```

**响应：**
```json
[
  {
    "name": "ChatAgent",
    "url": "http://localhost:8001"
  },
  {
    "name": "ConversationAgent",
    "url": "http://localhost:8002"
  }
]
```

## 错误处理

所有 API 接口都包含错误处理机制。当发生错误时，服务器将返回相应的 HTTP 状态码和错误信息：

```json
{
  "detail": "Failed to get capabilities: Database connection error"
}
```

常见状态码：
- `200`: 请求成功
- `400`: 请求参数错误
- `404`: 资源未找到
- `500`: 服务器内部错误

## 使用示例

### Python 客户端示例

```python
import requests
import json

# 服务器地址
BASE_URL = "http://76358938.r8.cpolar.top/http://10.244.9.104:5050"

# 1. 获取所有能力
print("=== 获取所有能力 ===")
response = requests.get(f"{BASE_URL}/capabilities")
capabilities = response.json()
print(f"Found {len(capabilities)} capabilities:")
for cap in capabilities:
    print(f"- {cap.get('name')}")

# 2. 搜索相似能力
print("\n=== 搜索相似能力 ===")
search_data = {
    "name": "Chat Assistant",
    "definition": "A conversational AI service",
    "top_k": 3,
    "threshold": 0.6
}
response = requests.post(f"{BASE_URL}/capabilities/search", json=search_data)
similar_capabilities = response.json()
print(f"Found {len(similar_capabilities)} similar capabilities:")
for cap in similar_capabilities:
    print(f"- {cap.get('name')} (similarity: {cap.get('similarity_score', 'N/A')})")

# 3. 获取能力的代理
print("\n=== 获取能力的代理 ===")
if capabilities:
    capability = capabilities[-2]
    agent_data = {
        "name": capability.get("name"),
        "definition": capability.get("definition")
    }
    response = requests.post(f"{BASE_URL}/capabilities/agents", json=agent_data)
    agents = response.json()
    print(f"Agents for capability '{capability.get('name')}':" )
    for agent in agents:
        print(f"- {agent.get('name')} ({agent.get('url')})")

# 4. 获取所有代理
print("\n=== 获取所有代理 ===")
response = requests.get(f"{BASE_URL}/agents")
all_agents = response.json()
print(f"All agents in system:")
for agent in all_agents:
    print(f"- {agent.get('name')} ({agent.get('url')})")

# 5. 获取代理的能力
print("\n=== 获取代理的能力 ===")
if all_agents:
    agent = all_agents[0]  # 使用第一个代理作为示例
    response = requests.get(f"{BASE_URL}/agents/{agent['name']}/capabilities?agent_url={agent['url']}")
    agent_capabilities = response.json()
    print(f"Capabilities for agent '{agent.get('name')}':" )
    for cap in agent_capabilities:
        print(f"- {cap.get('name')}")

# 6. 添加任务结果
print("\n=== 添加任务结果 ===")
task_data = {
    "agent_name": "ChatAgent",
    "agent_url": "http://localhost:8001",
    "capability_name": "Chat with Conversational AI Assistant",
    "capability_definition": "This service enables users to engage in natural language conversation...",
    "success": True,
    "task_content": "User asked: Hello",
    "task_result": "Hello! How can I help you today?"
}
response = requests.post(f"{BASE_URL}/task-result", json=task_data)
result = response.json()
print(f"Task result: {result.get('message')}")
print(f"Success: {result.get('success')}")

# 7. 获取能力表现
print("\n=== 获取能力表现 ===")
if capabilities:
    performance_data = {
        "name": capability.get("name"),
        "definition": capability.get("definition")
    }
    response = requests.post(f"{BASE_URL}/capabilities/performance", json=performance_data)
    performance = response.json()
    print(f"Performance for capability '{capability.get('name')}':" )
    for perf in performance:
        print(f"- {perf['name']}: {perf.get('success_count')}/{perf.get('total_count')} ({perf.get('success_rate'):.2%})")

# 8. 获取能力历史
print("\n=== 获取能力历史 ===")
if capabilities:
    history_data = {
        "name": capability.get("name"),
        "definition": capability.get("definition")
    }
    response = requests.post(f"{BASE_URL}/capabilities/history", json=history_data)
    history = response.json()
    print(f"History for capability '{capability.get('name')}':" )
    print(history)
```

### cURL 示例

```bash
# 1. 获取所有能力
echo "=== 获取所有能力 ==="
curl -X GET "http://10.244.9.104:5050/capabilities" | jq '.'

# 2. 搜索相似能力
echo -e "\n=== 搜索相似能力 ==="
curl -X POST "http://10.244.9.104:5050/capabilities/search" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Chat Assistant",
    "definition": "A conversational AI service",
    "top_k": 3,
    "threshold": 0.6
  }' | jq '.'

# 3. 获取能力的代理
echo -e "\n=== 获取能力的代理 ==="
curl -X POST "http://10.244.9.104:5050/capabilities/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Chat with Conversational AI Assistant",
    "definition": "This service enables users to engage in natural language conversation..."
  }' | jq '.'

# 4. 获取所有代理
echo -e "\n=== 获取所有代理 ==="
curl -X GET "http://10.244.9.104:5050/agents" | jq '.'

# 5. 获取代理的能力
echo -e "\n=== 获取代理的能力 ==="
curl -X GET "http://10.244.9.104:5050/agents/ChatAgent/capabilities?agent_url=http://localhost:8001" | jq '.'

# 6. 添加任务结果
echo -e "\n=== 添加任务结果 ==="
curl -X POST "http://10.244.9.104:5050/task-result" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "ChatAgent",
    "agent_url": "http://localhost:8001",
    "capability_name": "Chat with Conversational AI Assistant",
    "capability_definition": "This service enables users to engage in natural language conversation...",
    "success": true,
    "task_content": "User asked: Hello",
    "task_result": "Hello! How can I help you today?"
  }' | jq '.'

# 7. 获取能力表现
echo -e "\n=== 获取能力表现 ==="
curl -X POST "http://10.244.9.104:5050/capabilities/performance" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Chat with Conversational AI Assistant",
    "definition": "This service enables users to engage in natural language conversation..."
  }' | jq '.'

# 8. 获取能力历史
echo -e "\n=== 获取能力历史 ==="
curl -X POST "http://10.244.9.104:5050/capabilities/history" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Chat with Conversational AI Assistant",
    "definition": "This service enables users to engage in natural language conversation..."
  }' | jq '.'

# 9. 删除任务结果（可选）
echo -e "\n=== 删除任务结果 ==="
curl -X DELETE "http://10.244.9.104:5050/task-result" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "ChatAgent",
    "agent_url": "http://localhost:8001",
    "capability_name": "Chat with Conversational AI Assistant",
    "capability_definition": "This service enables users to engage in natural language conversation...",
    "task_content": "User asked: Hello",
    "task_result": "Hello! How can I help you today?"
  }' | jq '.'

# 注意：如果没有安装 jq，可以移除 | jq '.' 来查看原始 JSON 输出
```

## 注意事项

1. 服务器支持 CORS，允许跨域请求
2. 所有 API 接口都是异步的，支持高并发访问
3. 请确保在发送请求前服务器已经启动并正常运行
4. 对于任务结果操作，系统会严格验证能力和代理的存在性
5. 删除操作会返回布尔值 `success` 字段，表示操作是否成功
6. 时间戳格式为 ISO 8601 标准（如：`2024-01-15T10:30:00`）
