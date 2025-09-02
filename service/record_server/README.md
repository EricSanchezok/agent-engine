# Record Memory Server API

Record Memory Server 是一个基于 FastAPI 的 REST API 服务器，提供对 RecordMemory 系统的远程访问功能。该服务器运行在端口 5050 上，支持能力管理、代理管理和任务结果记录等操作。

## 服务器信息

- **服务器地址**: http://localhost:5050
- **API 文档**: http://localhost:5050/docs
- **健康检查**: http://localhost:5050/health

## API 接口文档

### 1. 获取所有能力 (GET /capabilities)

获取系统中所有已注册的能力。

**请求：**
```http
GET http://localhost:5050/capabilities
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
POST http://localhost:5050/capabilities/search
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
POST http://localhost:5050/capabilities/agents
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
GET http://localhost:5050/agents/ChatAgent/capabilities?agent_url=http://localhost:8001
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
POST http://localhost:5050/task-result
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
DELETE http://localhost:5050/task-result
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

### 6.1. 删除代理任务历史 (DELETE /agents/{agent_name}/task-history)

删除特定代理的所有任务历史记录。

**请求：**
```http
DELETE http://localhost:5050/agents/ChatAgent/task-history?agent_url=http://localhost:8001
```

**响应：**
```json
{
  "message": "Task history deleted successfully for agent ChatAgent",
  "success": true
}
```

### 6.2. 删除所有任务历史 (DELETE /task-history)

删除所有代理的所有任务历史记录。

**请求：**
```http
DELETE http://localhost:5050/task-history
```

**响应：**
```json
{
  "message": "All task history deleted successfully",
  "success": true
}
```

### 7. 获取能力表现 (POST /capabilities/performance)

获取执行特定能力的所有代理的表现统计。

**请求：**
```http
POST http://localhost:5050/capabilities/performance
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

### 9. 获取能力历史 (POST /capabilities/history)

获取特定能力的所有相关代理的历史记录。

**请求：**
```http
POST http://localhost:5050/capabilities/history
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

### 10. 获取所有代理 (GET /agents)

获取系统中所有唯一的代理。

**请求：**
```http
GET http://localhost:5050/agents
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
BASE_URL = "http://localhost:5050"

# 获取所有能力
response = requests.get(f"{BASE_URL}/capabilities")
capabilities = response.json()
print("All capabilities:", capabilities)

# 搜索相似能力
search_data = {
    "name": "Chat Assistant",
    "definition": "A conversational AI service",
    "top_k": 3,
    "threshold": 0.6
}
response = requests.post(f"{BASE_URL}/capabilities/search", json=search_data)
similar_capabilities = response.json()
print("Similar capabilities:", similar_capabilities)

# 添加任务结果
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
print("Task result added:", result["message"])
print("Operation successful:", result["success"])

# 删除代理任务历史
response = requests.delete(f"{BASE_URL}/agents/ChatAgent/task-history?agent_url=http://localhost:8001")
result = response.json()
print("Agent task history deleted:", result["message"])
print("Operation successful:", result["success"])
```

### cURL 示例

```bash
# 获取所有能力
curl -X GET "http://localhost:5050/capabilities"

# 搜索相似能力
curl -X POST "http://localhost:5050/capabilities/search" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Chat Assistant",
    "definition": "A conversational AI service",
    "top_k": 3,
    "threshold": 0.6
  }'

# 添加任务结果
curl -X POST "http://localhost:5050/task-result" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "ChatAgent",
    "agent_url": "http://localhost:8001",
    "capability_name": "Chat with Conversational AI Assistant",
    "capability_definition": "This service enables users to engage in natural language conversation...",
    "success": true,
    "task_content": "User asked: Hello",
    "task_result": "Hello! How can I help you today?"
  }'

# 删除代理任务历史
curl -X DELETE "http://localhost:5050/agents/ChatAgent/task-history?agent_url=http://localhost:8001"

# 删除所有任务历史
curl -X DELETE "http://localhost:5050/task-history"
```

## 注意事项

1. 服务器支持 CORS，允许跨域请求
2. 所有 API 接口都是异步的，支持高并发访问
3. 请确保在发送请求前服务器已经启动并正常运行
