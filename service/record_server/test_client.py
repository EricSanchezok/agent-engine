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