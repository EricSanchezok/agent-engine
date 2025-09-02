import asyncio
import json
from pprint import pprint

# Agent Engine imports
from agent_engine.a2a_client import send_message_to_a2a_agent

# Core imports
from core.holos.config import PROXY_URL

# Local imports
from agents.JudgeAgent.config import ROUTING_AGENT_BASE_URL

async def test_messages():
    """Test different message formats"""
    
    base_url = ROUTING_AGENT_BASE_URL
    proxy_url = PROXY_URL
    
    print("Testing different message formats...")
    print(f"Base URL: {base_url}")
    print(f"Proxy URL: {proxy_url}")
    print("-" * 50)
    
    # Test 1: Simple text message
    print("Test 1: Simple text message")
    try:
        response = await send_message_to_a2a_agent(
            base_url=base_url,
            message="Hello, how are you?",
            proxy_url=proxy_url
        )
        print("✅ Simple message successful")
        print(f"Response ID: {response.get('id', 'Unknown')}")
    except Exception as e:
        print(f"❌ Simple message failed: {e}")
    print()
    
    # Test 2: Simple JSON message
    print("Test 2: Simple JSON message")
    try:
        simple_json = {
            "task": "Search for papers on AI"
        }
        response = await send_message_to_a2a_agent(
            base_url=base_url,
            message=json.dumps(simple_json, ensure_ascii=False),
            proxy_url=proxy_url
        )
        print("✅ Simple JSON message successful")
        print(f"Response ID: {response.get('id', 'Unknown')}")
    except Exception as e:
        print(f"❌ Simple JSON message failed: {e}")
    print()
    
    # Test 3: Complex JSON message (like the one causing 500 error)
    print("Test 3: Complex JSON message")
    try:
        complex_json = {
            "test_task": {
                "task_description": "Search for academic papers on quantum computing published in the last two years. Return paper titles, authors, abstracts, and direct PDF links.",
                "depends_on": [],
                "agent": "ArxivSearchAgent",
                "agent_url": "http://10.244.9.104:9900/"
            }
        }
        response = await send_message_to_a2a_agent(
            base_url=base_url,
            message=json.dumps(complex_json, ensure_ascii=False, indent=4),
            proxy_url=proxy_url
        )
        print("✅ Complex JSON message successful")
        print(f"Response ID: {response.get('id', 'Unknown')}")
    except Exception as e:
        print(f"❌ Complex JSON message failed: {e}")
    print()

if __name__ == "__main__":
    asyncio.run(test_messages())
