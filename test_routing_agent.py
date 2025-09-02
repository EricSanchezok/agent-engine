import asyncio
import json
import httpx
from pprint import pprint

async def test_routing_agent():
    """Test routing agent connection and message handling"""
    
    # Test different URL formats
    urls = [
        "http://76358938.r8.cpolar.top/http://10.244.20.48:10002",
        "http://76358938.r8.cpolar.top/proxy/10.244.20.48/10002",
        "http://10.244.20.48:10002"  # Direct connection
    ]
    
    print("Testing routing agent...")
    print("-" * 50)
    
    # Test 1: Health check
    for url in urls:
        try:
            print(f"Test 1: Health check for {url}")
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{url}/health")
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    print(f"Response: {response.json()}")
                else:
                    print(f"Response: {response.text[:200]}...")
        except Exception as e:
            print(f"Failed: {e}")
        print()
    
    # Test 2: Agent card
    for url in urls:
        try:
            print(f"Test 2: Agent card for {url}")
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{url}/.well-known/agent.json")
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    agent_card = response.json()
                    print(f"Agent: {agent_card.get('name', 'Unknown')}")
                    print(f"Transport: {agent_card.get('preferredTransport', 'Unknown')}")
                else:
                    print(f"Response: {response.text[:200]}...")
        except Exception as e:
            print(f"Failed: {e}")
        print()
    
    # Test 3: Simple message
    simple_message = "Hello, how are you?"
    for url in urls:
        try:
            print(f"Test 3: Simple message for {url}")
            async with httpx.AsyncClient(timeout=30) as client:
                payload = {
                    "jsonrpc": "2.0",
                    "id": "test-123",
                    "method": "sendMessage",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [
                                {"kind": "text", "text": simple_message}
                            ],
                            "message_id": "test-message-123"
                        }
                    }
                }
                
                response = await client.post(url, json=payload, timeout=30)
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    print("Success!")
                    result = response.json()
                    print(f"Response ID: {result.get('id', 'Unknown')}")
                else:
                    print(f"Response: {response.text[:500]}...")
        except Exception as e:
            print(f"Failed: {e}")
        print()

if __name__ == "__main__":
    asyncio.run(test_routing_agent())
