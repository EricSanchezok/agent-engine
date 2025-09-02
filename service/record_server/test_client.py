#!/usr/bin/env python3
"""
Test client for Record Memory Server

This script demonstrates how to use the Record Memory Server API.
"""

import requests
import json
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Server configuration
BASE_URL = "http://localhost:5050"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_get_all_capabilities():
    """Test getting all capabilities"""
    print("\nTesting get all capabilities...")
    try:
        response = requests.get(f"{BASE_URL}/capabilities")
        capabilities = response.json()
        print(f"Found {len(capabilities)} capabilities")
        for cap in capabilities[:3]:  # Show first 3
            print(f"  - {cap['name']}")
        return True
    except Exception as e:
        print(f"Get capabilities failed: {e}")
        return False

def test_search_capabilities():
    """Test searching for similar capabilities"""
    print("\nTesting search capabilities...")
    try:
        search_data = {
            "name": "Chat Assistant",
            "definition": "A conversational AI service for user interaction",
            "top_k": 3,
            "threshold": 0.5
        }
        response = requests.post(f"{BASE_URL}/capabilities/search", json=search_data)
        results = response.json()
        print(f"Found {len(results)} similar capabilities")
        for result in results:
            print(f"  - {result['name']} (score: {result.get('similarity_score', 'N/A')})")
        return True
    except Exception as e:
        print(f"Search capabilities failed: {e}")
        return False

def test_get_agents_for_capability():
    """Test getting agents for a capability"""
    print("\nTesting get agents for capability...")
    try:
        # First get all capabilities to find one to test with
        response = requests.get(f"{BASE_URL}/capabilities")
        capabilities = response.json()
        
        if capabilities:
            capability = capabilities[0]
            agent_data = {
                "name": capability["name"],
                "definition": capability["definition"]
            }
            response = requests.post(f"{BASE_URL}/capabilities/agents", json=agent_data)
            agents = response.json()
            print(f"Found {len(agents)} agents for capability '{capability['name']}'")
            for agent in agents:
                print(f"  - {agent['name']} ({agent['url']})")
            return True
        else:
            print("No capabilities found to test with")
            return False
    except Exception as e:
        print(f"Get agents for capability failed: {e}")
        return False

def test_add_task_result():
    """Test adding a task result"""
    print("\nTesting add task result...")
    try:
        task_data = {
            "agent_name": "Simple Test Agent",
            "agent_url": "http://10.245.134.199:10001",
            "capability_name": "Chat with Conversational AI Assistant",
            "capability_definition": "This service enables users to engage in a natural language conversation with a conversational AI assistant. It accepts text queries and provides concise, clear, and helpful responses that may include answers, explanations, summaries, or clarifications. It supports multi-turn dialogues and handling of complex topics.",
            "success": True,
            "task_content": "Test task content",
            "task_result": "Test task result"
        }
        response = requests.post(f"{BASE_URL}/task-result", json=task_data)
        result = response.json()
        print(f"Task result added: {result}")
        return True
    except Exception as e:
        print(f"Add task result failed: {e}")
        return False

def test_get_capability_performance():
    """Test getting capability performance"""
    print("\nTesting get capability performance...")
    try:
        performance_data = {
            "name": "Test Capability",
            "definition": "A test capability for demonstration purposes"
        }
        response = requests.post(f"{BASE_URL}/capabilities/performance", json=performance_data)
        performance = response.json()
        print(f"Found performance data for {len(performance)} agents")
        for perf in performance:
            print(f"  - {perf['name']}: {perf['success_count']}/{perf['total_count']} ({perf['success_rate']:.2%})")
        return True
    except Exception as e:
        print(f"Get capability performance failed: {e}")
        return False

def test_get_all_agents():
    """Test getting all agents"""
    print("\nTesting get all agents...")
    try:
        response = requests.get(f"{BASE_URL}/agents")
        agents = response.json()
        print(f"Found {len(agents)} unique agents")
        for agent in agents:
            print(f"  - {agent['name']} ({agent['url']})")
        return True
    except Exception as e:
        print(f"Get all agents failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Record Memory Server Test Client")
    print("=" * 40)
    
    # Check if server is running
    if not test_health_check():
        print("Server is not running. Please start the server first.")
        return
    
    # Run tests
    tests = [
        # test_get_all_capabilities,
        # test_search_capabilities,
        # test_get_agents_for_capability,
        test_add_task_result,
        # test_get_capability_performance,
        # test_get_all_agents
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The server is working correctly.")
    else:
        print("Some tests failed. Please check the server logs for more details.")

if __name__ == "__main__":
    main()
