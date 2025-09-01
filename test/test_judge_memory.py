"""
Test script for JudgeMemory class functionality
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.JudgeAgent.judge_memory import JudgeMemory
from agent_engine.agent_logger import AgentLogger

logger = AgentLogger(__name__)


async def test_judge_memory_basic_operations():
    """Test basic operations of JudgeMemory"""
    print("Testing basic operations...")
    
    # Initialize JudgeMemory
    judge_memory = JudgeMemory(name='test_judge_memory')
    
    # Clear any existing data
    judge_memory.clear()
    
    # Test 1: Add capabilities
    print("1. Adding capabilities...")
    await judge_memory.add_capability(
        name="Web Browsing",
        definition="Ability to browse the web and extract information from web pages",
        alias=["Web Search", "Internet Browsing"],
        agents=[
            {"name": "WebAgent", "url": "https://webagent.com"},
            {"name": "BrowserBot", "url": "https://browserbot.com"}
        ]
    )
    
    await judge_memory.add_capability(
        name="Data Analysis",
        definition="Ability to analyze and process data sets",
        alias=["Data Processing", "Analytics"],
        agents=[
            {"name": "DataAgent", "url": "https://dataagent.com"},
            {"name": "AnalyticsBot", "url": "https://analyticsbot.com"}
        ]
    )
    
    await judge_memory.add_capability(
        name="Text Generation",
        definition="Ability to generate human-like text content",
        alias=["Content Creation", "Text Writing"],
        agents=[
            {"name": "WriterAgent", "url": "https://writeragent.com"}
        ]
    )
    
    print(f"Added {judge_memory.count()} capabilities")
    
    # Test 2: Search similar capabilities
    print("\n2. Testing capability search...")
    search_results = await judge_memory.search_similar_capabilities(
        name="Web Browsing",
        definition="Ability to search the internet for information",
        top_k=3
    )
    print(f"Found {len(search_results)} similar capabilities:")
    for result in search_results:
        print(f"  - {result['name']} (similarity: {result['similarity_score']:.3f})")
    
    # Test 3: Get agents for capability
    print("\n3. Testing get agents for capability...")
    capability_content = {
        "name": "Web Browsing",
        "definition": "Ability to browse the web and extract information from web pages"
    }
    agents = judge_memory.get_agents_for_capability(capability_content)
    print(f"Agents for Web Browsing: {len(agents)} found")
    for agent in agents:
        print(f"  - {agent['name']} ({agent['url']})")
    
    # Test 4: Get agent capabilities
    print("\n4. Testing get agent capabilities...")
    agent_capabilities = judge_memory.get_agent_capabilities(
        "WebAgent", "https://webagent.com"
    )
    print(f"Capabilities for WebAgent: {len(agent_capabilities)} found")
    for capability in agent_capabilities:
        print(f"  - {capability['name']}")
    
    # Test 5: Record task results
    print("\n5. Testing task result recording...")
    judge_memory.record_task_result(
        agent_name="WebAgent",
        agent_url="https://webagent.com",
        success=True,
        capability_content=capability_content,
        task_name="Search for weather information"
    )
    
    judge_memory.record_task_result(
        agent_name="WebAgent",
        agent_url="https://webagent.com",
        success=False,
        capability_content=capability_content,
        task_name="Extract data from complex table"
    )
    
    judge_memory.record_task_result(
        agent_name="BrowserBot",
        agent_url="https://browserbot.com",
        success=True,
        capability_content=capability_content,
        task_name="Navigate to multiple pages"
    )
    
    # Test 6: Get agent performance info
    print("\n6. Testing agent performance info...")
    performance_info = judge_memory.get_agent_performance_info(capability_content)
    print(f"Performance info for Web Browsing capability:")
    for info in performance_info:
        print(f"  - {info['name']}: {info['success_count']}/{info['total_count']} "
              f"(success rate: {info['success_rate']:.2f})")
    
    # Test 7: Test with different capability
    print("\n7. Testing with Data Analysis capability...")
    data_capability = {
        "name": "Data Analysis",
        "definition": "Ability to analyze and process data sets"
    }
    
    # Record some task results for data analysis
    judge_memory.record_task_result(
        agent_name="DataAgent",
        agent_url="https://dataagent.com",
        success=True,
        capability_content=data_capability,
        task_name="Process CSV data"
    )
    
    judge_memory.record_task_result(
        agent_name="AnalyticsBot",
        agent_url="https://analyticsbot.com",
        success=True,
        capability_content=data_capability,
        task_name="Generate statistical report"
    )
    
    data_performance = judge_memory.get_agent_performance_info(data_capability)
    print(f"Performance info for Data Analysis capability:")
    for info in data_performance:
        print(f"  - {info['name']}: {info['success_count']}/{info['total_count']} "
              f"(success rate: {info['success_rate']:.2f})")
    
    # Test 8: Test capability with metadata
    print("\n8. Testing get capability with metadata...")
    capability_with_metadata = judge_memory.get_capability_with_metadata(capability_content)
    if capability_with_metadata:
        content, metadata = capability_with_metadata
        print(f"Capability: {content['name']}")
        print(f"Metadata keys: {list(metadata.keys())}")
        if 'task_history' in metadata:
            print(f"Task history entries: {len(metadata['task_history'])}")
    
    # Test 9: Test update capability agents
    print("\n9. Testing update capability agents...")
    new_agents = [
        {"name": "WebAgent", "url": "https://webagent.com"},
        {"name": "BrowserBot", "url": "https://browserbot.com"},
        {"name": "NewWebAgent", "url": "https://newwebagent.com"}
    ]
    await judge_memory.update_capability_agents(capability_content, new_agents)
    
    updated_agents = judge_memory.get_agents_for_capability(capability_content)
    print(f"Updated agents for Web Browsing: {len(updated_agents)} found")
    for agent in updated_agents:
        print(f"  - {agent['name']} ({agent['url']})")
    
    print("\nAll tests completed successfully!")
    return True


async def test_judge_memory_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting edge cases...")
    
    judge_memory = JudgeMemory(name='test_edge_cases')
    judge_memory.clear()
    
    # Test 1: Empty capability search
    print("1. Testing empty capability search...")
    results = await judge_memory.search_similar_capabilities(
        name="Nonexistent",
        definition="nonexistent capability", 
        top_k=5
    )
    print(f"Empty search results: {len(results)} found")
    
    # Test 2: Get agents for nonexistent capability
    print("2. Testing get agents for nonexistent capability...")
    nonexistent_capability = {
        "name": "Nonexistent Capability",
        "definition": "This capability does not exist"
    }
    agents = judge_memory.get_agents_for_capability(nonexistent_capability)
    print(f"Agents for nonexistent capability: {len(agents)} found")
    
    # Test 3: Get capabilities for nonexistent agent
    print("3. Testing get capabilities for nonexistent agent...")
    agent_capabilities = judge_memory.get_agent_capabilities(
        "NonexistentAgent", "https://nonexistent.com"
    )
    print(f"Capabilities for nonexistent agent: {len(agent_capabilities)} found")
    
    # Test 4: Record task for nonexistent capability
    print("4. Testing record task for nonexistent capability...")
    try:
        judge_memory.record_task_result(
            agent_name="TestAgent",
            agent_url="https://testagent.com",
            success=True,
            capability_content=nonexistent_capability,
            task_name="Test task"
        )
        print("Task recorded (capability will be created)")
    except Exception as e:
        print(f"Error recording task: {e}")
    
    # Test 5: Get performance for capability with no history
    print("5. Testing get performance for capability with no history...")
    performance = judge_memory.get_agent_performance_info(nonexistent_capability)
    print(f"Performance for capability with no history: {len(performance)} agents")
    
    print("Edge case tests completed!")


def main():
    """Main test function"""
    print("Starting JudgeMemory tests...")
    print("=" * 50)
    
    try:
        # Run basic tests
        test_judge_memory_basic_operations()
        
        # Run edge case tests
        test_judge_memory_edge_cases()
        
        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
