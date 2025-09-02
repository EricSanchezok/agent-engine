"""
Test script for delete_task_result method in RecordMemory
"""

import asyncio
import json
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.record_server.record_memory import RecordMemory


async def test_delete_task_result():
    """Test the delete_task_result method"""
    print("Starting test for delete_task_result method...")
    
    # Initialize RecordMemory
    memory = RecordMemory()
    
    # Test data from capabilities.json
    capability_name = "Chat with Conversational AI Assistant"
    capability_definition = "This service enables users to engage in a natural language conversation with a conversational AI assistant. It accepts text queries and provides concise, clear, and helpful responses that may include answers, explanations, summaries, or clarifications. It supports multi-turn dialogues and handling of complex topics."
    
    agent_name = "OpenManus Agent"
    agent_url = "http://10.245.133.169:10003/"
    
    # Test task data
    task_content = "Hello, how are you?"
    task_result = "I'm doing well, thank you for asking!"
    
    print(f"\n1. Adding test task result...")
    print(f"   Capability: {capability_name}")
    print(f"   Agent: {agent_name}")
    print(f"   Task Content: {task_content}")
    print(f"   Task Result: {task_result}")
    
    # Add a task result
    await memory.add_task_result(
        agent_name=agent_name,
        agent_url=agent_url,
        capability_name=capability_name,
        capability_definition=capability_definition,
        success=True,
        task_content=task_content,
        task_result=task_result
    )
    
    print("   Task result added successfully!")
    
    # Check the performance before deletion
    print(f"\n2. Checking performance before deletion...")
    performance = await memory.get_capability_performance(capability_name, capability_definition)
    for perf in performance:
        if perf['name'] == agent_name:
            print(f"   Success Count: {perf['success_count']}")
            print(f"   Total Count: {perf['total_count']}")
            print(f"   Success Rate: {perf['success_rate']:.2f}")
            break
    
    # Get task history to verify the task was added
    print(f"\n3. Getting task history to verify task was added...")
    history = await memory.get_capability_history(capability_name, capability_definition)
    agent_key = f"{agent_name}_{agent_url}"
    if agent_key in history:
        tasks = history[agent_key]['tasks']
        print(f"   Found {len(tasks)} task(s) in history")
        for i, task in enumerate(tasks):
            print(f"   Task {i+1}: {task['task_content']} -> {task['task_result']} (Success: {task['success']})")
    
    # Delete the task result
    print(f"\n4. Deleting the task result...")
    await memory.delete_task_result(
        agent_name=agent_name,
        agent_url=agent_url,
        capability_name=capability_name,
        capability_definition=capability_definition,
        task_content=task_content,
        task_result=task_result
    )
    
    print("   Task result deleted!")
    
    # Check the performance after deletion
    print(f"\n5. Checking performance after deletion...")
    performance = await memory.get_capability_performance(capability_name, capability_definition)
    for perf in performance:
        if perf['name'] == agent_name:
            print(f"   Success Count: {perf['success_count']}")
            print(f"   Total Count: {perf['total_count']}")
            print(f"   Success Rate: {perf['success_rate']:.2f}")
            break
    
    # Get task history to verify the task was deleted
    print(f"\n6. Getting task history to verify task was deleted...")
    history = await memory.get_capability_history(capability_name, capability_definition)
    if agent_key in history:
        tasks = history[agent_key]['tasks']
        print(f"   Found {len(tasks)} task(s) in history after deletion")
        if len(tasks) == 0:
            print("   ✓ Task successfully deleted!")
        else:
            print("   ✗ Task still exists in history")
            for i, task in enumerate(tasks):
                print(f"   Remaining Task {i+1}: {task['task_content']} -> {task['task_result']}")
    else:
        print("   No task history found for this agent")
    
    # Test with non-existent task
    print(f"\n7. Testing deletion of non-existent task...")
    await memory.delete_task_result(
        agent_name=agent_name,
        agent_url=agent_url,
        capability_name=capability_name,
        capability_definition=capability_definition,
        task_content="Non-existent task",
        task_result="Non-existent result"
    )
    
    print("   Test completed!")
    
    # Test with different capability
    print(f"\n8. Testing with different capability...")
    capability_name_2 = "Answer Questions"
    capability_definition_2 = "This capability allows users to ask natural language questions and receive text-based responses. It covers both general inquiries and Gaia-related topics by leveraging the appropriate task agent (such as an OWL-based general AI assistant or a Gaia Task Agent) to provide accurate answers."
    
    agent_name_2 = "General Task Agent"
    agent_url_2 = "http://10.244.21.160:10003/"
    
    task_content_2 = "What is machine learning?"
    task_result_2 = "Machine learning is a subset of artificial intelligence..."
    
    print(f"   Adding task for different capability...")
    await memory.add_task_result(
        agent_name=agent_name_2,
        agent_url=agent_url_2,
        capability_name=capability_name_2,
        capability_definition=capability_definition_2,
        success=False,
        task_content=task_content_2,
        task_result=task_result_2
    )
    
    print(f"   Deleting task for different capability...")
    await memory.delete_task_result(
        agent_name=agent_name_2,
        agent_url=agent_url_2,
        capability_name=capability_name_2,
        capability_definition=capability_definition_2,
        task_content=task_content_2,
        task_result=task_result_2
    )
    
    print("   ✓ All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_delete_task_result())
