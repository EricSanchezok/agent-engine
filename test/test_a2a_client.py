"""
Test script for A2A client functionality.

This script tests the A2A client wrapper with a real A2A agent service.
"""

import asyncio
import json
from typing import Dict, Any

from agent_engine.a2a_client import A2AClientWrapper, send_message_to_a2a_agent, send_message_to_a2a_agent_streaming


async def test_a2a_client_wrapper():
    """Test the A2AClientWrapper class."""
    base_url = "http://10.12.16.139:9900"
    test_message = "Hello! How much is 10 USD in INR?"
    
    print("Testing A2AClientWrapper...")
    print(f"Base URL: {base_url}")
    print(f"Test message: {test_message}")
    print("-" * 50)
    
    try:
        async with A2AClientWrapper(base_url) as client:
            print("Successfully connected to A2A agent")
            
            # Test regular message sending
            print("\nSending regular message...")
            response = await client.send_message(test_message)
            print("Response received:")
            print(json.dumps(response, indent=2, ensure_ascii=False))
            
            # Test streaming message sending
            print("\nSending streaming message...")
            print("Streaming response chunks:")
            chunk_count = 0
            async for chunk in client.send_message_streaming(test_message):
                chunk_count += 1
                print(f"Chunk {chunk_count}:")
                print(json.dumps(chunk, indent=2, ensure_ascii=False))
                if chunk_count >= 5:  # Limit to first 5 chunks for demo
                    break
                    
    except Exception as e:
        print(f"Error testing A2AClientWrapper: {e}")
        raise


async def test_simple_functions():
    """Test the simple helper functions."""
    base_url = "http://10.12.16.139:9900"
    test_message = "What is the weather like today?"
    
    print("\n" + "=" * 60)
    print("Testing simple helper functions...")
    print(f"Base URL: {base_url}")
    print(f"Test message: {test_message}")
    print("-" * 50)
    
    try:
        # Test send_message_to_a2a_agent function
        print("\nTesting send_message_to_a2a_agent function...")
        response = await send_message_to_a2a_agent(base_url, test_message)
        print("Response received:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        
        # Test streaming function
        print("\nTesting send_message_to_a2a_agent_streaming function...")
        print("Streaming response chunks:")
        chunk_count = 0
        async for chunk in send_message_to_a2a_agent_streaming(base_url, test_message):
            chunk_count += 1
            print(f"Chunk {chunk_count}:")
            print(json.dumps(chunk, indent=2, ensure_ascii=False))
            if chunk_count >= 3:  # Limit to first 3 chunks for demo
                break
                
    except Exception as e:
        print(f"Error testing simple functions: {e}")
        raise


async def test_error_handling():
    """Test error handling with invalid URL."""
    invalid_url = "http://invalid-url-that-does-not-exist:9999"
    test_message = "This should fail"
    
    print("\n" + "=" * 60)
    print("Testing error handling...")
    print(f"Invalid URL: {invalid_url}")
    print(f"Test message: {test_message}")
    print("-" * 50)
    
    try:
        response = await send_message_to_a2a_agent(invalid_url, test_message)
        print("Unexpected success - should have failed!")
    except Exception as e:
        print(f"Expected error caught: {e}")
        print("Error handling test passed!")


async def main():
    """Main test function."""
    print("A2A Client Test Suite")
    print("=" * 60)
    
    # Test with valid URL
    await test_a2a_client_wrapper()
    await test_simple_functions()
    
    # Test error handling
    await test_error_handling()
    
    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
