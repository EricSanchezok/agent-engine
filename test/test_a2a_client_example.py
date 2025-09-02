"""
Complete example of using A2A client functionality.

This script demonstrates various ways to use the A2A client wrapper.
"""

import asyncio
import json
from typing import Dict, Any

from agent_engine.a2a_client import (
    A2AClientWrapper, 
    send_message_to_a2a_agent, 
    send_message_to_a2a_agent_streaming
)


async def example_using_context_manager():
    """Example using the context manager approach."""
    print("Example 1: Using Context Manager")
    print("=" * 50)
    
    base_url = "http://10.12.16.139:9900"
    message = "What is the current time?"
    
    try:
        async with A2AClientWrapper(base_url, timeout=15.0) as client:
            print(f"Connected to: {base_url}")
            print(f"Sending message: {message}")
            
            response = await client.send_message(message)
            print("Response received:")
            print(json.dumps(response, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"Error: {e}")


async def example_using_simple_function():
    """Example using the simple helper function."""
    print("\nExample 2: Using Simple Helper Function")
    print("=" * 50)
    
    base_url = "http://10.12.16.139:9900"
    message = "Tell me a joke"
    
    try:
        print(f"Sending message to {base_url}: {message}")
        
        response = await send_message_to_a2a_agent(
            base_url=base_url,
            message=message,
            timeout=15.0
        )
        
        print("Response received:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error: {e}")


async def example_streaming_response():
    """Example using streaming response."""
    print("\nExample 3: Using Streaming Response")
    print("=" * 50)
    
    base_url = "http://10.12.16.139:9900"
    message = "Explain quantum computing in simple terms"
    
    try:
        print(f"Sending streaming message to {base_url}: {message}")
        
        chunk_count = 0
        async for chunk in send_message_to_a2a_agent_streaming(
            base_url=base_url,
            message=message,
            timeout=30.0
        ):
            chunk_count += 1
            print(f"\nChunk {chunk_count}:")
            print(json.dumps(chunk, indent=2, ensure_ascii=False))
            
            # Limit to first 3 chunks for demo
            if chunk_count >= 3:
                print("... (showing first 3 chunks only)")
                break
                
    except Exception as e:
        print(f"Error: {e}")


async def example_multiple_messages():
    """Example sending multiple messages to the same agent."""
    print("\nExample 4: Multiple Messages to Same Agent")
    print("=" * 50)
    
    base_url = "http://10.12.16.139:9900"
    messages = [
        "Hello!",
        "How are you today?",
        "What can you help me with?"
    ]
    
    try:
        async with A2AClientWrapper(base_url, timeout=15.0) as client:
            print(f"Connected to: {base_url}")
            
            for i, message in enumerate(messages, 1):
                print(f"\nMessage {i}: {message}")
                
                try:
                    response = await client.send_message(message)
                    print(f"Response {i} received successfully")
                    
                    # Extract the response text if available
                    if 'result' in response and 'history' in response['result']:
                        history = response['result']['history']
                        if history and 'parts' in history[-1]:
                            parts = history[-1]['parts']
                            if parts and 'text' in parts[0]:
                                print(f"Response text: {parts[0]['text'][:100]}...")
                    
                except Exception as e:
                    print(f"Error sending message {i}: {e}")
                    
    except Exception as e:
        print(f"Error: {e}")


async def example_error_handling():
    """Example demonstrating error handling."""
    print("\nExample 5: Error Handling")
    print("=" * 50)
    
    # Test with invalid URL
    invalid_url = "http://invalid-server-that-does-not-exist:9999"
    message = "This should fail"
    
    try:
        print(f"Testing with invalid URL: {invalid_url}")
        response = await send_message_to_a2a_agent(invalid_url, message, timeout=5.0)
        print("Unexpected success!")
    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}: {e}")
        print("✅ Error handling works correctly")


async def main():
    """Main example function."""
    print("A2A Client Complete Example")
    print("=" * 60)
    
    # Run all examples
    await example_using_context_manager()
    await example_using_simple_function()
    await example_streaming_response()
    await example_multiple_messages()
    await example_error_handling()
    
    print("\n" + "=" * 60)
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
