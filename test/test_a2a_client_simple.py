"""
Simple test script for A2A client functionality.

This script tests basic connectivity and message sending to A2A agents.
"""

import asyncio
import json
from typing import Dict, Any

from agent_engine.a2a_client import A2AClientWrapper


async def test_connection_only():
    """Test only the connection to the A2A agent."""
    base_url = "http://10.12.16.139:9900"
    
    print("Testing A2A Client Connection")
    print("=" * 50)
    print(f"Base URL: {base_url}")
    print("-" * 50)
    
    try:
        async with A2AClientWrapper(base_url, timeout=10.0) as client:
            print("✅ Successfully connected to A2A agent")
            print(f"Agent card fetched successfully")
            
            # Try a simple message
            test_message = "Hello! Can you respond with a simple greeting?"
            print(f"\nSending test message: {test_message}")
            
            try:
                response = await client.send_message(test_message)
                print("✅ Successfully received response:")
                print(json.dumps(response, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"❌ Error sending message: {e}")
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try to provide more helpful error information
        if "timeout" in str(e).lower():
            print("\n💡 Suggestion: The server might be slow or unreachable.")
            print("   Try increasing the timeout or check if the server is running.")
        elif "connection" in str(e).lower():
            print("\n💡 Suggestion: Check if the server is running and accessible.")
            print("   Verify the URL and port number.")


async def test_localhost():
    """Test with localhost to see if the code works with a local server."""
    base_url = "http://localhost:9999"
    
    print("\n" + "=" * 60)
    print("Testing with localhost (for comparison)")
    print(f"Base URL: {base_url}")
    print("-" * 50)
    
    try:
        async with A2AClientWrapper(base_url, timeout=5.0) as client:
            print("✅ Successfully connected to localhost A2A agent")
            
            test_message = "Hello from localhost test!"
            print(f"\nSending test message: {test_message}")
            
            try:
                response = await client.send_message(test_message)
                print("✅ Successfully received response:")
                print(json.dumps(response, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"❌ Error sending message: {e}")
                
    except Exception as e:
        print(f"❌ Localhost connection failed: {e}")
        print("This is expected if no local A2A server is running.")


async def main():
    """Main test function."""
    print("A2A Client Simple Test")
    print("=" * 60)
    
    # Test the target server
    await test_connection_only()
    
    # Test localhost for comparison
    await test_localhost()
    
    print("\n" + "=" * 60)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
