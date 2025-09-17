#!/usr/bin/env python3
"""
Test Concurrency for Inno-Researcher Service
This script tests multiple concurrent users to demonstrate that the service
can handle multiple users simultaneously without blocking.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import List, Dict, Any

from agent_engine.agent_logger import agent_logger


class ConcurrentTester:
    """Test concurrent user interactions"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
    
    async def simulate_user(self, user_id: str, messages: List[str], delay: float = 0.1) -> Dict[str, Any]:
        """Simulate a single user's conversation"""
        start_time = time.time()
        user_results = {
            "user_id": user_id,
            "start_time": start_time,
            "messages": [],
            "total_time": 0,
            "success": True,
            "error": None
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                session_id = None
                
                for i, message in enumerate(messages):
                    message_start = time.time()
                    
                    payload = {
                        "message": message,
                        "user_id": user_id,
                        "session_id": session_id
                    }
                    
                    async with session.post(
                        f"{self.base_url}/chat",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        message_end = time.time()
                        response_time = message_end - message_start
                        
                        if response.status == 200:
                            data = await response.json()
                            session_id = data["session_id"]
                            
                            user_results["messages"].append({
                                "message": message,
                                "response": data["response"][:100] + "..." if len(data["response"]) > 100 else data["response"],
                                "response_time": response_time,
                                "timestamp": data["timestamp"]
                            })
                            
                            agent_logger.info(f"User {user_id} - Message {i+1}: {response_time:.2f}s")
                        else:
                            error_text = await response.text()
                            user_results["success"] = False
                            user_results["error"] = f"HTTP {response.status}: {error_text}"
                            agent_logger.error(f"User {user_id} - Message {i+1} failed: {response.status}")
                            break
                    
                    # Small delay between messages
                    if delay > 0:
                        await asyncio.sleep(delay)
                
                user_results["total_time"] = time.time() - start_time
                
        except Exception as e:
            user_results["success"] = False
            user_results["error"] = str(e)
            user_results["total_time"] = time.time() - start_time
            agent_logger.error(f"User {user_id} simulation failed: {e}")
        
        return user_results
    
    async def test_concurrent_users(self, num_users: int = 3, messages_per_user: int = 3):
        """Test multiple concurrent users"""
        agent_logger.info(f"Starting concurrent test with {num_users} users, {messages_per_user} messages each")
        
        # Create test messages for each user
        test_messages = [
            f"What is the latest research in artificial intelligence?",
            f"Can you help me find papers about machine learning?",
            f"What are the current trends in natural language processing?",
            f"How does deep learning work?",
            f"What is reinforcement learning?"
        ]
        
        # Create user tasks
        tasks = []
        for i in range(num_users):
            user_id = f"test_user_{i+1}"
            user_messages = test_messages[:messages_per_user]
            task = self.simulate_user(user_id, user_messages, delay=0.1)
            tasks.append(task)
        
        # Run all users concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Process results
        successful_users = 0
        total_messages = 0
        total_response_time = 0
        
        for result in results:
            if isinstance(result, Exception):
                agent_logger.error(f"Task failed with exception: {result}")
                continue
            
            if result["success"]:
                successful_users += 1
                total_messages += len(result["messages"])
                total_response_time += sum(msg["response_time"] for msg in result["messages"])
            
            self.results.append(result)
        
        # Print summary
        print("\n" + "="*60)
        print("CONCURRENT TEST RESULTS")
        print("="*60)
        print(f"Total test time: {total_time:.2f} seconds")
        print(f"Successful users: {successful_users}/{num_users}")
        print(f"Total messages processed: {total_messages}")
        print(f"Average response time: {total_response_time/total_messages:.2f}s" if total_messages > 0 else "N/A")
        print(f"Messages per second: {total_messages/total_time:.2f}")
        
        # Print individual user results
        print("\nIndividual User Results:")
        print("-" * 40)
        for result in self.results:
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            print(f"User {result['user_id']}: {status}")
            print(f"  Total time: {result['total_time']:.2f}s")
            print(f"  Messages: {len(result['messages'])}")
            if result["error"]:
                print(f"  Error: {result['error']}")
            print()
        
        return {
            "total_time": total_time,
            "successful_users": successful_users,
            "total_users": num_users,
            "total_messages": total_messages,
            "average_response_time": total_response_time/total_messages if total_messages > 0 else 0,
            "messages_per_second": total_messages/total_time if total_time > 0 else 0
        }
    
    async def test_service_health(self) -> bool:
        """Test if service is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        agent_logger.info(f"Service health check passed: {data}")
                        return True
                    else:
                        agent_logger.error(f"Service health check failed: {response.status}")
                        return False
        except Exception as e:
            agent_logger.error(f"Service health check error: {e}")
            return False


async def main():
    """Main test function"""
    print("🧪 Inno-Researcher Concurrency Test")
    print("=" * 50)
    
    tester = ConcurrentTester()
    
    # Check service health first
    print("🔍 Checking service health...")
    if not await tester.test_service_health():
        print("❌ Service is not available. Please start the service first.")
        print("Run: uv run agents/ResearchAgent/start_service.py")
        return
    
    print("✅ Service is healthy!")
    
    # Run concurrent test
    print("\n🚀 Starting concurrent user test...")
    results = await tester.test_concurrent_users(num_users=5, messages_per_user=3)
    
    # Final summary
    print("\n📊 FINAL SUMMARY")
    print("=" * 30)
    print(f"✅ Concurrent users handled: {results['successful_users']}/{results['total_users']}")
    print(f"⚡ Total messages processed: {results['total_messages']}")
    print(f"⏱️  Average response time: {results['average_response_time']:.2f}s")
    print(f"🚀 Throughput: {results['messages_per_second']:.2f} messages/second")
    
    if results['successful_users'] == results['total_users']:
        print("\n🎉 SUCCESS: All concurrent users were handled without blocking!")
    else:
        print(f"\n⚠️  WARNING: {results['total_users'] - results['successful_users']} users failed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        agent_logger.error(f"Test error: {e}")
        print(f"❌ Test failed: {e}")
