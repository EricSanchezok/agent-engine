#!/usr/bin/env python3
"""
Inno-Researcher Command Line Client
A simple command-line interface for interacting with the Inno-Researcher service.
"""

import asyncio
import aiohttp
import json
import sys
import argparse
from typing import Optional, Dict, Any
from datetime import datetime

from agent_engine.agent_logger import agent_logger


class InnoResearcherClient:
    """Client for interacting with Inno-Researcher service"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_id: Optional[str] = None
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    agent_logger.info(f"Service health check: {data}")
                    return True
                else:
                    agent_logger.error(f"Health check failed with status: {response.status}")
                    return False
        except Exception as e:
            agent_logger.error(f"Health check error: {e}")
            return False
    
    async def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Send a chat message to the service"""
        try:
            payload = {
                "message": message,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "context": context or {}
            }
            
            async with self.session.post(
                f"{self.base_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.session_id = data["session_id"]
                    return data["response"]
                else:
                    error_text = await response.text()
                    agent_logger.error(f"Chat request failed: {response.status} - {error_text}")
                    return f"Error: {response.status} - {error_text}"
        
        except Exception as e:
            agent_logger.error(f"Chat error: {e}")
            return f"Error: {e}"
    
    async def research(self, query: str, research_type: str = "general", 
                      parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Conduct research using the service"""
        try:
            payload = {
                "query": query,
                "user_id": self.user_id,
                "research_type": research_type,
                "parameters": parameters or {}
            }
            
            async with self.session.post(
                f"{self.base_url}/research",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    agent_logger.error(f"Research request failed: {response.status} - {error_text}")
                    return {"error": f"{response.status} - {error_text}"}
        
        except Exception as e:
            agent_logger.error(f"Research error: {e}")
            return {"error": str(e)}
    
    async def get_sessions(self) -> Dict[str, Any]:
        """Get user sessions"""
        try:
            async with self.session.get(f"{self.base_url}/sessions/{self.user_id}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    agent_logger.error(f"Get sessions failed: {response.status} - {error_text}")
                    return {"error": f"{response.status} - {error_text}"}
        
        except Exception as e:
            agent_logger.error(f"Get sessions error: {e}")
            return {"error": str(e)}


async def interactive_chat(client: InnoResearcherClient):
    """Interactive chat mode"""
    print("🤖 Welcome to Inno-Researcher!")
    print("Type your research questions or chat messages. Type 'quit' to exit.")
    print("Commands:")
    print("  /research <query> - Conduct research")
    print("  /sessions - Show your sessions")
    print("  /help - Show this help")
    print("  /quit - Exit the program")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.startswith('/'):
                await handle_command(client, user_input)
                continue
            
            # Regular chat
            print("🤖 Inno-Researcher: ", end="", flush=True)
            response = await client.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            agent_logger.error(f"Interactive chat error: {e}")
            print(f"❌ Error: {e}")


async def handle_command(client: InnoResearcherClient, command: str):
    """Handle special commands"""
    parts = command.split(' ', 1)
    cmd = parts[0].lower()
    
    if cmd == '/research':
        if len(parts) < 2:
            print("❌ Please provide a research query: /research <query>")
            return
        
        query = parts[1]
        print(f"🔍 Conducting research: {query}")
        print("⏳ Please wait...")
        
        result = await client.research(query)
        
        if "error" in result:
            print(f"❌ Research failed: {result['error']}")
        else:
            print("✅ Research completed!")
            print(f"📊 Results: {json.dumps(result['results'], indent=2, ensure_ascii=False)}")
    
    elif cmd == '/sessions':
        sessions = await client.get_sessions()
        if "error" in sessions:
            print(f"❌ Failed to get sessions: {sessions['error']}")
        else:
            print(f"📋 Your sessions ({sessions['total_sessions']} total):")
            for session_id, session_data in sessions['sessions'].items():
                print(f"  - {session_id}: {len(session_data['messages'])} messages")
    
    elif cmd == '/help':
        print("Available commands:")
        print("  /research <query> - Conduct research")
        print("  /sessions - Show your sessions")
        print("  /help - Show this help")
        print("  /quit - Exit the program")
    
    else:
        print(f"❌ Unknown command: {cmd}. Type /help for available commands.")


async def single_chat(client: InnoResearcherClient, message: str):
    """Single message mode"""
    print(f"👤 You: {message}")
    print("🤖 Inno-Researcher: ", end="", flush=True)
    response = await client.chat(message)
    print(response)


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Inno-Researcher Command Line Client")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Service URL (default: http://localhost:8000)")
    parser.add_argument("--message", "-m", help="Single message to send")
    parser.add_argument("--research", "-r", help="Research query to conduct")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive mode (default)")
    
    args = parser.parse_args()
    
    async with InnoResearcherClient(args.url) as client:
        # Health check
        print("🔍 Checking service health...")
        if not await client.health_check():
            print("❌ Service is not available. Please make sure the service is running.")
            sys.exit(1)
        
        print("✅ Service is healthy!")
        
        if args.research:
            # Research mode
            print(f"🔍 Conducting research: {args.research}")
            result = await client.research(args.research)
            if "error" in result:
                print(f"❌ Research failed: {result['error']}")
            else:
                print("✅ Research completed!")
                print(f"📊 Results: {json.dumps(result['results'], indent=2, ensure_ascii=False)}")
        
        elif args.message:
            # Single message mode
            await single_chat(client, args.message)
        
        else:
            # Interactive mode (default)
            await interactive_chat(client)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        agent_logger.error(f"Client error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)
