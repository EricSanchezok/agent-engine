"""
Research Agent Client

A command-line client for interacting with the Research Agent Server.
Supports both regular and streaming chat responses.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

import asyncio
import json
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

import aiohttp
import click

from agent_engine.agent_logger.agent_logger import AgentLogger

logger = AgentLogger(__name__)


class ResearchAgentClient:
    """
    Client for interacting with the Research Agent Server.
    
    Features:
    - Regular chat requests
    - Streaming chat responses
    - Session management
    - Error handling
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the Research Agent Server
        """
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"ResearchAgentClient initialized with base URL: {self.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check server health.
        
        Returns:
            Health status information
        """
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Health check failed with status {response.status}")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    async def chat(
        self,
        user_id: str,
        session_id: str,
        message: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Send a chat message and get a response.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session
            message: User's message
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            
        Returns:
            Chat response
        """
        try:
            payload = {
                "user_id": user_id,
                "session_id": session_id,
                "message": message,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            async with self.session.post(
                f"{self.base_url}/chat",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Chat request failed with status {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            raise
    
    async def chat_stream(
        self,
        user_id: str,
        session_id: str,
        message: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """
        Send a chat message and get a streaming response.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session
            message: User's message
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            
        Returns:
            Complete response text
        """
        try:
            payload = {
                "user_id": user_id,
                "session_id": session_id,
                "message": message,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response_chunks = []
            
            async with self.session.post(
                f"{self.base_url}/chat/stream",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Streaming chat request failed with status {response.status}: {error_text}")
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        try:
                            data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                            
                            if 'chunk' in data:
                                chunk = data['chunk']
                                response_chunks.append(chunk)
                                print(chunk, end='', flush=True)
                            elif 'done' in data:
                                break
                            elif 'error' in data:
                                raise Exception(f"Server error: {data['error']}")
                        except json.JSONDecodeError:
                            continue
            
            return ''.join(response_chunks)
                    
        except Exception as e:
            logger.error(f"Streaming chat request failed: {e}")
            raise
    
    async def get_session_stats(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session
            
        Returns:
            Session statistics
        """
        try:
            async with self.session.get(
                f"{self.base_url}/sessions/{user_id}/{session_id}/stats"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Get session stats failed with status {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Get session stats failed: {e}")
            raise
    
    async def clear_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Clear a session.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session
            
        Returns:
            Clear session response
        """
        try:
            payload = {
                "user_id": user_id,
                "session_id": session_id
            }
            
            async with self.session.post(
                f"{self.base_url}/sessions/clear",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Clear session failed with status {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Clear session failed: {e}")
            raise
    
    async def list_active_sessions(self) -> Dict[str, Any]:
        """
        List all active sessions.
        
        Returns:
            List of active sessions
        """
        try:
            async with self.session.get(f"{self.base_url}/sessions") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"List sessions failed with status {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"List sessions failed: {e}")
            raise


async def interactive_chat(
    base_url: str = "http://localhost:8000",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    stream: bool = False
):
    """
    Interactive chat session with the Research Agent.
    
    Args:
        base_url: Base URL of the Research Agent Server
        user_id: User ID (generated if not provided)
        session_id: Session ID (generated if not provided)
        stream: Whether to use streaming responses
    """
    # Generate IDs if not provided
    if user_id is None:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
    if session_id is None:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    print(f"🤖 Research Agent Client")
    print(f"📡 Server: {base_url}")
    print(f"👤 User ID: {user_id}")
    print(f"💬 Session ID: {session_id}")
    print(f"🌊 Streaming: {'Yes' if stream else 'No'}")
    print("=" * 50)
    
    async with ResearchAgentClient(base_url) as client:
        try:
            # Check server health
            health = await client.health_check()
            print(f"✅ Server is healthy: {health['status']}")
            print()
            
            # Interactive chat loop
            while True:
                try:
                    # Get user input
                    user_input = input("You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if user_input.lower() in ['/quit', '/exit', '/q']:
                        print("👋 Goodbye!")
                        break
                    elif user_input.lower() == '/stats':
                        try:
                            stats = await client.get_session_stats(user_id, session_id)
                            print(f"📊 Session Stats:")
                            print(json.dumps(stats['stats'], indent=2))
                            continue
                        except Exception as e:
                            print(f"❌ Failed to get stats: {e}")
                            continue
                    elif user_input.lower() == '/clear':
                        try:
                            result = await client.clear_session(user_id, session_id)
                            print(f"🗑️  {result['message']}")
                            continue
                        except Exception as e:
                            print(f"❌ Failed to clear session: {e}")
                            continue
                    elif user_input.lower() == '/sessions':
                        try:
                            sessions = await client.list_active_sessions()
                            print(f"📋 Active Sessions:")
                            for session in sessions['sessions']:
                                print(f"  - User: {session['user_id']}, Session: {session['session_id']}")
                            continue
                        except Exception as e:
                            print(f"❌ Failed to list sessions: {e}")
                            continue
                    
                    # Send message to agent
                    print("🤖 Assistant: ", end='', flush=True)
                    
                    if stream:
                        # Streaming response
                        try:
                            response = await client.chat_stream(
                                user_id=user_id,
                                session_id=session_id,
                                message=user_input
                            )
                            print()  # New line after streaming
                        except Exception as e:
                            print(f"❌ Error: {e}")
                    else:
                        # Regular response
                        try:
                            result = await client.chat(
                                user_id=user_id,
                                session_id=session_id,
                                message=user_input
                            )
                            print(result['response'])
                        except Exception as e:
                            print(f"❌ Error: {e}")
                    
                    print()  # Empty line for readability
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
                    
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("Make sure the Research Agent Server is running.")


@click.command()
@click.option('--base-url', default='http://localhost:8000', help='Base URL of the Research Agent Server')
@click.option('--user-id', help='User ID (generated if not provided)')
@click.option('--session-id', help='Session ID (generated if not provided)')
@click.option('--stream/--no-stream', default=True, help='Use streaming responses')
def main(base_url: str, user_id: Optional[str], session_id: Optional[str], stream: bool):
    """
    Research Agent Client - Interactive chat with the Research Agent Server.
    
    Examples:
        python client.py
        python client.py --base-url http://localhost:8000 --stream
        python client.py --user-id myuser --session-id mysession --no-stream
    """
    try:
        asyncio.run(interactive_chat(base_url, user_id, session_id, stream))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
