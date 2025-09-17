"""
Researcher class for handling research conversations with LLM.

This module provides:
- Integration with AzureClient for LLM communication
- Support for streaming responses
- Session-based conversation management
- Context-aware responses using SessionMemory
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
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime

from agent_engine.llm_client.azure_client import AzureClient
from agent_engine.agent_logger.agent_logger import AgentLogger
from session_memory import SessionMemory

logger = AgentLogger(__name__)


class Researcher:
    """
    Research agent that handles conversations with users.
    
    Features:
    - Integration with AzureClient for LLM communication
    - Support for streaming responses
    - Session-based conversation management
    - Context-aware responses using SessionMemory
    - Error handling and logging
    """
    
    def __init__(
        self,
        azure_api_key: str,
        azure_base_url: str = 'https://gpt.yunstorm.com/',
        azure_api_version: str = '2025-04-01-preview',
        model_name: str = 'gpt-4.1',
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the Researcher.
        
        Args:
            azure_api_key: Azure OpenAI API key
            azure_base_url: Azure OpenAI base URL
            azure_api_version: Azure OpenAI API version
            model_name: Model name to use (default: gpt-4.1)
            system_prompt: System prompt for the LLM
        """
        self.model_name = model_name
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Initialize Azure client
        self.azure_client = AzureClient(
            api_key=azure_api_key,
            base_url=azure_base_url,
            api_version=azure_api_version
        )
        
        # Session memory instances (keyed by user_id:session_id)
        self.session_memories: Dict[str, SessionMemory] = {}
        
        logger.info(f"Researcher initialized with model: {model_name}")
        logger.info(f"Azure endpoint: {azure_base_url}")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for the research agent"""
        return """You are a helpful research assistant. You can help users with various research tasks including:

1. Answering questions about various topics
2. Providing explanations and clarifications
3. Helping with analysis and synthesis of information
4. Assisting with research methodology and approaches

Please provide accurate, helpful, and well-structured responses. If you're unsure about something, please say so rather than guessing.

Always respond in a clear and professional manner, and feel free to ask follow-up questions if you need more information to provide a better answer."""

    def _get_session_key(self, user_id: str, session_id: str) -> str:
        """Generate session key for memory lookup"""
        return f"{user_id}:{session_id}"
    
    def _get_or_create_session_memory(self, user_id: str, session_id: str) -> SessionMemory:
        """Get existing session memory or create new one"""
        session_key = self._get_session_key(user_id, session_id)
        
        if session_key not in self.session_memories:
            logger.info(f"Creating new session memory for user={user_id}, session={session_id}")
            self.session_memories[session_key] = SessionMemory(
                user_id=user_id,
                session_id=session_id
            )
        
        return self.session_memories[session_key]
    
    async def chat(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        stream: bool = False,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Process a chat message from a user.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session
            user_message: User's message
            stream: Whether to use streaming response
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            
        Returns:
            Assistant's response (None if streaming)
        """
        try:
            logger.info(f"Processing chat message from user={user_id}, session={session_id}")
            logger.info(f"User message: {user_message[:100]}...")
            
            # Get session memory
            session_memory = self._get_or_create_session_memory(user_id, session_id)
            
            # Get context from session memory
            context = session_memory.get_context_for_llm()
            
            # Prepare messages for LLM
            messages = []
            
            # Add system prompt
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
            
            # Add context if available
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Previous conversation context:\n{context}"
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Get response from LLM
            if stream:
                # For streaming, we'll collect the full response
                response_chunks = []
                async for chunk in self.azure_client.chat_stream(
                    system_prompt=self.system_prompt,
                    user_prompt=user_message,
                    model_name=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature
                ):
                    response_chunks.append(chunk)
                
                response = "".join(response_chunks)
            else:
                # Non-streaming response
                response = await self.azure_client.call_llm(
                    model_name=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            if response:
                # Add Q&A to session memory
                session_memory.add_qa(user_message, response)
                
                logger.info(f"Response generated successfully. Length: {len(response)} chars")
                return response
            else:
                logger.error("Failed to generate response from LLM")
                return "I apologize, but I'm having trouble generating a response right now. Please try again."
                
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    async def chat_stream(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> AsyncIterator[str]:
        """
        Process a chat message with streaming response.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session
            user_message: User's message
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            
        Yields:
            Response chunks as they are generated
        """
        try:
            logger.info(f"Processing streaming chat message from user={user_id}, session={session_id}")
            logger.info(f"User message: {user_message[:100]}...")
            
            # Get session memory
            session_memory = self._get_or_create_session_memory(user_id, session_id)
            
            # Get context from session memory
            context = session_memory.get_context_for_llm()
            
            # Prepare user prompt with context
            user_prompt = user_message
            if context:
                user_prompt = f"Previous conversation context:\n{context}\n\nCurrent question: {user_message}"
            
            # Stream response from LLM
            response_chunks = []
            async for chunk in self.azure_client.chat_stream(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                model_name=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                response_chunks.append(chunk)
                yield chunk
            
            # Add complete Q&A to session memory
            if response_chunks:
                full_response = "".join(response_chunks)
                session_memory.add_qa(user_message, full_response)
                logger.info(f"Streaming response completed. Length: {len(full_response)} chars")
            
        except Exception as e:
            logger.error(f"Error in streaming chat processing: {e}")
            yield f"I apologize, but I encountered an error: {str(e)}"
    
    def get_session_stats(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific session.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session
            
        Returns:
            Dictionary with session statistics
        """
        try:
            session_key = self._get_session_key(user_id, session_id)
            if session_key in self.session_memories:
                return self.session_memories[session_key].get_session_stats()
            else:
                return {"error": "Session not found"}
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {"error": str(e)}
    
    def clear_session(self, user_id: str, session_id: str) -> bool:
        """
        Clear a specific session.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session
            
        Returns:
            True if session was cleared, False otherwise
        """
        try:
            session_key = self._get_session_key(user_id, session_id)
            if session_key in self.session_memories:
                self.session_memories[session_key].clear_session()
                logger.info(f"Cleared session for user={user_id}, session={session_id}")
                return True
            else:
                logger.warning(f"Session not found for user={user_id}, session={session_id}")
                return False
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False
    
    def list_active_sessions(self) -> List[Dict[str, str]]:
        """
        List all active sessions.
        
        Returns:
            List of dictionaries with user_id and session_id
        """
        try:
            sessions = []
            for session_key in self.session_memories.keys():
                user_id, session_id = session_key.split(":", 1)
                sessions.append({
                    "user_id": user_id,
                    "session_id": session_id
                })
            return sessions
        except Exception as e:
            logger.error(f"Error listing active sessions: {e}")
            return []
    
    async def close(self):
        """Close the researcher and cleanup resources"""
        try:
            # Close all session memories
            for session_memory in self.session_memories.values():
                session_memory.close()
            
            # Close Azure client
            await self.azure_client.close()
            
            logger.info("Researcher closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing researcher: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        asyncio.create_task(self.close())
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
