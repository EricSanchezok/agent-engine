"""
A2A Client implementation for agent-engine

This module provides a simplified interface for communicating with A2A protocol agents.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)

# AgentEngine imports
from agent_engine.agent_logger import AgentLogger

logger = AgentLogger(__name__)


class A2AClientWrapper:
    """
    A simplified wrapper for A2A client communication.
    
    This class provides easy-to-use methods for sending messages to A2A protocol agents.
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize the A2A client wrapper.
        
        Args:
            base_url: Base URL for the A2A agent service
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.httpx_client: Optional[httpx.AsyncClient] = None
        self.resolver: Optional[A2ACardResolver] = None
        self.client: Optional[A2AClient] = None
        self.agent_card: Optional[AgentCard] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    async def connect(self) -> None:
        """
        Connect to the A2A agent and fetch the agent card.
        
        Raises:
            RuntimeError: If unable to fetch the agent card
        """
        if self.httpx_client is None:
            self.httpx_client = httpx.AsyncClient(timeout=self.timeout)
            
        # Initialize A2ACardResolver
        self.resolver = A2ACardResolver(
            httpx_client=self.httpx_client,
            base_url=self.base_url,
        )
        
        # Fetch agent card
        await self._fetch_agent_card()
        
        # Initialize A2AClient
        self.client = A2AClient(
            httpx_client=self.httpx_client,
            agent_card=self.agent_card
        )
        
        logger.info(f"A2A client connected to: {self.base_url}")
        
    async def _fetch_agent_card(self) -> None:
        """
        Fetch the agent card from the A2A service.
        
        Raises:
            RuntimeError: If unable to fetch the agent card
        """
        PUBLIC_AGENT_CARD_PATH = '/.well-known/agent.json'
        EXTENDED_AGENT_CARD_PATH = '/agent/authenticatedExtendedCard'
        
        try:
            logger.info(f"Fetching public agent card from: {self.base_url}{PUBLIC_AGENT_CARD_PATH}")
            
            # Fetch public agent card
            public_card = await self.resolver.get_agent_card()
            logger.info("Successfully fetched public agent card")
            self.agent_card = public_card
            
            # Try to fetch extended card if supported
            if public_card.supports_authenticated_extended_card:
                try:
                    logger.info(f"Attempting to fetch extended agent card from: {self.base_url}{EXTENDED_AGENT_CARD_PATH}")
                    auth_headers_dict = {
                        'Authorization': 'Bearer dummy-token-for-extended-card'
                    }
                    extended_card = await self.resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={'headers': auth_headers_dict},
                    )
                    logger.info("Successfully fetched authenticated extended agent card")
                    self.agent_card = extended_card
                except Exception as e:
                    logger.warning(f"Failed to fetch extended agent card: {e}. Using public card.")
                    
        except Exception as e:
            logger.error(f"Critical error fetching agent card: {e}")
            raise RuntimeError(f"Failed to fetch the agent card from {self.base_url}") from e
            
    async def send_message(
        self, 
        message: str, 
        message_id: Optional[str] = None,
        role: str = "user"
    ) -> Dict[str, Any]:
        """
        Send a message to the A2A agent and get response.
        
        Args:
            message: The message text to send
            message_id: Optional message ID (will generate one if not provided)
            role: The role of the message sender (default: "user")
            
        Returns:
            Dictionary containing the response from the agent
            
        Raises:
            RuntimeError: If client is not connected
        """
        if self.client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
            
        if message_id is None:
            message_id = uuid4().hex
            
        send_message_payload = {
            'message': {
                'role': role,
                'parts': [
                    {'kind': 'text', 'text': message}
                ],
                'message_id': message_id,
            },
        }
        
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**send_message_payload)
        )
        
        logger.info(f"Sending message to A2A agent: {message[:50]}...")
        try:
            response = await self.client.send_message(request)
            
            # Convert response to dictionary
            result = response.model_dump(mode='json', exclude_none=True)
            logger.info("Successfully received response from A2A agent")
            
            return result
        except Exception as e:
            logger.error(f"Error sending message to A2A agent: {e}")
            raise
        
    async def send_message_streaming(
        self, 
        message: str, 
        message_id: Optional[str] = None,
        role: str = "user"
    ):
        """
        Send a message to the A2A agent and get streaming response.
        
        Args:
            message: The message text to send
            message_id: Optional message ID (will generate one if not provided)
            role: The role of the message sender (default: "user")
            
        Yields:
            Dictionary containing streaming response chunks from the agent
            
        Raises:
            RuntimeError: If client is not connected
        """
        if self.client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
            
        if message_id is None:
            message_id = uuid4().hex
            
        send_message_payload = {
            'message': {
                'role': role,
                'parts': [
                    {'kind': 'text', 'text': message}
                ],
                'message_id': message_id,
            },
        }
        
        streaming_request = SendStreamingMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**send_message_payload)
        )
        
        logger.info(f"Sending streaming message to A2A agent: {message[:50]}...")
        try:
            stream_response = self.client.send_message_streaming(streaming_request)
            
            async for chunk in stream_response:
                yield chunk.model_dump(mode='json', exclude_none=True)
        except Exception as e:
            logger.error(f"Error sending streaming message to A2A agent: {e}")
            raise
            
    async def close(self) -> None:
        """Close the client and clean up resources."""
        if self.httpx_client:
            await self.httpx_client.aclose()
            self.httpx_client = None
            self.resolver = None
            self.client = None
            self.agent_card = None
            logger.info("A2A client closed")


async def send_message_to_a2a_agent(
    base_url: str, 
    message: str, 
    message_id: Optional[str] = None,
    role: str = "user",
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Simple function to send a message to an A2A agent without creating a client instance.
    
    Args:
        base_url: Base URL for the A2A agent service
        message: The message text to send
        message_id: Optional message ID (will generate one if not provided)
        role: The role of the message sender (default: "user")
        timeout: Request timeout in seconds (default: 30.0)
        
    Returns:
        Dictionary containing the response from the agent
    """
    async with A2AClientWrapper(base_url, timeout) as client:
        return await client.send_message(message, message_id, role)


async def send_message_to_a2a_agent_streaming(
    base_url: str, 
    message: str, 
    message_id: Optional[str] = None,
    role: str = "user",
    timeout: float = 30.0
):
    """
    Simple function to send a streaming message to an A2A agent without creating a client instance.
    
    Args:
        base_url: Base URL for the A2A agent service
        message: The message text to send
        message_id: Optional message ID (will generate one if not provided)
        role: The role of the message sender (default: "user")
        timeout: Request timeout in seconds (default: 30.0)
        
    Yields:
        Dictionary containing streaming response chunks from the agent
    """
    async with A2AClientWrapper(base_url, timeout) as client:
        async for chunk in client.send_message_streaming(message, message_id, role):
            yield chunk

if __name__ == "__main__":
    from pprint import pprint
    message = "Hello, how are you?"
    response = asyncio.run(send_message_to_a2a_agent(
        base_url="http://10.12.16.139:9900",
        message=message,
        message_id=None,
        role="user",
        timeout=30.0
    ))
    pprint(response)