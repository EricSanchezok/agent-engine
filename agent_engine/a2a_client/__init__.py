"""
A2A Client module for agent-engine

This module provides client implementations for A2A (Agent-to-Agent) protocol communication.
"""
from .a2a_client import (
    A2AClientWrapper,
    send_message_to_a2a_agent,
    send_message_to_a2a_agent_streaming
)

__all__ = [
    'A2AClientWrapper',
    'send_message_to_a2a_agent',
    'send_message_to_a2a_agent_streaming'
]
