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
