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


Memory module for agent-engine

This module provides vector-based memory storage for agents.
"""

from .memory import Memory
from .embedder import Embedder, get_recommended_models
from .scalable_memory import ScalableMemory

__all__ = ['Memory', 'Embedder', 'get_recommended_models', 'ScalableMemory']
