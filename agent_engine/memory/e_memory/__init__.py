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


EMemory - A lightweight vector memory implementation using SQLite and ChromaDB.

This module provides a simplified memory implementation with:
- SQLite database storage for metadata
- ChromaDB for vector similarity search
- Basic CRUD operations
- PodEMemory for sharded large-scale storage
- Lightweight design without built-in embedding models
"""

from .core import EMemory
from .pod_ememory import PodEMemory
from .models import Record

__all__ = ["EMemory", "PodEMemory", "Record"]
