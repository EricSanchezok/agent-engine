"""
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
