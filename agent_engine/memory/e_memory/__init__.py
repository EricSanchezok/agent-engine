"""
EMemory - A lightweight vector memory implementation using SQLite and HNSWIndex.

This module provides a simplified memory implementation with:
- SQLite database storage
- HNSW indexing for vector similarity search
- Basic CRUD operations
- Lightweight design without built-in embedding models
"""

from .core import EMemory
from .models import Record

__all__ = ["EMemory", "Record"]
