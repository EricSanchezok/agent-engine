"""
Data models for EMemory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Record:
    """A record in EMemory."""
    id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = None
    vector: Optional[List[float]] = None
    timestamp: Optional[str] = None  # ISO8601 format
