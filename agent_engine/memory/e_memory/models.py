"""
Data models for EMemory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Record:
    """A record in EMemory."""
    id: Optional[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = field(default_factory=str)
    vector: Optional[List[float]] = field(default_factory=list)
    timestamp: Optional[str] = field(default_factory=str)
