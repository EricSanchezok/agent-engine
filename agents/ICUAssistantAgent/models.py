"""
Data models for ICU Assistant Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    """A event in ICU Assistant Agent."""
    id: Optional[str] = None
    timestamp: Optional[str] = None  # ISO8601 format
    event_type: Optional[str] = None
    sub_type: Optional[str] = None
    event_content: Optional[str] = None
    risks: Optional[List[Dict]] = None
    flag: Any = None
    metadata: Optional[Dict] = None