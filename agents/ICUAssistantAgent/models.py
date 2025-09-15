"""
from __future__ import annotations
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


Data models for ICU Assistant Agent.
"""


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