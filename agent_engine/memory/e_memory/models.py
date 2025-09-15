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


Data models for EMemory.
"""


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
