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



from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CollectionSpec:
    name: str
    mode: str  # vector | timeseries | general | mixed
    vector_dimensions: Dict[str, int] = field(default_factory=dict)
    metric: str = "cosine"  # cosine | l2 | ip
    index_params: Dict[str, Any] = field(default_factory=dict)
    elevate_fields: List[str] = field(default_factory=list)
    json_fields: List[str] = field(default_factory=list)
    default_metric: Optional[str] = None


@dataclass
class Record:
    id: Optional[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = None
    vector: Optional[List[float]] = None
    timestamp: Optional[str] = None  # ISO8601


@dataclass
class Point:
    metric: str
    timestamp: str  # ISO8601
    tags: Dict[str, str] = field(default_factory=dict)
    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Filter:
    expr: Dict[str, Any] = field(default_factory=dict)
    order_by: List[Tuple[str, str]] = field(default_factory=list)  # [(field, 'asc'|'desc')]
    limit: Optional[int] = None
    offset: Optional[int] = None


