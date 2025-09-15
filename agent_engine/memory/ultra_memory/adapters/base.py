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



from typing import Any, Dict, List, Optional, Tuple

from ..models import CollectionSpec, Record, Point, Filter


class StorageAdapter:
    def init(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def create_collection(self, spec: CollectionSpec) -> None:
        raise NotImplementedError

    def describe_collection(self, name: str) -> Dict[str, Any]:
        raise NotImplementedError

    def drop_collection(self, name: str) -> None:
        raise NotImplementedError

    def upsert(self, collection: str, records: List[Record], *, dedupe_key: Optional[str] = None) -> List[str]:
        raise NotImplementedError

    def query(self, collection: str, flt: Filter) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def count(self, collection: str, flt: Filter) -> int:
        raise NotImplementedError

    def exists(self, collection: str, flt: Filter) -> bool:
        raise NotImplementedError

    def search_vectors(self, collection: str, vector_or_text: Any, *, top_k: int, flt: Optional[Filter] = None, threshold: float = 0.0, ef_search: Optional[int] = None, vector_field: Optional[str] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        raise NotImplementedError

    def append_points(self, metric: str, points: List[Point]) -> int:
        raise NotImplementedError

    def query_timeseries(self, metric: str, flt: Filter) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:
        raise NotImplementedError

    # Optional delete APIs
    def delete_by_id(self, collection: str, item_id: str) -> int:
        raise NotImplementedError

    def delete_by_filter(self, collection: str, flt: Filter) -> int:
        raise NotImplementedError


