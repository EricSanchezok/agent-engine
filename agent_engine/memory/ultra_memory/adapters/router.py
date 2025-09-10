from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import StorageAdapter
from ..models import CollectionSpec, Record, Point, Filter


# Deprecated after simplifying config model, kept for compatibility if referenced elsewhere.
class MixedRouterAdapter(StorageAdapter):
    def __init__(self, pg_adapter: StorageAdapter, ts_adapter: StorageAdapter) -> None:
        self.pg = pg_adapter
        self.ts = ts_adapter

    def init(self) -> None:
        self.pg.init()
        self.ts.init()

    def close(self) -> None:
        self.pg.close()
        self.ts.close()

    def create_collection(self, spec: CollectionSpec) -> None:
        raise NotImplementedError("MixedRouterAdapter is deprecated in simplified config; use explicit backends")

    def describe_collection(self, name: str) -> Dict[str, Any]:
        return {"backend": "auto", "name": name}

    def drop_collection(self, name: str) -> None:
        pass

    def upsert(self, collection: str, records: List[Record], *, dedupe_key: Optional[str] = None) -> List[str]:
        return []

    def query(self, collection: str, flt: Filter) -> List[Dict[str, Any]]:
        return []

    def search_vectors(self, collection: str, vector_or_text: Any, *, top_k: int, flt: Optional[Filter] = None, threshold: float = 0.0, ef_search: Optional[int] = None, vector_field: Optional[str] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        return []

    def append_points(self, metric: str, points: List[Point]) -> int:
        return 0

    def query_timeseries(self, metric: str, flt: Filter) -> List[Dict[str, Any]]:
        return []

    def stats(self) -> Dict[str, Any]:
        return {"backend": "auto"}

    def health(self) -> Dict[str, Any]:
        return {"status": "deprecated"}


