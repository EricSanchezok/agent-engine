from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent_engine.agent_logger.agent_logger import AgentLogger

from .models import CollectionSpec, Record, Point, Filter
from .adapters.base import StorageAdapter
from .adapters.postgres_pgvector import PostgresPgvectorAdapter
from .adapters.timescaledb import TimescaleAdapter


@dataclass
class UltraMemoryConfig:
    backend: str = "postgres_pgvector"  # postgres_pgvector | timescaledb
    dsn: Optional[str] = None
    pool_min: int = 2
    pool_max: int = 16


class UltraMemory:
    def __init__(self, config: UltraMemoryConfig) -> None:
        self.config = config
        self.logger = AgentLogger(self.__class__.__name__)
        self.adapter: StorageAdapter = self._create_adapter(config)
        self.adapter.init()

    def _create_adapter(self, config: UltraMemoryConfig) -> StorageAdapter:
        backend = (config.backend or "").lower()
        if backend == "postgres_pgvector":
            if not config.dsn:
                raise ValueError("DSN is required for PostgresPgvector backend")
            return PostgresPgvectorAdapter(dsn=config.dsn, pool_min=config.pool_min, pool_max=config.pool_max)
        if backend == "timescaledb":
            if not config.dsn:
                raise ValueError("DSN is required for TimescaleDB backend")
            return TimescaleAdapter(dsn=config.dsn, pool_min=config.pool_min, pool_max=config.pool_max)
        raise ValueError(f"Unsupported backend: {backend}")

    # ---- Collection lifecycle ----
    def create_collection(self, spec: CollectionSpec) -> None:
        self.adapter.create_collection(spec)

    def describe_collection(self, name: str) -> Dict[str, Any]:
        return self.adapter.describe_collection(name)

    def drop_collection(self, name: str) -> None:
        self.adapter.drop_collection(name)

    # ---- Data ops ----
    def upsert(self, collection: str, records: List[Record], *, dedupe_key: Optional[str] = None) -> List[str]:
        return self.adapter.upsert(collection, records, dedupe_key=dedupe_key)

    # ---- Delete ops ----
    def delete_by_id(self, collection: str, item_id: str) -> int:
        if hasattr(self.adapter, "delete_by_id"):
            return int(getattr(self.adapter, "delete_by_id")(collection, item_id))
        raise NotImplementedError("delete_by_id not implemented for this backend")

    def delete_by_filter(self, collection: str, flt: Filter) -> int:
        if hasattr(self.adapter, "delete_by_filter"):
            return int(getattr(self.adapter, "delete_by_filter")(collection, flt))
        raise NotImplementedError("delete_by_filter not implemented for this backend")

    def query(self, collection: str, flt: Filter) -> List[Dict[str, Any]]:
        return self.adapter.query(collection, flt)

    def search_vectors(self, collection: str, vector_or_text: Any, *, top_k: int, flt: Optional[Filter] = None, threshold: float = 0.0, ef_search: Optional[int] = None, vector_field: Optional[str] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        return self.adapter.search_vectors(collection, vector_or_text, top_k=top_k, flt=flt, threshold=threshold, ef_search=ef_search, vector_field=vector_field)

    def append_points(self, metric: str, points: List[Point]) -> int:
        return self.adapter.append_points(metric, points)

    def query_timeseries(self, metric: str, flt: Filter) -> List[Dict[str, Any]]:
        return self.adapter.query_timeseries(metric, flt)

    # ---- Admin ----
    def stats(self) -> Dict[str, Any]:
        return self.adapter.stats()

    def health(self) -> Dict[str, Any]:
        return self.adapter.health()

    def close(self) -> None:
        self.adapter.close()


