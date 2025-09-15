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



import json
from typing import Any, Dict, List, Optional, Tuple

from agent_engine.agent_logger.agent_logger import AgentLogger

from ..models import CollectionSpec, Record, Point, Filter
from .base import StorageAdapter
from ..dsl import build_where


class TimescaleAdapter(StorageAdapter):
    def __init__(self, dsn: str, *, pool_min: int = 1, pool_max: int = 8) -> None:
        self.dsn = dsn
        self.pool_min = int(pool_min)
        self.pool_max = int(pool_max)
        self.logger = AgentLogger(self.__class__.__name__)
        self._ready = False
        self._conn = None
        # Minimal in-memory fallback
        self._series: Dict[str, List[Point]] = {}

    def init(self) -> None:
        try:
            import psycopg  # type: ignore

            self._conn = psycopg.connect(self.dsn)
            self._conn.autocommit = True
            self._ensure_extensions()
            self._ready = True
            self.logger.info("TimescaleAdapter initialized (timescaledb)")
        except Exception as e:
            self._conn = None
            self._ready = True
            self.logger.warning(f"Falling back to in-memory timeseries (psycopg connect failed: {e})")

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None
        self._ready = False

    # Collections are not used directly for timeseries; metrics define tables
    def create_collection(self, spec: CollectionSpec) -> None:
        # In timeseries mode, a collection represents a metric namespace.
        # We'll create a metric table named by collection name for convenience.
        if self._conn is None:
            return
        self._ensure_meta_table()
        metric = spec.name
        tbl = self._ts_tbl(metric)
        self._exec(f'CREATE TABLE IF NOT EXISTS "{tbl}" (ts TIMESTAMPTZ NOT NULL, tags JSONB, fields JSONB)')
        try:
            self._exec(f"SELECT create_hypertable('{tbl}', 'ts', if_not_exists => TRUE)")
        except Exception:
            pass
        self._upsert_or_validate_meta(spec)

    def describe_collection(self, name: str) -> Dict[str, Any]:
        return {"name": name, "ready": self._ready, "exists": True, "mode": "timeseries"}

    def drop_collection(self, name: str) -> None:
        # noop for timeseries; use metric tables
        pass

    def upsert(self, collection: str, records: List[Record], *, dedupe_key: Optional[str] = None) -> List[str]:
        raise NotImplementedError("upsert is not supported by TimescaleAdapter (timeseries-only)")

    def query(self, collection: str, flt: Filter) -> List[Dict[str, Any]]:
        raise NotImplementedError("query is not supported by TimescaleAdapter (timeseries-only)")

    def search_vectors(self, collection: str, vector_or_text: Any, *, top_k: int, flt: Optional[Filter] = None, threshold: float = 0.0, ef_search: Optional[int] = None, vector_field: Optional[str] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        raise NotImplementedError("search_vectors is not supported by TimescaleAdapter (timeseries-only)")

    def append_points(self, metric: str, points: List[Point]) -> int:
        if self._conn is None:
            arr = self._series.setdefault(metric, [])
            arr.extend(points)
            return len(points)
        tbl = self._ts_tbl(metric)
        self._exec(f'CREATE TABLE IF NOT EXISTS "{tbl}" (ts TIMESTAMPTZ NOT NULL, tags JSONB, fields JSONB)')
        # Create hypertable if not exists
        try:
            self._exec(f"SELECT create_hypertable('{tbl}', 'ts', if_not_exists => TRUE)")
        except Exception:
            pass
        for p in points:
            self._exec(
                f'INSERT INTO "{tbl}" (ts, tags, fields) VALUES (%s::timestamptz, %s::jsonb, %s::jsonb)',
                (p.timestamp, json.dumps(p.tags or {}), json.dumps(p.fields or {})),
            )
        return len(points)

    def query_timeseries(self, metric: str, flt: Filter) -> List[Dict[str, Any]]:
        if self._conn is None:
            arr = self._series.get(metric, [])
            out: List[Dict[str, Any]] = []
            from ..adapters.postgres_pgvector import _match_point  # reuse predicate

            for p in arr:
                if _match_point(p, flt.expr or {}):
                    out.append({"metric": metric, "timestamp": p.timestamp, "tags": p.tags, "fields": p.fields})
            return out
        tbl = self._ts_tbl(metric)
        # Basic: support timestamp range; tags/fields contains
        where = "TRUE"
        params: List[Any] = []
        expr = flt.expr or {}
        if "range" in expr and isinstance(expr["range"], dict):
            ((field, bounds),) = expr["range"].items()
            if field in ("timestamp", "ts"):
                where = "ts BETWEEN %s::timestamptz AND %s::timestamptz"
                params.extend([bounds[0], bounds[1]])
        rows = self._fetchall(f'SELECT ts, tags, fields FROM "{tbl}" WHERE {where}', tuple(params))
        return [{"metric": metric, "timestamp": r[0], "tags": r[1], "fields": r[2]} for r in rows]

    def stats(self) -> Dict[str, Any]:
        return {"backend": "timescaledb", "ready": self._ready}

    def health(self) -> Dict[str, Any]:
        return {"status": "ok" if self._ready else "init"}

    # Internals
    def _exec(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> None:
        if self._conn is None:
            return
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params or ())
        finally:
            cur.close()

    def _fetchall(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> List[Tuple[Any, ...]]:
        if self._conn is None:
            return []
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params or ())
            return list(cur.fetchall())
        finally:
            cur.close()

    def _ensure_extensions(self) -> None:
        try:
            self._exec('CREATE EXTENSION IF NOT EXISTS timescaledb')
        except Exception as e:
            self.logger.warning(f"Failed to ensure extension timescaledb: {e}")

    def _ensure_meta_table(self) -> None:
        self._exec(
            """
            CREATE TABLE IF NOT EXISTS um_collections (
                name TEXT PRIMARY KEY,
                mode TEXT,
                vector_dimensions JSONB,
                metric TEXT,
                index_params JSONB,
                elevate_fields JSONB,
                json_fields JSONB,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )

    def _upsert_or_validate_meta(self, spec: CollectionSpec) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute('SELECT mode FROM um_collections WHERE name=%s', (spec.name,))
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    'INSERT INTO um_collections(name, mode, vector_dimensions, metric, index_params, elevate_fields, json_fields) VALUES (%s,%s,%s::jsonb,%s,%s::jsonb,%s::jsonb,%s::jsonb)',
                    (
                        spec.name,
                        spec.mode,
                        json.dumps(spec.vector_dimensions or {}),
                        (spec.metric or ""),
                        json.dumps(spec.index_params or {}),
                        json.dumps(spec.elevate_fields or []),
                        json.dumps(spec.json_fields or []),
                    ),
                )
                return
            existing_mode = row[0]
            if str(existing_mode) != str(spec.mode):
                raise ValueError(f"Collection {spec.name} mode mismatch: {existing_mode} vs {spec.mode}")
        finally:
            cur.close()

    def _ts_tbl(self, metric: str) -> str:
        return f"um_ts_{self._sanitize(metric)}"

    def _sanitize(self, s: str) -> str:
        out = []
        for ch in str(s):
            if ch.isalnum() or ch == '_':
                out.append(ch)
            else:
                out.append('_')
        return ''.join(out)[:63]


