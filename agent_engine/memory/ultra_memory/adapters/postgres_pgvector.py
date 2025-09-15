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
import pyinstrument

from ..models import CollectionSpec, Record, Point, Filter
from .base import StorageAdapter
from ..dsl import build_select, build_where, build_count, build_exists


class PostgresPgvectorAdapter(StorageAdapter):
    def __init__(self, dsn: str, *, pool_min: int = 1, pool_max: int = 8) -> None:
        self.dsn = dsn
        self.pool_min = int(pool_min)
        self.pool_max = int(pool_max)
        self.logger = AgentLogger(self.__class__.__name__)
        self._ready = False
        # In-memory placeholder stores for prototyping without a live DB
        self._collections: Dict[str, CollectionSpec] = {}
        self._store: Dict[str, Dict[str, Record]] = {}
        self._timeseries: Dict[str, List[Point]] = {}
        self._conn = None

    def init(self) -> None:
        # Try to establish real PostgreSQL connection via psycopg (v3)
        try:
            import psycopg  # type: ignore

            self._conn = psycopg.connect(self.dsn)
            self._conn.autocommit = True
            self._ensure_extensions()
            self._ready = True
            self.logger.info("PostgresPgvectorAdapter initialized (postgres)")
            return
        except Exception as e:
            self._conn = None
            self._ready = True
            self.logger.warning(f"Falling back to in-memory adapter (psycopg connect failed: {e})")

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None
        self._ready = False

    def create_collection(self, spec: CollectionSpec) -> None:
        # Placeholder: real impl should create tables, indexes, vector columns
        self.logger.info(f"Create collection {spec.name} with mode={spec.mode}")
        self._collections[spec.name] = spec
        if self._conn is None:
            self._store.setdefault(spec.name, {})
            return
        self._ensure_meta_table()
        tbl = self._tbl(spec.name)
        cols = [
            'id TEXT PRIMARY KEY',
            'content TEXT',
            'attributes JSONB',
            'ts TIMESTAMPTZ',
        ]
        for col, dim in (spec.vector_dimensions or {}).items():
            safe = self._col(col)
            cols.append(f'"{safe}" vector({int(dim)})')
        sql = f'CREATE TABLE IF NOT EXISTS "{tbl}" (' + ", ".join(cols) + ")"
        self._exec(sql)
        # Validate metadata compatibility and record spec
        self._upsert_or_validate_meta(spec)
        # Indexes
        try:
            self._exec(f'CREATE INDEX IF NOT EXISTS idx_{tbl}_attrs ON "{tbl}" USING GIN (attributes)')
        except Exception:
            pass
        # Time index for range queries
        try:
            self._exec(f'CREATE INDEX IF NOT EXISTS idx_{tbl}_ts ON "{tbl}" (ts)')
        except Exception:
            pass
        # Optional vector index (IVFFLAT/HNSW) on first vector column
        if spec.vector_dimensions:
            first_vec = self._col(next(iter(spec.vector_dimensions.keys())))
            metric = (spec.metric or "cosine").lower()
            opclass = {
                "cosine": "vector_cosine_ops",
                "l2": "vector_l2_ops",
                "ip": "vector_ip_ops",
            }.get(metric, "vector_cosine_ops")
            idx_params = spec.index_params or {}
            lists = int(idx_params.get("lists", 100))
            try:
                self._exec(
                    f'CREATE INDEX IF NOT EXISTS idx_{tbl}_{first_vec}_ivf ON "{tbl}" USING ivfflat ("{first_vec}" {opclass}) WITH (lists={lists})'
                )
            except Exception:
                pass

    def describe_collection(self, name: str) -> Dict[str, Any]:
        spec = self._collections.get(name)
        if self._conn is None:
            return {"name": name, "ready": self._ready, "exists": bool(spec), "mode": (spec.mode if spec else None)}
        # Check table existence
        try:
            tbl = self._tbl(name)
            rows = self._fetchall('SELECT 1 FROM information_schema.tables WHERE table_name=%s', (tbl,))
            exists = bool(rows)
        except Exception:
            exists = False
        return {"name": name, "ready": self._ready, "exists": exists, "mode": (spec.mode if spec else None)}

    def drop_collection(self, name: str) -> None:
        self.logger.info(f"Drop collection {name}")
        self._collections.pop(name, None)
        self._store.pop(name, None)
        if self._conn is not None:
            try:
                self._exec(f'DROP TABLE IF EXISTS "{self._tbl(name)}"')
            except Exception:
                pass

    def upsert(self, collection: str, records: List[Record], *, dedupe_key: Optional[str] = None) -> List[str]:
        ids: List[str] = []
        if self._conn is None:
            col = self._store.setdefault(collection, {})
            for r in records:
                rid = self._derive_id(r, dedupe_key)
                r.id = rid
                col[rid] = r
                ids.append(rid)
            self.logger.debug(f"Upsert into {collection}: {len(records)} records (in-memory)")
            return ids
        # Postgres path
        tbl = self._tbl(collection)
        spec = self._collections.get(collection)
        vec_cols = list((spec.vector_dimensions or {}).keys()) if spec else []
        for r in records:
            rid = self._derive_id(r, dedupe_key)
            ids.append(rid)
            cols = ['id', 'content', 'attributes', 'ts'] + [self._col(v) for v in vec_cols]
            placeholders = ["%s", "%s", "%s::jsonb", "%s::timestamptz"] + ["%s::vector" for _ in vec_cols]
            values: List[Any] = [rid, r.content, json.dumps(r.attributes or {}, ensure_ascii=False), r.timestamp]
            for vname in vec_cols:
                vec = r.vector if r.vector is not None else None
                # validate length if provided
                if vec is not None:
                    dim = int((spec.vector_dimensions or {}).get(vname, len(vec)))
                    if len(vec) != dim:
                        raise ValueError(f"Vector length mismatch for column {vname}: expected {dim}, got {len(vec)}")
                values.append(self._vec_literal(vec) if vec is not None else None)
            insert_sql = f'INSERT INTO "{tbl}" (' + ", ".join(f'"{c}"' for c in cols) + ") VALUES (" + ", ".join(placeholders) + ") "
            update_set = ", ".join([f'"{c}"=EXCLUDED."{c}"' for c in cols if c != 'id'])
            sql = insert_sql + f"ON CONFLICT (id) DO UPDATE SET {update_set}"
            self._exec(sql, tuple(values))
        self.logger.debug(f"Upsert into {collection}: {len(records)} records (postgres)")
        return ids

    @pyinstrument.profile()
    def query(self, collection: str, flt: Filter) -> List[Dict[str, Any]]:
        if self._conn is None:
            self.logger.debug(f"Query on {collection}: {flt.expr} (in-memory)")
            res: List[Dict[str, Any]] = []
            col = self._store.get(collection, {})
            for r in col.values():
                if _match(r, flt.expr or {}):
                    res.append({
                        "id": r.id,
                        "attributes": r.attributes,
                        "content": r.content,
                        "vector": r.vector,
                        "timestamp": r.timestamp,
                    })
            for ob in reversed(flt.order_by or []):
                key, direc = ob[0], str(ob[1]).lower()
                res.sort(key=lambda x: _get_field(x, key), reverse=(direc == "desc"))
            if flt.offset:
                res = res[int(flt.offset):]
            if flt.limit is not None:
                res = res[: int(flt.limit)]
            return res
        # Postgres path
        tbl = self._tbl(collection)
        sql, params = build_select(tbl, ["id", "content", "attributes", "ts"], flt)
        rows = self._fetchall(sql, tuple(params))
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append({
                "id": row[0],
                "content": row[1],
                "attributes": row[2],
                "timestamp": row[3],
            })
        return out

    def count(self, collection: str, flt: Filter) -> int:
        if self._conn is None:
            # In-memory fallback
            col = self._store.get(collection, {})
            return sum(1 for r in col.values() if _match(r, flt.expr or {}))
        tbl = self._tbl(collection)
        sql, params = build_count(tbl, flt)
        rows = self._fetchall(sql, tuple(params))
        if not rows:
            return 0
        try:
            return int(rows[0][0])
        except Exception:
            return 0

    def exists(self, collection: str, flt: Filter) -> bool:
        if self._conn is None:
            col = self._store.get(collection, {})
            for r in col.values():
                if _match(r, flt.expr or {}):
                    return True
            return False
        tbl = self._tbl(collection)
        sql, params = build_exists(tbl, flt)
        rows = self._fetchall(sql, tuple(params))
        return bool(rows)

    def search_vectors(self, collection: str, vector_or_text: Any, *, top_k: int, flt: Optional[Filter] = None, threshold: float = 0.0, ef_search: Optional[int] = None, vector_field: Optional[str] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        if self._conn is None:
            self.logger.debug(f"Vector search on {collection}, top_k={top_k}, threshold={threshold} (in-memory)")
            if not isinstance(vector_or_text, list):
                return []
            q = vector_or_text
            col = self._store.get(collection, {})
            scored: List[Tuple[str, float, Dict[str, Any]]] = []
            for r in col.values():
                if r.vector is None:
                    continue
                if flt and not _match(r, flt.expr or {}):
                    continue
                sim = _cosine_similarity(q, r.vector)
                if sim < threshold:
                    continue
                meta = {"id": r.id, **r.attributes}
                scored.append((r.content or "", float(sim), meta))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[: max(0, int(top_k))]
        # Postgres path
        spec = self._collections.get(collection)
        if not spec or not spec.vector_dimensions:
            return []
        vec_key = vector_field if vector_field else next(iter(spec.vector_dimensions.keys()))
        if vec_key not in (spec.vector_dimensions or {}):
            raise ValueError(f"Unknown vector field {vec_key}")
        vec_col = self._col(vec_key)
        tbl = self._tbl(collection)
        where_sql = "TRUE"
        params: List[Any] = []
        if flt and flt.expr:
            where_sql, params = build_where(Filter(expr=flt.expr))
        vec_literal = self._vec_literal(vector_or_text if isinstance(vector_or_text, list) else [])
        metric = (spec.metric or "cosine").lower()
        if metric == "cosine":
            score_sql = f'(1 - ("{vec_col}" <=> %s::vector))'
            order_sql = f'ORDER BY ("{vec_col}" <=> %s::vector) ASC'
        elif metric == "l2":
            score_sql = f'(- ("{vec_col}" <-> %s::vector))'
            order_sql = f'ORDER BY ("{vec_col}" <-> %s::vector) ASC'
        else:  # ip
            score_sql = f'("{vec_col}" <#> %s::vector)'
            order_sql = f'ORDER BY {score_sql} DESC'

        sql = (
            f'SELECT id, content, attributes, {score_sql} AS sim '
            f'FROM "{tbl}" WHERE {where_sql} '
            f'{order_sql} LIMIT %s'
        )
        # Placeholder order differs by metric:
        # SELECT score(%s::vector) then ORDER uses same vector placeholder; final LIMIT
        rows = self._fetchall(sql, tuple([vec_literal] + params + [vec_literal, int(top_k)]))
        out: List[Tuple[str, float, Dict[str, Any]]] = []
        for row in rows:
            if threshold is not None:
                if metric == "cosine" and float(row[3]) < float(threshold):
                    continue
                # l2/ip 可在后续扩展更精细的阈值语义
            meta = {"id": row[0], **(row[2] or {})}
            out.append((row[1] or "", float(row[3]), meta))
        return out

    def append_points(self, metric: str, points: List[Point]) -> int:
        self.logger.debug(f"Append {len(points)} points to metric={metric} (in-memory)")
        if self._conn is None:
            arr = self._timeseries.setdefault(metric, [])
            arr.extend(points)
            return len(points)
        tbl = self._ts_tbl(metric)
        self._exec(f'CREATE TABLE IF NOT EXISTS "{tbl}" (ts TIMESTAMPTZ, tags JSONB, fields JSONB)')
        for p in points:
            self._exec(
                f'INSERT INTO "{tbl}" (ts, tags, fields) VALUES (%s::timestamptz, %s::jsonb, %s::jsonb)',
                (p.timestamp, json.dumps(p.tags or {}), json.dumps(p.fields or {})),
            )
        return len(points)

    def query_timeseries(self, metric: str, flt: Filter) -> List[Dict[str, Any]]:
        self.logger.debug(f"Query timeseries metric={metric}, filter={flt.expr} (in-memory)")
        if self._conn is None:
            arr = self._timeseries.get(metric, [])
            out: List[Dict[str, Any]] = []
            for p in arr:
                if _match_point(p, flt.expr or {}):
                    out.append({
                        "metric": p.metric,
                        "timestamp": p.timestamp,
                        "tags": p.tags,
                        "fields": p.fields,
                    })
            return out
        tbl = self._ts_tbl(metric)
        # Minimal support: range on timestamp
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
        if self._conn is None:
            return {
                "backend": "postgres_pgvector",
                "ready": self._ready,
                "collections": len(self._collections),
                "records": sum(len(v) for v in self._store.values()),
                "metrics": len(self._timeseries),
                "mode": "in_memory",
            }
        return {
            "backend": "postgres_pgvector",
            "ready": self._ready,
            "collections": len(self._collections),
            "mode": "postgres",
        }

    def health(self) -> Dict[str, Any]:
        return {"status": "ok" if self._ready else "init"}

    # ---------- Internal helpers (postgres) ----------
    def _derive_id(self, r: Record, dedupe_key: Optional[str]) -> str:
        if isinstance(r.id, str) and r.id:
            return r.id
        if dedupe_key and isinstance(r.attributes, dict) and (dedupe_key in r.attributes):
            v = r.attributes.get(dedupe_key)
            if isinstance(v, (str, int)):
                return str(v)
        # fallback: stable json of attributes
        return json.dumps(r.attributes or {}, ensure_ascii=False, sort_keys=True)

    def _exec(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> None:
        if self._conn is None:
            return
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params or ())
        finally:
            cur.close()

    @pyinstrument.profile()
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
            self._exec('CREATE EXTENSION IF NOT EXISTS vector')
        except Exception as e:
            self.logger.warning(f"Failed to ensure extension vector: {e}")

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
            cur.execute('SELECT mode, vector_dimensions, metric FROM um_collections WHERE name=%s', (spec.name,))
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    'INSERT INTO um_collections(name, mode, vector_dimensions, metric, index_params, elevate_fields, json_fields) VALUES (%s,%s,%s::jsonb,%s,%s::jsonb,%s::jsonb,%s::jsonb)',
                    (
                        spec.name,
                        spec.mode,
                        json.dumps(spec.vector_dimensions or {}),
                        (spec.metric or "cosine"),
                        json.dumps(spec.index_params or {}),
                        json.dumps(spec.elevate_fields or []),
                        json.dumps(spec.json_fields or []),
                    ),
                )
                return
            existing_mode, existing_vec_dims, existing_metric = row[0], row[1] or {}, row[2]
            # Validate compatibility
            if str(existing_mode) != str(spec.mode):
                raise ValueError(f"Collection {spec.name} mode mismatch: {existing_mode} vs {spec.mode}")
            if (existing_vec_dims or spec.vector_dimensions) and str(existing_metric or "") != str(spec.metric or "cosine"):
                raise ValueError(f"Collection {spec.name} metric mismatch: {existing_metric} vs {spec.metric}")
            # Vector dimension compatibility: existing cols must keep same dim; new cols allowed
            try:
                existing = dict(existing_vec_dims) if isinstance(existing_vec_dims, dict) else {}
            except Exception:
                existing = {}
            for k, v in existing.items():
                if k in (spec.vector_dimensions or {}) and int(spec.vector_dimensions[k]) != int(v):
                    raise ValueError(f"Collection {spec.name} vector dim mismatch for {k}: {v} vs {spec.vector_dimensions[k]}")
            # Optionally update meta if new vector columns added
            merged = dict(existing)
            merged.update(spec.vector_dimensions or {})
            if merged != existing:
                cur.execute('UPDATE um_collections SET vector_dimensions=%s::jsonb WHERE name=%s', (json.dumps(merged), spec.name))
        finally:
            cur.close()

    def _tbl(self, name: str) -> str:
        return f"um_{self._sanitize(name)}"

    def _ts_tbl(self, metric: str) -> str:
        return f"um_ts_{self._sanitize(metric)}"

    def _col(self, name: str) -> str:
        return self._sanitize(name)

    def _sanitize(self, s: str) -> str:
        out = []
        for ch in str(s):
            if ch.isalnum() or ch == '_':
                out.append(ch)
            else:
                out.append('_')
        return ''.join(out)[:63]

    def _vec_literal(self, vec: List[float]) -> Optional[str]:
        if vec is None:
            return None
        return '[' + ','.join(str(float(x)) for x in vec) + ']'

    # Deletes
    def delete_by_id(self, collection: str, item_id: str) -> int:
        if self._conn is None:
            col = self._store.get(collection, {})
            return 1 if col.pop(item_id, None) is not None else 0
        tbl = self._tbl(collection)
        cur = self._conn.cursor()
        try:
            cur.execute(f'DELETE FROM "{tbl}" WHERE id = %s', (item_id,))
            return int(cur.rowcount or 0)
        finally:
            cur.close()

    def delete_by_filter(self, collection: str, flt: Filter) -> int:
        if self._conn is None:
            col = self._store.get(collection, {})
            keys = [k for k, r in list(col.items()) if _match(r, flt.expr or {})]
            for k in keys:
                col.pop(k, None)
            return len(keys)
        where_sql, params = build_where(flt)
        tbl = self._tbl(collection)
        cur = self._conn.cursor()
        try:
            cur.execute(f'DELETE FROM "{tbl}" WHERE {where_sql}', tuple(params))
            return int(cur.rowcount or 0)
        finally:
            cur.close()


def _get_field(rec: Dict[str, Any], key: str) -> Any:
    if key in ("id", "content", "timestamp"):
        return rec.get(key)
    if key.startswith("attributes."):
        return rec.get("attributes", {}).get(key.split(".", 1)[1])
    return rec.get("attributes", {}).get(key)


def _match(r: Record, expr: Dict[str, Any]) -> bool:
    if not expr:
        return True
    if "and" in expr:
        return all(_match(r, e) for e in expr["and"] or [])
    if "or" in expr:
        return any(_match(r, e) for e in expr["or"] or [])
    if "not" in expr:
        return not _match(r, expr["not"])
    if "eq" in expr:
        field, val = expr["eq"][0], expr["eq"][1]
        return _value_of(r, field) == val
    if "in" in expr:
        field, arr = expr["in"][0], expr["in"][1]
        return _value_of(r, field) in set(arr)
    if "range" in expr:
        ((field, bounds),) = expr["range"].items()
        v = _value_of(r, field)
        if v is None:
            return False
        lo, hi = bounds[0], bounds[1]
        return (v >= lo) and (v <= hi)
    if "like" in expr:
        field, pattern = expr["like"][0], expr["like"][1]
        v = str(_value_of(r, field) or "")
        pat = str(pattern).replace("%", "")
        return pat in v
    if "contains" in expr:
        subset = expr["contains"] or {}
        attrs = r.attributes or {}
        return all(attrs.get(k) == v for k, v in subset.items())
    return True


def _match_point(p: Point, expr: Dict[str, Any]) -> bool:
    if not expr:
        return True
    # Minimal: support range on timestamp and eq on tags/fields
    if "and" in expr:
        return all(_match_point(p, e) for e in expr["and"] or [])
    if "or" in expr:
        return any(_match_point(p, e) for e in expr["or"] or [])
    if "not" in expr:
        return not _match_point(p, expr["not"])
    if "eq" in expr:
        field, val = expr["eq"][0], expr["eq"][1]
        return _value_of_point(p, field) == val
    if "in" in expr:
        field, arr = expr["in"][0], expr["in"][1]
        return _value_of_point(p, field) in set(arr)
    if "range" in expr:
        ((field, bounds),) = expr["range"].items()
        v = _value_of_point(p, field)
        if v is None:
            return False
        lo, hi = bounds[0], bounds[1]
        return (v >= lo) and (v <= hi)
    return True


def _value_of(r: Record, field: str) -> Any:
    if field == "id":
        return r.id
    if field == "content":
        return r.content
    if field in ("ts", "timestamp"):
        return r.timestamp
    # dotted under attributes
    cur: Any = r.attributes
    for p in str(field).split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _value_of_point(p: Point, field: str) -> Any:
    if field in ("ts", "timestamp"):
        return p.timestamp
    if field.startswith("tags."):
        return p.tags.get(field.split(".", 1)[1])
    if field.startswith("fields."):
        return p.fields.get(field.split(".", 1)[1])
    return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    import math

    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / math.sqrt(na * nb))


