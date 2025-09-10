from __future__ import annotations

import os
import re
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from agent_engine.agent_logger.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.utils import get_current_file_dir
from agent_engine.memory.ultra_memory import UltraMemory, UltraMemoryConfig, CollectionSpec, Record, Filter

from core.arxiv.arxiv import ArXivFetcher
from core.arxiv.paper_db import Paper


@dataclass
class PaperMemoryConfig:
    """Config for PaperMemory.

    dsn_template: e.g. "postgresql://user:pass@host:port/{db}". The {db} placeholder will be
    replaced by remote database name derived from segment key:
      - "2024H1" -> "h1_2024"
      - "2024H2" -> "h2_2024"

    If dsn_map is provided, it overrides the template for given segment keys.
    """

    dsn_template: str
    dsn_map: Optional[Dict[str, str]] = None
    collection_name: str = "papers"
    vector_field: str = "text_vec"
    vector_dim: int = 3072
    metric: str = "cosine"
    index_params: Dict[str, Any] = None  # type: ignore[assignment]


class PaperMemory:
    """UltraMemory-backed segmented memory for arXiv papers (half-year segments).

    Segment policy: YYYYH1 or YYYYH2. Each segment maps to a dedicated Postgres database
    named as:
      - YYYYH1 -> h1_YYYY
      - YYYYH2 -> h2_YYYY
    already present on the remote server. We connect to the DB and create a collection
    named by config.collection_name (default: "papers").

    Record mapping:
    - id: arXiv id (may include version suffix)
    - content: summary
    - vector: Azure text-embedding-3-large
    - attributes: metadata from paper, flattened as JSON (keeps original keys)
    - timestamp: ISO8601 string derived from metadata timestamp/submittedDate/published
    """

    def __init__(self, cfg: PaperMemoryConfig) -> None:
        load_dotenv()
        self.logger = AgentLogger(self.__class__.__name__)

        self.cfg = cfg
        if self.cfg.index_params is None:
            self.cfg.index_params = {}

        # Azure client
        api_key = os.getenv("AZURE_API_KEY", "")
        base_url = os.getenv("AZURE_BASE_URL", "https://gpt.yunstorm.com/")
        api_version = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
        if not api_key:
            self.logger.error("AZURE_API_KEY not found in environment variables")
            raise ValueError("AZURE_API_KEY is required")
        self.llm_client = AzureClient(api_key=api_key, base_url=base_url, api_version=api_version)
        self.embed_model: str = "text-embedding-3-large"

        # Cache for segment UltraMemory instances
        self._segments: Dict[str, UltraMemory] = {}

        # Read-cache for queries
        self._query_cache: Dict[str, List[Tuple[str, List[float], Dict[str, Any]]]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl: float = 300.0

        # Local directory for detecting known segments (compat with ArxivMemory tooling)
        self.persist_dir: Path = get_current_file_dir() / 'database'
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self.logger.info("PaperMemory initialized with UltraMemory (pgvector) and segmented storage")

    # ------------------------------ DSN & UltraMemory helpers ------------------------------
    def _dsn_for_segment(self, segment_key: str) -> str:
        if self.cfg.dsn_map and segment_key in self.cfg.dsn_map:
            return self.cfg.dsn_map[segment_key]
        tpl = self.cfg.dsn_template
        # Map YYYYH1/YYYYH2 -> h1_YYYY/h2_YYYY
        db_name = segment_key
        m = re.match(r"^(\d{4})H([12])$", segment_key)
        if m:
            year = m.group(1)
            half = m.group(2)
            db_name = f"h{half}_{year}"
        if "{db}" in tpl:
            return tpl.replace("{db}", db_name)
        # Fallback: append database at the end
        if tpl.rstrip('/').count('/') >= 3:
            # Looks like already includes a database; replace the last path segment
            try:
                prefix = tpl.rsplit('/', 1)[0]
                return prefix + '/' + db_name
            except Exception:
                return tpl.rstrip('/') + '/' + db_name
        return tpl.rstrip('/') + '/' + db_name

    def _get_segment_um(self, segment_key: str) -> UltraMemory:
        um = self._segments.get(segment_key)
        if um:
            return um
        dsn = self._dsn_for_segment(segment_key)
        um = UltraMemory(UltraMemoryConfig(backend="postgres_pgvector", dsn=dsn))
        # Ensure collection exists with vector schema
        spec = CollectionSpec(
            name=self.cfg.collection_name,
            mode="vector",
            vector_dimensions={self.cfg.vector_field: int(self.cfg.vector_dim)},
            metric=self.cfg.metric,
            index_params=dict(self.cfg.index_params or {}),
        )
        try:
            um.create_collection(spec)
        except Exception as e:
            self.logger.warning(f"Create collection failed for segment {segment_key}: {e}")
        self._segments[segment_key] = um
        return um

    def _list_existing_segments(self) -> List[str]:
        """Detect segments similarly to ArxivMemory by scanning local database directory.
        Fallback to a reasonable range when none found.
        """
        out: List[str] = []
        if self.persist_dir.exists():
            try:
                for p in self.persist_dir.iterdir():
                    if p.is_dir() and (re.match(r"^\d{4}H[12]$", p.name) or p.name == "undated"):
                        out.append(p.name)
            except Exception:
                pass
        if not out:
            # Default range 2022..2030
            try:
                for y in range(2022, 2031):
                    out.append(f"{y}H1")
                    out.append(f"{y}H2")
            except Exception:
                pass
        out.sort()
        return out

    # ------------------------------ Date helpers ------------------------------
    def _extract_date_str(self, md: Dict[str, Any]) -> Optional[str]:
        ts = md.get("timestamp") or md.get("submittedDate") or md.get("published") or ""
        if not isinstance(ts, str):
            ts = str(ts)
        m = re.match(r"(\d{8})", ts)
        if m:
            return m.group(1)
        m2 = re.match(r"(\d{4})[-/]?(\d{2})[-/]?(\d{2})", ts)
        if m2:
            return f"{m2.group(1)}{m2.group(2)}{m2.group(3)}"
        return None

    def _to_iso_ts(self, md: Dict[str, Any]) -> Optional[str]:
        s = md.get("timestamp") or md.get("submittedDate") or md.get("published")
        if s is None:
            d = self._extract_date_str(md)
            if not d:
                return None
            s = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        if isinstance(s, (int, float)):
            try:
                from datetime import datetime
                return datetime.utcfromtimestamp(float(s)).strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                pass
        st = str(s)
        # Normalize a few common formats
        m = re.match(r"^(\d{4})(\d{2})(\d{2})$", st)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T00:00:00Z"
        m2 = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T].*)?$", st)
        if m2:
            # Keep date part; if time exists, assume UTC
            return f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}T00:00:00Z"
        # Fallback best-effort
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(st.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
        return None

    def _segment_key_from_date(self, date_str: str) -> Optional[str]:
        if not isinstance(date_str, str) or len(date_str) < 6:
            return None
        try:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            half = "H1" if 1 <= month <= 6 else "H2"
            return f"{year}{half}"
        except Exception:
            return None

    # ------------------------------ Existence checks ------------------------------
    def _exists_with_any_segment(self, item_id: str) -> bool:
        for seg in self._list_existing_segments():
            try:
                um = self._get_segment_um(seg)
                rows = um.query(self.cfg.collection_name, Filter(expr={"eq": ["id", str(item_id)]}, limit=1))
                if rows:
                    return True
            except Exception as e:
                self.logger.warning(f"Existence check failed in segment {seg}: {e}")
        return False

    # ------------------------------ Public APIs ------------------------------
    async def store_papers(self, papers: List[Paper], *, max_concurrency: int = 32) -> List[str]:
        if not papers:
            return []

        semaphore = asyncio.Semaphore(max_concurrency)

        async def embed_one(idx: int, paper: Paper) -> Tuple[int, Optional[List[float]], Optional[Exception]]:
            async with semaphore:
                try:
                    summary = paper.info.get("summary", "")
                    text = str(summary) if summary is not None else ""
                    vec = await self.llm_client.embedding(text, model_name=self.embed_model)
                    if vec is None:
                        return idx, None, RuntimeError("Embedding returned None")
                    if isinstance(vec, list) and vec and isinstance(vec[0], list):
                        vec = vec[0]
                    return idx, [float(x) for x in vec], None
                except Exception as e:
                    self.logger.warning(f"Embedding failed for paper {paper.id}: {e}")
                    return idx, None, e

        items: List[Record] = []
        to_embed: List[Tuple[int, Paper]] = []

        for p in papers:
            pid = p.id
            md = dict(p.info)
            md["id"] = pid
            if self._exists_with_any_segment(pid):
                self.logger.info(f"Skip id={pid}: already stored (any segment)")
                continue
            idx = len(items)
            # vector filled after embedding
            items.append(Record(id=pid, attributes=md, content=str(p.info.get("summary", "") or ""), vector=None, timestamp=None))
            to_embed.append((idx, p))

        # Embed concurrently
        if to_embed:
            tasks: List[asyncio.Task] = [asyncio.create_task(embed_one(idx, paper)) for idx, paper in to_embed]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            for idx, vec, err in results:
                if vec is not None:
                    items[idx].vector = vec
                else:
                    self.logger.warning(f"Skip storing paper due to missing vector: {items[idx].id}")

        # Upsert by segment
        stored: List[str] = []
        by_segment: Dict[str, List[Record]] = {}
        for it in items:
            if it.vector is None:
                continue
            date_str = self._extract_date_str(it.attributes) or ""
            seg = self._segment_key_from_date(date_str) or "undated"
            it.timestamp = self._to_iso_ts(it.attributes)
            by_segment.setdefault(seg, []).append(it)

        for seg, recs in by_segment.items():
            try:
                um = self._get_segment_um(seg)
                ids = um.upsert(self.cfg.collection_name, recs)
                stored.extend(ids)
            except Exception as e:
                self.logger.error(f"Upsert failed for segment {seg}: {e}")

        return stored

    def get_by_id(self, arxiv_id: str) -> List[Dict[str, Any]]:
        cid = str(arxiv_id).strip()
        if not cid:
            return []
        out: List[Dict[str, Any]] = []
        # If version specified, return exact first hit
        if re.search(r"v\d+$", cid):
            for seg in self._list_existing_segments():
                try:
                    um = self._get_segment_um(seg)
                    rows = um.query(self.cfg.collection_name, Filter(expr={"eq": ["id", cid]}, limit=1))
                    if rows:
                        r = rows[0]
                        out.append({"id": r.get("id"), "content": r.get("content"), "metadata": r.get("attributes", {})})
                        return out
                except Exception:
                    continue
            return []

        base = cid
        for seg in self._list_existing_segments():
            try:
                um = self._get_segment_um(seg)
                # Fetch by scanning ids in this segment; use LIKE to minimize load
                rows = um.query(self.cfg.collection_name, Filter(expr={"like": ["id", f"{base}%"]}, limit=10000))
                for r in rows:
                    rid = str(r.get("id") or "")
                    if rid == base or rid.startswith(base + "v"):
                        out.append({"id": rid, "content": r.get("content"), "metadata": r.get("attributes", {})})
            except Exception:
                continue

        def _version_key(x: Dict[str, Any]) -> Tuple[int, int]:
            xid = x.get("id", "")
            if xid == base:
                return (0, 0)
            m = re.search(r"v(\d+)$", xid)
            if m:
                return (1, int(m.group(1)))
            return (2, 0)

        out.sort(key=_version_key)
        return out

    def _fetch_vector_row(self, seg: str, item_id: str) -> Optional[Tuple[str, List[float], Dict[str, Any]]]:
        """Fetch a single row including vector via adapter connection if available."""
        try:
            um = self._get_segment_um(seg)
            adapter = getattr(um, "adapter", None)
            conn = getattr(adapter, "_conn", None)
            if conn is None:
                # Fallback: no direct vector, return basic via query
                rows = um.query(self.cfg.collection_name, Filter(expr={"eq": ["id", item_id]}, limit=1))
                if not rows:
                    return None
                r = rows[0]
                return r.get("content"), [], r.get("attributes", {})
            tbl = adapter._tbl(self.cfg.collection_name)  # type: ignore[attr-defined]
            vec_col = adapter._col(self.cfg.vector_field)  # type: ignore[attr-defined]
            cur = conn.cursor()
            try:
                cur.execute(f'SELECT content, attributes, "{vec_col}" FROM "{tbl}" WHERE id=%s', (item_id,))
                row = cur.fetchone()
                if not row:
                    return None
                content = row[0]
                attrs = row[1] or {}
                v = row[2]
                vec: List[float] = []
                if v is not None:
                    # psycopg returns vector as string like '[0.1,0.2]' or a memoryview depending on adapter
                    try:
                        if isinstance(v, str):
                            s = v.strip().lstrip('[').rstrip(']')
                            if s:
                                vec = [float(x) for x in s.split(',')]
                        elif isinstance(v, (list, tuple)):
                            vec = [float(x) for x in v]
                        else:
                            s = str(v).strip().lstrip('[').rstrip(']')
                            if s:
                                vec = [float(x) for x in s.split(',')]
                    except Exception:
                        vec = []
                return content, vec, attrs
            finally:
                cur.close()
        except Exception as e:
            self.logger.warning(f"Vector fetch failed in {seg} for id={item_id}: {e}")
            return None

    def get_item_by_id(self, arxiv_id: str) -> Optional[Tuple[str, List[float], Dict[str, Any]]]:
        cid = str(arxiv_id).strip()
        if not cid:
            return None
        if re.search(r"v\d+$", cid):
            for seg in self._list_existing_segments():
                row = self._fetch_vector_row(seg, cid)
                if row is not None:
                    return row
            return None

        # No version, prefer base id then highest version
        best: Optional[Tuple[str, List[float], Dict[str, Any]]] = None
        best_ver = -1
        base = cid
        for seg in self._list_existing_segments():
            try:
                um = self._get_segment_um(seg)
                rows = um.query(self.cfg.collection_name, Filter(expr={"like": ["id", f"{base}%"]}, limit=10000))
                for r in rows:
                    rid = str(r.get("id") or "")
                    if rid == base:
                        # fetch vector precisely
                        got = self._fetch_vector_row(seg, rid)
                        if got is not None:
                            return got
                        return (r.get("content"), [], r.get("attributes", {}))
                    m = re.search(r"v(\d+)$", rid)
                    if m:
                        ver = int(m.group(1))
                        if ver > best_ver:
                            best_ver = ver
                            got = self._fetch_vector_row(seg, rid)
                            if got is not None:
                                best = got
                            else:
                                best = (r.get("content"), [], r.get("attributes", {}))
            except Exception:
                continue
        return best

    def get_items_by_ids(self, arxiv_ids: List[str]) -> Dict[str, Tuple[str, List[float], Dict[str, Any]]]:
        out: Dict[str, Tuple[str, List[float], Dict[str, Any]]] = {}
        if not arxiv_ids:
            return out
        for aid in arxiv_ids:
            try:
                row = self.get_item_by_id(aid)
            except Exception as e:
                self.logger.warning(f"Failed to fetch item for id={aid}: {e}")
                row = None
            if row is not None:
                out[str(aid)] = row
        return out

    def histogram_by_day(self) -> List[Tuple[str, int]]:
        counts: Dict[str, int] = {}

        for seg in self._list_existing_segments():
            try:
                um = self._get_segment_um(seg)
                # Query all timestamps ordered for this segment in pages
                offset = 0
                page = 10000
                while True:
                    rows = um.query(self.cfg.collection_name, Filter(expr={}, order_by=[("timestamp", "asc")], limit=page, offset=offset))
                    if not rows:
                        break
                    for r in rows:
                        md = r.get("attributes", {})
                        d = self._extract_date_str(md)
                        if d:
                            counts[d] = counts.get(d, 0) + 1
                    if len(rows) < page:
                        break
                    offset += page
            except Exception as e:
                self.logger.warning(f"Histogram scan failed for {seg}: {e}")

        return sorted(counts.items(), key=lambda x: x[0])

    def get_by_month(self, yyyymm: str, categories: Optional[List[str]] = None, *, include_vector: bool = True, limit: Optional[int] = None, batch_size: int = 10000) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        cid = str(yyyymm).strip()
        if not (len(cid) == 6 and cid.isdigit()):
            raise ValueError("yyyymm should be 'YYYYMM')")

        cache_key = f"{cid}_{categories}_{include_vector}_{limit}"
        now = time.time()
        if cache_key in self._query_cache:
            if now - self._cache_timestamps.get(cache_key, 0.0) < self._cache_ttl:
                return self._query_cache[cache_key]
            self._query_cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)

        year = int(cid[:4])
        month = int(cid[4:6])
        if not (1 <= month <= 12):
            raise ValueError("Month should be between 01 and 12")
        seg_key = f"{year}{'H1' if month <= 6 else 'H2'}"

        out: List[Tuple[str, List[float], Dict[str, Any]]] = []
        seen: set = set()

        # Compute range [start, next_month)
        from datetime import datetime, timezone
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end.strftime("%Y-%m-%dT%H:%M:%SZ")

        def _categories_match(md: Dict[str, Any]) -> bool:
            if not categories:
                return True
            cats = md.get("categories", [])
            if isinstance(cats, str):
                cats_list = [cats]
            elif isinstance(cats, list):
                cats_list = [str(x) for x in cats]
            else:
                cats_list = []
            return bool(set(cats_list).intersection(set([c.strip() for c in categories])))

        try:
            um = self._get_segment_um(seg_key)
        except Exception as e:
            self.logger.warning(f"No UltraMemory for segment {seg_key}: {e}")
            self._query_cache[cache_key] = out
            self._cache_timestamps[cache_key] = now
            return out

        adapter = getattr(um, "adapter", None)
        conn = getattr(adapter, "_conn", None)
        tbl = adapter._tbl(self.cfg.collection_name) if adapter else None  # type: ignore[attr-defined]
        vec_col = adapter._col(self.cfg.vector_field) if adapter else None  # type: ignore[attr-defined]

        if include_vector and conn is not None and tbl and vec_col:
            # Use direct SQL to fetch vectors
            cur = conn.cursor()
            try:
                base_sql = f'SELECT id, content, attributes, "{vec_col}" FROM "{tbl}" WHERE ts BETWEEN %s::timestamptz AND %s::timestamptz'
                params = [start_iso, end_iso]
                if limit is not None:
                    base_sql += " LIMIT %s"
                    params.append(int(limit))
                cur.execute(base_sql, tuple(params))
                rows = cur.fetchall()
                for r in rows:
                    mid = r[0]
                    if mid in seen:
                        continue
                    md = r[2] or {}
                    if not _categories_match(md):
                        continue
                    content = r[1]
                    v = r[3]
                    vec: List[float] = []
                    if v is not None:
                        try:
                            if isinstance(v, str):
                                s = v.strip().lstrip('[').rstrip(']')
                                if s:
                                    vec = [float(x) for x in s.split(',')]
                            elif isinstance(v, (list, tuple)):
                                vec = [float(x) for x in v]
                            else:
                                s = str(v).strip().lstrip('[').rstrip(']')
                                if s:
                                    vec = [float(x) for x in s.split(',')]
                        except Exception:
                            vec = []
                    seen.add(mid)
                    out.append((content, vec, md))
                    if limit is not None and len(out) >= int(limit):
                        break
            finally:
                cur.close()
        else:
            # Fallback through query API (no vectors)
            rows = um.query(self.cfg.collection_name, Filter(expr={"range": {"timestamp": [start_iso, end_iso]}}, limit=limit))
            for r in rows:
                mid = r.get("id")
                if mid in seen:
                    continue
                md = r.get("attributes", {})
                if not _categories_match(md):
                    continue
                seen.add(mid)
                out.append((r.get("content"), [], md))

        self._query_cache[cache_key] = out
        self._cache_timestamps[cache_key] = now
        # Cleanup if large
        if len(self._query_cache) > 50:
            self._cleanup_query_cache()
        return out

    async def store_one_day(self, date_str: str, categories: Optional[List[str]] = None, *, max_results: int = 10000, max_concurrency: int = 32) -> List[str]:
        self.logger.info(f"Storing papers for {date_str} with {categories} and {max_results} results")
        if not isinstance(date_str, str) or len(date_str) != 8 or not date_str.isdigit():
            raise ValueError("date_str should be 'YYYYMMDD'")

        try:
            from datetime import datetime, timedelta
            dt = datetime.strptime(date_str, "%Y%m%d")
            next_day = (dt + timedelta(days=1)).strftime("%Y%m%d")
        except Exception:
            next_day = date_str

        if categories:
            cats = " OR ".join([f"cat:{c}" for c in categories])
            q = f"({cats}) AND submittedDate:[{date_str} TO {next_day}]"
        else:
            q = f"submittedDate:[{date_str} TO {next_day}]"
        fetcher = ArXivFetcher()
        papers = await fetcher.search(query_string=q, max_results=max_results)
        self.logger.info(f"Found {len(papers)} papers")
        return await self.store_papers(papers, max_concurrency=max_concurrency)

    def clear_query_cache(self) -> None:
        self._query_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Query cache cleared")

    def _cleanup_query_cache(self) -> None:
        now = time.time()
        expired = [k for k, ts in self._cache_timestamps.items() if (now - ts) >= self._cache_ttl]
        for k in expired:
            self._query_cache.pop(k, None)
            self._cache_timestamps.pop(k, None)
        if expired:
            self.logger.debug(f"Cleaned up {len(expired)} expired query cache entries")

    # --------------- Migration helper ---------------
    def upsert_records(self, segment_key: str, records: List[Record]) -> List[str]:
        if not records:
            return []
        um = self._get_segment_um(segment_key)
        return um.upsert(self.cfg.collection_name, records)


if __name__ == "__main__":
    # Example usage (requires valid DSN template)
    cfg = PaperMemoryConfig(dsn_template=os.getenv("PAPERMEMORY_DSN_TEMPLATE", "postgresql://user:pass@host:port/{db}"))
    pm = PaperMemory(cfg)
    # print(pm.histogram_by_day())
    # rows = pm.get_by_month("202406", categories=["cs.AI"])  # will not include vectors unless DB is available
    # print(len(rows))


