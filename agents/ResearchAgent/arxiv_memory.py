from __future__ import annotations

import os
import re
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
import pyinstrument
from dotenv import load_dotenv

from agent_engine.agent_logger.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.memory import ScalableMemory
from agent_engine.utils import get_current_file_dir

from core.arxiv.arxiv import ArXivFetcher
from core.arxiv.paper_db import Paper


@dataclass
class ArxivItem:
    """Lightweight DTO for memory IO."""
    id: str
    content: str
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None


class ArxivMemory:
    """Segmented arXiv metadata memory for ResearchAgent.

    Storage strategy:
    - Base location: agents/ResearchAgent/database/
    - Segmented DBs (read/write): base_dir/segments/<YYYYH1|YYYYH2>/
        Each segment directory contains a ScalableMemory instance (DuckDB + index)

    Item mapping:
    - id: Paper.id (may include version suffix like 'v2')
    - content: Paper.summary
    - vector: Azure text-embedding-3-large
    """

    def __init__(self) -> None:
        load_dotenv()
        self.logger = AgentLogger(self.__class__.__name__)

        # Azure client configuration via environment variables (consistent with other agents)
        api_key = os.getenv("AZURE_API_KEY", "")
        base_url = os.getenv("AZURE_BASE_URL", "https://gpt.yunstorm.com/")
        api_version = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
        if not api_key:
            self.logger.error("AZURE_API_KEY not found in environment variables")
            raise ValueError("AZURE_API_KEY is required")

        self.llm_client = AzureClient(api_key=api_key, base_url=base_url, api_version=api_version)
        # Embedding model fixed as requested
        self.embed_model: str = "text-embedding-3-large"

        # Base dir for all arXiv memories (segments)
        self.persist_dir: Path = get_current_file_dir() / 'database'
        self.segments_dir: Path = self.persist_dir
        try:
            self.segments_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Cache for segment ScalableMemory instances
        self._segment_memories: Dict[str, ScalableMemory] = {}
        
        # Query result cache
        self._query_cache: Dict[str, List[Tuple[str, List[float], Dict[str, Any]]]] = {}
        self._cache_ttl: float = 300.0  # 5 minutes
        self._cache_timestamps: Dict[str, float] = {}

        self.logger.info("ArxivMemory initialized with segmented storage (half-year) and Azure embeddings")

    # ------------------------------ Internal helpers ------------------------------
    def _extract_date_str(self, md: Dict[str, Any]) -> Optional[str]:
        """Extract YYYYMMDD from metadata best-effort."""
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

    def _segment_key_from_date(self, date_str: str) -> Optional[str]:
        """Return segment key like '2022H1' or '2022H2' from YYYYMMDD."""
        if not isinstance(date_str, str) or len(date_str) < 6:
            return None
        try:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            half = "H1" if 1 <= month <= 6 else "H2"
            return f"{year}{half}"
        except Exception:
            return None

    def _get_segment_memory(self, segment_key: str) -> ScalableMemory:
        """Get or create ScalableMemory for a specific segment under segments_dir/segment_key.

        Name is kept as 'arxiv_metadata' to keep filenames consistent inside each segment directory.
        """
        if segment_key in self._segment_memories:
            return self._segment_memories[segment_key]
        seg_dir = self.segments_dir / segment_key
        try:
            seg_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        mem = ScalableMemory(
            name="arxiv_metadata",
            llm_client=self.llm_client,
            embed_model=self.embed_model,
            persist_dir=seg_dir,
        )
        self._segment_memories[segment_key] = mem
        return mem

    def _list_existing_segments(self) -> List[str]:
        """List existing segment keys by scanning the segments directory, including 'undated' if present."""
        if not self.segments_dir.exists():
            return []
        out: List[str] = []
        try:
            for p in self.segments_dir.iterdir():
                if not p.is_dir():
                    continue
                if re.match(r"^\d{4}H[12]$", p.name) or p.name == "undated":
                    out.append(p.name)
        except Exception:
            return out
        out.sort()
        return out

    def _iter_all_memories_for_read(self) -> List[ScalableMemory]:
        """Return all memories to search (segments in chronological order)."""
        mems: List[ScalableMemory] = []
        for seg in self._list_existing_segments():
            mems.append(self._get_segment_memory(seg))
        return mems

    def _exists_with_vector(self, item_id: str) -> bool:
        """Check across segments if the item exists with a stored vector."""
        for mem in self._iter_all_memories_for_read():
            try:
                _c, vec, _m = mem.get_by_id(item_id)
            except Exception as e:
                self.logger.warning(f"Lookup failed for id={item_id} in a memory: {e}")
                continue
            if isinstance(vec, list) and len(vec) > 0:
                return True
        return False

    # ------------------------------ Public APIs ------------------------------
    async def store_papers(self, papers: List[Paper], *, max_concurrency: int = 32) -> List[str]:
        """Store a batch of arXiv papers' metadata into memory.

        - id: paper.id (can include version)
        - content: paper.summary
        - vector: Azure embedding for summary
        - metadata: paper.info (dict)

        Optimization:
        - Before embedding, check if the id already exists in any segment AND has a stored vector; if so, skip to avoid token cost.
        """
        if not papers:
            return []

        semaphore = asyncio.Semaphore(max_concurrency)

        async def embed_one(idx: int, paper: Paper) -> Tuple[int, Optional[List[float]], Optional[Exception]]:
            async with semaphore:
                try:
                    summary = paper.info.get("summary", "")
                    # Ensure string
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

        # Prepare inputs: skip items that already have vectors in any DB
        items: List[ArxivItem] = []
        to_embed: List[Tuple[int, Paper]] = []

        for p in papers:
            pid = p.id
            content = str(p.info.get("summary", "") or "")
            md = dict(p.info)
            md["id"] = pid

            # Check existing record and vector across segments
            try:
                exists = self._exists_with_vector(pid)
            except Exception as e:
                self.logger.warning(f"Lookup failed for id={pid}: {e}")
                exists = False

            if exists:
                self.logger.info(f"Skip id={pid}: already stored with vector (any segment)")
                continue

            # Need embedding
            idx = len(items)
            items.append(ArxivItem(id=pid, content=content, metadata=md, vector=None))
            to_embed.append((idx, p))

        # Run concurrent embeddings
        if to_embed:
            tasks: List[asyncio.Task] = [asyncio.create_task(embed_one(idx, paper)) for idx, paper in to_embed]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            for idx, vec, err in results:
                if vec is not None:
                    items[idx].vector = vec
                else:
                    # If embedding failed, leave vector None; it will be filtered out
                    self.logger.warning(f"Skip storing paper due to missing vector: {items[idx].id}")

        # Group payloads by segment key
        payload_by_segment: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            if it.vector is None:
                continue
            date_str = self._extract_date_str(it.metadata) or ""
            seg = self._segment_key_from_date(date_str) or "unknown"
            if seg == "unknown":
                # Fallback: put undated items into new segment 'unknown'
                seg = "undated"
            payload_by_segment.setdefault(seg, []).append({
                "id": it.id,
                "content": it.content,
                "vector": it.vector,
                "metadata": it.metadata,
            })

        if not payload_by_segment:
            return []

        stored_ids: List[str] = []
        for seg_key, payload in payload_by_segment.items():
            mem = self._get_segment_memory(seg_key)
            ids = await mem.add_many(payload)
            stored_ids.extend(ids)
        return stored_ids

    def get_by_id(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """Get metadata by id across segmented DBs.

        If version is omitted, return all versions including non-version form.

        Examples:
            - input '2402.11163v2' -> return that exact version if present
            - input '2402.11163'   -> list including '2402.11163', '2402.11163v1', '2402.11163v2', ...
        """
        cid = str(arxiv_id).strip()
        if not cid:
            return []

        # If contains version suffix 'v<digits>', return exact (first hit wins: segments in order)
        if re.search(r"v\d+$", cid):
            for mem in self._iter_all_memories_for_read():
                content, _vec, md = mem.get_by_id(cid)
                if content is not None:
                    return [{"id": cid, "content": content, "metadata": md}]
            return []

        # No version specified: collect all matching ids across memories
        base = cid
        results: List[Dict[str, Any]] = []
        for mem in self._iter_all_memories_for_read():
            for content, _vec, md in mem.get_all():
                mid = md.get("id")
                if not isinstance(mid, str):
                    continue
                if mid == base or mid.startswith(base + "v"):
                    results.append({"id": mid, "content": content, "metadata": md})

        # Stable sort by version number if present, with base id first
        def _version_key(x: Dict[str, Any]) -> Tuple[int, int]:
            xid = x.get("id", "")
            if xid == base:
                return (0, 0)
            m = re.search(r"v(\d+)$", xid)
            if m:
                return (1, int(m.group(1)))
            return (2, 0)

        results.sort(key=_version_key)
        return results

    def histogram_by_day(self) -> List[Tuple[str, int]]:
        """Return a histogram of papers per day across segmented DBs."""
        counts: Dict[str, int] = {}

        def _consume_md(md_list: List[Dict[str, Any]]) -> None:
            for md in md_list:
                d = self._extract_date_str(md)
                if d:
                    counts[d] = counts.get(d, 0) + 1

        try:
            # Segments first
            for mem in self._iter_all_memories_for_read():
                try:
                    _consume_md(mem.get_all_metadata())
                except Exception as e:
                    self.logger.warning(f"Failed to load metadata from a memory: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load metadata for histogram: {e}")
            return []

        return sorted(counts.items(), key=lambda x: x[0])

    def get_by_month(self, yyyymm: str, categories: Optional[List[str]] = None, *, include_vector: bool = True, limit: Optional[int] = None, batch_size: int = 10000) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        """Return all (content, vector, metadata) for a specific month and optional categories.

        Args:
            yyyymm: Month in 'YYYYMM' format.
            categories: Optional list of category strings (e.g., ['cs.AI', 'cs.LG']). If provided,
                an item is included when it has any overlap with the given categories.
            include_vector: If False, vectors are not loaded (returned as empty lists).
            limit: Optional maximum number of results to return across all memories.
            batch_size: Rows per page per memory when scanning DB.

        Notes:
            - Searches the corresponding half-year segment first if it exists.
            - Date is extracted from metadata using the same logic as ingestion.
        """
        cid = str(yyyymm).strip()
        if not (len(cid) == 6 and cid.isdigit()):
            raise ValueError("yyyymm should be 'YYYYMM'")
        
        # Check cache first
        cache_key = f"{cid}_{categories}_{include_vector}_{limit}"
        current_time = time.time()
        
        if cache_key in self._query_cache:
            if current_time - self._cache_timestamps[cache_key] < self._cache_ttl:
                self.logger.debug(f"Cache hit for query {cache_key}")
                return self._query_cache[cache_key]
            else:
                # Cache expired
                del self._query_cache[cache_key]
                del self._cache_timestamps[cache_key]
        
        self.logger.debug(f"Cache miss for query {cache_key}")

        # Determine target half-year segment
        try:
            year = int(cid[:4])
            month = int(cid[4:6])
        except Exception:
            raise ValueError("Invalid yyyymm format")
        if not (1 <= month <= 12):
            raise ValueError("Month should be between 01 and 12")
        seg_key = f"{year}{'H1' if month <= 6 else 'H2'}"

        # Build list of memories to read: target segment (if exists)
        memories: List[ScalableMemory] = []
        try:
            existing = set(self._list_existing_segments())
        except Exception:
            existing = set()
        if seg_key in existing:
            memories.append(self._get_segment_memory(seg_key))

        # Normalize category filter
        category_set = set([c.strip() for c in categories]) if categories else None

        results: List[Tuple[str, List[float], Dict[str, Any]]] = []
        seen_ids: set = set()

        def _build_where_and_params(mem_backend: str) -> Tuple[str, List[str]]:
            params: List[str] = []
            if mem_backend == "duckdb":
                # Use JSON extraction for faster predicate evaluation with optimized patterns
                month_clause = (
                    "("
                    "json_extract_string(metadata, '$.timestamp') LIKE ? OR "
                    "json_extract_string(metadata, '$.submittedDate') LIKE ? OR "
                    "json_extract_string(metadata, '$.published') LIKE ?"
                    ")"
                )
                params.extend([f"{cid}%", f"{cid}%", f"{cid}%"])
            else:
                # Optimized SQLite patterns - use more specific patterns first
                month_patterns = [
                    f'%"timestamp":"{cid}%',  # Most common format first
                    f'%"timestamp": "{cid}%',
                    f'%"submittedDate":"{cid}%',
                    f'%"submittedDate": "{cid}%',
                    f'%"published":"{cid}%',
                    f'%"published": "{cid}%'
                ]
                month_clause = "(" + " OR ".join(["metadata LIKE ?" for _ in month_patterns]) + ")"
                params.extend(month_patterns)

            if category_set is not None and len(category_set) > 0:
                # Use more efficient category matching
                cat_patterns = [f'%"{c}"%' for c in category_set]
                cat_clause = "(" + " OR ".join(["metadata LIKE ?" for _ in cat_patterns]) + ")"
                where = month_clause + " AND " + cat_clause
                params.extend(cat_patterns)
            else:
                where = month_clause
            return where, params

        for mem in memories:
            try:
                mem_backend = getattr(mem.db, "backend", "sqlite")
                where_clause, params = _build_where_and_params(mem_backend)
                # Decide projected columns
                cols = "id, content, metadata, vector" if include_vector else "id, content, metadata"
                
                # Use larger batch size for better performance
                sql = f"SELECT {cols} FROM items WHERE {where_clause} LIMIT {int(batch_size)}"
                cur = mem.db.execute(sql, tuple(params))
                rows = mem.db.fetchall(cur)
                
                if not rows:
                    continue
                    
                # Process all rows at once for better performance
                batch_results = []
                for r in rows:
                    mid = r[0]
                    if mid in seen_ids:
                        continue
                        
                    # Parse metadata and verify month/categories precisely
                    try:
                        md = r[2]
                        if isinstance(md, dict):
                            md_dict = md
                        elif isinstance(md, str) and md:
                            # Use more efficient JSON parsing
                            md_dict = json.loads(md)
                        else:
                            md_dict = {}
                    except (json.JSONDecodeError, TypeError):
                        md_dict = {}
                        
                    d = self._extract_date_str(md_dict)
                    if not (isinstance(d, str) and len(d) >= 6 and d[:6] == cid):
                        continue
                        
                    if category_set is not None:
                        cats = md_dict.get("categories", [])
                        if isinstance(cats, str):
                            cats_list = [cats]
                        elif isinstance(cats, list):
                            cats_list = [str(x) for x in cats]
                        else:
                            cats_list = []
                        if not set(cats_list).intersection(category_set):
                            continue

                    content = r[1]
                    if include_vector:
                        try:
                            vblob = r[3]
                            if vblob is None:
                                vec: List[float] = []
                            else:
                                # More efficient vector conversion
                                vec_array = np.frombuffer(vblob, dtype=np.float32)
                                vec = vec_array.tolist()
                        except (ValueError, TypeError):
                            vec = []
                    else:
                        vec = []

                    seen_ids.add(mid)
                    batch_results.append((content, vec, md_dict))
                    if limit is not None and len(results) + len(batch_results) >= int(limit):
                        results.extend(batch_results)
                        return results
                
                results.extend(batch_results)

                # If no rows matched via SQL, fallback to Python metadata scan (no vectors)
                if not results:
                    for md in mem.get_all_metadata():
                        d = self._extract_date_str(md)
                        if not (isinstance(d, str) and len(d) >= 6 and d[:6] == cid):
                            continue
                        if category_set is not None:
                            cats = md.get("categories", [])
                            if isinstance(cats, str):
                                cats_list = [cats]
                            elif isinstance(cats, list):
                                cats_list = [str(x) for x in cats]
                            else:
                                cats_list = []
                            if not set(cats_list).intersection(category_set):
                                continue
                        mid = md.get("id")
                        if not isinstance(mid, str) or mid in seen_ids:
                            continue
                        content, vector, _md = mem.get_by_id(mid)
                        if content is None:
                            continue
                        vec = vector or []
                        seen_ids.add(mid)
                        results.append((content, vec if include_vector else [], md))
                        if limit is not None and len(results) >= int(limit):
                            return results
            except Exception as e:
                self.logger.warning(f"Failed to read from a memory for month {cid}: {e}")

        # Cache the results
        self._query_cache[cache_key] = results
        self._cache_timestamps[cache_key] = current_time
        
        # Clean up expired cache entries if cache is getting large
        if len(self._query_cache) > 50:
            self._cleanup_query_cache()

        return results

    async def store_one_day(self, date_str: str, categories: Optional[List[str]] = None, *, max_results: int = 10000, max_concurrency: int = 32) -> List[str]:
        """Fetch and store all arXiv papers for a day and optional categories.

        Args:
            date_str: 'YYYYMMDD'
            categories: List like ['cs.AI', 'cs.LG']; if provided, query will OR them.
            max_results: limit
            max_concurrency: embedding concurrency
        """
        self.logger.info(f"Storing papers for {date_str} with {categories} and {max_results} results")
        if not isinstance(date_str, str) or len(date_str) != 8 or not date_str.isdigit():
            raise ValueError("date_str should be 'YYYYMMDD'")

        # Build query string over [date_str, next_day]
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
    
    def _cleanup_query_cache(self) -> None:
        """Clean up expired query cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp >= self._cache_ttl
        ]
        for key in expired_keys:
            self._query_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired query cache entries")
    
    def clear_query_cache(self) -> None:
        """Clear all query cache."""
        self._query_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Query cache cleared")


if __name__ == "__main__":
    mem = ArxivMemory()
    # print(mem.histogram_by_day())
    triples = mem.get_by_month("202406")
    print(len(triples))