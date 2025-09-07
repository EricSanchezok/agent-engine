from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    """A scalable memory for arXiv metadata for ResearchAgent.

    - Storage location: agents/ResearchAgent/database/
    - Item id: Paper.id (can include version suffix like 'v2')
    - Content: Paper.summary
    - Vector: Azure text-embedding-3-large
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

        # Persist dir for ScalableMemory under ResearchAgent/database
        self.persist_dir: Path = get_current_file_dir() / 'database'

        # ScalableMemory for arxiv metadata; name fixed to 'arxiv_metadata'
        # Persist dir rule: treat persist_dir as final storage directory (no extra subdir)
        self.memory = ScalableMemory(
            name="arxiv_metadata",
            llm_client=self.llm_client,
            embed_model=self.embed_model,
            persist_dir=self.persist_dir,
        )

        self.logger.info("ArxivMemory initialized with Azure embeddings and ScalableMemory")

    # ------------------------------ Public APIs ------------------------------
    async def store_papers(self, papers: List[Paper], *, max_concurrency: int = 32) -> List[str]:
        """Store a batch of arXiv papers' metadata into memory.

        - id: paper.id (can include version)
        - content: paper.summary
        - vector: Azure embedding for summary
        - metadata: paper.info (dict)

        Optimization:
        - Before embedding, check if the id already exists AND has a stored vector; if so, skip to avoid token cost.
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

        # Prepare inputs: skip items that already have vectors in DB
        items: List[ArxivItem] = []
        to_embed: List[Tuple[int, Paper]] = []

        for p in papers:
            pid = p.id
            content = str(p.info.get("summary", "") or "")
            md = dict(p.info)
            md["id"] = pid

            # Check existing record and vector to avoid redundant embeddings
            try:
                _existing_content, existing_vec, _existing_md = self.memory.get_by_id(pid)
            except Exception as e:
                self.logger.warning(f"Lookup failed for id={pid}: {e}")
                existing_vec = None

            if isinstance(existing_vec, list) and len(existing_vec) > 0:
                # Already stored with vector; skip
                self.logger.info(f"Skip id={pid}: already stored with vector")
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

        payload: List[Dict[str, Any]] = []
        for it in items:
            if it.vector is None:
                continue
            payload.append({
                "id": it.id,
                "content": it.content,
                "vector": it.vector,
                "metadata": it.metadata,
            })

        if not payload:
            return []

        ids = await self.memory.add_many(payload)
        return ids

    def get_by_id(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """Get metadata by id. If version is omitted, return all versions including non-version form.

        Examples:
            - input '2402.11163v2' -> return that exact version if present
            - input '2402.11163'   -> return list including '2402.11163', '2402.11163v1', '2402.11163v2', ...
        """
        cid = str(arxiv_id).strip()
        if not cid:
            return []

        # If contains version suffix 'v<digits>', return exact
        import re
        if re.search(r"v\d+$", cid):
            content, _vec, md = self.memory.get_by_id(cid)
            return [{"id": cid, "content": content, "metadata": md}] if content is not None else []

        # No version specified: return all matching ids
        base = cid
        results: List[Dict[str, Any]] = []
        for content, _vec, md in self.memory.get_all():
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
        """Return a histogram of papers per day from the memory database.

        Returns:
            List of (date_str, count) where date_str is 'YYYYMMDD', sorted ascending by date.
            Only dates that have at least one paper are included.
        """
        import re

        counts: Dict[str, int] = {}
        try:
            metadatas = self.memory.get_all_metadata()
        except Exception as e:
            self.logger.error(f"Failed to load metadata for histogram: {e}")
            return []

        for md in metadatas:
            ts = md.get("timestamp") or md.get("submittedDate") or md.get("published") or ""
            if not isinstance(ts, str):
                ts = str(ts)

            date_str: Optional[str] = None
            m = re.match(r"(\d{8})", ts)
            if m:
                date_str = m.group(1)
            else:
                m2 = re.match(r"(\d{4})[-/]?(\d{2})[-/]?(\d{2})", ts)
                if m2:
                    date_str = f"{m2.group(1)}{m2.group(2)}{m2.group(3)}"

            if date_str:
                counts[date_str] = counts.get(date_str, 0) + 1

        return sorted(counts.items(), key=lambda x: x[0])

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


if __name__ == "__main__":
    asyncio.run(ArxivMemory().store_one_day("20250904"))