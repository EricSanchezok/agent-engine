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

        # Prepare inputs
        items: List[ArxivItem] = []
        texts_to_embed_idx: List[int] = []
        tasks: List[asyncio.Task] = []

        for i, p in enumerate(papers):
            pid = p.id
            content = str(p.info.get("summary", "") or "")
            md = dict(p.info)
            md["id"] = pid
            items.append(ArxivItem(id=pid, content=content, metadata=md, vector=None))
            tasks.append(asyncio.create_task(embed_one(i, p)))
            texts_to_embed_idx.append(i)

        # Run concurrent embeddings
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for idx, vec, err in results:
            if vec is not None:
                items[idx].vector = vec
            else:
                # If embedding failed, skip this paper
                self.logger.warning(f"Skip storing paper due to missing vector: {papers[idx].id}")

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

    async def store_one_day(self, date_str: str, categories: Optional[List[str]] = None, *, max_results: int = 10000, max_concurrency: int = 32) -> List[str]:
        """Fetch and store all arXiv papers for a day and optional categories.

        Args:
            date_str: 'YYYYMMDD'
            categories: List like ['cs.AI', 'cs.LG']; if provided, query will OR them.
            max_results: limit
            max_concurrency: embedding concurrency
        """
        if not isinstance(date_str, str) or len(date_str) != 8 or not date_str.isdigit():
            raise ValueError("date_str should be 'YYYYMMDD'")

        # Build query string
        if categories:
            cats = " OR ".join([f"cat:{c}" for c in categories])
            q = f"({cats}) AND submittedDate:[{date_str} TO {date_str}]"
        else:
            q = f"submittedDate:[{date_str} TO {date_str}]"

        fetcher = ArXivFetcher()
        papers = await fetcher.search(query_string=q, max_results=max_results)
        return await self.store_papers(papers, max_concurrency=max_concurrency)


