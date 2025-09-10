from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent_engine.agent_logger import AgentLogger

from core.arxiv.arxiv import ArXivFetcher
from core.arxiv.paper_db import Paper


def _build_date_query(date_str: str, categories: Optional[List[str]] = None) -> str:
    """Build arXiv query string for a single day.

    arXiv submittedDate is a closed interval [start TO end]. To fetch one day,
    set end to next day.
    """
    from datetime import datetime, timedelta

    dt = datetime.strptime(date_str, "%Y%m%d")
    next_day = (dt + timedelta(days=1)).strftime("%Y%m%d")
    if categories:
        cats = " OR ".join([f"cat:{c}" for c in categories])
        return f"({cats}) AND submittedDate:[{date_str} TO {next_day}]"
    return f"submittedDate:[{date_str} TO {next_day}]"


class PaperSearchModule:
    """Search papers from arXiv and optionally return minimal metadata.

    Unified entry:
        await run({
            "query": Optional[str],
            "date_str": Optional[str],  # YYYYMMDD
            "categories": Optional[List[str]],
            "max_results": Optional[int] = 10000,
        }) -> {
            "code": 0,
            "message": "success",
            "data": {"arxiv_ids": List[str], "papers": List[Dict[str, Any]]}
        }
    """

    def __init__(self) -> None:
        self.logger = AgentLogger(self.__class__.__name__)
        self.fetcher = ArXivFetcher()

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query: Optional[str] = payload.get("query")
            date_str: Optional[str] = payload.get("date_str")
            categories: Optional[List[str]] = payload.get("categories")
            max_results: int = int(payload.get("max_results", 10000))

            if not query:
                if date_str:
                    query = _build_date_query(date_str, categories)
                else:
                    raise ValueError("Either 'query' or 'date_str' must be provided")

            self.logger.info(f"Searching arXiv with query: {query}")
            papers: List[Paper] = await self.fetcher.search(query_string=query, max_results=max_results)

            arxiv_ids: List[str] = [p.id for p in papers]
            infos: List[Dict[str, Any]] = [p.info for p in papers]

            self.logger.info(f"Search returned {len(arxiv_ids)} papers")
            return {
                "code": 0,
                "message": "success",
                "data": {
                    "arxiv_ids": arxiv_ids,
                    "papers": infos,
                },
            }
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {"code": -1, "message": f"search error: {e}", "data": {}}


