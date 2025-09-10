from __future__ import annotations

import asyncio
import random
import ssl
import socket
from typing import Any, Dict, List

import aiohttp
import certifi
from tqdm.asyncio import tqdm

from agent_engine.agent_logger import AgentLogger

from core.arxiv.arxiv import ArXivFetcher
from core.arxiv.paper_db import Paper


class PaperDownloadModule:
    """Download PDFs for arXiv ids and return file bytes in memory.

    Unified entry:
        await run({
            "arxiv_ids": List[str],
        }) -> {
            "code": 0,
            "message": "success",
            "data": {"files": List[Dict[name, bytes_b64]]}
        }
    """

    def __init__(self) -> None:
        self.logger = AgentLogger(self.__class__.__name__)
        self.fetcher = ArXivFetcher()
        self.semaphore = asyncio.Semaphore(32)

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            arxiv_ids: List[str] = list(payload.get("arxiv_ids") or [])
            if not arxiv_ids:
                raise ValueError("'arxiv_ids' is required and should be non-empty")

            papers: List[Paper] = await self.fetcher.search(id_list=arxiv_ids)
            if not papers:
                raise ValueError("No papers found for given ids")

            async def download_with_semaphore(paper: Paper, session: aiohttp.ClientSession) -> Paper:
                async with self.semaphore:
                    await asyncio.sleep(random.uniform(0.05, 0.2))
                    return await self.fetcher.download(paper, session)

            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(
                ssl=ssl_ctx,
                limit=32,
                limit_per_host=16,
                family=socket.AF_INET,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )

            downloaded: List[Paper] = []
            async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
                tasks = [download_with_semaphore(p, session) for p in papers]
                for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Downloading Papers"):
                    try:
                        p = await coro
                        downloaded.append(p)
                    except Exception as e:
                        self.logger.error(f"Download task failed: {e}")

            files = []
            for p in downloaded:
                if isinstance(p, Paper) and p.pdf_bytes:
                    files.append({"name": f"{p.id}.pdf", "bytes_b64": p.pdf_bytes})

            if not files:
                raise ValueError("No papers downloaded successfully")

            self.logger.info(f"Downloaded {len(files)} papers")
            return {"code": 0, "message": "success", "data": {"files": files}}
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            return {"code": -1, "message": f"download error: {e}", "data": {}}


