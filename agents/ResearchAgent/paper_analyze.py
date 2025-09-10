from __future__ import annotations

from typing import Any, Dict, List

from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient

from core.arxiv.arxiv import ArXivFetcher
from core.arxiv.paper_db import Paper, ArxivPaperDB


class PaperAnalyzeModule:
    """Analyze PDFs and generate reports.

    Unified entry:
        await run({
            "arxiv_ids": List[str],
            "write_back_to_db": bool = True,
        }) -> {
            "code": 0,
            "message": "success",
            "data": {"reports": Dict[id, str]}
        }
    """

    def __init__(self) -> None:
        self.logger = AgentLogger(self.__class__.__name__)
        self.llm_client = AzureClient()
        self.fetcher = ArXivFetcher()
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[2]
        self.db = ArxivPaperDB(str(project_root / "database" / "arxiv_paper_db.sqlite"))

    async def _analyze_one(self, pdf_b64: str) -> str:
        # 这里保持简单：直接将 PDF 文本交给 LLM 生成报告。
        # 可在后续集成更复杂的解析/分段/草稿+复审流程。
        try:
            system_prompt = "You are a helpful research assistant. Generate a concise technical report for the provided paper content."
            # 使用小模型草拟，节省成本
            result = await self.llm_client.chat(system_prompt, pdf_b64, model_name='o3-mini', max_tokens=8000)
            return result or ""
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return ""

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ids: List[str] = list(payload.get("arxiv_ids") or [])
            write_back: bool = bool(payload.get("write_back_to_db", True))
            if not ids:
                raise ValueError("'arxiv_ids' is required and should be non-empty")

            papers: List[Paper] = await self.fetcher.search(id_list=ids)
            if not papers:
                raise ValueError("No papers found for given ids")

            # Ensure PDFs
            import ssl, certifi, socket, aiohttp, asyncio
            from tqdm.asyncio import tqdm
            import random

            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(
                ssl=ssl_ctx,
                limit=32,
                limit_per_host=16,
                family=socket.AF_INET,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )

            async def ensure_pdf(p: Paper, session: aiohttp.ClientSession) -> Paper:
                if p.pdf_bytes:
                    return p
                await asyncio.sleep(random.uniform(0.05, 0.2))
                return await self.fetcher.download(p, session)

            ensured: List[Paper] = []
            async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
                tasks = [ensure_pdf(p, session) for p in papers]
                for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Ensuring PDFs"):
                    try:
                        ensured.append(await coro)
                    except Exception as e:
                        self.logger.error(f"Ensure pdf task failed: {e}")

            reports: Dict[str, str] = {}
            for p in ensured:
                if not p.pdf_bytes:
                    self.logger.warning(f"Skip {p.id}: no pdf bytes")
                    continue
                text = p.pdf_bytes  # 简化：直接传 base64 给 LLM，后续可替换为抽取文本
                report = await self._analyze_one(text)
                if report:
                    reports[p.id] = report
                    if write_back:
                        try:
                            dbp = self.db.get(p.id)
                            dbp.metadata["report"] = report
                            dbp.metadata["report_source"] = "daily"
                            self.db.add(dbp, overwrite=True)
                        except Exception as e:
                            self.logger.error(f"Write back failed for {p.id}: {e}")

            if not reports:
                raise ValueError("No reports generated")

            return {"code": 0, "message": "success", "data": {"reports": reports}}
        except Exception as e:
            self.logger.error(f"Analyze failed: {e}")
            return {"code": -1, "message": f"analyze error: {e}", "data": {}}


