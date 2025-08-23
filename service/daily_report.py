"""SignalFrontier 日常流程调度脚本

本文件提供两大功能：
1. `run_for_date(date_str: str)`
   立即对指定日期（YYYYMMDD）执行完整流程：
   - 检索 arXiv 论文
   - 内容过滤
   - PDF 下载
   - 生成解析报告
   - 结果写入数据库

2. `schedule_daily()`
   使用 APScheduler 在北京时间每天 09:00 自动执行前一天的流程。
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import logging
from typing import List, Tuple

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Agent imports -------------------------------------------------------------
from agents.ArxivSearchAgent.agent import ArxivSearchAgent  # type: ignore
from agents.PaperFilterAgent.agent import PaperFilterAgent  # type: ignore
from agents.PaperFetchAgent.agent import PaperFetchAgent  # type: ignore
from agents.PaperAnalysisAgent.agent import PaperAnalysisAgent  # type: ignore

# Core imports --------------------------------------------------------------
from core.arxiv.arxiv import ArXivFetcher  # type: ignore
from core.arxiv.paper_db import Paper, ArxivPaperDB  # type: ignore
from core.arxiv.config import CATEGORIES_QUERY_STRING  # type: ignore

logger = logging.getLogger("SignalFrontier")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _build_query(date_str: str) -> str:
    """根据日期构造 arXiv 查询字符串。

    arXiv 的 submittedDate 区间查询语法为闭区间 `[start TO end]`，因此
    若想检索某一天（date_str）内的全部论文，需要将结束日期设置为
    次日的字符串，以确保 `date_str <= submittedDate < next_day`。
    """
    start = _dt.datetime.strptime(date_str, "%Y%m%d")
    end = (start + _dt.timedelta(days=1)).strftime("%Y%m%d")
    return f"submittedDate:[{date_str} TO {end}] AND {CATEGORIES_QUERY_STRING}"


def _extract_ids(papers: List[Paper]) -> List[str]:
    return [p.id.replace("_", ".") for p in papers]

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def _run_flow_for_date(date_str: str):
    """异步执行完整流程。"""
    logger.info(f"===== Start SignalFrontier pipeline for {date_str} =====")

    # 1. 检索 --------------------------------------------------------------
    query = _build_query(date_str)
    logger.info(f"ArxivSearchAgent 查询: {query}")

    fetcher = ArXivFetcher()
    papers: List[Paper] = await fetcher.search(query)
    logger.info(f"ArxivSearchAgent 返回 {len(papers)} 篇论文")

    if not papers:
        logger.warning("检索结果为空，流程结束")
        return

    arxiv_ids = _extract_ids(papers)

    # 2. 过滤 --------------------------------------------------------------
    logger.info("调用 PaperFilterAgent 进行向量过滤…")
    filter_agent = PaperFilterAgent()

    filter_input = {"arxiv_ids": arxiv_ids, "max_recommendations": 8}
    filter_ids: List[str] = await _invoke_filter_agent(filter_agent, filter_input)

    if not filter_ids:
        logger.warning("过滤后无论文，流程结束")
        return

    logger.info(f"PaperFilterAgent 留下 {len(filter_ids)} 篇论文")

    # 3. 下载 --------------------------------------------------------------
    logger.info("调用 PaperFetchAgent 下载 PDF…")
    fetch_agent = PaperFetchAgent()
    papers_with_pdf: List[Paper] = await _invoke_fetch_agent(fetch_agent, filter_ids)
    logger.info(f"成功下载 {len(papers_with_pdf)} 篇论文 PDF")

    if not papers_with_pdf:
        logger.warning("下载后无有效 PDF，流程结束")
        return

    # 4. 解析 --------------------------------------------------------------
    logger.info("调用 PaperAnalysisAgent 生成报告…")
    analysis_agent = PaperAnalysisAgent()
    await _invoke_analysis_agent(analysis_agent, papers_with_pdf)

    logger.info("===== Pipeline finished =====")

# ---------------------------------------------------------------------------
# 内部调用封装（利用 Agent 内部 API，但绕过 A2A 框架）
# ---------------------------------------------------------------------------

async def _invoke_filter_agent(agent: PaperFilterAgent, payload: dict) -> List[str]:
    """利用 PaperFilterAgent 逻辑返回筛选 id 列表。"""
    import json
    from a2a.server.events import EventQueue  # type: ignore
    from a2a.server.agent_execution import RequestContext  # type: ignore
    from a2a.utils import new_agent_text_message  # type: ignore
    from a2a.types import MessageSendParams  # type: ignore
    import uuid

    queue = EventQueue()
    msg = new_agent_text_message(json.dumps(payload, ensure_ascii=False))
    context = RequestContext(request=MessageSendParams(message=msg))
    await agent.execute(context, queue)

    event = await queue.dequeue_event()
    if not event:
        return []
    # The dequeued object is already a Task instance in the current a2a version,
    # hence we no longer need to access the deprecated ``event.task`` attribute.
    task = event
    if not task.artifacts:
        return []
    text_json = task.artifacts[0].parts[0].root.text  # type: ignore
    try:
        ids = json.loads(text_json)
        return ids
    except Exception:
        return []


async def _invoke_fetch_agent(agent: PaperFetchAgent, ids: List[str]) -> List[Paper]:
    """利用 PaperFetchAgent 下载，并返回包含 pdf_bytes 的 Paper 列表。"""
    import json
    from a2a.server.events import EventQueue  # type: ignore
    from a2a.server.agent_execution import RequestContext  # type: ignore
    from a2a.utils import new_agent_text_message  # type: ignore
    from a2a.types import MessageSendParams  # type: ignore
    import uuid

    queue = EventQueue()
    msg = new_agent_text_message(json.dumps({"arxiv_ids": ids}, ensure_ascii=False))
    context = RequestContext(request=MessageSendParams(message=msg))
    await agent.execute(context, queue)

    # 与 ArXivFetcher 使用相同的数据库文件保持一致
    db = ArxivPaperDB("database/arxiv_paper_db.sqlite")
    papers = [db.get(pid) for pid in ids]
    return [p for p in papers if p and p.pdf_bytes]


async def _invoke_analysis_agent(agent: PaperAnalysisAgent, papers: List[Paper]):
    """利用 PaperAnalysisAgent 为每篇论文生成报告并保存到数据库（metadata['report']）。"""
    from a2a.server.events import EventQueue  # type: ignore
    from a2a.server.agent_execution import RequestContext  # type: ignore
    from a2a.types import FilePart, Part, FileWithBytes, Message, Role, MessageSendParams  # type: ignore
    import uuid

    queue = EventQueue()

    parts = [
        Part(root=FilePart(file=FileWithBytes(name=f"{p.id}.pdf", mime_type="application/pdf", bytes=p.pdf_bytes)))  # type: ignore
        for p in papers
    ]
    user_message = Message(
        role=Role.user,
        task_id=str(uuid.uuid4()),
        message_id=str(uuid.uuid4()),
        content_id=str(uuid.uuid4()),
        parts=parts,
    )
    context = RequestContext(request=MessageSendParams(message=user_message))

    await agent.execute(context, queue)

    # 解析返回结果并写入数据库的 metadata['report']
    event = await queue.dequeue_event()
    if not event or not event.artifacts:
        logger.warning("PaperAnalysisAgent 没有返回任何 artifacts，跳过报告写入")
        return

    # 获取文本列表（一个 part 对应一篇论文的报告）
    parts = event.artifacts[0].parts
    reports = [p.root.text for p in parts]  # type: ignore

    if not reports:
        logger.warning("PaperAnalysisAgent 返回的报告为空，跳过写入")
        return

    # 将报告写入对应 Paper.metadata['report'] 并保存到数据库
    db = ArxivPaperDB("database/arxiv_paper_db.sqlite")
    for paper, report in zip(papers, reports):
        if report:
            paper.metadata["report"] = report
            db.add(paper, overwrite=True)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def run_for_date(date_str: str):
    """同步接口：对指定日期 (YYYYMMDD) 执行完整流程。"""
    asyncio.run(_run_flow_for_date(date_str))


def schedule_daily():
    """启动定时任务：每天 09:00 (Asia/Shanghai) 处理前一天数据。"""
    tz = pytz.timezone("Asia/Shanghai")

    def _job_wrapper():
        yesterday = (_dt.datetime.now(tz) - _dt.timedelta(days=1)).strftime("%Y%m%d")
        run_for_date(yesterday)

    scheduler = BackgroundScheduler(timezone=tz)
    trigger = CronTrigger(hour=9, minute=0)
    scheduler.add_job(_job_wrapper, trigger, id="signal_frontier_daily")
    scheduler.start()
    logger.info("Daily SignalFrontier pipeline scheduled at 09:00 Asia/Shanghai")

    # 防止脚本退出
    try:
        while True:
            asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


if __name__ == "__main__":
    run_for_date("20250821")