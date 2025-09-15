"""SignalFrontier 日常流程调度脚本

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


# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

import asyncio
import datetime as _dt
import logging
from typing import List

# Third-party imports
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import time
import datetime as _dt

# Agent imports -------------------------------------------------------------
from agents.PaperFilterAgent.agent import PaperFilterAgent  # type: ignore
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

db = ArxivPaperDB("database/arxiv_paper_db.sqlite")

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
# Database helpers
# ---------------------------------------------------------------------------
def _flush_daily_reports(date_str: str) -> None:
    """Update `report_source` from *daily* to *agent* for all papers of the day.

    This function scans the database for papers whose timestamp matches
    ``date_str`` and whose ``metadata['report_source']`` is ``"daily"``. When
    such a paper is found, the value is replaced with ``"agent"`` and the row
    is overwritten. The business logic of the pipeline is **not** affected –
    this is merely a data-consistency pre-processing step.

    Parameters
    ----------
    date_str : str
        The target date in ``YYYYMMDD`` format.
    """
    papers = db.get_by_date(date_str)

    updated = 0
    for paper in papers:
        if paper.metadata.get("report_source") == "daily":
            paper.metadata["report_source"] = "agent"
            db.add(paper, overwrite=True)
            updated += 1

    if updated:
        logger.info(
            f"Updated report_source to 'agent' for {updated} papers on {date_str}"
        )
    else:
        logger.info("No papers required report_source update.")

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def _run_flow_for_date(date_str: str):
    """
    异步执行完整流程的核心逻辑。
    如果发生任何预料之外的错误，这个函数会抛出异常。
    """
    logger.info(f"===== Start SignalFrontier pipeline for {date_str} =====")

    # 1. 检索 --------------------------------------------------------------
    query = _build_query(date_str)
    logger.info(f"ArxivSearchAgent 查询: {query}")

    fetcher = ArXivFetcher()
    papers: List[Paper] = await fetcher.search(query)
    logger.info(f"ArxivSearchAgent 返回 {len(papers)} 篇论文")

    if not papers:
        # 如果没有检索到论文，则抛出异常以触发重试
        raise ValueError("检索结果为空，触发重试")

    arxiv_ids = _extract_ids(papers)
    await asyncio.sleep(10)

    # 2. 过滤 --------------------------------------------------------------
    logger.info("调用 PaperFilterAgent 进行向量过滤…")
    filter_agent = PaperFilterAgent()

    filter_input = {"arxiv_ids": arxiv_ids, "max_recommendations": 8}
    filter_ids: List[str] = await _invoke_filter_agent(filter_agent, filter_input)

    if not filter_ids:
        # 如果过滤后没有论文，同样抛出异常以触发重试
        raise ValueError("过滤后无论文，触发重试")

    logger.info(f"PaperFilterAgent 留下 {len(filter_ids)} 篇论文")
    await asyncio.sleep(10)
    
    # 3. 解析 --------------------------------------------------------------
    logger.info("调用 PaperAnalysisAgent 生成报告…")
    analysis_agent = PaperAnalysisAgent()

    analysis_input = {"arxiv_ids": filter_ids}
    await _invoke_analysis_agent(analysis_agent, analysis_input)

    logger.info(f"===== Pipeline for {date_str} finished successfully =====")


async def run_flow_for_date_with_retry(date_str: str, max_retries: int = 72):
    """
    带有重试逻辑的包装函数。
    它会调用 _run_flow_for_date，并在失败时重试。

    :param date_str: 执行流程的日期字符串。
    :param max_retries: 最大重试次数。
    """
    for attempt in range(max_retries):
        try:
            # 尝试执行核心流程
            await _run_flow_for_date(date_str)
            # 如果成功，就跳出循环
            return
        except Exception as e:
            # 捕获所有可能的异常
            logger.error(f"流程执行失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # 如果不是最后一次尝试，就等待一段时间再重试
                wait_time = 300
                logger.info(f"将在 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                # 如果所有重试都失败了，记录最终的失败信息
                logger.critical(f"流程在 {max_retries} 次尝试后彻底失败。")

async def _invoke_filter_agent(agent: PaperFilterAgent, payload: dict) -> List[str]:
    """利用 PaperFilterAgent 逻辑返回筛选 id 列表。
    现在直接使用 BaseA2AAgent 提供的 `run_user_input` 高阶封装，
    避免重复构造 EventQueue / RequestContext 等底层细节。
    """
    import json

    # 直接调用高级 API
    event = await agent.run_user_input(json.dumps(payload, ensure_ascii=False))

    if not event or not event.artifacts:
        return []

    # artifacts[0] 中应包含单个文本 part，内容为 JSON 数组字符串
    try:
        text_json = event.artifacts[0].parts[0].root.text  # type: ignore
        ids = json.loads(text_json)
        return ids if isinstance(ids, list) else []
    except Exception:
        return []


async def _invoke_analysis_agent(agent: PaperAnalysisAgent, payload: dict):
    import json

    event = await agent.run_user_input(json.dumps(payload, ensure_ascii=False))

    # 解析返回结果并写入数据库的 metadata['report']
    if not event or not event.artifacts:
        raise ValueError("PaperAnalysisAgent 没有返回任何 artifacts，跳过报告写入")

    result_parts = event.artifacts[0].parts
    reports = [p.root.text for p in result_parts]  # type: ignore

    if not reports:
        raise ValueError("PaperAnalysisAgent 返回的报告为空，跳过写入")

    # 将报告写入对应 Paper.metadata['report'] 并保存到数据库
    ids = payload["arxiv_ids"]
    for id, report in zip(ids, reports):
        if report:
            paper = db.get(id)
            paper.metadata["report"] = report
            paper.metadata["report_source"] = "daily"
            db.add(paper, overwrite=True)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def run_for_date(date_str: str):
    """同步接口：对指定日期 (YYYYMMDD) 执行完整流程。"""
    # Ensure data consistency before pipeline starts
    _flush_daily_reports(date_str)

    # Execute the main pipeline
    asyncio.run(run_flow_for_date_with_retry(date_str))

def schedule_daily():
    """启动定时任务：每天 10:30 (Asia/Shanghai) 处理前一天数据。"""
    tz = pytz.timezone("Asia/Shanghai")

    def _job_wrapper():
        # 在实际任务中获取时间，而不是在定义时
        yesterday = (_dt.datetime.now(tz) - _dt.timedelta(days=1)).strftime("%Y%m%d")
        run_for_date(yesterday)

    scheduler = BackgroundScheduler(timezone=tz)
    trigger = CronTrigger(hour=10, minute=30, second=0) 
    scheduler.add_job(_job_wrapper, trigger, id="signal_frontier_daily")
    scheduler.start()
    logger.info("Daily SignalFrontier pipeline scheduled at 10:30 Asia/Shanghai")

    # 防止主线程退出
    try:
        while True:
            # 使用 time.sleep() 来阻塞主线程
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down...")
        scheduler.shutdown()
        logger.info("Scheduler has been shut down.")


if __name__ == "__main__":
    # run_for_date("20250902")
    schedule_daily()