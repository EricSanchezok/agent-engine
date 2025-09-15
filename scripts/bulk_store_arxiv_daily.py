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


"""
Bulk store arXiv papers day by day starting from 2022-01-01 up to today.

Usage:
  run.bat scripts\bulk_store_arxiv_daily.py

Notes:
  - Stores all categories (no category filter).
  - Sequentially iterates day by day to avoid overwhelming external services.
  - Uses Azure embeddings via ArxivMemory and ScalableMemory for persistence.
"""

import asyncio
from datetime import date, timedelta
from typing import List

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ResearchAgent.arxiv_memory import ArxivMemory


def generate_date_strs(start: date, end: date) -> List[str]:
    """Generate YYYYMMDD strings for dates in [start, end], inclusive."""
    out: List[str] = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


async def main() -> int:
    logger = AgentLogger("BulkArxivDaily")
    mem = ArxivMemory()

    start = date(2024, 6, 14)
    end = date.today()
    date_list = generate_date_strs(start, end)

    logger.info(f"Starting bulk store from {date_list[0]} to {date_list[-1]} ({len(date_list)} days)")

    total_ok = 0
    for ds in date_list:
        try:
            logger.info(f"Processing date {ds}")
            ids = await mem.store_one_day(ds, categories=None)
            total_ok += len(ids)
            logger.info(f"Done {ds}: stored {len(ids)} items; total stored so far: {total_ok}")
        except Exception as e:
            logger.error(f"Failed {ds}: {e}")

    logger.info(f"Bulk store completed. Total stored items: {total_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


