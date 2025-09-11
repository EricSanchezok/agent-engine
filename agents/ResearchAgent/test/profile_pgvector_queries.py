from __future__ import annotations

import os
import time
from typing import List
import pyinstrument

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ResearchAgent.config import PAPER_DSN_TEMPLATE
from agents.ResearchAgent.paper_memory import PaperMemory, PaperMemoryConfig
from agent_engine.memory.ultra_memory import Filter


logger = AgentLogger("ProfilePgvectorQueries")


@pyinstrument.profile()
def run_small_scan(segment: str = "2023H1", limit: int = 1000) -> None:
    logger.info(f"Profiling segment={segment}, limit={limit}")

    if not PAPER_DSN_TEMPLATE or "USER:PASS@HOST:PORT" in PAPER_DSN_TEMPLATE:
        logger.error("PAPER_DSN_TEMPLATE is not properly configured")
        return

    pm = PaperMemory(PaperMemoryConfig(
        dsn_template=PAPER_DSN_TEMPLATE,
        collection_name="papers",
        vector_field="text_vec",
        vector_dim=3072,
        metric="cosine",
        index_params={"lists": 100},
    ))

    um = pm._get_segment_um(segment)

    t0 = time.time()
    rows = um.query("papers", Filter(limit=limit))
    t1 = time.time()
    logger.info(f"Scan rows={len(rows)} elapsed={t1 - t0:.3f}s")

    if rows:
        # fetch a sample record
        sid = rows[0].get("id")
        logger.info(f"Sample id={sid}")
        t2 = time.time()
        one = um.query("papers", Filter(expr={"eq": ["id", sid]}, limit=1))
        t3 = time.time()
        logger.info(f"Fetch by id elapsed={t3 - t2:.3f}s found={len(one)}")


if __name__ == "__main__":
    run_small_scan()


