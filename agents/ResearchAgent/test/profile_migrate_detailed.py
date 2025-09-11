from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
import pyinstrument

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ResearchAgent.paper_memory import PaperMemory, PaperMemoryConfig
from agents.ResearchAgent.scripts.migrate_arxiv_to_ultra import (
    _bulk_upsert_segment, _extract_date_str, _to_iso_ts, _segment_key_from_date,
    _iter_source_segments, _iter_items_from_sqlite
)
from agent_engine.memory.ultra_memory import Record
from pathlib import Path


logger = AgentLogger("ProfileMigrateDetailed")


@pyinstrument.profile()
def profile_migrate_steps() -> None:
    """分段测试迁移的各个步骤性能"""
    from agents.ResearchAgent.config import PAPER_DSN_TEMPLATE
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

    # Step 1: 读取本地数据
    t0 = time.time()
    base_dir = Path(__file__).resolve().parents[1] / 'database'
    items: List = []
    for seg_dir in _iter_source_segments(base_dir):
        sqlite_path = seg_dir / "arxiv_metadata.sqlite3"
        if sqlite_path.exists():
            for row in _iter_items_from_sqlite(sqlite_path):
                items.append(row)
                if len(items) >= 500:  # 测试500条
                    break
            break
    t1 = time.time()
    logger.info(f"Step 1 - Read local data: {len(items)} items in {t1 - t0:.3f}s")

    # Step 2: 数据转换
    t0 = time.time()
    records: List[Record] = []
    for rid, content, md, vec in items:
        date_str = _extract_date_str(md) or ""
        seg = _segment_key_from_date(date_str) or "undated"
        ts = _to_iso_ts(md)
        rec = Record(id=str(rid), attributes=md, content=str(content or ""), vector=vec, timestamp=ts)
        records.append(rec)
    t1 = time.time()
    logger.info(f"Step 2 - Data transformation: {len(records)} records in {t1 - t0:.3f}s")

    # Step 3: 获取分片UltraMemory（含DDL）
    t0 = time.time()
    um = pm._get_segment_um("2022H1")
    t1 = time.time()
    logger.info(f"Step 3 - Get segment UM (DDL): {t1 - t0:.3f}s")

    # Step 4: 批量写入（分小批次测试）
    batch_sizes = [50, 100, 200, 500]
    for batch_size in batch_sizes:
        if batch_size > len(records):
            continue
        batch = records[:batch_size]
        t0 = time.time()
        count, ids = _bulk_upsert_segment(pm, "2022H1", batch)
        t1 = time.time()
        rate = batch_size / (t1 - t0) if (t1 - t0) > 0 else 0
        logger.info(f"Step 4 - Bulk upsert {batch_size} records: {t1 - t0:.3f}s ({rate:.1f} records/s)")
        
        # 清理写入的数据
        for rid in ids:
            try:
                um.delete_by_id(pm.cfg.collection_name, rid)
            except Exception:
                pass

    # Step 5: 测试不同COPY vs VALUES路径
    if len(records) >= 200:
        batch = records[:200]
        
        # 测试COPY路径
        import agents.ResearchAgent.scripts.migrate_arxiv_to_ultra as mig_mod
        orig_copy = mig_mod.USE_COPY_LOAD
        try:
            mig_mod.USE_COPY_LOAD = True
            t0 = time.time()
            count, ids = _bulk_upsert_segment(pm, "2022H1", batch)
            t1 = time.time()
            logger.info(f"Step 5a - COPY path 200 records: {t1 - t0:.3f}s")
            # 清理
            for rid in ids:
                try:
                    um.delete_by_id(pm.cfg.collection_name, rid)
                except Exception:
                    pass
            
            # 测试VALUES路径
            mig_mod.USE_COPY_LOAD = False
            t0 = time.time()
            count, ids = _bulk_upsert_segment(pm, "2022H1", batch)
            t1 = time.time()
            logger.info(f"Step 5b - VALUES path 200 records: {t1 - t0:.3f}s")
            # 清理
            for rid in ids:
                try:
                    um.delete_by_id(pm.cfg.collection_name, rid)
                except Exception:
                    pass
        finally:
            mig_mod.USE_COPY_LOAD = orig_copy


if __name__ == "__main__":
    profile_migrate_steps()
