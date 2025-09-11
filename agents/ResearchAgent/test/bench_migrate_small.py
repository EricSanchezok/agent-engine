from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import pyinstrument

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ResearchAgent.paper_memory import PaperMemory, PaperMemoryConfig
from agents.ResearchAgent.scripts.migrate_arxiv_to_ultra import (
    _iter_source_segments, _iter_items_from_sqlite, _iter_items_from_duckdb,
    _extract_date_str, _to_iso_ts, _segment_key_from_date,
)
from agent_engine.memory.ultra_memory import Record


logger = AgentLogger("BenchMigrateSmall")


def _take_items(base_dir, max_items: int = 1000) -> List[Tuple[str, str, Dict[str, Any], Optional[List[float]]]]:
    out: List[Tuple[str, str, Dict[str, Any], Optional[List[float]]]] = []
    for seg_dir in _iter_source_segments(base_dir):
        sqlite_path = seg_dir / "arxiv_metadata.sqlite3"
        duckdb_path = seg_dir / "arxiv_metadata.duckdb"
        it: Iterable[Tuple[str, str, Dict[str, Any], Optional[List[float]]]]
        if sqlite_path.exists():
            it = _iter_items_from_sqlite(sqlite_path)
        elif duckdb_path.exists():
            it = _iter_items_from_duckdb(duckdb_path)
        else:
            continue
        for row in it:
            out.append(row)
            if len(out) >= max_items:
                return out
    return out


@pyinstrument.profile()
def migrate_1000_no_dirty() -> None:
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

    base_dir = Path(__file__).resolve().parents[1] / 'database'
    items = _take_items(base_dir, 1000)
    if not items:
        logger.warning("No local items found for test")
        return

    # Prepare records grouped by remote segment
    by_seg: Dict[str, List[Record]] = {}
    for rid, content, md, vec in items:
        date_str = _extract_date_str(md) or ""
        seg = _segment_key_from_date(date_str) or "undated"
        ts = _to_iso_ts(md)
        rec = Record(id=str(rid), attributes=md, content=str(content or ""), vector=vec, timestamp=ts)
        by_seg.setdefault(seg, []).append(rec)

    # Measure per segment bulk upsert, then delete to avoid dirty data
    total = 0
    t0 = time.time()
    for seg, recs in by_seg.items():
        if not recs:
            continue
        logger.info(f"Testing segment {seg} with {len(recs)} records")
        um = pm._get_segment_um(seg)
        # Bulk upsert via adapter direct SQL path as在迁移脚本中
        from agents.ResearchAgent.scripts.migrate_arxiv_to_ultra import _bulk_upsert_segment
        t1 = time.time()
        n, ids = _bulk_upsert_segment(pm, seg, recs)
        t2 = time.time()
        logger.info(f"Upsert {n} records elapsed={t2 - t1:.3f}s")
        total += n
        # Clean up inserted ids to avoid leaving data
        td1 = time.time()
        for rid in ids:
            try:
                um.delete_by_id(pm.cfg.collection_name, rid)
            except Exception:
                pass
        td2 = time.time()
        logger.info(f"Delete {len(recs)} records elapsed={td2 - td1:.3f}s")
    t3 = time.time()
    logger.info(f"Total upserted (and deleted)={total} elapsed={t3 - t0:.3f}s")


if __name__ == "__main__":
    migrate_1000_no_dirty()


