from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import pyinstrument

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ResearchAgent.config import PAPER_DSN_TEMPLATE
from agents.ResearchAgent.paper_memory import PaperMemory, PaperMemoryConfig
from agent_engine.memory.ultra_memory import Filter, Record


logger = AgentLogger("BenchInteractions")


def _pick_segment_with_data(pm: PaperMemory, segments: List[str]) -> Optional[str]:
    for seg in segments:
        try:
            um = pm._get_segment_um(seg)
            rows = um.query("papers", Filter(limit=1))
            if rows:
                return seg
        except Exception:
            continue
    return None


def _get_sample_row(pm: PaperMemory, seg: str) -> Optional[Dict[str, Any]]:
    um = pm._get_segment_um(seg)
    rows = um.query("papers", Filter(limit=1))
    return rows[0] if rows else None


def _sql_count(pm: PaperMemory, seg: str) -> int:
    um = pm._get_segment_um(seg)
    adapter = getattr(um, "adapter", None)
    conn = getattr(adapter, "_conn", None)
    if conn is None:
        return 0
    tbl = adapter._tbl(pm.cfg.collection_name)  # type: ignore[attr-defined]
    cur = conn.cursor()
    try:
        cur.execute(f'SELECT COUNT(*) FROM "{tbl}"')
        row = cur.fetchone()
        return int(row[0]) if row else 0
    finally:
        cur.close()


@pyinstrument.profile()
def bench_segment(seg: str = "2023H1", limit_scan: int = 2000, month: str = "202301") -> None:
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

    # Select a segment with data if the provided default doesn't have it
    segs = [seg] + ["2022H1", "2023H1", "2024H1", "2025H1"]
    use_seg = _pick_segment_with_data(pm, segs) or seg
    logger.info(f"Using segment: {use_seg}")

    um = pm._get_segment_um(use_seg)

    # Warm-up: fetch one row
    sample = _get_sample_row(pm, use_seg)
    if not sample:
        logger.warning("No sample row available; skipping benchmarks")
        return
    sample_id = sample.get("id")
    logger.info(f"Sample id: {sample_id}")

    # 1) Count via SQL COUNT(*)
    t0 = time.time()
    n = _sql_count(pm, use_seg)
    t1 = time.time()
    logger.info(f"COUNT(*) rows={n} elapsed={t1 - t0:.3f}s")

    # 2) Current-style scan with limit (approx throughput)
    t0 = time.time()
    rows = um.query("papers", Filter(limit=limit_scan))
    t1 = time.time()
    logger.info(f"Scan(limit={limit_scan}) rows={len(rows)} elapsed={t1 - t0:.3f}s")

    # 3) Fetch by id
    t0 = time.time()
    by_id = um.query("papers", Filter(expr={"eq": ["id", sample_id]}, limit=1))
    t1 = time.time()
    logger.info(f"Fetch by id elapsed={t1 - t0:.3f}s found={len(by_id)}")

    # 4) Date range: get_by_month (no vectors)
    t0 = time.time()
    month_rows = pm.get_by_month(month, include_vector=False, limit=2000)
    t1 = time.time()
    logger.info(f"get_by_month {month} include_vector=False returned={len(month_rows)} elapsed={t1 - t0:.3f}s")

    # 5) Date range: get_by_month (with vectors, small limit)
    t0 = time.time()
    month_rows_vec = pm.get_by_month(month, include_vector=True, limit=200)
    t1 = time.time()
    logger.info(f"get_by_month {month} include_vector=True returned={len(month_rows_vec)} elapsed={t1 - t0:.3f}s")

    # 6) Existence across segments
    t0 = time.time()
    exists_any = pm._exists_with_any_segment(sample_id)
    t1 = time.time()
    logger.info(f"_exists_with_any_segment elapsed={t1 - t0:.3f}s exists={exists_any}")

    # 7) Safe overwrite upsert (no dirty data): re-upsert same content/vector/timestamp
    #    Fetch vector and attrs via private helper
    from agents.ResearchAgent.paper_memory import PaperMemory as _PM  # for type access
    content, vector, attrs = pm._fetch_vector_row(use_seg, sample_id) or (sample.get("content"), [], sample.get("attributes", {}))
    ts = sample.get("timestamp")
    rec = Record(id=sample_id, attributes=attrs or {}, content=content or "", vector=vector or None, timestamp=ts)
    t0 = time.time()
    pm.upsert_records(use_seg, [rec])
    t1 = time.time()
    logger.info(f"Safe overwrite upsert elapsed={t1 - t0:.3f}s (same values)")


if __name__ == "__main__":
    bench_segment()


