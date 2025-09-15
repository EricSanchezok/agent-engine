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
Migrate legacy single arXiv database into half-year segmented databases.

Usage:
  run.bat scripts\migrate_arxiv_legacy_to_segments.py

Notes:
  - This script DOES NOT delete the legacy DB or its index.
  - Streaming migration: reads rows in batches from legacy DB and writes per-segment.
  - Uses upsert semantics in target segments; safe to re-run.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ResearchAgent.arxiv_memory import ArxivMemory
from agent_engine.memory.scalable_memory import _from_blob


logger = AgentLogger("ArxivMigrateLegacy")


def extract_date_str(md: Dict[str, Any]) -> Optional[str]:
    ts = md.get("timestamp") or md.get("submittedDate") or md.get("published") or ""
    if not isinstance(ts, str):
        ts = str(ts)
    m = re.match(r"(\d{8})", ts)
    if m:
        return m.group(1)
    m2 = re.match(r"(\d{4})[-/]?(\d{2})[-/]?(\d{2})", ts)
    if m2:
        return f"{m2.group(1)}{m2.group(2)}{m2.group(3)}"
    return None


def segment_key_from_date(date_str: str) -> Optional[str]:
    if not isinstance(date_str, str) or len(date_str) < 6:
        return None
    try:
        year = int(date_str[:4])
        month = int(date_str[4:6])
        half = "H1" if 1 <= month <= 6 else "H2"
        return f"{year}{half}"
    except Exception:
        return None


async def main() -> int:
    mem = ArxivMemory()
    legacy = mem._get_legacy_memory()
    if legacy is None:
        logger.error("No legacy arXiv database found (neither arxiv_metada.duckdb nor arxiv_metadata.duckdb)")
        return 1

    logger.info("Starting streaming migration from legacy database...")

    # Stream rows from legacy DB in sorted id order using an increasing cursor (no OFFSET)
    db = legacy.db
    batch_read_size = 5000
    batch_write_size = 2000
    last_id: Optional[str] = None
    total_seen = 0
    total_written = 0
    skipped_no_vector = 0

    while True:
        try:
            if last_id is None:
                cur = db.execute(
                    "SELECT id, content, metadata, vector FROM items WHERE vector IS NOT NULL ORDER BY id LIMIT ?",
                    (batch_read_size,),
                )
            else:
                cur = db.execute(
                    "SELECT id, content, metadata, vector FROM items WHERE vector IS NOT NULL AND id > ? ORDER BY id LIMIT ?",
                    (last_id, batch_read_size),
                )
            rows = db.fetchall(cur)
        except Exception as e:
            logger.error(f"Failed to read from legacy DB: {e}")
            return 1

        if not rows:
            break

        # Build per-segment payloads for this read batch
        payload_by_segment: Dict[str, List[Dict[str, Any]]] = {}
        for rid, content, md_json, vec_blob in rows:
            total_seen += 1
            try:
                if vec_blob is None:
                    skipped_no_vector += 1
                    continue
                try:
                    md: Dict[str, Any] = json.loads(md_json) if md_json else {}
                except Exception:
                    md = {}
                item_id = md.get("id") or rid
                if not isinstance(item_id, str) or not item_id:
                    continue
                vector = _from_blob(vec_blob)
                if not isinstance(vector, list) or not vector:
                    skipped_no_vector += 1
                    continue
                d = extract_date_str(md)
                seg = segment_key_from_date(d or "") or "undated"
                payload_by_segment.setdefault(seg, []).append({
                    "id": item_id,
                    "content": content,
                    "vector": vector,
                    "metadata": md,
                })
            except Exception as e:
                logger.warning(f"Row {rid}: skipped due to error: {e}")
                continue

        # Flush writes per segment in write-sized batches
        for seg_key, payload in sorted(payload_by_segment.items()):
            seg_mem = mem._get_segment_memory(seg_key)
            if not payload:
                continue
            try:
                for i in range(0, len(payload), batch_write_size):
                    batch = payload[i : i + batch_write_size]
                    _ids = await seg_mem.add_many(batch)
                    total_written += len(_ids)
                logger.info(f"Segment {seg_key}: wrote {len(payload)} items (cumulative={total_written})")
            except Exception as e:
                logger.error(f"Segment {seg_key}: write failed: {e}")

        # Advance cursor
        last_id = rows[-1][0]

    if skipped_no_vector:
        logger.warning(f"Skipped {skipped_no_vector} items without vectors")

    logger.info(f"Migration completed. Total seen: {total_seen}, total newly written items: {total_written}")
    logger.info("Legacy DB and index are preserved as-is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


