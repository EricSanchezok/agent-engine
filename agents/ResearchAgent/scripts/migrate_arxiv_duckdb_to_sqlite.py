from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agent_engine.agent_logger.agent_logger import AgentLogger
from agent_engine.memory.scalable_memory import ScalableMemory


logger = AgentLogger("ArxivMemoryMigrator")


def _decode_vector(blob: Optional[bytes]) -> Optional[List[float]]:
    if blob is None:
        return None
    try:
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        return np.frombuffer(blob, dtype=np.float32).tolist()  # type: ignore[arg-type]
    except Exception:
        return None


async def migrate_segment(seg_dir: Path, *, batch_size: int = 10000) -> None:
    """Migrate one segment directory from DuckDB to SQLite in-place.

    Layout:
    - DuckDB source: seg_dir / 'arxiv_metadata.duckdb'
    - SQLite dest:   seg_dir / 'arxiv_metadata.sqlite3'
    - Index file(s): seg_dir / 'index_*_arxiv_metadata.*' (will be regenerated)
    """
    duck_path = seg_dir / "arxiv_metadata.duckdb"
    sqlite_path = seg_dir / "arxiv_metadata.sqlite3"

    if not duck_path.exists():
        logger.info(f"Skip segment {seg_dir.name}: no DuckDB file found")
        return

    try:
        import duckdb  # type: ignore
    except Exception as e:
        logger.error(
            f"DuckDB python module is required to read source DBs. Install duckdb and retry. Error: {e}"
        )
        raise

    logger.info(f"Starting migration for segment {seg_dir.name}")

    # Connect source (DuckDB) strictly in read-only mode if supported
    try:
        con = duckdb.connect(str(duck_path))
    except Exception as e:
        logger.error(f"Failed to open DuckDB at {duck_path}: {e}")
        raise

    try:
        src_count = int(con.execute("SELECT COUNT(*) FROM items").fetchone()[0])
    except Exception as e:
        logger.error(f"Failed to count rows in source DB: {e}")
        con.close()
        raise

    # Prepare destination (SQLite)
    mem_dst = ScalableMemory(
        name="arxiv_metadata",
        persist_dir=str(seg_dir),
        db_backend="sqlite",
        enable_vectors=True,
    )

    try:
        dst_count = int(mem_dst.count())
    except Exception:
        dst_count = 0

    if dst_count >= src_count and sqlite_path.exists():
        logger.info(
            f"Segment {seg_dir.name} already migrated: dst_count={dst_count} >= src_count={src_count}. Skipping."
        )
        con.close()
        return

    # Move aside legacy index files to avoid label-ID mismatch during rebuild
    # They will be regenerated for SQLite as we insert.
    timestamp = int(time.time())
    for idx_name in [
        "index_hnsw_arxiv_metadata.bin",
        "index_annoy_arxiv_metadata.ann",
        "index_bruteforce_arxiv_metadata.json",
    ]:
        idx_path = seg_dir / idx_name
        if idx_path.exists():
            try:
                backup = seg_dir / f"{idx_path.stem}.duckdb.{timestamp}{idx_path.suffix}"
                os.replace(idx_path, backup)
                logger.info(f"Backed up legacy index file to {backup.name}")
            except Exception as e:
                logger.warning(f"Failed to back up index {idx_path.name}: {e}")

    logger.info(
        f"Migrating {src_count - dst_count} rows (src={src_count}, dst={dst_count}) in batches of {batch_size}"
    )

    # If destination already has some items (resumable), build a set of existing IDs for quick skip.
    existing_ids: Optional[set[str]] = None
    if dst_count > 0:
        try:
            cur = mem_dst.db.execute("SELECT id FROM items")
            existing_ids = {r[0] for r in mem_dst.db.fetchall(cur)}
        except Exception:
            existing_ids = None

    inserted = 0
    offset = 0
    while offset < src_count:
        try:
            rows = con.execute(
                "SELECT id, content, vector, metadata FROM items ORDER BY id LIMIT ? OFFSET ?",
                [batch_size, offset],
            ).fetchall()
        except Exception as e:
            logger.error(f"Failed to read batch at offset {offset}: {e}")
            raise

        if not rows:
            break

        items: List[Dict[str, Any]] = []
        for rid, content, vblob, md in rows:
            if existing_ids is not None and rid in existing_ids:
                continue
            vec = _decode_vector(vblob)
            if vec is None or len(vec) == 0:
                continue
            try:
                if isinstance(md, str) and md:
                    md_dict: Dict[str, Any] = json.loads(md)
                elif isinstance(md, dict):
                    md_dict = md
                else:
                    md_dict = {}
            except Exception:
                md_dict = {}
            md_dict["id"] = rid
            items.append({
                "id": rid,
                "content": content or "",
                "vector": vec,
                "metadata": md_dict,
            })

        if items:
            await mem_dst.add_many(items)
            inserted += len(items)
            logger.info(
                f"Segment {seg_dir.name}: migrated batch size={len(items)} (total inserted={inserted})"
            )

        offset += len(rows)

    # Persist index and verify counts
    try:
        mem_dst._maybe_persist_index(force=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        final_dst = mem_dst.count()
    except Exception:
        final_dst = -1

    con.close()

    if final_dst == src_count:
        logger.info(f"Segment {seg_dir.name}: migration completed. count={final_dst}")
    else:
        logger.warning(
            f"Segment {seg_dir.name}: migration finished with count mismatch src={src_count}, dst={final_dst}"
        )


async def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent / "database"
    if not base_dir.exists():
        logger.error(f"Database directory not found: {base_dir}")
        return

    segments = [p for p in base_dir.iterdir() if p.is_dir() and (p / "arxiv_metadata.duckdb").exists()]
    if not segments:
        logger.info("No DuckDB segments found. Nothing to migrate.")
        return

    logger.info(f"Found {len(segments)} segments with DuckDB databases to migrate")
    for seg in sorted(segments, key=lambda x: x.name):
        await migrate_segment(seg)


if __name__ == "__main__":
    asyncio.run(main())


