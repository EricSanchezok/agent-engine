from __future__ import annotations

import os
import re
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import csv
import io
from dotenv import load_dotenv

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ResearchAgent.paper_memory import PaperMemory, PaperMemoryConfig
from agent_engine.memory.ultra_memory import Record

from agents.ResearchAgent.config import DSN_TEMPLATE_LOCAL


# ----------------------------- Local server configuration -----------------------------
# Local database connection since we're running on the same server as the database
DSN_TEMPLATE: str = DSN_TEMPLATE_LOCAL

# Optional allow-list of target databases present on the remote server.
ALLOWED_SEGMENTS: Optional[List[str]] = [
    "2022H1", "2022H2",
    "2023H1", "2023H2",
    "2024H1", "2024H2",
    "2025H1", "2025H2",
]

# Collection & vector schema in UltraMemory
COLLECTION_NAME = "papers"
VECTOR_FIELD = "text_vec"
VECTOR_DIM = 3072
VECTOR_METRIC = "cosine"

# Batch & behavior controls - optimized for local database access
BATCH_FLUSH = 10000                # Larger batch size for local database
BULK_UPSERT_CHUNK = 5000           # Larger chunk size for better performance
OVERWRITE_EXISTING = False         # False: INSERT ... ON CONFLICT DO NOTHING; True: DO UPDATE
MIGRATION_MAX_RECORDS: Optional[int] = None   # No limit for full migration
USE_COPY_LOAD = True               # Use COPY into temp table + merge for faster loads
PREFETCH_EXISTING_IDS = True       # Query database for existing ids to skip before upsert
PREFETCH_BATCH = 10000             # Larger prefetch batch for local database
DISABLE_INDEXES_DURING_LOAD = True # Drop heavy indexes before load and recreate after


logger = AgentLogger("MigrateArxivToUltraLocal")


def _extract_date_str(md: Dict[str, Any]) -> Optional[str]:
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


def _segment_key_from_date(date_str: str) -> Optional[str]:
    if not isinstance(date_str, str) or len(date_str) < 6:
        return None
    try:
        year = int(date_str[:4])
        month = int(date_str[4:6])
        half = "H1" if 1 <= month <= 6 else "H2"
        return f"{year}{half}"
    except Exception:
        return None


def _to_iso_ts(md: Dict[str, Any]) -> Optional[str]:
    s = md.get("timestamp") or md.get("submittedDate") or md.get("published")
    if s is None:
        d = _extract_date_str(md)
        if not d:
            return None
        s = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    if isinstance(s, (int, float)):
        try:
            from datetime import datetime
            return datetime.utcfromtimestamp(float(s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    st = str(s)
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", st)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T00:00:00Z"
    m2 = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T].*)?$", st)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}T00:00:00Z"
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(st.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass
    return None


def _decode_vector(v: Any) -> Optional[List[float]]:
    if v is None:
        return None
    try:
        if isinstance(v, (bytes, bytearray, memoryview)):
            arr = np.frombuffer(v, dtype=np.float32)
            return arr.tolist()
        if isinstance(v, str):
            s = v.strip().lstrip('[').rstrip(']')
            if not s:
                return None
            return [float(x) for x in s.split(',')]
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
        # Fallback: try string conversion
        s = str(v).strip().lstrip('[').rstrip(']')
        if not s:
            return None
        return [float(x) for x in s.split(',')]
    except Exception:
        return None


def _iter_source_segments(base_dir: Path) -> Iterable[Path]:
    if not base_dir.exists():
        return []
    for p in sorted(base_dir.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        if re.match(r"^\d{4}H[12]$", p.name) or p.name == "undated":
            yield p


def _iter_items_from_sqlite(db_path: Path) -> Iterable[Tuple[str, str, Dict[str, Any], Optional[List[float]]]]:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        try:
            for row in cur.execute("SELECT id, content, metadata, vector FROM items"):
                rid = row[0]
                content = row[1]
                md_raw = row[2]
                vec_raw = row[3]
                try:
                    md = json.loads(md_raw) if isinstance(md_raw, str) else (md_raw or {})
                except Exception:
                    md = {}
                vec = _decode_vector(vec_raw)
                yield rid, content, md, vec
        finally:
            cur.close()
    finally:
        conn.close()


def _iter_items_from_duckdb(db_path: Path) -> Iterable[Tuple[str, str, Dict[str, Any], Optional[List[float]]]]:
    try:
        import duckdb  # type: ignore
    except Exception as e:
        logger.error(f"duckdb driver not available: {e}")
        return []
    conn = duckdb.connect(str(db_path))
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, content, metadata, vector FROM items")
            while True:
                rows = cur.fetchmany(4096)
                if not rows:
                    break
                for row in rows:
                    rid = row[0]
                    content = row[1]
                    md_raw = row[2]
                    vec_raw = row[3]
                    try:
                        if isinstance(md_raw, dict):
                            md = md_raw
                        elif isinstance(md_raw, str):
                            md = json.loads(md_raw)
                        else:
                            md = {}
                    except Exception:
                        md = {}
                    vec = _decode_vector(vec_raw)
                    yield rid, content, md, vec
        finally:
            cur.close()
    finally:
        conn.close()


def _vec_literal(vec: Optional[List[float]]) -> Optional[str]:
    if vec is None:
        return None
    try:
        return '[' + ','.join(str(float(x)) for x in vec) + ']'
    except Exception:
        return None


def _get_existing_record_count(pm: PaperMemory, seg: str) -> int:
    """Get the count of existing records in the target segment"""
    try:
        um = pm._get_segment_um(seg)
        adapter = getattr(um, "adapter", None)
        conn = getattr(adapter, "_conn", None)
        if conn is None:
            return 0
        
        tbl = adapter._tbl(pm.cfg.collection_name)
        cur = conn.cursor()
        try:
            cur.execute(f'SELECT COUNT(*) FROM "{tbl}"')
            result = cur.fetchone()
            return result[0] if result else 0
        finally:
            cur.close()
    except Exception as e:
        logger.warning(f"Failed to get existing record count for segment {seg}: {e}")
        return 0


def _bulk_upsert_segment(pm: PaperMemory, seg: str, recs: List[Record]) -> Tuple[int, List[str], int]:
    """
    Returns: (inserted_count, inserted_ids, skipped_count)
    """
    if not recs:
        return 0, [], 0
    
    # Ensure UM and collection exist; grab adapter & table/column names
    um = pm._get_segment_um(seg)
    adapter = getattr(um, "adapter", None)
    conn = getattr(adapter, "_conn", None)
    if conn is None:
        # Fallback to per-record upsert through adapter API (slower)
        inserted = um.upsert(pm.cfg.collection_name, recs)
        return (len(inserted), [str(x) for x in inserted], 0)
    
    tbl = adapter._tbl(pm.cfg.collection_name)  # type: ignore[attr-defined]
    vec_col = adapter._col(pm.cfg.vector_field)  # type: ignore[attr-defined]

    # Prefetch existing ids to skip duplicates
    skipped_count = 0
    if PREFETCH_EXISTING_IDS and recs:
        try:
            existing: set = set()
            cur = conn.cursor()
            try:
                for i in range(0, len(recs), PREFETCH_BATCH):
                    chunk = recs[i:i + PREFETCH_BATCH]
                    ids = [r.id for r in chunk if r.id]
                    if not ids:
                        continue
                    cur.execute(f'SELECT id FROM "{tbl}" WHERE id = ANY(%s)', (ids,))
                    rows = cur.fetchall()
                    for row in rows:
                        existing.add(str(row[0]))
            finally:
                cur.close()
            
            if existing:
                original_count = len(recs)
                recs = [r for r in recs if str(r.id) not in existing]
                skipped_count = original_count - len(recs)
                logger.info(f"Skipping {skipped_count} existing records in segment {seg}")
                
            if not recs:
                return (0, [], skipped_count)
        except Exception as e:
            logger.warning(f"Prefetch existing ids failed, continue without prefilter: {e}")

    # Fast path: COPY into temp table then merge
    if USE_COPY_LOAD:
        old_autocommit = getattr(conn, "autocommit", True)
        try:
            conn.autocommit = False  # wrap in a single transaction
        except Exception:
            pass
        cur = conn.cursor()
        try:
            temp_tbl = f"tmp_{tbl}_load"
            cur.execute(f'CREATE TEMP TABLE "{temp_tbl}" AS TABLE "{tbl}" WITH NO DATA')
            copy_sql = f'COPY "{temp_tbl}" (id, content, attributes, ts, "{vec_col}") FROM STDIN WITH (FORMAT csv)'
            with cur.copy(copy_sql) as cp:
                # Stream rows chunked to avoid large memory spikes
                buf = io.StringIO()
                writer = csv.writer(buf)
                for r in recs:
                    md = r.attributes or {}
                    ts = r.timestamp or ""
                    vec_lit = _vec_literal(r.vector) or ""
                    writer.writerow([r.id, r.content or "", json.dumps(md, ensure_ascii=False), ts, vec_lit])
                    cp.write(buf.getvalue())
                    buf.seek(0)
                    buf.truncate(0)
            if OVERWRITE_EXISTING:
                update_set = ", ".join([
                    'content=EXCLUDED.content',
                    'attributes=EXCLUDED.attributes',
                    'ts=EXCLUDED.ts',
                    f'"{vec_col}"=EXCLUDED."{vec_col}"',
                ])
                merge_sql = (
                    f'INSERT INTO "{tbl}" (id, content, attributes, ts, "{vec_col}") '
                    f'SELECT id, content, attributes, ts, "{vec_col}" FROM "{temp_tbl}" '
                    f'ON CONFLICT (id) DO UPDATE SET {update_set}'
                )
            else:
                merge_sql = (
                    f'INSERT INTO "{tbl}" (id, content, attributes, ts, "{vec_col}") '
                    f'SELECT id, content, attributes, ts, "{vec_col}" FROM "{temp_tbl}" '
                    f'ON CONFLICT (id) DO NOTHING'
                )
            cur.execute(merge_sql)
            try:
                conn.commit()
            except Exception:
                pass
            finally:
                try:
                    # temp table drops on commit; explicit drop as safety
                    cur.execute(f'DROP TABLE IF EXISTS "{temp_tbl}"')
                except Exception:
                    pass
            ids = [str(r.id) for r in recs if r.id]
            return (len(recs), ids, skipped_count)
        except Exception as e:
            logger.warning(f"COPY path failed, falling back to VALUES upsert: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            try:
                conn.autocommit = old_autocommit
            except Exception:
                pass

    # Prepare batched multi-row upserts (single transaction)
    total_written = 0
    inserted_ids: List[str] = []
    cols = ['id', 'content', 'attributes', 'ts', vec_col]
    row_placeholders = ["%s", "%s", "%s::jsonb", "%s::timestamptz", "%s::vector"]

    # Update clause only when overwriting existing
    if OVERWRITE_EXISTING:
        update_set = ", ".join([f'"{c}"=EXCLUDED."{c}"' for c in cols if c != 'id'])
        conflict_sql = f" ON CONFLICT (id) DO UPDATE SET {update_set}"
    else:
        conflict_sql = " ON CONFLICT (id) DO NOTHING"

    cur = conn.cursor()
    try:
        old_autocommit = getattr(conn, "autocommit", True)
        try:
            conn.autocommit = False
        except Exception:
            pass
        for i in range(0, len(recs), BULK_UPSERT_CHUNK):
            chunk = recs[i:i + BULK_UPSERT_CHUNK]
            if not chunk:
                continue
            values: List[Any] = []
            values_sql_parts: List[str] = []
            for r in chunk:
                # Attributes to JSON, timestamp as ISO string
                md = r.attributes or {}
                ts = r.timestamp
                vec_lit = _vec_literal(r.vector)
                values.extend([r.id, r.content, json.dumps(md, ensure_ascii=False), ts, vec_lit])
                values_sql_parts.append("(" + ", ".join(row_placeholders) + ")")

            sql = (
                f'INSERT INTO "{tbl}" ('
                + ", ".join(f'"{c}"' for c in cols)
                + ") VALUES "
                + ", ".join(values_sql_parts)
                + conflict_sql
            )
            cur.execute(sql, tuple(values))
            total_written += len(chunk)
            inserted_ids.extend([str(r.id) for r in chunk if r.id])
        try:
            conn.commit()
        except Exception:
            pass
        finally:
            try:
                conn.autocommit = old_autocommit
            except Exception:
                pass
        return (total_written, inserted_ids, skipped_count)
    finally:
        cur.close()


def _flush(pm: PaperMemory, bucket: Dict[str, List[Record]]) -> Tuple[int, int]:
    """
    Returns: (total_inserted, total_skipped)
    """
    total_inserted = 0
    total_skipped = 0
    
    for seg, recs in list(bucket.items()):
        if not recs:
            continue
        if ALLOWED_SEGMENTS is not None and seg not in ALLOWED_SEGMENTS:
            logger.warning(f"Skip segment {seg}: not in allowed list")
            bucket[seg] = []
            continue
        try:
            existing_count = _get_existing_record_count(pm, seg)
            logger.info(f"Segment {seg}: {existing_count} records already exist in database")
            
            written_cnt, _, skipped_cnt = _bulk_upsert_segment(pm, seg, recs)
            total_inserted += written_cnt
            total_skipped += skipped_cnt
            
            new_count = _get_existing_record_count(pm, seg)
            logger.info(f"Segment {seg}: inserted {written_cnt} new records, skipped {skipped_cnt} existing records, total now: {new_count}")
        except Exception as e:
            logger.error(f"Failed to flush {len(recs)} records to {seg}: {e}")
        finally:
            bucket[seg] = []
    
    return total_inserted, total_skipped


def migrate() -> None:
    load_dotenv()

    logger.info("Starting local migration from ArXiv memory to UltraMemory database")
    logger.info(f"Database connection: {DSN_TEMPLATE.split('@')[1] if '@' in DSN_TEMPLATE else DSN_TEMPLATE}")

    pm = PaperMemory(PaperMemoryConfig(
        dsn_template=DSN_TEMPLATE,
        collection_name=COLLECTION_NAME,
        vector_field=VECTOR_FIELD,
        vector_dim=VECTOR_DIM,
        metric=VECTOR_METRIC,
        index_params={"lists": 100},
    ))

    base_dir = Path(__file__).resolve().parents[1] / "database"
    if not base_dir.exists():
        logger.error(f"Legacy database directory not found: {str(base_dir)}")
        return

    logger.info(f"Scanning source directory: {base_dir}")
    
    # Get initial statistics
    total_segments = 0
    for seg_dir in _iter_source_segments(base_dir):
        total_segments += 1
        sqlite_path = seg_dir / "arxiv_metadata.sqlite3"
        duckdb_path = seg_dir / "arxiv_metadata.duckdb"
        
        if sqlite_path.exists():
            logger.info(f"Found SQLite database: {seg_dir.name}")
        elif duckdb_path.exists():
            logger.info(f"Found DuckDB database: {seg_dir.name}")
        else:
            logger.warning(f"No database file found in {seg_dir.name}")
    
    logger.info(f"Total segments to process: {total_segments}")

    bucket: Dict[str, List[Record]] = {}
    scanned = 0
    written_total = 0
    skipped_total = 0

    segment_count = 0
    for seg_dir in _iter_source_segments(base_dir):
        segment_count += 1
        sqlite_path = seg_dir / "arxiv_metadata.sqlite3"
        duckdb_path = seg_dir / "arxiv_metadata.duckdb"
        
        it: Iterable[Tuple[str, str, Dict[str, Any], Optional[List[float]]]]
        if sqlite_path.exists():
            logger.info(f"[{segment_count}/{total_segments}] Processing SQLite segment: {seg_dir.name}")
            it = _iter_items_from_sqlite(sqlite_path)
        elif duckdb_path.exists():
            logger.info(f"[{segment_count}/{total_segments}] Processing DuckDB segment: {seg_dir.name}")
            it = _iter_items_from_duckdb(duckdb_path)
        else:
            logger.warning(f"[{segment_count}/{total_segments}] No DB file found in {seg_dir}")
            continue

        segment_scanned = 0
        for rid, content, md, vec in it:
            scanned += 1
            segment_scanned += 1
            
            # Decide target segment by metadata date
            date_str = _extract_date_str(md) or ""
            seg = _segment_key_from_date(date_str) or "undated"
            ts = _to_iso_ts(md)
            
            # Vector dim validation
            if vec is not None and len(vec) != VECTOR_DIM:
                # Keep content/metadata but drop vector if dimension mismatch
                vec = None
            
            rec = Record(id=str(rid), attributes=md, content=str(content or ""), vector=vec, timestamp=ts)
            arr = bucket.setdefault(seg, [])
            arr.append(rec)
            
            if len(arr) >= BATCH_FLUSH:
                batch_inserted, batch_skipped = _flush(pm, {seg: arr})
                written_total += batch_inserted
                skipped_total += batch_skipped
                arr.clear()
                
            if scanned % 5000 == 0:
                logger.info(f"Progress: scanned {scanned} records (current segment: {segment_scanned}), "
                          f"inserted {written_total}, skipped {skipped_total}")

        logger.info(f"Completed segment {seg_dir.name}: scanned {segment_scanned} records")
        
        # Flush remainder for this segment directory
        segment_inserted, segment_skipped = _flush(pm, bucket)
        written_total += segment_inserted
        skipped_total += segment_skipped

    logger.info("=" * 60)
    logger.info("Migration completed successfully!")
    logger.info(f"Total records scanned: {scanned}")
    logger.info(f"Total records inserted: {written_total}")
    logger.info(f"Total records skipped (already existed): {skipped_total}")
    logger.info(f"Total records processed: {written_total + skipped_total}")
    logger.info("=" * 60)


if __name__ == "__main__":
    migrate()
