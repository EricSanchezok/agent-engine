from __future__ import annotations

import os
import re
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ResearchAgent.paper_memory import PaperMemory, PaperMemoryConfig
from agent_engine.memory.ultra_memory import Record
try:
    from agents.ResearchAgent.config import PAPER_DSN_TEMPLATE as CFG_PAPER_DSN_TEMPLATE  # type: ignore
except Exception:
    CFG_PAPER_DSN_TEMPLATE = None


# ----------------------------- User configuration -----------------------------
# Set your remote Postgres DSN template here. {db} will be replaced by remote DB name
# derived from segment:
#   - 2024H1 -> h1_2024
#   - 2024H2 -> h2_2024
DSN_TEMPLATE: str = os.getenv("PAPER_DSN_TEMPLATE") or (CFG_PAPER_DSN_TEMPLATE or "postgresql://USER:PASS@HOST:PORT/{db}")

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

# Batch sizes
BATCH_FLUSH = 1000


logger = AgentLogger("MigrateArxivToUltra")


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


def _flush(pm: PaperMemory, bucket: Dict[str, List[Record]]) -> int:
    total = 0
    for seg, recs in list(bucket.items()):
        if not recs:
            continue
        if ALLOWED_SEGMENTS is not None and seg not in ALLOWED_SEGMENTS:
            logger.warning(f"Skip segment {seg}: not in allowed list")
            bucket[seg] = []
            continue
        try:
            written = pm.upsert_records(seg, recs)
            total += len(written)
            logger.info(f"Flushed {len(written)} records to segment {seg}")
        except Exception as e:
            logger.error(f"Failed to flush {len(recs)} records to {seg}: {e}")
        finally:
            bucket[seg] = []
    return total


def migrate() -> None:
    load_dotenv()

    if ("USER:PASS@HOST:PORT" in DSN_TEMPLATE) or ("USER" in DSN_TEMPLATE and "PASS" in DSN_TEMPLATE):
        logger.error("Please set PAPER_DSN_TEMPLATE in env or agents/ResearchAgent/config.py before running migration.")
        return

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

    bucket: Dict[str, List[Record]] = {}
    scanned = 0
    written_total = 0

    for seg_dir in _iter_source_segments(base_dir):
        sqlite_path = seg_dir / "arxiv_metadata.sqlite3"
        duckdb_path = seg_dir / "arxiv_metadata.duckdb"
        it: Iterable[Tuple[str, str, Dict[str, Any], Optional[List[float]]]]
        if sqlite_path.exists():
            logger.info(f"Scanning SQLite segment: {seg_dir.name}")
            it = _iter_items_from_sqlite(sqlite_path)
        elif duckdb_path.exists():
            logger.info(f"Scanning DuckDB segment: {seg_dir.name}")
            it = _iter_items_from_duckdb(duckdb_path)
        else:
            logger.warning(f"No DB file found in {seg_dir}")
            continue

        for rid, content, md, vec in it:
            scanned += 1
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
                written_total += _flush(pm, {seg: arr})
                arr.clear()
            if scanned % 5000 == 0:
                logger.info(f"Scanned {scanned} records, written {written_total}")

        # Flush remainder for this segment directory
        written_total += _flush(pm, bucket)

    logger.info(f"Migration finished. Scanned={scanned}, written={written_total}")


if __name__ == "__main__":
    migrate()


