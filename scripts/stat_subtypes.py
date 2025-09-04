"""
Stat sub_type categories across ICU raw datasets.

Usage:
  run.bat scripts\stat_subtypes.py

This script scans JSON files under database/icu_raw/, aggregates the counts of
`sub_type` from each event in `sequence`, and writes a JSON report to
database/icu_knowledge/subtypes_stats.json.

Notes:
  - Missing or empty `sub_type` are counted under key "_missing"
  - Large files are loaded via json.load (acceptable for current sizes)
  - All logs/prints are in English
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "database" / "icu_raw"
OUT_DIR = PROJECT_ROOT / "database" / "icu_knowledge"
OUT_PATH = OUT_DIR / "subtypes_stats.json"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_subtypes() -> Tuple[Counter, int, List[str]]:
    sub_counter: Counter = Counter()
    files_processed: List[str] = []
    total_events: int = 0

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory not found: {RAW_DIR}")

    for path in sorted(RAW_DIR.glob("*.json")):
        try:
            data = load_json(path)
        except Exception as e:
            print(f"WARN: Failed to load {path.name}: {e}")
            continue

        seq = data.get("sequence")
        if not isinstance(seq, list):
            print(f"WARN: File {path.name} has no valid 'sequence' list; skipping")
            continue

        for ev in seq:
            sub_type = ev.get("sub_type") if isinstance(ev, dict) else None
            key = (sub_type or "").strip() or "_missing"
            sub_counter[key] += 1
        total_events += len(seq)
        files_processed.append(path.name)

    return sub_counter, total_events, files_processed


def write_report(sub_counter: Counter, total_events: int, files_processed: List[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_files": len(files_processed),
        "total_events": total_events,
        "unique_sub_types": len(sub_counter),
        "sub_type_counts": dict(sorted(sub_counter.items(), key=lambda x: (-x[1], x[0]))),
        "files_processed": files_processed,
    }
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved subtypes stats to: {OUT_PATH}")


def main() -> int:
    sub_counter, total_events, files_processed = aggregate_subtypes()
    write_report(sub_counter, total_events, files_processed)
    print(
        f"Done. files={len(files_processed)} events={total_events} unique_sub_types={len(sub_counter)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


