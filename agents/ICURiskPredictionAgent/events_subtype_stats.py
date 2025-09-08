import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_engine.agent_logger.agent_logger import AgentLogger
from agent_engine.utils import find_project_root, get_relative_path_from_current_file


logger = AgentLogger(__name__)

# Simple token estimator: approximate tokens as words count (~ OpenAI tiktoken ~4 chars/token)
# We will use a mixed heuristic combining whitespace split and length/4 upper bound for robustness.

def estimate_tokens(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    # whitespace tokens
    ws_tokens = len(text.split())
    # char-based estimate
    char_tokens = (len(text) + 3) // 4
    # take max to avoid undercount for long words / CJK
    return max(ws_tokens, char_tokens)


def load_confirm_subtypes() -> List[str]:
    path = get_relative_path_from_current_file("risk_confirm_subtypes.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subtypes = [x.get("sub_type") for x in data if isinstance(x, dict) and x.get("sub_type")]
    return subtypes


def list_patient_files(root: Path) -> List[Path]:
    pdir = root / "database" / "icu_patients"
    if not pdir.exists():
        return []
    return sorted(pdir.glob("*.json"))


def process_file(path: Path, confirm_set: set) -> Tuple[int, int]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        seq = data.get("sequence", [])
        if not isinstance(seq, list):
            return 0, 0
        count = 0
        total_tokens = 0
        for ev in seq:
            st = str(ev.get("sub_type", ""))
            if st in confirm_set:
                count += 1
                content = ev.get("event_content", "")
                total_tokens += estimate_tokens(str(content))
        return count, total_tokens
    except Exception as e:
        logger.error(f"Failed processing {path}: {e}")
        return 0, 0


def main() -> None:
    project_root = find_project_root()
    patients = list_patient_files(project_root)
    if not patients:
        logger.info("No patient files found under database/icu_patients")
        print(json.dumps({"event_count": 0, "total_tokens": 0}, ensure_ascii=False))
        return

    confirm_subtypes = load_confirm_subtypes()
    confirm_set = set(confirm_subtypes)

    logger.info(f"Scanning {len(patients)} patient files, confirm_subtypes={len(confirm_set)}")

    total_events = 0
    total_tokens = 0

    max_workers = min(16, os.cpu_count() or 8)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_file, p, confirm_set): p for p in patients}
        for fut in as_completed(futures):
            c, t = fut.result()
            total_events += c
            total_tokens += t

    result: Dict[str, int] = {"event_count": total_events, "total_tokens": total_tokens}
    logger.info(f"Stats: events={total_events}, tokens={total_tokens}")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
