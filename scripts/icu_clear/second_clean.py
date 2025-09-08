import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from agent_engine.agent_logger import AgentLogger


LOGGER = AgentLogger("ICUSecondClean")


# Sub-types considered to contain scoring items and should have unified "分"
SCORING_SUB_TYPES = {
    "医院获得性肺炎风险因素评估表",
    "Caprini血栓风险评估单",
    "导尿管伴随性尿路感染风险监控",
    "Braden",
    "跌倒/坠床风险评估表",
    "Barthel指数量表",
    "内科血栓风险评估表",
    "重症疼痛观察量表CPOT",
    "改良早期预警评分表MEWS",
    "住院病人昏迷评定记录单",
    "谵妄评估单",
}


def normalize_commas(text: str) -> str:
    if not text:
        return text
    s = text.replace("，", ", ")
    s = re.sub(r"([,，])\s*\1+", r"\1 ", s)
    # normalize multiple spaces
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def tokenize_kv(text: str) -> List[Tuple[Optional[str], str]]:
    if not text:
        return []
    s = normalize_commas(text)
    raw_tokens = [t.strip() for t in s.split(",") if t.strip()]
    tokens: List[Tuple[Optional[str], str]] = []
    for tok in raw_tokens:
        if ":" in tok:
            k, v = tok.split(":", 1)
            tokens.append((k.strip(), v.strip()))
        else:
            tokens.append((None, tok))
    return tokens


def join_tokens(tokens: List[Tuple[Optional[str], str]]) -> str:
    parts: List[str] = []
    for k, v in tokens:
        if k is None:
            parts.append(v)
        else:
            parts.append(f"{k}: {v}")
    return ", ".join(parts)


def remove_prefix_in_keys(text: str, prefixes: List[str]) -> str:
    tokens = tokenize_kv(text)
    new_tokens: List[Tuple[Optional[str], str]] = []
    for k, v in tokens:
        if k is None:
            new_tokens.append((k, v))
            continue
        key = k
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
        new_tokens.append((key, v))
    return join_tokens(new_tokens)


def dedupe_repeated_block_strong(text: str) -> str:
    """
    Stronger duplicate remover that can catch patterns like A + A + tail.
    It tries to find two consecutive identical token blocks from the beginning,
    even if there are extra tokens appended after the second block.
    """
    if not text:
        return text

    s = text.strip()

    # Quick whole-string half duplication (exact)
    ss = s.strip(" ,\n")
    if len(ss) % 2 == 0:
        half = len(ss) // 2
        if ss[:half] == ss[half:]:
            return ss[:half]

    tokens = [t.strip() for t in normalize_commas(s).split(",") if t.strip()]
    n = len(tokens)
    if n < 6:
        return s

    # Exact contiguous duplication starting at 0: tokens[0:j] == tokens[j:2j]
    max_j = n // 2
    for j in range(max_j, 4, -1):  # prefer larger repeats
        left = tokens[0:j]
        mid = tokens[j:2 * j]
        if left == mid:
            deduped = left + tokens[2 * j :]
            return ", ".join(deduped)

    # General case: find any i where tokens[i:j] == tokens[j: j+(j-i)]
    # We require block length >= 4 to avoid false positives
    for i in range(0, n - 7):
        for j in range(n - 3, i + 3, -1):
            block_len = j - i
            if block_len < 4:
                break
            k = j + block_len
            if k > n:
                continue
            if tokens[i:j] == tokens[j:k]:
                dedup = tokens[:i] + tokens[i:j] + tokens[k:]
                return ", ".join(dedup)

    # Fallback: attempt fuzzy match by collapsing spaces within tokens
    norm = [re.sub(r"\s+", "", t) for t in tokens]
    for j in range(max_j, 4, -1):
        if norm[0:j] == norm[j:2 * j]:
            deduped = tokens[0:j] + tokens[2 * j :]
            return ", ".join(deduped)

    # Final cleanup of doubled commas
    s = re.sub(r"([,，])\s*\1+", r"\1 ", s)
    return s


def should_exclude_key_from_score(sub_type: str, key: Optional[str]) -> bool:
    if key is None:
        return True
    k = key.strip()
    common_exclude = {"性别", "年龄", "入院时间", "护理措施", "措施", "审阅", "分型结果"}
    if k in common_exclude:
        return True
    # Special cases by sub_type
    if sub_type == "谵妄评估单" and ("RASS评分" in k):
        return True
    if sub_type == "改良早期预警评分表MEWS":
        mews_exclude = {"心率", "收缩压", "呼吸频率", "体温", "意识", "风险分级"}
        if k in mews_exclude:
            return True
    return False


def add_fen_after_number(value: str) -> str:
    """
    Convert:
    - "60（高风险）" -> "60分（高风险）"
    - "50 重点防护" -> "50分 重点防护"
    - "11" -> "11分"
    Only applies to non-negative integers; decimals/negatives are skipped.
    """
    if not value:
        return value
    v = value.strip()

    # Skip negatives and decimals
    if re.match(r"^-", v) or re.search(r"\d+\.\d+", v):
        return value

    # If already contains "分" right after number-in-parentheses form, leave
    if re.search(r"\(\s*\d+\s*分\s*\)|（\s*\d+\s*分\s*）", v):
        return value

    # Parentheses case: number followed by '(' or '（'
    m = re.match(r"^(\d+)\s*(（|\().*$", v)
    if m:
        num = m.group(1)
        return v.replace(num, f"{num}分", 1)

    # Plain number + suffix text (e.g., "50 重点防护")
    m = re.match(r"^(\d+)\s*(\D.*)$", v)
    if m:
        num, tail = m.group(1), m.group(2)
        return f"{num}分 {tail.strip()}".strip()

    # Pure integer
    if re.fullmatch(r"\d+", v):
        return f"{v}分"

    return value


def unify_score_units(sub_type: str, text: str) -> str:
    if not text or sub_type not in SCORING_SUB_TYPES:
        return text

    tokens = tokenize_kv(text)
    new_tokens: List[Tuple[Optional[str], str]] = []
    for k, v in tokens:
        if should_exclude_key_from_score(sub_type, k):
            new_tokens.append((k, v))
            continue
        # Only consider adding "分" if value contains at least one digit
        if v and re.search(r"\d", v):
            v2 = add_fen_after_number(v)
            new_tokens.append((k, v2))
        else:
            new_tokens.append((k, v))
    return join_tokens(new_tokens)


def clean_event_content(sub_type: str, content: str) -> str:
    try:
        if content is None:
            return content
        if not isinstance(content, str):
            return content
        # Step 1: stronger de-duplication
        s1 = dedupe_repeated_block_strong(content)
        # Step 1.1: remove special prefixes by type
        if sub_type == "跌倒/坠床风险评估表":
            s1 = remove_prefix_in_keys(s1, ["morse_"])
        # Step 2: unify score units for scoring sub-types
        s2 = unify_score_units(sub_type, s1)
        return s2
    except Exception as e:
        LOGGER.exception(f"Second clean failed for sub_type={sub_type}: {e}")
        return content


def process_file(in_path: Path, out_path: Path) -> Tuple[int, int]:
    try:
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        LOGGER.exception(f"Failed to read {in_path}: {e}")
        return (0, 0)

    sequence = data.get("sequence", [])
    changed = 0
    for ev in sequence:
        sub_type = ev.get("sub_type")
        original = ev.get("event_content")
        new_content = clean_event_content(sub_type, original)
        if isinstance(new_content, str) and new_content != original:
            ev["event_content"] = new_content
            changed += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        LOGGER.exception(f"Failed to write {out_path}: {e}")

    return (len(sequence), changed)


def main():
    project_root = Path(__file__).resolve().parents[2]
    in_dir = project_root / "database" / "icu_first_clean"
    out_dir = project_root / "database" / "icu_second_clean"

    LOGGER.info(f"Starting ICU second clean. Input: {in_dir}, Output: {out_dir}")

    total_files = 0
    total_events = 0
    total_changed = 0

    for path in in_dir.glob("*.json"):
        total_files += 1
        out_path = out_dir / path.name
        num_events, num_changed = process_file(path, out_path)
        total_events += num_events
        total_changed += num_changed
        if total_files % 20 == 0:
            LOGGER.info(f"Processed {total_files} files... updated {total_changed} events so far")

    LOGGER.info(
        f"Done. Files: {total_files}, Events: {total_events}, Updated events: {total_changed}. Output at {out_dir}"
    )


if __name__ == "__main__":
    main()


