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


import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from agent_engine.agent_logger import AgentLogger


LOGGER = AgentLogger("ICUFirstClean")


def load_reasons(reason_path: Path) -> Dict[str, str]:
    try:
        with reason_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        reasons: Dict[str, str] = {}
        for k, v in data.items():
            if isinstance(v, str) and v.strip():
                reasons[k] = v.strip()
        return reasons
    except Exception as e:
        LOGGER.exception(f"Failed to load reasons from {reason_path}: {e}")
        return {}


def normalize_commas(text: str) -> str:
    if not text:
        return text
    # Normalize Chinese comma to English comma for easier tokenization
    normalized = text.replace("，", ", ")
    # Collapse multiple commas
    normalized = re.sub(r"([,，])\s*\1+", r"\1 ", normalized)
    return normalized


def tokenize_kv(text: str) -> List[Tuple[Optional[str], str]]:
    """
    Split a free-form "key: value, key: value, ..." string into tokens.
    Returns list of (key, value) where key can be None for non-kv fragments.
    """
    if not text:
        return []
    text_norm = normalize_commas(text)
    raw_tokens = [t.strip() for t in text_norm.split(",") if t.strip()]
    tokens: List[Tuple[Optional[str], str]] = []
    for tok in raw_tokens:
        if ":" in tok:
            key, value = tok.split(":", 1)
            tokens.append((key.strip(), value.strip()))
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


def dedupe_repeated_block(text: str) -> str:
    if not text:
        return text
    s = text.strip()

    # 1) Exact half duplication check
    if len(s) % 2 == 0:
        half = len(s) // 2
        left = s[:half].strip(" ,\n")
        right = s[half:].strip(" ,\n")
        if left and left == right:
            return left

    # 2) Token-based duplication check
    tokens = [t.strip() for t in normalize_commas(s).split(",") if t.strip()]
    if len(tokens) % 2 == 0:
        half = len(tokens) // 2
        if tokens[:half] == tokens[half:]:
            return ", ".join(tokens[:half])

    # 3) Anchor-based duplication check using the first token as anchor
    first_comma = s.find(",")
    if first_comma != -1:
        anchor = s[:first_comma].strip()
        second_anchor = s.find(anchor, first_comma + 1)
        if second_anchor != -1:
            part1 = s[:second_anchor].strip(" ,\n")
            part2 = s[second_anchor:].strip(" ,\n")
            if part1 and part1 == part2:
                return part1

    # 4) Cleanup doubled commas like ",,"
    s = re.sub(r"([,，])\s*\1+", r"\1 ", s)
    return s


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


def replace_measures_numbers(text: str, field_name: str = "措施") -> str:
    mapping = {
        "1": "健康教育",
        "2": "心理疏导",
        "3": "充足氧供",
        "4": "早期活动",
        "5": "疼痛控制",
        "6": "修养环境管理",
        "7": "水分营养摄入",
        "8": "预防并发症",
        "9": "增加家属探视",
        "10": "规范使用约束带",
        "11": "遵医嘱药物治疗",
    }
    tokens = tokenize_kv(text)
    changed = False
    for idx, (k, v) in enumerate(tokens):
        if k == field_name:
            nums = re.findall(r"\d+", v)
            if nums:
                items = [mapping.get(n, n) for n in nums]
                new_val = "；".join(items)
                tokens[idx] = (k, new_val)
                changed = True
    return join_tokens(tokens) if changed else text


def recalc_total_field(
    text: str,
    total_key_patterns: List[str],
    exclude_keys_prefixes: List[str],
    exclude_exact_keys: List[str],
    keep_total_suffix: bool = True,
) -> str:
    """
    Recalculate the total field by summing numeric values of other kv tokens.
    - total_key_patterns: list of keys treated as total (e.g., ["肺炎总分", "总分"])
    - exclude_keys_prefixes: keys that start with these prefixes will be excluded from sum
    - exclude_exact_keys: exact keys to exclude from sum
    - keep_total_suffix: keep the original suffix text after the number in total token
    """
    tokens = tokenize_kv(text)

    # Locate total token index and preserve suffix
    total_idx = -1
    total_suffix = ""
    for i, (k, v) in enumerate(tokens):
        if k is None:
            continue
        name = k.strip()
        if any(name == pat for pat in total_key_patterns):
            total_idx = i
            m = re.match(r"\s*(\d+)(.*)$", v)
            if m and keep_total_suffix:
                total_suffix = m.group(2)
            break

    # Sum eligible numeric values
    total_val = 0
    for k, v in tokens:
        if k is None:
            continue
        name = k.strip()
        if any(name == pat for pat in total_key_patterns):
            continue
        if any(name.startswith(pref) for pref in exclude_keys_prefixes):
            continue
        if name in exclude_exact_keys:
            continue
        m = re.search(r"-?\d+", v)
        if m:
            try:
                total_val += int(m.group(0))
            except Exception:
                pass

    # Update or append total token
    if total_idx >= 0:
        tokens[total_idx] = (tokens[total_idx][0], f"{total_val}{total_suffix}")
    else:
        tokens.append((total_key_patterns[0], str(total_val)))

    return join_tokens(tokens)


def clean_pneumonia(text: str) -> str:
    s = dedupe_repeated_block(text)
    s = remove_prefix_in_keys(s, ["肺炎_"])
    s = recalc_total_field(
        s,
        total_key_patterns=["肺炎总分", "总分"],
        exclude_keys_prefixes=["护理措施", "性别"],
        exclude_exact_keys=["护理措施", "性别"],
        keep_total_suffix=True,
    )
    return s


def clean_caprini(text: str) -> str:
    s = dedupe_repeated_block(text)
    # Remove wrong prefix "压疮_"
    s = remove_prefix_in_keys(s, ["压疮_"])
    # Fix key like "年龄3" -> "年龄"
    tokens = tokenize_kv(s)
    for i, (k, v) in enumerate(tokens):
        if k is None:
            continue
        if re.fullmatch(r"年龄\d+", k):
            tokens[i] = ("年龄", v)
    s = join_tokens(tokens)
    # Recalc total = 病史 + 实验室检查 + 手术
    tks = tokenize_kv(s)
    score = 0
    comp_keys = {"病史", "实验室检查", "手术"}
    for k, v in tks:
        if k in comp_keys:
            m = re.search(r"-?\d+", v)
            if m:
                try:
                    score += int(m.group(0))
                except Exception:
                    pass
    # Update 总分 token if exists (after removing prefix it should be "总分")
    updated = False
    for i, (k, v) in enumerate(tks):
        if k == "总分":
            # Keep any suffix description
            m = re.match(r"\s*(\d+)(.*)$", v)
            suffix = m.group(2) if m else ""
            tks[i] = (k, f"{score}{suffix}")
            updated = True
            break
    if not updated:
        tks.append(("总分", str(score)))
    return join_tokens(tks)


def clean_delirium(text: str) -> str:
    s = dedupe_repeated_block(text)
    s = replace_measures_numbers(s, field_name="措施")
    return s


def clean_urinary_cauti(text: str) -> str:
    s = dedupe_repeated_block(text)
    s = remove_prefix_in_keys(s, ["尿路_"])
    s = recalc_total_field(
        s,
        total_key_patterns=["导尿管感染总分"],
        exclude_keys_prefixes=["护理措施", "性别"],
        exclude_exact_keys=["护理措施", "性别"],
        keep_total_suffix=True,
    )
    return s


def clean_braden(text: str) -> str:
    s = dedupe_repeated_block(text)
    s = remove_prefix_in_keys(s, ["压疮_"])
    return s


def clean_simple_dedupe(text: str) -> str:
    return dedupe_repeated_block(text)


def clean_barthel(text: str) -> str:
    s = dedupe_repeated_block(text)
    # Remove trailing digits from item keys like "进食1" -> "进食"
    tokens = tokenize_kv(s)
    new_tokens: List[Tuple[Optional[str], str]] = []
    for k, v in tokens:
        if k is None:
            new_tokens.append((k, v))
            continue
        new_key = re.sub(r"(.*?)(\d+)$", r"\1", k)
        new_tokens.append((new_key, v))
    s = join_tokens(new_tokens)
    # Remove wrong prefix "压疮_" from keys (e.g., 总分、评估结果)
    s = remove_prefix_in_keys(s, ["压疮_"])
    # Recalculate 总分 by summing numbers inside parentheses, like "(5分)"
    tks = tokenize_kv(s)
    total = 0
    for k, v in tks:
        if k and k not in {"总分", "自理能力等级", "性别"}:
            m = re.search(r"[（(]\s*(\d+)\s*分[）)]", v)
            if m:
                try:
                    total += int(m.group(1))
                except Exception:
                    pass
    updated = False
    for i, (k, v) in enumerate(tks):
        if k == "总分":
            # Keep suffix if any
            m = re.match(r"\s*(\d+)(.*)$", v)
            suffix = m.group(2) if m else ""
            tks[i] = (k, f"{total}{suffix}")
            updated = True
            break
    if not updated:
        tks.append(("总分", str(total)))
    return join_tokens(tks)


def clean_cpot(text: str) -> str:
    s = dedupe_repeated_block(text)
    s = remove_prefix_in_keys(s, ["CPOT_"])
    return s


def clean_mews(text: str) -> str:
    s = dedupe_repeated_block(text)
    s = remove_prefix_in_keys(s, ["MEWS-"])
    return s


def clean_gcs(text: str) -> str:
    """住院病人昏迷评定记录单: recalc 昏迷评分"""
    s = dedupe_repeated_block(text)
    tokens = tokenize_kv(s)
    score = 0
    for k, v in tokens:
        if k and (k.startswith("睁眼-") or k.startswith("运动-") or k.startswith("言语-")):
            m = re.search(r"-?\d+", v)
            if m:
                try:
                    score += int(m.group(0))
                except Exception:
                    pass
    new_tokens: List[Tuple[Optional[str], str]] = []
    replaced = False
    for k, v in tokens:
        if k == "昏迷评分":
            m = re.match(r"\s*(\d+)(.*)$", v)
            suffix = m.group(2) if m else ""
            new_tokens.append((k, f"{score}{suffix}"))
            replaced = True
        else:
            new_tokens.append((k, v))
    if not replaced:
        new_tokens.append(("昏迷评分", str(score)))
    return join_tokens(new_tokens)


def clean_event_content(sub_type: str, content: str) -> str:
    try:
        if content is None:
            return content
        if sub_type == "导管观察及护理记录":
            return clean_simple_dedupe(content)
        if sub_type == "谵妄评估单":
            return clean_delirium(content)
        if sub_type == "约束护理记录单":
            return remove_prefix_in_keys(dedupe_repeated_block(content), ["约束_"])
        if sub_type == "医院获得性肺炎风险因素评估表":
            return clean_pneumonia(content)
        if sub_type == "Caprini血栓风险评估单":
            return clean_caprini(content)
        if sub_type == "疼痛评估及记录单":
            return clean_simple_dedupe(content)
        if sub_type == "导尿管伴随性尿路感染风险监控":
            return clean_urinary_cauti(content)
        if sub_type == "Braden":
            return clean_braden(content)
        if sub_type == "皮肤观察及护理记录单":
            return clean_simple_dedupe(content)
        if sub_type == "跌倒/坠床风险评估表":
            return clean_simple_dedupe(content)
        if sub_type == "压力性损伤观察及护理记录单":
            return clean_simple_dedupe(content)
        if sub_type == "Barthel指数量表":
            return clean_barthel(content)
        if sub_type == "内科血栓风险评估表":
            return clean_simple_dedupe(content)
        if sub_type == "重症疼痛观察量表CPOT":
            return clean_cpot(content)
        if sub_type == "改良早期预警评分表MEWS":
            return clean_mews(content)
        if sub_type == "住院病人昏迷评定记录单":
            return clean_gcs(content)
        if sub_type == "危重护理记录单":
            return clean_simple_dedupe(content)
        if sub_type == "肌力等级评估表":
            return clean_simple_dedupe(content)
        # Fallback (should rarely happen if reasons drive selection)
        return content
    except Exception as e:
        LOGGER.exception(f"Clean failed for sub_type={sub_type}: {e}")
        return content


def process_file(in_path: Path, out_path: Path, reasons: Dict[str, str]) -> Tuple[int, int]:
    """
    Returns (num_events, num_cleaned)
    """
    try:
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        LOGGER.exception(f"Failed to read {in_path}: {e}")
        return (0, 0)

    sequence = data.get("sequence", [])
    cleaned = 0
    for ev in sequence:
        sub_type = ev.get("sub_type")
        if not sub_type:
            continue
        # Only handle sub_types that have non-empty reasons
        if sub_type not in reasons:
            continue
        original = ev.get("event_content")
        new_content = clean_event_content(sub_type, original)
        if isinstance(new_content, str) and new_content != original:
            ev["event_content"] = new_content
            cleaned += 1

    # Write output JSON preserving the structure
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        LOGGER.exception(f"Failed to write {out_path}: {e}")

    return (len(sequence), cleaned)


def main():
    project_root = Path(__file__).resolve().parents[2]
    in_dir = project_root / "database" / "icu_raw"
    out_dir = project_root / "database" / "icu_first_clean"
    reason_path = project_root / "scripts" / "icu_clear" / "sub_type_reason.json"

    reasons = load_reasons(reason_path)
    if not reasons:
        LOGGER.warning("No cleaning reasons found; nothing to do.")
        return

    LOGGER.info(f"Starting ICU first clean. Input: {in_dir}, Output: {out_dir}")
    LOGGER.info(f"Sub-types to clean: {sorted(list(reasons.keys()))}")

    total_files = 0
    total_events = 0
    total_cleaned = 0

    for path in in_dir.glob("*.json"):
        total_files += 1
        out_path = out_dir / path.name
        num_events, num_cleaned = process_file(path, out_path, reasons)
        total_events += num_events
        total_cleaned += num_cleaned
        if total_files % 20 == 0:
            LOGGER.info(f"Processed {total_files} files... cleaned {total_cleaned} events so far")

    LOGGER.info(
        f"Done. Files: {total_files}, Events: {total_events}, Cleaned events: {total_cleaned}. Output at {out_dir}"
    )


if __name__ == "__main__":
    main()


