import os
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_engine.agent_logger.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.utils import get_relative_path_from_current_file, find_project_root
from dotenv import load_dotenv


logger = AgentLogger(__name__)

# Load environment variables from .env if present
load_dotenv()

RUN_ALL = True
# Concurrency control
CONCURRENCY: int = int(os.getenv("RISK_LABELING_CONCURRENCY", "64"))

# Models and limits
DEFAULT_MODEL: str = os.getenv("ICU_RISK_MODEL", "o3-mini")
MAX_TOKENS: int = int(os.getenv("ICU_RISK_MAX_TOKENS", "20000"))
TEMPERATURE: Optional[float] = 0.0

# Event content cap to avoid excessive token usage
MAX_EVENT_CHARS: int = int(os.getenv("ICU_RISK_MAX_EVENT_CHARS", "20000"))

# Default single patient to run on first execution
DEFAULT_SINGLE_PATIENT_ID: str = os.getenv("ICU_RISK_SINGLE_PATIENT_ID", "1125178539")

# Enforce strict JSON output
STRICT_JSON_SUFFIX: str = (
    "\n\nImportant: Return ONLY a single JSON object with keys 'risks' and 'summary'. "
    "No extra text, no code fences, no markdown, no commentary."
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _truncate_text(text: str, max_len: int) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= max_len:
        return text
    head = text[: max_len - 200]
    tail = text[-200:]
    return f"{head}\n... [TRUNCATED {len(text) - max_len} CHARS] ...\n{tail}"


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        # remove triple backticks and optional language tag
        t = t.strip("`")
        # after stripping, there might still be language tag at the start
        if "\n" in t:
            first_newline = t.find("\n")
            t = t[first_newline + 1 :]
    return t


def _extract_json_candidates(text: str) -> List[str]:
    # Remove code fences first
    buf = _strip_code_fences(text)
    candidates: List[str] = []
    depth = 0
    start: Optional[int] = None
    for i, ch in enumerate(buf):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(buf[start : i + 1])
                    start = None
    # Also add raw buffer as last resort
    if buf and buf not in candidates:
        candidates.append(buf)
    return candidates


def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    # 1) direct
    obj = _safe_json_loads(text)
    if isinstance(obj, dict):
        return obj
    # 2) try candidates
    for cand in _extract_json_candidates(text):
        obj = _safe_json_loads(cand)
        if isinstance(obj, dict):
            return obj
    return None


class RiskLabelingRunner:
    def __init__(self) -> None:
        # Resolve file paths relative to this module
        self.prompts_path: Path = get_relative_path_from_current_file("prompts.yaml")
        self.risks_table_path: Path = get_relative_path_from_current_file("risks_table.json")
        self.confirm_subtypes_path: Path = get_relative_path_from_current_file("risk_confirm_subtypes.json")

        # Load prompt templates and risks table into memory
        self.prompt_loader = PromptLoader(self.prompts_path)
        self.risks_table_str: str = json.dumps(_load_json(self.risks_table_path), ensure_ascii=False, indent=2)

        # Load confirmable subtypes set
        confirm_list = _load_json(self.confirm_subtypes_path)
        self.confirm_subtypes: List[str] = [x.get("sub_type") for x in confirm_list if isinstance(x, dict) and x.get("sub_type")]
        self.confirm_subtypes_set = set(self.confirm_subtypes)

        # LLM client
        api_key = os.getenv("AZURE_API_KEY")
        if not api_key:
            raise ValueError("AZURE_API_KEY is required in environment variables")
        self.llm_client = AzureClient(api_key=api_key)

        # Concurrency semaphore
        self.semaphore = asyncio.Semaphore(CONCURRENCY)

        # Compute database dir
        self.project_root: Path = find_project_root()
        self.icu_patients_dir: Path = self.project_root / "database" / "icu_patients"

    async def close(self) -> None:
        try:
            await self.llm_client.close()
        except Exception:
            pass

    # ----------------------- LLM invocations -----------------------
    async def _call_risk_diagnosis(self, event_content: str) -> Optional[Dict[str, Any]]:
        system_prompt = self.prompt_loader.get_prompt(
            section="risk_diagnosis",
            prompt_type="system",
            risks_table=self.risks_table_str,
        )
        user_prompt = self.prompt_loader.get_prompt(
            section="risk_diagnosis",
            prompt_type="user",
            event=event_content,
        )
        user_prompt = f"{user_prompt}{STRICT_JSON_SUFFIX}"
        try:
            resp = await self.llm_client.chat(system_prompt, user_prompt, model_name=DEFAULT_MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
            if not resp:
                return None
            # Log truncated raw response
            logger.info(f"diagnosis_raw: { _truncate_text(resp, 800) }")
            parsed = _parse_llm_json(resp)
            return parsed
        except Exception as e:
            logger.error(f"risk_diagnosis call failed: {e}")
            return None

    async def _call_risk_audit(self, initial_diagnosis: Dict[str, Any], event_content: str) -> Optional[Dict[str, Any]]:
        system_prompt = self.prompt_loader.get_prompt(
            section="risk_audit",
            prompt_type="system",
            risks_table=self.risks_table_str,
        )
        user_prompt = self.prompt_loader.get_prompt(
            section="risk_audit",
            prompt_type="user",
            initial_diagnosis=json.dumps(initial_diagnosis, ensure_ascii=False),
            event=event_content,
        )
        user_prompt = f"{user_prompt}{STRICT_JSON_SUFFIX}"
        try:
            resp = await self.llm_client.chat(system_prompt, user_prompt, model_name=DEFAULT_MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
            if not resp:
                return None
            logger.info(f"audit_raw: { _truncate_text(resp, 800) }")
            parsed = _parse_llm_json(resp)
            return parsed
        except Exception as e:
            logger.error(f"risk_audit call failed: {e}")
            return None

    # ----------------------- Patient IO -----------------------
    def _load_patient_json(self, patient_id: str) -> Tuple[Path, Dict[str, Any]]:
        path = (self.icu_patients_dir / f"{patient_id}.json").resolve()
        if not path.exists():
            raise FileNotFoundError(f"Patient file not found: {path}")
        data = _load_json(path)
        return path, data

    def _save_patient_json(self, path: Path, data: Dict[str, Any]) -> None:
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    # ----------------------- Event filtering -----------------------
    def _iter_confirmable_events(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        seq = data.get("sequence", [])
        if not isinstance(seq, list):
            return []
        results: List[Dict[str, Any]] = []
        for ev in seq:
            sub_type = str(ev.get("sub_type", ""))
            if sub_type in self.confirm_subtypes_set:
                results.append(ev)
        return results

    # ----------------------- Event processing -----------------------
    async def _process_event(self, patient_id: str, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with self.semaphore:
            event_id = event.get("id")
            raw_content = event.get("event_content", "")
            content = _truncate_text(str(raw_content), MAX_EVENT_CHARS)

            # Step 1: diagnosis
            diagnosis = await self._call_risk_diagnosis(content)
            if not diagnosis:
                logger.warning(f"patient={patient_id} event={event_id} diagnosis empty")
                return None

            # Step 2: audit
            audited = await self._call_risk_audit(diagnosis, content)
            if not audited:
                logger.warning(f"patient={patient_id} event={event_id} audit empty")
                return None

            # Normalize to list of {name: reason}
            risks = audited.get("risks")
            if not isinstance(risks, list):
                # tolerate dict mapping form
                if isinstance(risks, dict):
                    risks = [{k: v} for k, v in risks.items()]
                else:
                    risks = []

            normalized: List[Dict[str, str]] = []
            for item in risks:
                if isinstance(item, dict):
                    for k, v in item.items():
                        name = str(k).strip()
                        reason = str(v).strip()
                        if name:
                            normalized.append({"name": name, "reason": reason})

            result = {"event_id": event_id, "risks": normalized, "summary": audited.get("summary", "")}
            logger.info(f"patient={patient_id} event={event_id} risks={normalized} summary={audited.get('summary', '')}")
            return result

    async def label_patient(self, patient_id: str) -> None:
        path, data = self._load_patient_json(patient_id)
        events = self._iter_confirmable_events(data)
        if not events:
            logger.info(f"No confirmable events for patient {patient_id}")
            return

        logger.info(f"Start labeling patient {patient_id}, confirmable events={len(events)}")

        tasks: List[asyncio.Task] = []
        for ev in events:
            tasks.append(asyncio.create_task(self._process_event(patient_id, ev)))

        results: List[Optional[Dict[str, Any]]] = await asyncio.gather(*tasks, return_exceptions=False)

        # Write back risks into each event's 'risks' field
        event_id_to_result: Dict[str, Dict[str, Any]] = {}
        for r in results:
            if r and isinstance(r, dict) and r.get("event_id"):
                event_id_to_result[r["event_id"]] = r

        updated = 0
        for ev in data.get("sequence", []):
            ev_id = ev.get("id")
            if ev_id in event_id_to_result:
                r = event_id_to_result[ev_id]
                # risks field should include name and reason
                ev["risks"] = r.get("risks", [])
                updated += 1

        logger.info(f"Labeled patient {patient_id}: updated events={updated}")
        self._save_patient_json(path, data)

    # ----------------------- Batch helpers -----------------------
    def list_all_patient_ids(self) -> List[str]:
        ids: List[str] = []
        if not self.icu_patients_dir.exists():
            return ids
        for p in self.icu_patients_dir.glob("*.json"):
            ids.append(p.stem)
        ids.sort()
        return ids


async def _run_single_patient(patient_id: str) -> None:
    runner = RiskLabelingRunner()
    try:
        await runner.label_patient(patient_id)
    finally:
        await runner.close()


async def _run_all_patients() -> None:
    runner = RiskLabelingRunner()
    try:
        for pid in runner.list_all_patient_ids():
            try:
                await runner.label_patient(pid)
            except Exception as e:
                logger.error(f"Error labeling patient {pid}: {e}")
    finally:
        await runner.close()



def main() -> None:
    if RUN_ALL:
        logger.info("Running risk labeling for ALL patients")
        asyncio.run(_run_all_patients())
    else:
        logger.info(f"Running risk labeling for single patient: {DEFAULT_SINGLE_PATIENT_ID}")
        asyncio.run(_run_single_patient(DEFAULT_SINGLE_PATIENT_ID))


if __name__ == "__main__":
    main()
