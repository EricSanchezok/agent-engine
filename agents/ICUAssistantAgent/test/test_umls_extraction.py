from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_engine.agent_logger import AgentLogger

from agents.ICUDataIngestionAgent.agent import ICUDataIngestionAgent
from agents.ICUMemoryAgent.agent import ICUMemoryAgent
from agents.ICUMemoryAgent.umls_concept_extractor import UMLSConceptExtractor


PATIENT_ID = "1125112810"
UPDATES = 100


async def _replay_and_get_last_event(memory: ICUMemoryAgent, ingestion: ICUDataIngestionAgent, *, logger: AgentLogger) -> Optional[str]:
    total_written = 0
    last_event_id: Optional[str] = None
    for i in range(1, UPDATES + 1):
        batch = await ingestion.update()
        if not batch:
            logger.info("No more events from ingestion; stopping early at update %s", i)
            break
        ids = await memory.add_events(PATIENT_ID, batch)
        total_written += len(ids)
        last_event_id = ids[-1] if ids else last_event_id
        logger.info(f"Update {i}: wrote {len(ids)} events (total={total_written})")
    return last_event_id


async def main() -> int:
    logger = AgentLogger("ICUAssistantAgent")

    # 1) Prepare agents
    ingestion = ICUDataIngestionAgent()
    memory = ICUMemoryAgent()

    # 2) Load patient events
    patient_json_path = f"database/icu_raw/{PATIENT_ID}.json"
    ingestion.load_patient(patient_json_path)
    patient_id = ingestion.patient_id or PATIENT_ID

    # Fresh start for the specific patient memory
    memory.delete_patient_memory(patient_id)

    # 3) Replay updates and capture the last event id
    last_event_id = await _replay_and_get_last_event(memory, ingestion, logger=logger)
    if not last_event_id:
        logger.error("No event id produced during ingestion updates")
        return 1

    # 4) Fetch the last event content from memory
    ev = memory.get_event_by_id(patient_id, last_event_id)
    if not ev:
        logger.error(f"Event not found in memory: {last_event_id}")
        return 1
    content = ev.get("content") or ""

    # 5) Run UMLS concept extraction
    try:
        extractor = UMLSConceptExtractor()
    except Exception as e:
        logger.error(f"Failed to init UMLSConceptExtractor: {e}")
        return 1

    result = extractor.extract(str(content), debug=True)

    # 6) Save input + extraction result to JSON (overwrite each run)
    out = {
        "patient_id": patient_id,
        "event_id": last_event_id,
        "event_content": content,
        "extraction": result,
    }
    out_path = Path("agents/ICUAssistantAgent/test/test_umls_extraction.json")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved umls extraction to {out_path}")
    except Exception as e:
        logger.error(f"Failed to write JSON results: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


