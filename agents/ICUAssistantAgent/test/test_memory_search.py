from __future__ import annotations

import asyncio
from pprint import pprint
from typing import Optional, List, Dict, Any
from pathlib import Path
import json

from agent_engine.agent_logger import AgentLogger

from agents.ICUDataIngestionAgent.agent import ICUDataIngestionAgent
from agents.ICUMemoryAgent.agent import ICUMemoryAgent

PATIENT_ID = "1125112810"
UPDATES = 100
TOP_K = 10
WINDOW_HOURS = 24
TAU_HOURS = 6.0


async def _do_search(memory: ICUMemoryAgent, patient_id: str, last_event_id: Optional[str], *, logger: AgentLogger) -> List[Dict[str, Any]]:
    if not last_event_id:
        return []
    logger.info(f"Trigger search on last_event_id={last_event_id}")
    results = await memory.search_related_events(
        patient_id=patient_id,
        event_id=last_event_id,
        top_k=TOP_K,
        window_hours=WINDOW_HOURS,
        tau_hours=TAU_HOURS,
        version="v1",
    )
    for rank, r in enumerate(results, start=1):
        ev_id = r.get("event_id")
        score = r.get("score")
        reasons = r.get("reasons", {})
        logger.info(f"Rank {rank}: id={ev_id} score={score:.4f} reasons={reasons}")
    return results


async def main() -> int:
    logger = AgentLogger("ICUAssistantAgent")

    # 1) Prepare agents
    ingestion = ICUDataIngestionAgent()
    memory = ICUMemoryAgent()

    # 2) Load patient events
    patient_json_path = f"database/icu_raw/{PATIENT_ID}.json"
    ingestion.load_patient(patient_json_path)
    patient_id = ingestion.patient_id or PATIENT_ID

    memory.delete_patient_memory(patient_id)

    # 3) Replay updates and write into ICUMemoryAgent
    total_written = 0
    for i in range(1, UPDATES + 1):
        batch = await ingestion.update()
        if not batch:
            logger.info("No more events from ingestion; stopping early at update %s", i)
            break
        ids = await memory.add_events(patient_id, batch)
        total_written += len(ids)
        logger.info(f"Update {i}: wrote {len(ids)} events (total={total_written})")

    events = await ingestion.update()
    print(events[0])
    event_id = events[0]["event_id"]
    results = await _do_search(memory, patient_id, event_id, logger=logger)

    # 4) Write results to JSON file for easy inspection
    out = {
        "patient_id": patient_id,
        "event_id": event_id,
        "params": {
            "top_k": TOP_K,
            "window_hours": WINDOW_HOURS,
            "tau_hours": TAU_HOURS,
            "near_duplicate_delta": 0.0,
        },
        "results": results,
    }
    out_path = Path("agents/ICUAssistantAgent/test_search_v1.json")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved search results to {out_path}")
    except Exception as e:
        logger.error(f"Failed to write JSON results: {e}")

    logger.info(f"Replay finished. Total events written={total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


