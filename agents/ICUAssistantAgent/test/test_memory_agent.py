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

    # Use the latest event timestamp as reference time instead of 'now'
    recent = memory.get_recent_events(patient_id, n=1)
    ref_time = recent[0]["timestamp"] if recent else None
    events = memory.get_events_within_hours(patient_id, ref_time=ref_time, hours=1)
    print(events)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


