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


