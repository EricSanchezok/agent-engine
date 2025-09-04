from __future__ import annotations

import asyncio
from typing import Optional

from agent_engine.agent_logger import AgentLogger

from agents.ICUDataIngestionAgent.agent import ICUDataIngestionAgent
from agents.ICUMemoryAgent.agent import ICUMemoryAgent

# Fixed parameters (edit here as needed)
PATIENT_ID = "1125112810"
UPDATES = 100
SEARCH_AFTER = 0  # set 0 or None to skip in-loop search
TOP_K = 10
WINDOW_HOURS = 24
TAU_HOURS = 6.0


async def _do_search(memory: ICUMemoryAgent, patient_id: str, last_event_id: Optional[str], *, logger: AgentLogger) -> None:
    if not last_event_id:
        return
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


async def main() -> int:
    logger = AgentLogger("ICUAssistantAgent")

    # 1) Prepare agents
    ingestion = ICUDataIngestionAgent()
    memory = ICUMemoryAgent()

    # 2) Load patient events
    patient_json_path = f"database/icu_raw/{PATIENT_ID}.json"
    ingestion.load_patient(patient_json_path)
    patient_id = ingestion.patient_id or PATIENT_ID

    # 3) Replay updates and write into ICUMemoryAgent
    last_event_id: Optional[str] = None
    total_written = 0
    triggered = False
    for i in range(1, UPDATES + 1):
        batch = await ingestion.update()
        if not batch:
            logger.info("No more events from ingestion; stopping early at update %s", i)
            break
        ids = await memory.add_events(patient_id, batch)
        total_written += len(ids)
        if ids:
            last_event_id = ids[-1]
        logger.info(f"Update {i}: wrote {len(ids)} events (total={total_written})")

        if SEARCH_AFTER and i == SEARCH_AFTER:
            await _do_search(memory, patient_id, last_event_id, logger=logger)
            triggered = True

    # If not triggered in-loop, do a final search after replay for convenience
    if SEARCH_AFTER and not triggered:
        await _do_search(memory, patient_id, last_event_id, logger=logger)

    logger.info(f"Replay finished. Total events written={total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


