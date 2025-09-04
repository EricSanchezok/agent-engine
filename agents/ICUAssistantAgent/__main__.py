from __future__ import annotations

import argparse
import os
from typing import Optional

from agent_engine.agent_logger import AgentLogger

from agents.ICUDataIngestionAgent.agent import ICUDataIngestionAgent
from agents.ICUMemoryAgent.agent import ICUMemoryAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ICU Assistant Agent orchestrator")
    default_pid = os.getenv("ICU_PATIENT_ID", "1125112810")
    p.add_argument("patient_id", nargs="?", default=default_pid, help="Patient ID, default from env ICU_PATIENT_ID or 1125112810")
    p.add_argument("--updates", type=int, default=100, help="Number of updates to replay")
    p.add_argument("--search_after", type=int, default=101, help="Do a search after this update index (1-based)")
    p.add_argument("--top_k", type=int, default=20, help="Top K for related search")
    p.add_argument("--window_hours", type=int, default=24, help="Temporal window size in hours")
    p.add_argument("--tau_hours", type=float, default=6.0, help="Time decay tau in hours")
    return p.parse_args()


def _do_search(memory: ICUMemoryAgent, patient_id: str, last_event_id: Optional[str], *, logger: AgentLogger, args: argparse.Namespace) -> None:
    if not last_event_id:
        return
    logger.info(f"Trigger search on last_event_id={last_event_id}")
    results = memory.search_related_events(
        patient_id=patient_id,
        event_id=last_event_id,
        top_k=args.top_k,
        window_hours=args.window_hours,
        tau_hours=args.tau_hours,
        version="v1",
    )
    for rank, r in enumerate(results, start=1):
        ev_id = r.get("event_id")
        score = r.get("score")
        reasons = r.get("reasons", {})
        logger.info(f"Rank {rank}: id={ev_id} score={score:.4f} reasons={reasons}")


def main() -> int:
    args = parse_args()
    logger = AgentLogger("ICUAssistantAgent")

    # 1) Prepare agents
    ingestion = ICUDataIngestionAgent()
    memory = ICUMemoryAgent()

    # 2) Load patient events
    patient_json_path = f"database/icu_patients/{args.patient_id}.json"
    ingestion.load_patient(patient_json_path)
    patient_id = ingestion.patient_id or args.patient_id

    # 3) Replay updates and write into ICUMemoryAgent
    last_event_id: Optional[str] = None
    total_written = 0
    triggered = False
    for i in range(1, args.updates + 1):
        batch = ingestion.update()
        if not batch:
            logger.info("No more events from ingestion; stopping early at update %s", i)
            break
        ids = memory.add_events(patient_id, batch)
        total_written += len(ids)
        if ids:
            last_event_id = ids[-1]
        logger.info(f"Update {i}: wrote {len(ids)} events (total={total_written})")

        if args.search_after and i == args.search_after:
            _do_search(memory, patient_id, last_event_id, logger=logger, args=args)
            triggered = True

    # If not triggered in-loop, do a final search after replay for convenience
    if args.search_after and not triggered:
        _do_search(memory, patient_id, last_event_id, logger=logger, args=args)

    logger.info(f"Replay finished. Total events written={total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


