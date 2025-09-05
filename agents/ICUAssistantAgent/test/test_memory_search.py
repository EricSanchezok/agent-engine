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
UPDATES = 500
TOP_K = 20
TAU_HOURS = 6.0
VERSION = "v1"


async def _do_search(memory: ICUMemoryAgent, patient_id: str, last_event_id: Optional[str], *, logger: AgentLogger) -> List[Dict[str, Any]]:
    if not last_event_id:
        return []
    logger.info(f"Trigger search on last_event_id={last_event_id}")
    results = await memory.search_related_events(
        patient_id=patient_id,
        event_id=last_event_id,
        top_k=TOP_K,
        tau_hours=TAU_HOURS,
        version=VERSION,
        near_duplicate_delta=0.05,
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
    last_event_id = None
    for i in range(1, UPDATES + 1):
        batch = await ingestion.update()
        if not batch:
            logger.info("No more events from ingestion; stopping early at update %s", i)
            break
        ids = await memory.add_events(patient_id, batch)
        total_written += len(ids)
        logger.info(f"Update {i}: wrote {len(ids)} events (total={total_written})")
        # Print each event's timestamp and id to inspect ordering
        try:
            ts_list = [str(ev.get("timestamp")) for ev in batch]
            is_sorted = all(ts_list[j] <= ts_list[j + 1] for j in range(len(ts_list) - 1))
            # for idx, ev in enumerate(batch, start=1):
            #     logger.info(f"Update {i}: event[{idx}] ts={ev.get('timestamp')} id={ev.get('event_id')}")
            # logger.info(f"Update {i}: batch timestamps non-decreasing={is_sorted}")
        except Exception as e:
            logger.warning(f"Update {i}: failed to print timestamps: {e}")
        last_event_id = ids[-1]

    results = await _do_search(memory, patient_id, last_event_id, logger=logger)

    # 4) Write results to JSON file for easy inspection
    # enrich result items with stored content
    enriched_results: List[Dict[str, Any]] = []
    for r in results:
        ev_id = r.get("event_id")
        try:
            info = memory.get_event_by_id(patient_id, ev_id) if ev_id else None
            if info and isinstance(info, dict):
                content = info.get("content")
                if isinstance(content, str) and content:
                    r = dict(r)
                    r["content"] = content
        except Exception:
            pass
        enriched_results.append(r)

    last_event = memory.get_event_by_id(patient_id, last_event_id) if last_event_id else None
    out = {
        "patient_id": patient_id,
        "event_id": last_event_id,
        "metadata": last_event.get("metadata") if last_event else None,
        "event_content": last_event.get("content") if last_event else None,
        "params": {
            "top_k": TOP_K,
            "tau_hours": TAU_HOURS,
            "near_duplicate_delta": 0.0,
        },
        "results": enriched_results,
    }
    out_path = Path(f"agents/ICUAssistantAgent/test/test_search_{VERSION}.json")
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


