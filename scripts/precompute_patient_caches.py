from __future__ import annotations

"""
Precompute translation and vector caches for a single patient asynchronously.

Usage:
  run.bat scripts\precompute_patient_caches.py

Configuration:
  Edit PATIENT_ID and CONCURRENCY below as needed.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from agent_engine.agent_logger import AgentLogger
from agents.ICUDataIngestionAgent.agent import ICUDataIngestionAgent
from agents.ICUMemoryAgent.agent import ICUMemoryAgent


# Parameters
PATIENT_ID = "1125112810"
CONCURRENCY = 64


async def translate_all(ingestion: ICUDataIngestionAgent, *, logger: AgentLogger) -> Tuple[int, int]:
    sem = asyncio.Semaphore(CONCURRENCY)
    seq: List[Dict[str, Any]] = ingestion._sequence  # loaded by load_patient

    async def _one(ev: Dict[str, Any]) -> bool:
        ev_id = ev.get("id")
        content = ev.get("event_content")
        if not isinstance(content, str) or not content.strip():
            return False
        async with sem:
            try:
                res = await ingestion._translator.get_translation(ev_id, content, overwrite=True)
                return res is not None and len(res) > 0
            except Exception as e:
                logger.warning(f"Translate failed for id={ev_id}: {e}")
                return False

    tasks = [_one(ev) for ev in seq]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    ok = sum(1 for r in results if r)
    return ok, len(seq)


async def vectorize_all(memory: ICUMemoryAgent, ingestion: ICUDataIngestionAgent, *, logger: AgentLogger) -> Tuple[int, int]:
    sem = asyncio.Semaphore(CONCURRENCY)
    seq: List[Dict[str, Any]] = ingestion._sequence
    pid: str = ingestion.patient_id or ""

    async def _one(ev: Dict[str, Any]) -> bool:
        async with sem:
            try:
                await memory.add_event(pid, ev)
                return True
            except Exception as e:
                logger.warning(f"Vectorize failed for id={ev.get('id')}: {e}")
                return False

    tasks = [_one(ev) for ev in seq]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    ok = sum(1 for r in results if r)
    return ok, len(seq)


async def main() -> int:
    logger = AgentLogger("PrecomputePatientCaches")
    ingestion = ICUDataIngestionAgent()
    memory = ICUMemoryAgent()

    # Load patient
    patient_json_path = f"database/icu_raw/{PATIENT_ID}.json"
    ingestion.load_patient(patient_json_path)
    if ingestion.patient_id is None:
        ingestion.set_patient_id(PATIENT_ID)

    # Phase 1: translate all
    logger.info(f"Start translation for patient_id={PATIENT_ID} with concurrency={CONCURRENCY}")
    ok_t, total = await translate_all(ingestion, logger=logger)
    logger.info(f"Translation done: success={ok_t}/{total}")

    # Phase 2: vectorize all
    logger.info(f"Start vectorization for patient_id={PATIENT_ID} with concurrency={CONCURRENCY}")
    ok_v, total_v = await vectorize_all(memory, ingestion, logger=logger)
    logger.info(f"Vectorization done: success={ok_v}/{total_v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


