from __future__ import annotations

"""
Precompute translation and vector caches for ICU patient data.

Usage:
  # Single patient by id
  run.bat scripts\precompute_patient_caches.py --id 1125112810

  # All patients under database/icu_patients
  run.bat scripts\precompute_patient_caches.py --all

Notes:
  - If no arguments are provided, the script defaults to PATIENT_ID.
  - Edit PATIENT_ID and CONCURRENCY below as needed.
"""

import asyncio
import argparse
from pathlib import Path
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
                # Cache vectors only into global cache with overwrite control
                await memory.cache_event_vector_only(pid, ev, overwrite=True)
                return True
            except Exception as e:
                logger.warning(f"Vectorize failed for id={ev.get('id')}: {e}")
                return False

    tasks = [_one(ev) for ev in seq]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    ok = sum(1 for r in results if r)
    return ok, len(seq)


async def process_patient_file(patient_json_path: str, memory: ICUMemoryAgent, *, logger: AgentLogger) -> Tuple[int, int, int, int]:
    """Process a single patient JSON file: translate then vectorize.

    Returns a tuple: (ok_translate, total_translate, ok_vectorize, total_vectorize)
    """
    ingestion = ICUDataIngestionAgent()
    ingestion.load_patient(patient_json_path)
    if ingestion.patient_id is None:
        # Fallback to file stem as patient id when not provided by loader
        fallback_id = Path(patient_json_path).stem
        if fallback_id:
            ingestion.set_patient_id(fallback_id)

    pid: str = ingestion.patient_id or ""

    logger.info(f"Start translation for patient_id={pid} with concurrency={CONCURRENCY}")
    ok_t, total_t = await translate_all(ingestion, logger=logger)
    logger.info(f"Translation done: success={ok_t}/{total_t}")

    logger.info(f"Start vectorization for patient_id={pid} with concurrency={CONCURRENCY}")
    ok_v, total_v = await vectorize_all(memory, ingestion, logger=logger)
    logger.info(f"Vectorization done: success={ok_v}/{total_v}")

    return ok_t, total_t, ok_v, total_v


async def main() -> int:
    logger = AgentLogger("PrecomputePatientCaches")

    parser = argparse.ArgumentParser(description="Precompute translation and vector caches for ICU patients.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--id", dest="patient_id", type=str, help="Process a single patient by id")
    group.add_argument("--all", dest="process_all", action="store_true", help="Process all patient JSON files under database/icu_patients")
    args = parser.parse_args()

    memory = ICUMemoryAgent()

    # Process all patients mode
    if getattr(args, "process_all", False):
        patients_dir = Path("database/icu_patients")
        if not patients_dir.exists():
            logger.error(f"Patients directory not found: {patients_dir}")
            return 1

        files = sorted(patients_dir.glob("*.json"))
        if not files:
            logger.warning(f"No patient JSON files found in: {patients_dir}")
            return 0

        logger.info(f"Processing all patients: count={len(files)}")
        for file_path in files:
            await process_patient_file(str(file_path), memory, logger=logger)
        return 0

    # Default/single patient mode
    pid = getattr(args, "patient_id", None) or PATIENT_ID
    patient_json_path = f"database/icu_patients/{pid}.json"
    await process_patient_file(patient_json_path, memory, logger=logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


