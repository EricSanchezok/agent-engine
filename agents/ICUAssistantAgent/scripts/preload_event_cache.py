import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import json

from agent_engine.agent_logger import AgentLogger
from agent_engine.memory.e_memory import PodEMemory
from agent_engine.utils import get_current_file_dir

CONCURRENCY = 128
sem = asyncio.Semaphore(CONCURRENCY)
data_dir = "agents/ICUAssistantAgent/database/icu_patients"

logger = AgentLogger(__name__)

event_cache = PodEMemory(
    name="event_cache",
    persist_dir=Path(get_current_file_dir().parent / "database"),
    dimension=3072
)

def check_preload_status():
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)
            patient_id = file.split(".")[0]
            sequence = data["sequence"]
            in_ids = []
            out_ids = []
            for event in sequence:
                _id = event["id"]
                if event_cache.get(_id) is None:
                    out_ids.append(_id)
                else:
                    in_ids.append(_id)
            logger.info(f"patient_id: {patient_id}, {len(in_ids)}/{len(out_ids)+len(in_ids)}")


if __name__ == "__main__":
    check_preload_status()
