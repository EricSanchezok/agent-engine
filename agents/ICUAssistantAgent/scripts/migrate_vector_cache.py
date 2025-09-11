import os

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from agent_engine.agent_logger.agent_logger import AgentLogger
from agent_engine.memory import ScalableMemory
from agent_engine.llm_client import AzureClient
from agent_engine.utils import get_current_file_dir, generate_unique_id
from agent_engine.memory.e_memory import Record, PodEMemory

load_dotenv()

logger = AgentLogger(__name__)


def main():
    api_key = os.getenv("AZURE_API_KEY", "")
    if not api_key:
        logger.error("AZURE_API_KEY not found in environment variables")
        raise ValueError("AZURE_API_KEY is required")

    llm_client = AzureClient(api_key=api_key)
    embed_model = "text-embedding-3-large"

    old_persist_dir = "C:/Eric/projects/agent-engine/agents/ICUMemoryAgent/database"
    old_vector_cache = ScalableMemory(
        name="icu_vector_cache",
        llm_client=llm_client,
        embed_model=embed_model,
        persist_dir=old_persist_dir,
        db_backend="duckdb",
        enable_vectors=True
    )

    event_cache = PodEMemory(
        name="event_cache",
        persist_dir="C:/Eric/projects/agent-engine/agents/ICUAssistantAgent/database",
        dimension=3072
    )

    logger.info(f"Migrating {old_vector_cache.count()} items")
    index = 1
    total = 0
    for batch in old_vector_cache.iterate_all(5000):
        logger.info(f"Migrating batch {index}, {total} items migrated")
        items = []
        for item in batch:
            items.append(Record(
                id=item[2]["id"],
                vector=item[1]
            ))
        event_cache.add_batch(items)
        index += 1
        total += len(batch)
    logger.info(f"Migration complete, {total} items migrated")

if __name__ == "__main__":
    main()