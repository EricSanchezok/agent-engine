import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))


import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Agent Engine imports
from agent_engine.memory.e_memory import EMemory, Record, PodEMemory
from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client import QzClient
from agent_engine.utils import get_current_file_dir, generate_unique_id

# Local imports
from agents.ICUAssistantAgent.models import Event

load_dotenv()

logger = AgentLogger(__name__)

class PatientMemory:
    def __init__(self, patient_id: str, content_id: Optional[str] = None):
        if not content_id:
            content_id = generate_unique_id(patient_id)

        self.patient_id = patient_id
        self.content_id = content_id

        api_key = os.getenv("QZ_API_KEY", "")
        if not api_key:
            self.logger.error("QZ_API_KEY not found in environment variables")
            raise ValueError("QZ_API_KEY is required")

        base_url = os.getenv("QZ_BASE_URL", "http://eric-vpn.cpolar.top/r/eric_qwen3_embedding_8b")
        self.llm_client = QzClient(api_key=api_key, base_url=base_url)
        self.embed_model = "eric-qwen3-embedding-8b"

        self.persist_dir: str | Path = get_current_file_dir().parent / "database" / "patient_memory" / content_id
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.memory = EMemory(
            name=patient_id,
            persist_dir=self.persist_dir,
        )

        self._event_cache = PodEMemory(
            name="event_cache",
            persist_dir=Path(get_current_file_dir().parent / "database"),
        )

    async def add_event(self, event: Event) -> None:
        event_id = event.id
        # Check if vector is already cached in event_cache
        _vector_cache: Optional[Record] = self._event_cache.get(event_id)
        
        if _vector_cache:
            # Use cached vector to save tokens
            vector = _vector_cache.vector
        else:
            # Generate new embedding and cache it
            vector = await self.llm_client.embedding(model_name=self.embed_model, text=event.event_content)
            # Cache the vector for future use
            _cache_record = Record(
                id=event_id,
                vector=vector,
            )
            self._event_cache.add(_cache_record)

        # Create the main record for patient memory
        record = Record(
            id=event_id,
            attributes={
                "event_type": event.event_type,
                "sub_type": event.sub_type,
                "risks": event.risks,
                "flag": event.flag,
                "metadata": event.metadata
            },
            content=event.event_content,
            vector=vector,
            timestamp=event.timestamp
        )
        
        # Add to patient memory (this is separate from event_cache)
        self.memory.add(record)

if __name__ == "__main__":
    patient_memory = PatientMemory(patient_id="test")
    print(patient_memory._event_cache.count())
    print(patient_memory._event_cache.get("a2b59a07-03c8-470c-bd2b-4b7ab256c707"))
    # test_event = Event(
    #     id="a2b59a07-03c8-470c-bd2b-4b7ab256c707",
    #     timestamp="2024-05-21T15:54:11",
    #     event_type="order",
    #     sub_type="order",
    #     event_content="医嘱类型: long-term\n医嘱内容:\n- 神经外科护理常规",
    #     risks=[],
    #     flag=0,
    #     metadata={"end_timestamp": "医嘱结束时间:2024-05-27T08:18:00"}
    # )
    # patient_memory.add_event(test_event)
    # print(patient_memory._event_cache.count())