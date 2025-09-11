import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Agent Engine imports
from agent_engine.memory.e_memory import EMemory, Record
from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.utils import get_current_file_dir, generate_unique_id

load_dotenv()

logger = AgentLogger(__name__)

class PatientMemory:
    def __init__(self, patient_id: str, content_id: Optional[str] = None):
        if not content_id:
            content_id = generate_unique_id(patient_id)

        self.patient_id = patient_id
        self.content_id = content_id

        api_key = os.getenv("AZURE_API_KEY", "")
        if not api_key:
            self.logger.error("AZURE_API_KEY not found in environment variables")
            raise ValueError("AZURE_API_KEY is required")

        self.llm_client = AzureClient(api_key=api_key)
        self.embed_model = "text-embedding-3-large"

        self.persist_dir: str | Path = get_current_file_dir().parent / "database" / "patient_memory" / content_id
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.memory = EMemory(
            name=patient_id,
            llm_client=self.llm_client,
            embed_model=self.embed_model,
            persist_dir=self.persist_dir,
            enable_vectors=True
        )

        self._vector_cache = ScalableMemory(
            name="icu_vector_cache",
            llm_client=self.llm_client,
            embed_model=self.embed_model,
            persist_dir=Path(get_current_file_dir().parent / "database"),
            enable_vectors=True
        )

    def close(self) -> None:
        try:
            self.memory.db.close()
        except Exception:
            pass

        try:
            self._vector_cache.db.close()
        except Exception:
            pass