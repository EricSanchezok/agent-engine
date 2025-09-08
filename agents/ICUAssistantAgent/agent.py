from __future__ import annotations

from typing import List, Optional

from agent_engine.agent_logger import AgentLogger

from agents.ICUDataIngestionAgent.agent import ICUDataIngestionAgent
from agents.ICUMemoryAgent.agent import ICUMemoryAgent


class ICUAssistantAgent:
    """Coordinator agent for ICU system testing and user-facing orchestration.

    Responsibilities:
    - Load patient data via ICUDataIngestionAgent
    - Feed events into ICUMemoryAgent (batch updates)
    - Trigger related-event search on demand
    - Provide a simple programmable API for tests/demo
    """

    def __init__(self) -> None:
        self.logger = AgentLogger(self.__class__.__name__)
        self.ingestion_agent = ICUDataIngestionAgent()
        self.memory_agent = ICUMemoryAgent()

    def load_patient(self, patient_id: str) -> None:
        path = f"database/icu_patients/{patient_id}.json"
        self.ingestion_agent.load_patient(path)
        # align ingestion_agent patient_id with given
        if self.ingestion_agent.patient_id is None:
            self.ingestion_agent.set_patient_id(patient_id)

    def replay_updates(self, num_updates: int) -> int:
        pid = self.ingestion_agent.patient_id
        if not pid:
            raise ValueError("Patient not loaded")
        total_written = 0
        for i in range(1, num_updates + 1):
            batch = self.ingestion_agent.update()
            if not batch:
                self.logger.info(f"No more events at update {i}; stop.")
                break
            ids = self.memory_agent.add_events(pid, batch)
            total_written += len(ids)
            self.logger.info(f"Update {i}: wrote {len(ids)} events")
        return total_written

    def search_after_last(self, top_k: int = 20, window_hours: int = 24, tau_hours: float = 6.0) -> List[dict]:
        pid = self.ingestion_agent.patient_id
        if not pid:
            raise ValueError("Patient not loaded")
        # get latest event id in memory_agent
        latest = self.memory_agent.get_recent_events(pid, n=1)
        if not latest:
            return []
        ev_id = latest[0]["id"]
        return self.memory_agent.search_related_events(
            patient_id=pid,
            event_id=ev_id,
            top_k=top_k,
            window_hours=window_hours,
            tau_hours=tau_hours,
            version="v1",
        )


