import json
import os
import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_current_file_dir

logger = AgentLogger(__name__)

class Loader:
    def __init__(self):
        self._patient_id: Optional[str] = None
        self._meta_id: Optional[str] = None
        self._sequence: List[Dict[str, Any]] = []
        self._cursor: int = 0


    @property
    def patient_id(self) -> Optional[str]:
        return self._patient_id

    def load_patient(self, source: Any) -> None:
        """
        Load a patient JSON from file path/Path or a preloaded dict.
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Patient JSON not found: {path}")
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self._patient_id = path.stem
        elif isinstance(source, dict):
            data = source
            # patient_id must be provided separately if using dict; keep None otherwise
        else:
            raise TypeError("source must be a file path or a dict")

        meta_id = data.get("meta_id")
        sequence = data.get("sequence")
        if not isinstance(sequence, list):
            raise ValueError("Invalid patient JSON: 'sequence' must be a list")

        self._meta_id = meta_id
        self._sequence = sequence
        self._cursor = 0

        logger.info(
            f"Loaded patient: patient_id={self._patient_id}, meta_id={self._meta_id}, events={len(self._sequence)}"
        )

    def reset(self) -> None:
        self._cursor = 0
        logger.info("Cursor reset to the beginning of sequence")

    async def _wrap_envelope(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        ev_id = ev.get("id")
        content = ev.get("event_content")

        return {
            "patient_id": self._patient_id,
            "meta_id": self._meta_id,
            "event_id": ev_id,
            "timestamp": ev.get("timestamp"),
            "event_type": ev.get("event_type"),
            "sub_type": ev.get("sub_type"),
            "event_content": content
        }

    async def next_event(self) -> Optional[Dict[str, Any]]:
        """Return the next single event (enveloped)."""
        if self._cursor >= len(self._sequence):
            return None
        ev = self._sequence[self._cursor]
        self._cursor += 1
        return await self._wrap_envelope(ev)


    async def update(self) -> List[Dict[str, Any]]:
        """
        Return a list of events for the next update step.
        """
        batch: List[Dict[str, Any]] = []
        if self._cursor >= len(self._sequence):
            return batch

        # Anchor event
        anchor = self._sequence[self._cursor]
        anchor_type = anchor.get("event_type")
        anchor_subtype = anchor.get("sub_type")

        # Always consume at least the anchor
        batch.append(await self._wrap_envelope(anchor))
        self._cursor += 1

        # Greedy consume following events that match grouping key
        while self._cursor < len(self._sequence):
            nxt = self._sequence[self._cursor]
            if (
                nxt.get("event_type") == anchor_type
                and nxt.get("sub_type") == anchor_subtype
            ):
                batch.append(await self._wrap_envelope(nxt))
                self._cursor += 1
                continue
            break

        return batch


async def test():
    loader = Loader()
    loader.load_patient(get_current_file_dir().parent / "database" / "icu_patients" / "1125112810.json")
    for _ in range(10):
        print(await loader.next_event())

if __name__ == "__main__":
    asyncio.run(test())