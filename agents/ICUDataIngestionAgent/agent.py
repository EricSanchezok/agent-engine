import json
import os
import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Agent Engine imports
from agent_engine.agent_logger import AgentLogger

# Local imports
from agents.ICUDataIngestionAgent.umls_translator import UMLSClinicalTranslator

class ICUDataIngestionAgent:
    """
    Minimal, deterministic ICU data ingestor for replaying events from a single
    icu_patients JSON file.

    Core features:
    - load_patient(path_or_dict): load a patient JSON (filename is patient_id by convention)
    - reset(): reset read cursor to the beginning of the sequence
    - next_event(): return the next single event (raw envelope)
    - update(): return a list of events that should be emitted together
      Rule: when a new event is popped, if there are subsequent events with the
      same (event_type, sub_type) and identical timestamp, they are batched into
      the same update. This groups bursts like order/order.
    - Envelope: always include patient_id, event_id, timestamp, event_type, sub_type,
      event_content, raw (original event object).
    """

    def __init__(self) -> None:
        self.logger = AgentLogger(self.__class__.__name__)
        self._patient_id: Optional[str] = None
        self._meta_id: Optional[str] = None
        self._sequence: List[Dict[str, Any]] = []
        self._cursor: int = 0

        # Translator module encapsulating prompts, LLM, and cache
        self._translator = UMLSClinicalTranslator()

    @property
    def patient_id(self) -> Optional[str]:
        return self._patient_id

    def load_patient(self, source: Any) -> None:
        """
        Load a patient JSON from file path/Path or a preloaded dict.

        Expected JSON structure:
        {
          "meta_id": "...",
          "sequence": [ { "id": "...", "timestamp": "...", "event_type": "...", "sub_type": "...", ... }, ... ]
        }
        Filename (without extension) is used as patient_id when source is a path.
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

        # Ensure deterministic ordering by timestamp then by index (assume input is already ordered)
        self._meta_id = meta_id
        self._sequence = sequence
        self._cursor = 0

        self.logger.info(
            f"Loaded patient: patient_id={self._patient_id}, meta_id={self._meta_id}, events={len(self._sequence)}"
        )

    def set_patient_id(self, patient_id: str) -> None:
        """Set patient_id when loading from a dict source."""
        self._patient_id = patient_id

    def reset(self) -> None:
        self._cursor = 0
        self.logger.info("Cursor reset to the beginning of sequence")

    async def _wrap_envelope(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        ev_id = ev.get("id")
        content_cn = ev.get("event_content")
        content_en = None
        if isinstance(content_cn, str) and content_cn.strip():
            try:
                content_en = await self._translator.get_translation(ev_id, content_cn, overwrite=False)
            except RuntimeError:
                pass

        return {
            "patient_id": self._patient_id,
            "meta_id": self._meta_id,
            "event_id": ev_id,
            "timestamp": ev.get("timestamp"),
            "event_type": ev.get("event_type"),
            "sub_type": ev.get("sub_type"),
            "event_content": content_en if content_en else content_cn,
            "raw": ev,
        }

    # Expose translator maintenance ops
    def clear_translation_cache(self) -> None:
        self._translator.clear_cache()

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

        Grouping rule:
        - Pop one event as the anchor.
        - While the following events share the same (timestamp, event_type, sub_type),
          include them in the same update batch. This captures bursts like many
          order/order lines at one timestamp.
        - Stop when any of timestamp, event_type or sub_type changes.
        - If sequence is exhausted, return [].
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
    agent = ICUDataIngestionAgent()
    agent.load_patient("database/icu_raw/1125112810.json")
    test_count = 10
    for _ in range(test_count):
        print("*"*100)
        print(await agent.update())

if __name__ == "__main__":
    asyncio.run(test())