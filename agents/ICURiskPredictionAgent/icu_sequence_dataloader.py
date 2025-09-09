import json
import asyncio
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ICUMemoryAgent.agent import ICUMemoryAgent
from agents.ICURiskPredictionAgent.risk_label_indexer import (
    RiskLabelIndexer,
    AmbiguousLabelError,
    LabelNotFoundError,
)


class ICUSequenceDataLoader:
    """Sequential ICU patient data loader for risk prediction training.

    Features:
    - Loads patients one by one from a directory of JSON files (no cross-patient shuffle).
    - For each event, fetches its embedding vector from ICUMemoryAgent's global `_vector_cache`.
      If missing, embeds once and stores it back to `_vector_cache` (no writes to patient memories).
    - Builds a multi-hot label vector of size = number of risks from `RiskLabelIndexer` (e.g., 363).
    - Provides patient-level generators for train/val/test splits, preserving time order within a patient.
    """

    def __init__(
        self,
        data_dir: str | Path,
        memory_agent: Optional[ICUMemoryAgent] = None,
        only_patient_id: Optional[str] = None,
        only_patient_ids: Optional[List[str]] = None,
        train_ratio: float = 0.6,
        val_ratio: Optional[float] = None,
        test_ratio: float = 0.0,
        shuffle_patients: bool = False,
        random_seed: int = 42,
    ) -> None:
        self.logger = AgentLogger(self.__class__.__name__)
        self.data_dir: Path = Path(data_dir).resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.memory_agent: ICUMemoryAgent = memory_agent or ICUMemoryAgent()
        self.indexer = RiskLabelIndexer()
        self.label_size: int = self.indexer.size

        # List all patient JSON files deterministically
        files = sorted([p for p in self.data_dir.glob("*.json") if p.is_file()])

        # Optional: restrict to specified patient(s) for testing
        target_ids: Optional[List[str]] = None
        if only_patient_id:
            target_ids = [str(only_patient_id)]
        if only_patient_ids:
            ids = [str(x) for x in only_patient_ids]
            target_ids = (target_ids or []) + [x for x in ids if x not in (target_ids or [])]
        if target_ids:
            wanted = set(target_ids)
            files = [p for p in files if p.stem in wanted]
            self.logger.info(
                f"Filtering to target patients: {sorted(wanted)} | matched_files={len(files)}"
            )

        if not files:
            raise FileNotFoundError(f"No JSON files found in {self.data_dir}")

        # Train/Val/Test split by patient files
        self._train_files: List[Path]
        self._val_files: List[Path]
        self._test_files: List[Path]

        import random

        if shuffle_patients:
            rnd = random.Random(random_seed)
            files = files.copy()
            rnd.shuffle(files)

        n = len(files)
        if target_ids:
            # Testing mode: put all selected patients into train split
            self._train_files = files
            self._val_files = []
            self._test_files = []
        else:
            if val_ratio is None:
                # Default: split remaining to val
                train_count = int(n * train_ratio)
                val_count = n - train_count
                test_count = 0
            else:
                # Explicit train/val and remainder to test (or use test_ratio if provided)
                train_count = int(n * train_ratio)
                val_count = int(n * val_ratio)
                rest = n - train_count - val_count
                if test_ratio > 0.0:
                    test_count = int(n * test_ratio)
                    # Adjust val to keep counts non-negative and use all files
                    if train_count + val_count + test_count > n:
                        test_count = max(0, n - train_count - val_count)
                else:
                    test_count = max(0, rest)

            # Ensure no negatives and use all files
            train_count = max(0, min(train_count, n))
            val_count = max(0, min(val_count, n - train_count))
            test_count = max(0, n - train_count - val_count)

            self._train_files = files[:train_count]
            self._val_files = files[train_count:train_count + val_count]
            self._test_files = files[train_count + val_count:train_count + val_count + test_count]

        self.logger.info(
            f"ICUSequenceDataLoader initialized | patients={n} | train={len(self._train_files)} | "
            f"val={len(self._val_files)} | test={len(self._test_files)} | labels={self.label_size}"
        )

    # ---------------------- Public API ----------------------
    def label_vocab_size(self) -> int:
        return self.label_size

    def label_version_hash(self) -> str:
        return self.indexer.version_hash

    def split_counts(self) -> Dict[str, int]:
        return {
            "train": len(self._train_files),
            "val": len(self._val_files),
            "test": len(self._test_files),
        }

    def iter_patient_sequences(self, split: str = "train") -> Iterator[Dict[str, Any]]:
        """Yield per-patient sequential data for the given split.

        Yields dict with keys:
            - patient_id: str
            - event_ids: List[str]
            - timestamps: List[str]
            - vectors: np.ndarray of shape (T, D)
            - labels: np.ndarray of shape (T, label_size)
        """
        files = self._get_files_for_split(split)
        for f in files:
            data = self._load_patient_file(f)
            if data is None:
                continue
            patient_id, events = data

            # Sort events by timestamp ascending to preserve temporal order
            events = [ev for ev in events if self._get_event_id(ev) is not None]
            events.sort(key=lambda e: self._to_datetime(self._get_timestamp(e)) or self._min_dt())
            if not events:
                continue

            # Ensure vectors exist in global cache (batch for missing ones)
            missing: List[Dict[str, Any]] = []
            for ev in events:
                ev_id = self._get_event_id(ev) or ""
                if not ev_id:
                    continue
                vec = self.memory_agent.get_vector_from_cache(ev_id)
                if vec is None:
                    missing.append(ev)

            if missing:
                try:
                    asyncio.run(self.memory_agent.cache_events_vectors_only(patient_id, missing, overwrite=False))
                except Exception as e:
                    self.logger.warning(f"Batch embedding missing vectors failed for patient {patient_id}: {e}")

            # Build sequences
            vectors: List[List[float]] = []
            labels: List[np.ndarray] = []
            event_ids: List[str] = []
            timestamps: List[str] = []

            for ev in events:
                ev_id = self._get_event_id(ev)
                if not ev_id:
                    continue
                vec = self.memory_agent.get_vector_from_cache(ev_id)
                if vec is None:
                    # As a last resort, try on-demand embed for this single event
                    try:
                        vec = asyncio.run(self.memory_agent.get_or_embed_vector_by_id(
                            event_id=ev_id,
                            event=ev,
                            patient_id=patient_id,
                            overwrite=False,
                        ))
                    except Exception as e:
                        self.logger.warning(f"Skip event without vector (id={ev_id}) for patient {patient_id}: {e}")
                        continue

                y = self._build_label_vector(ev)
                vectors.append([float(x) for x in vec])
                labels.append(y)
                event_ids.append(ev_id)
                timestamps.append(self._get_timestamp(ev) or "")

            if not vectors:
                continue

            X = np.asarray(vectors, dtype=np.float32)
            Y = np.stack(labels, axis=0).astype(np.float32)

            yield {
                "patient_id": patient_id,
                "event_ids": event_ids,
                "timestamps": timestamps,
                "vectors": X,
                "labels": Y,
            }

    # ---------------------- Internals ----------------------
    def _get_files_for_split(self, split: str) -> List[Path]:
        s = split.lower().strip()
        if s == "train":
            return self._train_files
        if s == "val" or s == "valid" or s == "validation":
            return self._val_files
        if s == "test":
            return self._test_files
        raise ValueError(f"Unknown split: {split}")

    def _load_patient_file(self, path: Path) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.warning(f"Failed to read JSON at {path}: {e}")
            return None

        # Determine patient id
        patient_id: str = str(raw.get("patient_id") or path.stem)
        seq = raw.get("sequence")
        if not isinstance(seq, list):
            self.logger.warning(f"Missing or invalid 'sequence' in {path}")
            return None
        return patient_id, seq

    def _get_event_id(self, event: Dict[str, Any]) -> Optional[str]:
        return (
            (event.get("event_id") if isinstance(event, dict) else None)
            or (event.get("id") if isinstance(event, dict) else None)
        )

    def _get_timestamp(self, event: Dict[str, Any]) -> Optional[str]:
        if not isinstance(event, dict):
            return None
        ts = event.get("timestamp")
        if isinstance(ts, str):
            return ts
        return None

    def _to_datetime(self, ts: Optional[str]) -> Optional[np.datetime64]:
        if ts is None:
            return None
        s = ts.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            # Use numpy for stable sorting keys
            return np.datetime64(s)
        except Exception:
            try:
                return np.datetime64(s.split(".")[0])
            except Exception:
                return None

    def _min_dt(self) -> np.datetime64:
        return np.datetime64("1970-01-01T00:00:00+00:00")

    def _build_label_vector(self, event: Dict[str, Any]) -> np.ndarray:
        y = np.zeros(self.label_size, dtype=np.float32)
        risks = event.get("risks") if isinstance(event, dict) else None
        if not isinstance(risks, list):
            return y
        for item in risks:
            if not isinstance(item, dict):
                continue
            for risk_name in item.keys():
                try:
                    idx = self.indexer.risk_to_index(str(risk_name))
                    if 0 <= idx < self.label_size:
                        y[idx] = 1.0
                except (AmbiguousLabelError, LabelNotFoundError) as e:
                    # Skip ambiguous or unknown risk names, but log once per occurrence
                    self.logger.warning(f"Skip risk '{risk_name}' for event due to mapping issue: {e}")
                except Exception as e:
                    self.logger.warning(f"Unexpected error mapping risk '{risk_name}': {e}")
        return y


if __name__ == "__main__":
    loader = ICUSequenceDataLoader(
        data_dir="database/icu_patients",
        only_patient_id="1125112810",
        train_ratio=0.6,
        shuffle_patients=False,
    )

    for sample in loader.iter_patient_sequences("train"):
        X = sample["vectors"]
        Y = sample["labels"]

    print(X.shape)
    print(Y.shape)