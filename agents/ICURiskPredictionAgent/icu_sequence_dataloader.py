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
        # Risk smoothing hyperparameters
        dilation_window_hours: float = 48.0,
        erosion_window_hours: float = 24.0,
        rise_max_window_hours: float = 168.0,
        risk_growth_rate: float = 2.0,
        risk_decay_rate: float = 4.0,
    ) -> None:
        self.logger = AgentLogger(self.__class__.__name__)
        self.data_dir: Path = Path(data_dir).resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.memory_agent: ICUMemoryAgent = memory_agent or ICUMemoryAgent()
        self.indexer = RiskLabelIndexer()
        self.label_size: int = self.indexer.size
        
        # Risk smoothing hyperparameters
        self.dilation_window_hours: float = dilation_window_hours
        self.erosion_window_hours: float = erosion_window_hours
        self.rise_max_window_hours: float = rise_max_window_hours
        self.risk_growth_rate: float = risk_growth_rate
        self.risk_decay_rate: float = risk_decay_rate

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

            # Calculate time deltas
            time_deltas = self._calculate_time_deltas(timestamps)
            
            # Add time delta as the last dimension to each vector
            vectors_with_time = []
            for i, vec in enumerate(vectors):
                vec_with_time = vec + [time_deltas[i]]
                vectors_with_time.append(vec_with_time)

            X = np.asarray(vectors_with_time, dtype=np.float32)
            Y = np.stack(labels, axis=0).astype(np.float32)
            
            # Apply temporal smoothing to risk labels
            Y_smoothed = self._smooth_risk_labels(Y, events)

            yield {
                "patient_id": patient_id,
                "event_ids": event_ids,
                "timestamps": timestamps,
                "time_deltas": time_deltas,
                "vectors": X,
                "labels": Y,
                "labels_smoothed": Y_smoothed,
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

    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp string to seconds since epoch."""
        dt = self._to_datetime(timestamp)
        if dt is None:
            return 0.0
        # Convert to seconds since epoch
        return float((dt - np.datetime64('1970-01-01T00:00:00+00:00')) / np.timedelta64(1, 's'))

    def _calculate_time_deltas(self, timestamps: List[str]) -> List[float]:
        """Calculate time deltas between consecutive timestamps in seconds."""
        if not timestamps:
            return []
        
        time_deltas = [0.0]  # First event has delta 0
        for i in range(1, len(timestamps)):
            prev_seconds = self._timestamp_to_seconds(timestamps[i-1])
            curr_seconds = self._timestamp_to_seconds(timestamps[i])
            delta = curr_seconds - prev_seconds
            time_deltas.append(max(0.0, delta))  # Ensure non-negative
        
        return time_deltas

    def _smooth_risk_labels(self, labels: np.ndarray, events: List[Dict[str, Any]]) -> np.ndarray:
        """
        Apply temporal smoothing to risk labels based on diagnostic breakpoints and time intervals.
        
        Logic:
        1. Identify diagnostic breakpoints based on risk_confirm_subtypes.json
        2. Convert timestamps to cumulative hours from admission
        3. For each risk, smooth between consecutive breakpoints based on time intervals
        4. Linear interpolation: 0.0 -> 1.0 when risk appears, 1.0 -> 0.0 when risk disappears
        """
        if labels.size == 0:
            return labels
        
        num_events, num_risks = labels.shape
        smoothed_labels = labels.copy().astype(np.float32)
        
        # Load diagnostic subtypes configuration
        confirm_subtypes = self._load_confirm_subtypes()
        
        # Find diagnostic breakpoints (events with confirm subtypes)
        breakpoints = self._find_diagnostic_breakpoints(events, confirm_subtypes)
        
        if len(breakpoints) == 0:
            # No breakpoints found, return original labels
            return smoothed_labels
        
        # Convert timestamps to cumulative hours from admission
        timestamps = [event.get("timestamp", "") for event in events]
        cumulative_hours = self._calculate_cumulative_hours_from_timestamps(timestamps)
        
        # Step 1: Apply dilation to remove short-term risk spikes
        dilated_labels = self._debounce_risk_labels(labels, cumulative_hours, window_hours=self.dilation_window_hours)
        
        # Step 2: Apply erosion to fill gaps between 0.0 values
        eroded_labels = self._erode_risk_labels(dilated_labels, cumulative_hours, window_hours=self.erosion_window_hours)
        
        # Process each risk separately with connected smoothing
        for risk_idx in range(num_risks):
            risk_sequence = eroded_labels[:, risk_idx]
            
            # Get risk states at each breakpoint
            risk_states = []
            breakpoint_times = []
            for bp_idx in breakpoints:
                risk_states.append(risk_sequence[bp_idx])
                breakpoint_times.append(cumulative_hours[bp_idx])
            
            # Apply connected smoothing logic
            smoothed_labels[:, risk_idx] = self._apply_connected_smoothing(
                risk_sequence, breakpoints, risk_states, breakpoint_times, 
                cumulative_hours, num_events
            )
        
        return smoothed_labels

    def _apply_connected_smoothing(self, risk_sequence: np.ndarray, breakpoints: List[int], 
                                 risk_states: List[float], breakpoint_times: List[float],
                                 cumulative_hours: List[float], num_events: int) -> np.ndarray:
        """
        Two-phase connected smoothing:
        Phase 1 (decline): For every 1.0→0.0 breakpoint pair (n→n+1), interpolate 1.0→0.0 across [n, n+1].
        Phase 2 (rise): For every 1.0 breakpoint at index n, find the previous 1.0 breakpoint m (< n-1).
          Use the immediate next breakpoint after m, i.e., (m+1), as the start (which must be 0.0 after phase 1),
          and interpolate 0.0→1.0 across [(m+1)→n]. If no previous 1.0 exists, start from the first breakpoint.
        Constant segments (adjacent equal states) remain constant (e.g., 1.0 between adjacent 1.0 breakpoints).
        
        Args:
            risk_sequence: Risk values at each event
            breakpoints: List of breakpoint indices
            risk_states: Risk values at each breakpoint
            breakpoint_times: Time values at each breakpoint
            cumulative_hours: Cumulative hours for all events
            num_events: Total number of events
        
        Returns:
            Smoothed risk sequence
        """
        smoothed_sequence = np.zeros(num_events, dtype=float)

        # Baseline: fill constant segments between consecutive breakpoints
        for i in range(len(breakpoints) - 1):
            current_bp = breakpoints[i]
            next_bp = breakpoints[i + 1]
            current_state = risk_states[i]
            for event_idx in range(current_bp, min(next_bp + 1, num_events)):
                smoothed_sequence[event_idx] = current_state

        # Events before first breakpoint
        if breakpoints[0] > 0:
            first_state = risk_states[0]
            for event_idx in range(0, breakpoints[0]):
                smoothed_sequence[event_idx] = first_state

        # Events after last breakpoint
        if breakpoints[-1] < num_events - 1:
            last_state = risk_states[-1]
            for event_idx in range(breakpoints[-1] + 1, num_events):
                smoothed_sequence[event_idx] = last_state

        # Phase 1: Declines (1.0 → 0.0) applied first
        for i in range(len(breakpoints) - 1):
            current_bp = breakpoints[i]
            next_bp = breakpoints[i + 1]
            current_state = risk_states[i]
            next_state = risk_states[i + 1]
            if current_state == 1.0 and next_state == 0.0:
                start_time = breakpoint_times[i]
                end_time = breakpoint_times[i + 1]
                for event_idx in range(current_bp, min(next_bp + 1, num_events)):
                    event_time = cumulative_hours[event_idx]
                    if end_time > start_time:
                        t = (event_time - start_time) / (end_time - start_time)
                    else:
                        t = 0.0
                    t = max(0.0, min(1.0, t))
                    smoothed_sequence[event_idx] = self._exponential_interpolation(
                        t, 1.0, 0.0, growth_rate=self.risk_growth_rate, decay_rate=self.risk_decay_rate
                    )

        # Phase 2: Rises (0.0 → 1.0) connected from (previous 1.0's next breakpoint) to current 1.0
        for n in range(len(breakpoints)):
            if risk_states[n] != 1.0:
                continue
            # Find previous 1.0 breakpoint m with m < n-1
            m: Optional[int] = None
            j = n - 1
            while j >= 0:
                if risk_states[j] == 1.0:
                    m = j
                    break
                j -= 1
            if m is not None and m >= n - 1:
                # Adjacent or invalid; skip because segment between them should already be constant 1.0
                continue
            # Determine start index in terms of breakpoint list
            if m is None:
                start_bp_idx = 0
            else:
                start_bp_idx = m + 1  # start at the breakpoint right after previous 1.0 (which is 0.0 after phase 1)
            end_bp_idx = n
            if start_bp_idx >= end_bp_idx:
                continue
            start_bp = breakpoints[start_bp_idx]
            end_bp = breakpoints[end_bp_idx]
            start_time = breakpoint_times[start_bp_idx]
            end_time = breakpoint_times[end_bp_idx]

            # Enforce a maximum rise window: if (end_time - start_time) > self.rise_max_window_hours,
            # advance start_bp_idx forward to the furthest breakpoint within the threshold window.
            if (end_time - start_time) > self.rise_max_window_hours:
                threshold_time = end_time - self.rise_max_window_hours
                k = start_bp_idx
                while k < end_bp_idx and breakpoint_times[k] < threshold_time:
                    k += 1
                if k < end_bp_idx:
                    start_bp_idx = k
                    start_bp = breakpoints[start_bp_idx]
                    start_time = breakpoint_times[start_bp_idx]

            if end_time <= start_time:
                continue
            for event_idx in range(start_bp, min(end_bp + 1, num_events)):
                event_time = cumulative_hours[event_idx]
                t = (event_time - start_time) / (end_time - start_time)
                t = max(0.0, min(1.0, t))
                smoothed_sequence[event_idx] = self._exponential_interpolation(
                    t, 0.0, 1.0, growth_rate=self.risk_growth_rate, decay_rate=self.risk_decay_rate
                )

        return smoothed_sequence

    def _find_furthest_zero_breakpoint(self, breakpoints: List[int], risk_states: List[float], 
                                     breakpoint_times: List[float], current_index: int) -> tuple[int, float]:
        """
        Find the furthest previous breakpoint with 0.0 value.
        
        Args:
            breakpoints: List of breakpoint indices
            risk_states: Risk values at each breakpoint
            breakpoint_times: Time values at each breakpoint
            current_index: Current breakpoint index
        
        Returns:
            Tuple of (breakpoint_index, breakpoint_time)
        """
        # Look backwards from current index
        for i in range(current_index - 1, -1, -1):
            if risk_states[i] == 0.0:
                return breakpoints[i], breakpoint_times[i]
        
        # If no previous 0.0 found, use the first breakpoint
        return breakpoints[0], breakpoint_times[0]

    def _debounce_risk_labels(self, labels: np.ndarray, cumulative_hours: List[float], window_hours: float = 48.0) -> np.ndarray:
        """
        Remove short-term risk spikes by filling gaps between risk occurrences within a time window.
        
        Args:
            labels: Original risk labels array (num_events, num_risks)
            cumulative_hours: Cumulative hours from admission for each event
            window_hours: Time window in hours for debouncing (default: 48 hours)
        
        Returns:
            Debounced labels array with short-term gaps filled
        """
        if labels.size == 0:
            return labels
        
        num_events, num_risks = labels.shape
        debounced_labels = labels.copy().astype(np.float32)
        
        for risk_idx in range(num_risks):
            risk_sequence = labels[:, risk_idx]
            
            # Find all positions where this risk is present (value = 1.0)
            risk_positions = np.where(risk_sequence == 1.0)[0]
            
            if len(risk_positions) < 2:
                # Not enough occurrences to debounce
                continue
            
            # Group consecutive risk occurrences within the time window
            risk_groups = []
            current_group = [risk_positions[0]]
            
            for i in range(1, len(risk_positions)):
                current_pos = risk_positions[i]
                prev_pos = risk_positions[i-1]
                
                # Calculate time difference
                time_diff = cumulative_hours[current_pos] - cumulative_hours[prev_pos]
                
                if time_diff <= window_hours:
                    # Within window, add to current group
                    current_group.append(current_pos)
                else:
                    # Outside window, start new group
                    risk_groups.append(current_group)
                    current_group = [current_pos]
            
            # Add the last group
            risk_groups.append(current_group)
            
            # Fill gaps within each group
            for group in risk_groups:
                if len(group) < 2:
                    continue
                
                # Get start and end positions of the group
                start_pos = group[0]
                end_pos = group[-1]
                
                # Fill all positions between start and end with 1.0
                for pos in range(start_pos, end_pos + 1):
                    if pos < num_events:
                        debounced_labels[pos, risk_idx] = 1.0
        
        return debounced_labels

    def _erode_risk_labels(self, labels: np.ndarray, cumulative_hours: List[float], window_hours: float = 48.0) -> np.ndarray:
        """
        Apply erosion to risk labels: fill gaps between 0.0 values within window_hours.
        This is the "corrosion" step after "dilation" (debouncing).
        
        Args:
            labels: Risk labels array (num_events, num_risks)
            cumulative_hours: Cumulative hours from admission for each event
            window_hours: Time window for erosion (default: 48.0)
        
        Returns:
            Eroded labels array
        """
        num_events, num_risks = labels.shape
        eroded_labels = labels.copy()
        
        for risk_idx in range(num_risks):
            risk_sequence = labels[:, risk_idx]
            
            # Find positions where risk is 0.0
            zero_positions = [i for i, val in enumerate(risk_sequence) if val == 0.0]
            
            if len(zero_positions) < 2:
                continue
            
            # Group consecutive 0.0 positions within window_hours
            zero_groups = []
            current_group = [zero_positions[0]]
            
            for i in range(1, len(zero_positions)):
                current_pos = zero_positions[i]
                prev_pos = zero_positions[i-1]
                
                # Calculate time difference
                time_diff = cumulative_hours[current_pos] - cumulative_hours[prev_pos]
                
                if time_diff <= window_hours:
                    # Within window, add to current group
                    current_group.append(current_pos)
                else:
                    # Outside window, start new group
                    zero_groups.append(current_group)
                    current_group = [current_pos]
            
            # Add the last group
            zero_groups.append(current_group)
            
            # Fill gaps within each group with 0.0
            for group in zero_groups:
                if len(group) < 2:
                    continue
                
                # Get start and end positions of the group
                start_pos = group[0]
                end_pos = group[-1]
                
                # Fill all positions between start and end with 0.0
                for pos in range(start_pos, end_pos + 1):
                    if pos < num_events:
                        eroded_labels[pos, risk_idx] = 0.0
        
        return eroded_labels

    def _exponential_interpolation(self, t: float, start_state: float, end_state: float, 
                                 growth_rate: float = 2.0, decay_rate: float = 4.0) -> float:
        """
        Apply exponential interpolation between two states.
        
        Args:
            t: Interpolation factor (0.0 to 1.0)
            start_state: Starting state value
            end_state: Ending state value
            growth_rate: Controls the steepness of risk growth (higher = steeper)
            decay_rate: Controls the steepness of risk decay (higher = steeper)
        
        Returns:
            Interpolated value using exponential function
        """
        if start_state == end_state:
            return start_state
        
        # Clamp t to [0, 1] range
        t = max(0.0, min(1.0, t))
        
        if start_state == 0.0 and end_state == 1.0:
            # Risk appears: 0.0 -> 1.0
            # Use exponential growth: starts slow, accelerates near 1.0
            # Formula: t^growth_rate (higher growth_rate = faster acceleration)
            return t ** growth_rate
        elif start_state == 1.0 and end_state == 0.0:
            # Risk disappears: 1.0 -> 0.0
            # Use exponential decay: starts slow, accelerates near 0.0 (symmetric to growth)
            # Formula: (1-t)^decay_rate (higher decay_rate = faster acceleration toward 0)
            return (1.0 - t) ** decay_rate
        else:
            # No state change or other cases
            return start_state

    def _calculate_cumulative_hours_from_timestamps(self, timestamps: List[str]) -> List[float]:
        """Calculate cumulative hours from admission for each timestamp."""
        if not timestamps:
            return []
        
        cumulative_hours = []
        start_time = None
        
        for ts in timestamps:
            if not ts:
                cumulative_hours.append(0.0)
                continue
            
            try:
                # Parse timestamp
                if ts.endswith('Z'):
                    ts = ts[:-1] + '+00:00'
                
                try:
                    dt = self._to_datetime(ts)
                except Exception:
                    dt = None
                
                if dt is None:
                    cumulative_hours.append(0.0)
                    continue
                
                # Convert to seconds since epoch
                seconds = self._timestamp_to_seconds(ts)
                
                if start_time is None:
                    start_time = seconds
                
                # Calculate hours from start
                hours = (seconds - start_time) / 3600.0
                cumulative_hours.append(hours)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse timestamp '{ts}': {e}")
                cumulative_hours.append(0.0)
        
        return cumulative_hours

    def _load_confirm_subtypes(self) -> List[str]:
        """Load diagnostic subtypes from risk_confirm_subtypes.json."""
        try:
            config_path = Path(__file__).parent / "risk_confirm_subtypes.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return [item["sub_type"] for item in config]
        except Exception as e:
            self.logger.warning(f"Failed to load confirm subtypes config: {e}")
            return ["exam", "lab", "blood_gas", "surgery"]  # Default fallback

    def _find_diagnostic_breakpoints(self, events: List[Dict[str, Any]], confirm_subtypes: List[str]) -> List[int]:
        """Find event indices that are diagnostic breakpoints based on event_type."""
        breakpoints = []
        
        for i, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            
            # Check if event has event_type information that matches confirm subtypes
            event_type = event.get("sub_type")
            if event_type in confirm_subtypes:
                breakpoints.append(i)
        
        return breakpoints

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
        Y_smoothed = sample["labels_smoothed"]
        time_deltas = sample["time_deltas"]
        timestamps = sample["timestamps"]
        
        print(f"Vector shape: {X.shape}")
        print(f"Label shape: {Y.shape}")
        print(f"Smoothed label shape: {Y_smoothed.shape}")
        print(f"Time deltas shape: {len(time_deltas)}")
        print()
        
        # Show first 10 time deltas and corresponding timestamps
        print("First 10 time deltas (in seconds):")
        for i in range(min(10, len(time_deltas))):
            print(f"  Event {i}: {time_deltas[i]:.2f} seconds (timestamp: {timestamps[i]})")