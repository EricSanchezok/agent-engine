from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SearchWeights:
    w_sim_vec: float = 0.5
    w_time: float = 0.5
    tau_hours: float = 6.0


def _parse_dt(s: Optional[str | datetime]) -> Optional[datetime]:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    v = str(s).strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(v)
    except Exception:
        try:
            dt = datetime.fromisoformat(v.split(".")[0])
        except Exception:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class BaseSearchAlgorithm:
    """Interface for ICUMemoryAgent search algorithms.

    Implementations should provide a `search_related_events` method that returns a
    list of dicts with scores and reasons.
    """

    def search_related_events(
        self,
        *,
        patient_mem,
        query_event_id: str,
        top_k: int = 20,
        window_hours: int = 24,
        weights: Optional[Dict[str, float]] = None,
        tau_hours: float = 6.0,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


