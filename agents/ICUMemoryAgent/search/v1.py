from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Agent Engine imports
from agent_engine.agent_logger import AgentLogger
from agent_engine.memory import ScalableMemory

# Search imports
from .base import BaseSearchAlgorithm, _parse_dt

logger = AgentLogger("ICUMemorySearchV1")

class SearchV1(BaseSearchAlgorithm):
    """V1 search: sim_vec + time_score.

    - sim_vec: use ScalableMemory.search similarity as cosine similarity
    - time_score: exp(-|Δt|/τ)
    """

    async def search_related_events(
        self,
        *,
        patient_mem: ScalableMemory,
        query_event_id: str,
        top_k: int = 20,
        window_hours: int = 24,
        weights: Optional[Dict[str, float]] = None,
        tau_hours: float = 6.0,
    ) -> List[Dict[str, Any]]:
        if not query_event_id:
            return []

        w1 = float((weights or {}).get("w1", 0.5))
        w2 = float((weights or {}).get("w2", 0.5))
        tau = float(tau_hours)

        # 1) Fetch query vector/time from patient memory
        q_content, q_vec, q_md = patient_mem.get_by_id(query_event_id)
        if q_vec is None:
            logger.warning(f"Query event not found or missing vector: {query_event_id}")
            return []
        q_ts = _parse_dt(q_md.get("timestamp")) if isinstance(q_md, dict) else None

        # 2) Vector recall (TopN)
        topn_vec = max(top_k * 5, 50)
        vec_hits = await patient_mem.search(q_vec, top_k=topn_vec)  # (content, sim, md)

        # 3) Temporal window recall
        candidates_by_id: Dict[str, Dict[str, Any]] = {}

        if q_ts is not None and window_hours > 0:
            start_dt = q_ts - timedelta(hours=window_hours)
            end_dt = q_ts + timedelta(hours=window_hours)
        else:
            start_dt = None
            end_dt = None

        # Add vector hits
        for content, sim, md in vec_hits:
            ev_id = md.get("id")
            if not ev_id or ev_id == query_event_id:
                continue
            ts = _parse_dt(md.get("timestamp"))
            in_window = False
            if start_dt is None or ts is None:
                in_window = True
            else:
                in_window = (start_dt <= ts <= end_dt)

            candidates_by_id.setdefault(ev_id, {
                "event_id": ev_id,
                "content": content,
                "metadata": md,
                "sim_vec": float(sim),
                "in_window": in_window,
            })
            # Keep max similarity if duplicates
            if candidates_by_id[ev_id]["sim_vec"] < float(sim):
                candidates_by_id[ev_id]["sim_vec"] = float(sim)

        # Add all events in the time window as candidates (even if not in vec hits)
        if start_dt is not None and end_dt is not None:
            for content, vector, md in patient_mem.get_all():
                ev_id = md.get("id")
                if not ev_id or ev_id == query_event_id:
                    continue
                ts = _parse_dt(md.get("timestamp"))
                if ts is None:
                    continue
                if start_dt <= ts <= end_dt:
                    can = candidates_by_id.setdefault(ev_id, {
                        "event_id": ev_id,
                        "content": content,
                        "metadata": md,
                        "sim_vec": 0.0,
                        "in_window": True,
                    })

        # 4) Re-rank with time_score
        results: List[Dict[str, Any]] = []
        for ev_id, it in candidates_by_id.items():
            md = it["metadata"]
            ts = _parse_dt(md.get("timestamp"))
            if q_ts is None or ts is None:
                time_score = 0.0
            else:
                dt_hours = abs((ts - q_ts).total_seconds()) / 3600.0
                time_score = float(pow(2.718281828, -dt_hours / max(1e-6, tau)))

            sim_vec = float(it.get("sim_vec", 0.0))
            score = w1 * sim_vec + w2 * time_score
            results.append({
                "event_id": ev_id,
                "score": float(score),
                "reasons": {
                    "sim_vec": sim_vec,
                    "time_score": time_score,
                },
                "metadata": md,
            })

        # 5) Sort and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


