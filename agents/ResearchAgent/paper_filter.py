from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from agent_engine.agent_logger import AgentLogger

from .arxiv_memory import ArxivMemory


class PaperFilterModule:
    """Filter arXiv papers using preference ids from signals_qiji list and ArxivMemory vectors.

    Unified entry:
        await run({
            "arxiv_ids": List[str],  # candidates
            "signals_qiji_ids_path": str,  # JSON file containing list[str]
            "top_k": int = 8,
        }) -> {
            "code": 0,
            "message": "success",
            "data": {"filtered_ids": List[str]}
        }
    """

    def __init__(self) -> None:
        self.logger = AgentLogger(self.__class__.__name__)
        self.memory = ArxivMemory()

    def _load_signals_qiji_ids(self, path: str) -> List[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                ids = json.load(f)
            if not isinstance(ids, list):
                return []
            return [str(x) for x in ids if isinstance(x, (str, int))]
        except Exception as e:
            self.logger.error(f"Failed to load signals_qiji ids from {path}: {e}")
            return []

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            candidate_ids: List[str] = list(payload.get("arxiv_ids") or [])
            ids_path: Optional[str] = payload.get("signals_qiji_ids_path")
            top_k: int = int(payload.get("top_k", 8))

            if not candidate_ids:
                raise ValueError("'arxiv_ids' is required and should be non-empty")
            if not ids_path:
                raise ValueError("'signals_qiji_ids_path' is required")

            # Load preference ids
            pref_ids: List[str] = self._load_signals_qiji_ids(ids_path)
            if not pref_ids:
                raise ValueError("signals_qiji ids list is empty")

            # Fetch vectors from memory
            pref_items = self.memory.get_items_by_ids(pref_ids)
            cand_items = self.memory.get_items_by_ids(candidate_ids)

            # Build matrices
            pref_vecs: List[List[float]] = [v for (_c, v, _m) in pref_items.values() if v]
            cand_pairs: List[Tuple[str, List[float]]] = [
                (cid, v) for cid, (_c, v, _m) in cand_items.items() if v
            ]

            if not pref_vecs:
                raise ValueError("No vectors found for signals_qiji ids in ArxivMemory")
            if not cand_pairs:
                raise ValueError("No candidate vectors found in ArxivMemory")

            pref_np = np.array(pref_vecs, dtype=np.float32)
            cand_ids_sorted: List[str] = []
            cand_np = np.array([v for (_id, v) in cand_pairs], dtype=np.float32)

            # Compute similarity: for each candidate, max similarity over pref set
            sim_matrix = cosine_similarity(cand_np, pref_np)
            max_sims = np.max(sim_matrix, axis=1)

            # Sort by similarity desc
            order = np.argsort(-max_sims)
            for idx in order[:top_k]:
                cand_id = cand_pairs[int(idx)][0]
                cand_ids_sorted.append(cand_id)

            return {"code": 0, "message": "success", "data": {"filtered_ids": cand_ids_sorted}}
        except Exception as e:
            self.logger.error(f"Filter failed: {e}")
            return {"code": -1, "message": f"filter error: {e}", "data": {}}


