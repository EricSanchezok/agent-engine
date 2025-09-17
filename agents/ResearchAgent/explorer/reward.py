from __future__ import annotations
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



"""
Reward interfaces for ResearchArxivEnv.

Design goals:
- Reward may depend on entire episode history (all past states and actions).
- The reward object maintains its own internal state across steps.
- The environment calls reward.reset(initial_state, ctx) at env.reset.
- The environment calls reward.update(prev_state, action, next_state, ctx) each step to obtain reward.

You can subclass ArxivRewardBase to implement custom logic. Example:

class MyReward(ArxivRewardBase):
    def update(self, prev_state, action, next_state, ctx):
        # compute reward using self.history and ctx["density_estimator"], etc.
        r = 1.0
        self._append_transition(prev_state, action, next_state, r, ctx)
        return r
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from agent_engine.agent_logger import AgentLogger
import numpy as np  # for type annotations and simple means
from agents.ResearchAgent.explorer.vector_density import HNSWVectorDensityEstimator


class ArxivRewardBase(ABC):
    def __init__(self) -> None:
        self.logger = AgentLogger(self.__class__.__name__)
        self.history: List[Dict[str, Any]] = []

    def reset(self, initial_state: Dict[str, Any], ctx: Dict[str, Any]) -> None:
        """Reset internal episode state.

        Args:
            initial_state: The state returned by env.reset().
            ctx: Additional context (e.g., density_estimator, density_k, next_month).
        """
        self.history.clear()
        # Optionally record the starting point
        self.history.append({
            "type": "reset",
            "state": initial_state,
            "ctx": {k: ctx.get(k) for k in ("density_k", "next_month")},
        })

    @abstractmethod
    def update(
        self,
        prev_state: Dict[str, Any],
        action: Sequence[Sequence[float]],
        next_state: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> float:
        """Compute reward and update internal state.

        Implementations may consult self.history for past transitions and should
        append the current transition using _append_transition().
        """
        raise NotImplementedError

    def _append_transition(
        self,
        prev_state: Dict[str, Any],
        action: Sequence[Sequence[float]],
        next_state: Dict[str, Any],
        reward: float,
        ctx: Dict[str, Any],
    ) -> None:
        self.history.append({
            "type": "transition",
            "prev_month": prev_state.get("month"),
            "next_month": next_state.get("month"),
            "action_size": len(action) if hasattr(action, "__len__") else None,
            "reward": float(reward),
            "ctx": {k: ctx.get(k) for k in ("density_k",)},
        })


class ConstantReward(ArxivRewardBase):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = float(value)

    def update(
        self,
        prev_state: Dict[str, Any],
        action: Sequence[Sequence[float]],
        next_state: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> float:
        r = self.value
        self._append_transition(prev_state, action, next_state, r, ctx)
        return r



class NoveltyForwardReward(ArxivRewardBase):
    """Composite reward using HNSW density.

    Components (weights configurable):
    - Novelty (pioneering): For current action vectors, compute average density against
      the corpus of all past states' vectors (excluding current/next). Lower density -> better,
      so we invert: reward_novelty = 1 / (eps + avg_density).
      If there is no historical corpus yet, reward_novelty = 0.

    - Forward-looking: For all past actions (excluding current), compute their average density
      within the new state's vectors (next_state). Higher density -> better alignment with the new state.
      reward_forward = avg_density_in_next_state. If there are no past actions or next state is empty, 0.

    Total reward = weight_novelty * reward_novelty + weight_forward * reward_forward.
    """

    def __init__(
        self,
        *,
        weight_novelty: float = 1.0,
        weight_forward: float = 1.0,
        k: int = 100,
        eps: float = 1e-9,
        max_historic_vectors: int | None = None,
        max_past_action_vectors: int | None = None,
        # HNSW params
        hnsw_M: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        metric: str = "cosine",
        method: str = "knn_average",
    ) -> None:
        super().__init__()
        self.weight_novelty = float(weight_novelty)
        self.weight_forward = float(weight_forward)
        self.k = int(k)
        self.eps = float(eps)
        self.max_historic_vectors = max_historic_vectors if (max_historic_vectors is None) else int(max_historic_vectors)
        self.max_past_action_vectors = max_past_action_vectors if (max_past_action_vectors is None) else int(max_past_action_vectors)
        self.hnsw_M = int(hnsw_M)
        self.hnsw_ef_construction = int(hnsw_ef_construction)
        self.hnsw_ef_search = int(hnsw_ef_search)
        self.metric = str(metric)
        self.method = str(method)

        self._historic_vectors: list[list[float]] | None = None
        self._past_action_vectors: list[list[float]] | None = None

    def reset(self, initial_state: dict, ctx: dict) -> None:  # type: ignore[override]
        super().reset(initial_state, ctx)
        self._historic_vectors = None
        self._past_action_vectors = None

    def _ensure_2d_float32(self, arr_like) -> "np.ndarray":
        import numpy as np
        arr = np.asarray(arr_like, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("expected a 2D array-like with shape (N, D)")
        return arr

    def _append_to_historic(self, vectors_2d: "np.ndarray") -> None:
        import numpy as np
        if vectors_2d.size == 0:
            return
        if self._historic_vectors is None:
            self._historic_vectors = vectors_2d.tolist()
        else:
            # Extend list to avoid frequent big-copy
            self._historic_vectors.extend(v.tolist() for v in vectors_2d)
        if self.max_historic_vectors is not None and len(self._historic_vectors) > self.max_historic_vectors:
            # Keep most recent tail
            self._historic_vectors = self._historic_vectors[-self.max_historic_vectors :]

    def _append_to_past_actions(self, vectors_2d: "np.ndarray") -> None:
        if vectors_2d.size == 0:
            return
        if self._past_action_vectors is None:
            self._past_action_vectors = vectors_2d.tolist()
        else:
            self._past_action_vectors.extend(v.tolist() for v in vectors_2d)
        if self.max_past_action_vectors is not None and len(self._past_action_vectors) > self.max_past_action_vectors:
            self._past_action_vectors = self._past_action_vectors[-self.max_past_action_vectors :]

    def _avg_density_hnsw(self, corpus_vectors: "np.ndarray", queries_2d: "np.ndarray", k: int) -> float:
        import numpy as np
        if corpus_vectors.shape[0] == 0 or queries_2d.shape[0] == 0:
            return 0.0
        est = HNSWVectorDensityEstimator(
            metric=self.metric,
            method=self.method,
            M=self.hnsw_M,
            ef_construction=self.hnsw_ef_construction,
            ef_search=self.hnsw_ef_search,
            self_exclude=True,
        )
        est.build(corpus_vectors)
        vals: list[float] = []
        for i in range(queries_2d.shape[0]):
            vals.append(float(est.estimate_density(queries_2d[i], k=k)))
        return float(np.mean(vals)) if vals else 0.0

    def update(
        self,
        prev_state: dict,
        action: Sequence[Sequence[float]],
        next_state: dict,
        ctx: dict,
    ) -> float:  # type: ignore[override]
        import numpy as np

        # Prepare data
        prev_vecs = self._ensure_2d_float32(prev_state.get("vectors", np.empty((0, 0), dtype=np.float32)))
        action_vecs = self._ensure_2d_float32(action)
        next_vecs = self._ensure_2d_float32(next_state.get("vectors", np.empty((0, 0), dtype=np.float32)))

        k = int(ctx.get("density_k", self.k))

        # Part 1: Novelty (pioneering) against historical states (exclude current/next)
        if self._historic_vectors is None or len(self._historic_vectors) == 0:
            novelty_reward = 0.0
        else:
            hist_arr = np.asarray(self._historic_vectors, dtype=np.float32)
            avg_density_hist = self._avg_density_hnsw(hist_arr, action_vecs, k=k)
            novelty_reward = 1.0 / (self.eps + float(avg_density_hist))

        # Part 2: Forward-looking: past actions measured in new state
        if next_vecs.shape[0] == 0:
            forward_reward = 0.0
        else:
            # Prefer env's density estimator if provided and usable
            est = ctx.get("density_estimator")
            if est is None:
                # Build our own on next_vecs
                est = HNSWVectorDensityEstimator(
                    metric=self.metric,
                    method=self.method,
                    M=self.hnsw_M,
                    ef_construction=self.hnsw_ef_construction,
                    ef_search=self.hnsw_ef_search,
                    self_exclude=True,
                )
                est.build(next_vecs)

            vals: list[float] = []
            if self._past_action_vectors is not None and len(self._past_action_vectors) > 0:
                past_actions_arr = np.asarray(self._past_action_vectors, dtype=np.float32)
                for i in range(past_actions_arr.shape[0]):
                    try:
                        vals.append(float(est.estimate_density(past_actions_arr[i], k=k)))
                    except Exception as e:
                        self.logger.warning(f"Failed to estimate forward density for past action: {e}")
            forward_reward = float(np.mean(vals)) if vals else 0.0

        # Weighted sum
        reward = (self.weight_novelty * float(novelty_reward)) + (self.weight_forward * float(forward_reward))

        # Record and update internal state AFTER computing reward
        self._append_transition(prev_state, action, next_state, reward, ctx)
        # Add prev_state vectors into historic corpus
        if prev_vecs.size > 0:
            self._append_to_historic(prev_vecs)
        # Track actions (exclude current in forward computation by adding after)
        if action_vecs.size > 0:
            self._append_to_past_actions(action_vecs)

        return float(reward)

