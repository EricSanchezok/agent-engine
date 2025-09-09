from __future__ import annotations

"""
ResearchArxivEnv: A Gym-style environment for month-wise arXiv vector streams.

Environment semantics (Gymnasium-like):
- reset() -> (observation, info)
- step(action) -> (observation, reward, terminated, truncated, info)

State/Observation:
- Entire month's vectors as a NumPy float32 array of shape (num_papers, vector_dim).
- Observation is a dict with keys:
  - "month": YYYYMM string
  - "vectors": np.ndarray of shape (N, D)

Action:
- A batch of vectors: array-like of shape (n, vector_dim). The environment does not
  enforce semantics; it is validated for dimensionality only.

Transition:
- Each step advances to the next available month. If already at the last month,
  the environment returns the current state again with terminated=True.

Reward:
- Default constant reward = 1.0. A pluggable reward_fn(prev_state, action, next_state, ctx)
  can be injected at init-time or via set_reward_fn().

Density Estimator (for future reward design):
- The environment accepts an optional density_estimator_factory callable that returns
  an object with build(corpus_vectors) and estimate_density(query_vector, k) methods
  (e.g., HNSWVectorDensityEstimator from vector_density.py). If provided, an estimator
  is (re)built on each month's vectors and passed to reward_fn via ctx["density_estimator"].

Note:
- This module does not require gym/gymnasium; it only mimics core APIs.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyinstrument

from agent_engine.agent_logger import AgentLogger
from agents.ResearchAgent.arxiv_memory import ArxivMemory
from agents.ResearchAgent.reward import ArxivRewardBase, ConstantReward


logger = AgentLogger("ResearchArxivEnv")


RewardFn = Callable[[Dict[str, Any], Sequence[Sequence[float]], Dict[str, Any], Dict[str, Any]], float]
DensityEstimatorFactory = Callable[[], Any]


@dataclass
class _MonthCache:
    month: str
    vectors: np.ndarray  # (N, D) float32
    paper_ids: List[str]


class ResearchArxivEnv:
    def __init__(
        self,
        *,
        categories: Optional[List[str]] = None,
        vector_dim: int = 3072,
        max_months: Optional[int] = None,
        max_vectors_per_month: Optional[int] = None,
        include_content: bool = False,
        reward_fn: Optional[RewardFn] = None,
        reward_obj: Optional[ArxivRewardBase] = None,
        density_estimator_factory: Optional[DensityEstimatorFactory] = None,
        density_k: int = 50,
    ) -> None:
        """Initialize the environment.

        Args:
            categories: Optional category filter for memory.get_by_month.
            vector_dim: Expected vector dimensionality for both state and action.
            max_months: Optional cap on number of months to iterate (starting from the earliest).
            max_vectors_per_month: Optional cap on number of vectors loaded per month (head-only).
            include_content: Whether to keep paper content in info (can be large). Default False.
            reward_fn: Optional callable(prev_state, action, next_state, ctx) -> float.
            density_estimator_factory: Optional factory that produces a density estimator instance.
            density_k: Default k used for density estimator queries (passed into ctx).
        """
        self.categories = list(categories) if categories else None
        self.vector_dim = int(vector_dim)
        self.max_months = int(max_months) if max_months is not None else None
        self.max_vectors_per_month = int(max_vectors_per_month) if max_vectors_per_month is not None else None
        self.include_content = bool(include_content)
        # Reward: prefer object if provided; otherwise fallback to function or constant
        self._reward_fn: RewardFn = reward_fn if reward_fn is not None else self._default_reward_fn
        self._reward_obj: ArxivRewardBase = reward_obj if reward_obj is not None else ConstantReward(1.0)
        self._density_estimator_factory = density_estimator_factory
        self._density_k = int(density_k)

        # Storage
        self._memory = ArxivMemory()
        self._months: List[str] = self._compute_available_months()
        if self.max_months is not None:
            self._months = self._months[: self.max_months]

        if not self._months:
            raise RuntimeError("No available months found in ArxivMemory.")

        self._month_index: int = 0
        self._current_cache: Optional[_MonthCache] = None
        self._density_estimator: Any = None

        logger.info(
            f"ResearchArxivEnv initialized: months={len(self._months)}, vector_dim={self.vector_dim}, categories={self.categories}"
        )

    # ------------------------------ Public API ------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment to the earliest available month.

        Returns:
            observation: dict with keys {"month", "vectors"}
            info: dict with metadata (paper_ids, counts, etc.)
        """
        del seed, options
        self._month_index = 0
        self._current_cache = self._load_month_cache(self._months[self._month_index])
        self._maybe_build_density(self._current_cache)
        obs, info = self._make_observation_and_info(self._current_cache)
        # Reset reward object state
        ctx_reset: Dict[str, Any] = {
            "density_estimator": self._density_estimator,
            "density_k": self._density_k,
            "next_month": self._current_cache.month,
        }
        try:
            self._reward_obj.reset(self._cache_to_state(self._current_cache), ctx_reset)
        except Exception as e:
            logger.warning(f"Reward object reset failed: {e}")
        return obs, info

    @pyinstrument.profile()
    def step(self, action: Sequence[Sequence[float]]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance to the next month.

        Args:
            action: array-like of shape (n, vector_dim).

        Returns:
            observation, reward, terminated, truncated, info
        """
        self._validate_action(action)

        prev_cache = self._current_cache or self._load_month_cache(self._months[self._month_index])

        # Prepare next state
        terminated = False
        truncated = False
        if self._month_index + 1 < len(self._months):
            self._month_index += 1
            next_cache = self._load_month_cache(self._months[self._month_index])
            self._current_cache = next_cache
            self._maybe_build_density(next_cache)
        else:
            # No further month available
            terminated = True
            next_cache = prev_cache
            self._current_cache = prev_cache

        # Prepare observation
        obs, info = self._make_observation_and_info(next_cache)

        # Compute reward (default 1.0) with context
        ctx: Dict[str, Any] = {
            "density_estimator": self._density_estimator,
            "density_k": self._density_k,
            "prev_month": prev_cache.month,
            "next_month": next_cache.month,
        }
        reward: float
        # Prefer class-based reward
        try:
            reward = float(self._reward_obj.update(self._cache_to_state(prev_cache), action, self._cache_to_state(next_cache), ctx))
        except Exception as e:
            logger.warning(f"Reward object update failed, falling back to function: {e}")
            reward = float(self._reward_fn(self._cache_to_state(prev_cache), action, self._cache_to_state(next_cache), ctx))

        return obs, reward, terminated, truncated, info

    def set_reward_fn(self, reward_fn: RewardFn) -> None:
        """Replace reward function at runtime."""
        if not callable(reward_fn):
            raise ValueError("reward_fn must be callable")
        self._reward_fn = reward_fn

    def set_reward_obj(self, reward_obj: ArxivRewardBase) -> None:
        """Replace reward object at runtime."""
        if not isinstance(reward_obj, ArxivRewardBase):
            raise ValueError("reward_obj must be an instance of ArxivRewardBase")
        self._reward_obj = reward_obj

    def months(self) -> List[str]:
        """Return list of available months (YYYYMM)."""
        return list(self._months)

    def current_month(self) -> str:
        """Return current month (YYYYMM)."""
        return self._months[self._month_index]

    # ------------------------------ Internals ------------------------------
    def _default_reward_fn(
        self,
        prev_state: Dict[str, Any],
        action: Sequence[Sequence[float]],
        next_state: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> float:
        # Placeholder reward; replace with custom logic later
        return 1.0

    def _compute_available_months(self) -> List[str]:
        # Aggregate days into months using histogram_by_day
        pairs = self._memory.histogram_by_day()  # List[(YYYYMMDD, count)]
        months: List[str] = []
        seen: set[str] = set()
        for d, _cnt in pairs:
            if isinstance(d, str) and len(d) >= 6:
                mm = d[:6]
                if mm not in seen:
                    seen.add(mm)
                    months.append(mm)
        months.sort()
        return months

    def _load_month_cache(self, yyyymm: str) -> _MonthCache:
        triples = self._memory.get_by_month(yyyymm, categories=self.categories, include_vector=True)
        # Filter and limit
        vecs: List[List[float]] = []
        pids: List[str] = []
        count = 0
        for content, vec, md in triples:
            if not vec:
                continue
            if len(vec) != self.vector_dim:
                # Skip malformed vectors rather than failing hard
                continue
            vecs.append([float(x) for x in vec])
            pid = str(md.get("id", f"{yyyymm}-{len(pids)}"))
            pids.append(pid)
            count += 1
            if self.max_vectors_per_month is not None and count >= self.max_vectors_per_month:
                break

        if not vecs:
            logger.warning(f"Month {yyyymm} contains no valid vectors; substituting empty array with shape (0, {self.vector_dim}).")
            arr = np.empty((0, self.vector_dim), dtype=np.float32)
        else:
            arr = np.asarray(vecs, dtype=np.float32)

        return _MonthCache(month=yyyymm, vectors=arr, paper_ids=pids)

    def _maybe_build_density(self, cache: _MonthCache) -> None:
        # Build density estimator if provided
        if self._density_estimator_factory is None:
            self._density_estimator = None
            return
        try:
            est = self._density_estimator_factory()
            if hasattr(est, "build"):
                est.build(cache.vectors)
            self._density_estimator = est
            logger.info(
                f"Density estimator built for month {cache.month}: size={cache.vectors.shape[0]} dim={cache.vectors.shape[1] if cache.vectors.ndim==2 else 'NA'}"
            )
        except Exception as e:
            logger.warning(f"Failed to build density estimator for month {cache.month}: {e}")
            self._density_estimator = None

    def _make_observation_and_info(self, cache: _MonthCache) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs = {
            "month": cache.month,
            "vectors": cache.vectors,
        }
        info: Dict[str, Any] = {
            "month": cache.month,
            "num_vectors": int(cache.vectors.shape[0]),
            "vector_dim": int(cache.vectors.shape[1]) if cache.vectors.ndim == 2 else self.vector_dim,
            "paper_ids": list(cache.paper_ids),
        }
        if self._density_estimator is not None:
            info["density_estimator"] = type(self._density_estimator).__name__
        return obs, info

    def _cache_to_state(self, cache: _MonthCache) -> Dict[str, Any]:
        return {
            "month": cache.month,
            "vectors": cache.vectors,
        }

    def _validate_action(self, action: Sequence[Sequence[float]]) -> None:
        # Accept lists or np arrays
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("action must be a 2D array-like of shape (n, vector_dim)")
        if arr.shape[1] != self.vector_dim:
            raise ValueError(f"action vector_dim mismatch: expected {self.vector_dim}, got {arr.shape[1]}")


if __name__ == "__main__":
    # Minimal quickstart
    env = ResearchArxivEnv()
    obs, info = env.reset()
    logger.info(f"Reset to month={obs['month']}, vectors={obs['vectors'].shape}")

    # Dummy random action with correct dim
    rng = np.random.default_rng(0)
    terminated = False
    action = rng.normal(size=(8, env.vector_dim)).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    logger.info(
        f"Step -> month={obs['month']}, vectors={obs['vectors'].shape}, reward={reward}, terminated={terminated}, truncated={truncated}"
    )


