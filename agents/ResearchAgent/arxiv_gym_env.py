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
import time
import asyncio
from functools import lru_cache

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
    timestamp: float = 0.0  # Cache timestamp for TTL


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
        month_range: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Initialize the environment.

        Args:
            categories: Optional category filter for memory.get_by_month.
            vector_dim: Expected vector dimensionality for both state and action.
            max_months: Optional cap on number of months to iterate (starting from the earliest).
            max_vectors_per_month: Optional cap on number of vectors loaded per month (head-only).
            include_content: Whether to keep paper content in info (can be large). Default False.
            reward_fn: Optional callable(prev_state, action, next_state, ctx) -> float.
            month_range: Optional tuple of (start_month, end_month) in 'YYYYMM' format to limit the range.
        """
        self.categories = list(categories) if categories else None
        self.vector_dim = int(vector_dim)
        self.max_months = int(max_months) if max_months is not None else None
        self.max_vectors_per_month = int(max_vectors_per_month) if max_vectors_per_month is not None else None
        self.include_content = bool(include_content)
        self.month_range = month_range
        # Reward: prefer object if provided; otherwise fallback to function or constant
        self._reward_fn: RewardFn = reward_fn if reward_fn is not None else self._default_reward_fn
        self._reward_obj: ArxivRewardBase = reward_obj if reward_obj is not None else ConstantReward(1.0)

        # Storage
        self._memory = ArxivMemory()
        self._months: List[str] = self._compute_available_months()
        if self.max_months is not None:
            self._months = self._months[: self.max_months]

        if not self._months:
            raise RuntimeError("No available months found in ArxivMemory.")

        self._month_index: int = 0
        self._current_cache: Optional[_MonthCache] = None
        
        # Cache configuration
        self._cache_ttl: float = 300.0  # 5 minutes cache TTL
        self._month_cache: Dict[str, _MonthCache] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        
        # Preloading configuration
        self._preload_enabled: bool = True
        self._preload_next_n: int = 2  # Preload next 2 months
        self._preload_task: Optional[asyncio.Task] = None
        self._preload_all: bool = True  # Preload all months at startup
        
        # Performance monitoring
        self._performance_stats: Dict[str, Any] = {
            "step_count": 0,
            "total_step_time": 0.0,
            "total_load_time": 0.0,
            "total_reward_time": 0.0,
            "avg_step_time": 0.0,
            "avg_load_time": 0.0,
            "avg_reward_time": 0.0
        }

        logger.info(
            f"ResearchArxivEnv initialized: months={len(self._months)}, vector_dim={self.vector_dim}, categories={self.categories}, month_range={self.month_range}"
        )
        
        # Preload all months if enabled
        if self._preload_all:
            logger.info("Preloading all months...")
            self._preload_all_months()

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
        obs, info = self._make_observation_and_info(self._current_cache)
        # Reset reward object state
        ctx_reset: Dict[str, Any] = {
            "next_month": self._current_cache.month,
        }
        try:
            self._reward_obj.reset(self._cache_to_state(self._current_cache), ctx_reset)
        except Exception as e:
            logger.warning(f"Reward object reset failed: {e}")
        return obs, info

    # @pyinstrument.profile()
    def step(self, action: Sequence[Sequence[float]]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance to the next month.

        Args:
            action: array-like of shape (n, vector_dim).

        Returns:
            observation, reward, terminated, truncated, info
        """
        step_start_time = time.time()
        self._validate_action(action)

        prev_cache = self._current_cache or self._load_month_cache(self._months[self._month_index])

        # Prepare next state
        terminated = False
        truncated = False
        if self._month_index + 1 < len(self._months):
            self._month_index += 1
            next_cache = self._load_month_cache(self._months[self._month_index])
            self._current_cache = next_cache
            
            # Trigger preloading for next months
            self._trigger_preload()
        else:
            # No further month available
            terminated = True
            next_cache = prev_cache
            self._current_cache = prev_cache

        # Prepare observation
        obs, info = self._make_observation_and_info(next_cache)

        # Compute reward (default 1.0) with context
        reward_start_time = time.time()
        ctx: Dict[str, Any] = {
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
        
        # Update performance statistics
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        reward_time = step_end_time - reward_start_time
        
        self._performance_stats["step_count"] += 1
        self._performance_stats["total_step_time"] += step_time
        self._performance_stats["total_reward_time"] += reward_time
        
        # Update averages
        count = self._performance_stats["step_count"]
        self._performance_stats["avg_step_time"] = self._performance_stats["total_step_time"] / count
        self._performance_stats["avg_reward_time"] = self._performance_stats["total_reward_time"] / count

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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cached_months": len(self._month_cache),
            "cache_ttl": self._cache_ttl
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._month_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Cache cleared")
    
    def _cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, cache_entry in self._month_cache.items()
            if current_time - cache_entry.timestamp >= self._cache_ttl
        ]
        for key in expired_keys:
            del self._month_cache[key]
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _preload_all_months(self) -> None:
        """Preload all months at startup for maximum performance."""
        logger.info(f"Preloading {len(self._months)} months...")
        start_time = time.time()
        
        for i, month in enumerate(self._months):
            if month not in self._month_cache:
                try:
                    cache_entry = self._load_month_cache(month)
                    logger.info(f"Preloaded month {month} ({i+1}/{len(self._months)})")
                except Exception as e:
                    logger.warning(f"Failed to preload month {month}: {e}")
        
        preload_time = time.time() - start_time
        logger.info(f"Preloading completed in {preload_time:.2f}s, cached {len(self._month_cache)} months")
    
    async def _preload_next_months(self) -> None:
        """Preload next N months in background."""
        if not self._preload_enabled:
            return
        
        current_idx = self._month_index
        next_indices = range(current_idx + 1, min(current_idx + 1 + self._preload_next_n, len(self._months)))
        
        for idx in next_indices:
            month = self._months[idx]
            if month not in self._month_cache:
                try:
                    # Load in background without blocking
                    cache_entry = self._load_month_cache(month)
                    logger.debug(f"Preloaded month {month}")
                except Exception as e:
                    logger.warning(f"Failed to preload month {month}: {e}")
    
    def _trigger_preload(self) -> None:
        """Trigger background preloading."""
        if self._preload_enabled and self._preload_task is None:
            try:
                loop = asyncio.get_event_loop()
                self._preload_task = loop.create_task(self._preload_next_months())
            except RuntimeError:
                # No event loop running, skip preloading
                pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Return comprehensive performance statistics."""
        stats = self._performance_stats.copy()
        stats.update(self.get_cache_stats())
        return stats
    
    def reset_performance_stats(self) -> None:
        """Reset all performance statistics."""
        self._performance_stats = {
            "step_count": 0,
            "total_step_time": 0.0,
            "total_load_time": 0.0,
            "total_reward_time": 0.0,
            "avg_step_time": 0.0,
            "avg_load_time": 0.0,
            "avg_reward_time": 0.0
        }
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Performance statistics reset")

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
        
        # Apply month range filter if specified
        if self.month_range is not None:
            start_month, end_month = self.month_range
            if not (isinstance(start_month, str) and isinstance(end_month, str) and 
                    len(start_month) == 6 and len(end_month) == 6 and 
                    start_month.isdigit() and end_month.isdigit()):
                raise ValueError("month_range must be a tuple of two 6-digit strings (YYYYMM)")
            
            if start_month > end_month:
                raise ValueError("start_month must be <= end_month")
            
            # Filter months within the specified range
            filtered_months = [mm for mm in months if start_month <= mm <= end_month]
            logger.info(f"Filtered months from {len(months)} to {len(filtered_months)} using range {start_month}-{end_month}")
            return filtered_months
        
        return months

    def _load_month_cache(self, yyyymm: str) -> _MonthCache:
        # Check cache first
        current_time = time.time()
        if yyyymm in self._month_cache:
            cached = self._month_cache[yyyymm]
            if current_time - cached.timestamp < self._cache_ttl:
                self._cache_hits += 1
                return cached
            else:
                # Cache expired, remove it
                del self._month_cache[yyyymm]
        
        self._cache_misses += 1
        
        # Load from database with timing
        load_start_time = time.time()
        triples = self._memory.get_by_month(yyyymm, categories=self.categories, include_vector=True)
        
        # Pre-allocate arrays for better performance
        max_items = self.max_vectors_per_month or len(triples)
        vecs: List[List[float]] = []
        pids: List[str] = []
        
        count = 0
        for content, vec, md in triples:
            if not vec:
                continue
            if len(vec) != self.vector_dim:
                # Skip malformed vectors rather than failing hard
                continue
            
            # Optimize vector conversion - avoid unnecessary conversions
            if isinstance(vec, np.ndarray):
                # If already numpy array, just ensure correct dtype
                if vec.dtype != np.float32:
                    vec_float = vec.astype(np.float32).tolist()
                else:
                    vec_float = vec.tolist()
            else:
                # Convert list to numpy array first for efficiency
                vec_array = np.array(vec, dtype=np.float32)
                vec_float = vec_array.tolist()
            vecs.append(vec_float)
            
            pid = str(md.get("id", f"{yyyymm}-{len(pids)}"))
            pids.append(pid)
            count += 1
            if self.max_vectors_per_month is not None and count >= self.max_vectors_per_month:
                break

        if not vecs:
            logger.warning(f"Month {yyyymm} contains no valid vectors; substituting empty array with shape (0, {self.vector_dim}).")
            arr = np.empty((0, self.vector_dim), dtype=np.float32)
        else:
            # Use more efficient array creation - allow copy when needed
            arr = np.asarray(vecs, dtype=np.float32)

        # Create cache entry
        cache_entry = _MonthCache(month=yyyymm, vectors=arr, paper_ids=pids, timestamp=current_time)
        self._month_cache[yyyymm] = cache_entry
        
        # Update load time statistics
        load_time = time.time() - load_start_time
        self._performance_stats["total_load_time"] += load_time
        if self._performance_stats["step_count"] > 0:
            self._performance_stats["avg_load_time"] = self._performance_stats["total_load_time"] / self._performance_stats["step_count"]
        
        # Clean up expired cache entries periodically
        if len(self._month_cache) > 10:  # Only clean when cache is large
            self._cleanup_expired_cache()
        
        return cache_entry

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
    env = ResearchArxivEnv(month_range=("202201", "202205"))
    obs, info = env.reset()
    logger.info(f"Reset to month={obs['month']}, vectors={obs['vectors'].shape}")

    # Dummy random action with correct dim
    rng = np.random.default_rng(0)
    terminated = False
    while not terminated:
        action = rng.normal(size=(8, env.vector_dim)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(
            f"Step -> month={obs['month']}, vectors={obs['vectors'].shape}, reward={reward}, terminated={terminated}, truncated={truncated}"
        )


