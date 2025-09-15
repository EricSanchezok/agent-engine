from __future__ import annotations

"""
Vector density estimation utilities for high-dimensional embeddings.

This module provides two interchangeable estimators:

1) KNNVectorDensityEstimator (exact)
   - Computes density using k-nearest neighbors from the full corpus
   - Suitable for small-to-medium corpora or offline evaluation

2) HNSWVectorDensityEstimator (approximate)
   - Uses hnswlib to accelerate KNN search
   - Suitable for large corpora where exact search is expensive

Density definitions supported (set by `method`):
 - "knn_average":  density = 1.0 / (epsilon + mean_k(distances))
 - "knn_radius":   density = k / (epsilon + r_k)  where r_k is distance to the k-th neighbor

Both estimators default to cosine distance which is appropriate for
embedding vectors like OpenAI text-embedding-3-large.

Notes:
 - If the query vector is taken from the same corpus, set self_exclude=True
   to ignore the zero-distance self neighbor.
 - For reproducibility and stability, distances are clamped to be non-negative.
"""

import math
from typing import List, Sequence, Tuple, Optional
import pyinstrument
import numpy as np

from agent_engine.agent_logger import AgentLogger  # [[memory:8092102]]


logger = AgentLogger(__name__)


def _ensure_2d_array(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("vectors must be a 2D array-like of shape (num_vectors, dim)")
    return arr


def _ensure_1d_array(vector: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("vector must be a 1D array-like of shape (dim,)")
    return arr


def _cosine_distances(query: np.ndarray, corpus: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute cosine distances between query (dim,) and corpus (N, dim).

    Returns an array of shape (N,) with values in [0, 2].
    """
    q = query
    X = corpus
    q_norm = np.linalg.norm(q) + eps
    X_norms = np.linalg.norm(X, axis=1) + eps
    sims = (X @ q) / (X_norms * q_norm)
    dists = 1.0 - sims
    # Clamp for numerical safety
    dists = np.maximum(dists, 0.0)
    return dists.astype(np.float32)


def _l2_distances(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    diffs = corpus - query[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    return dists.astype(np.float32)


def _knn_density_from_dists(
    dists: np.ndarray,
    k: int,
    method: str = "knn_average",
    eps: float = 1e-12,
) -> float:
    if k <= 0:
        return 0.0
    if dists.size == 0:
        return 0.0
    k = int(min(k, dists.size))
    # Use partial selection for efficiency
    part = np.partition(dists, k - 1)[:k]
    if method == "knn_radius":
        r_k = float(np.max(part))
        return float(k) / (eps + r_k)
    # default: knn_average
    mean_k = float(np.mean(part))
    return 1.0 / (eps + mean_k)


class KNNVectorDensityEstimator:
    """Exact KNN-based density estimator using vectorized NumPy operations.

    Parameters
    ----------
    metric : str
        Distance metric: "cosine" (default) or "euclidean".
    method : str
        Density calculation: "knn_average" (default) or "knn_radius".
    eps : float
        Small constant for numerical stability.
    self_exclude : bool
        If True, attempts to drop a zero-distance self neighbor.
    """

    def __init__(
        self,
        *,
        metric: str = "cosine",
        method: str = "knn_average",
        eps: float = 1e-12,
        self_exclude: bool = True,
    ) -> None:
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        if method not in ("knn_average", "knn_radius"):
            raise ValueError("method must be 'knn_average' or 'knn_radius'")
        self.metric = metric
        self.method = method
        self.eps = float(eps)
        self.self_exclude = bool(self_exclude)

    @pyinstrument.profile()
    def estimate_density(
        self,
        query_vector: Sequence[float],
        corpus_vectors: Sequence[Sequence[float]],
        k: int = 50,
    ) -> float:
        q = _ensure_1d_array(query_vector)
        X = _ensure_2d_array(corpus_vectors)

        if X.shape[0] == 0:
            logger.warning("Empty corpus provided to KNNVectorDensityEstimator; returning 0.0 density")
            return 0.0
        if q.shape[0] != X.shape[1]:
            raise ValueError(f"Dimension mismatch: query dim={q.shape[0]} vs corpus dim={X.shape[1]}")

        if self.metric == "cosine":
            dists = _cosine_distances(q, X, eps=self.eps)
        else:
            dists = _l2_distances(q, X)

        # Optionally drop self neighbor (distance ~ 0)
        if self.self_exclude and dists.size > 0:
            # Remove one zero (or near-zero) if present
            zero_mask = dists <= (10.0 * self.eps)
            if np.any(zero_mask):
                dists = dists[~zero_mask]
                if dists.size == 0:
                    return 0.0

        return _knn_density_from_dists(dists, k=k, method=self.method, eps=self.eps)


class HNSWVectorDensityEstimator:
    """Approximate KNN-based density estimator using hnswlib.

    Build once with `build(corpus_vectors)` and call `estimate_density(query_vector, k)`.

    Parameters
    ----------
    metric : str
        Distance metric: "cosine" (default) or "euclidean" (L2).
    method : str
        Density calculation: "knn_average" (default) or "knn_radius".
    eps : float
        Small constant for numerical stability.
    M : int
        HNSW construction parameter controlling graph degree.
    ef_construction : int
        HNSW construction parameter for index building quality.
    ef_search : int
        ef parameter used at query time; larger -> better recall, slower.
    self_exclude : bool
        If True, attempts to drop the zero-distance self neighbor when present.
    """

    def __init__(
        self,
        *,
        metric: str = "cosine",
        method: str = "knn_average",
        eps: float = 1e-12,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        self_exclude: bool = True,
    ) -> None:
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        if method not in ("knn_average", "knn_radius"):
            raise ValueError("method must be 'knn_average' or 'knn_radius'")
        self.metric = metric
        self.method = method
        self.eps = float(eps)
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.self_exclude = bool(self_exclude)

        self._index = None
        self._dim: Optional[int] = None
        self._normalized_for_cosine: bool = False

    def _require_hnsw(self):
        try:
            import hnswlib  # type: ignore
            return hnswlib
        except Exception as e:
            raise ImportError(
                "hnswlib is required for HNSWVectorDensityEstimator. Add it to pyproject.toml and run 'uv sync' to install."
            ) from e

    def build(self, corpus_vectors: Sequence[Sequence[float]]) -> None:
        hnswlib = self._require_hnsw()
        X = _ensure_2d_array(corpus_vectors)
        self._dim = int(X.shape[1])

        space = "cosine" if self.metric == "cosine" else "l2"
        index = hnswlib.Index(space=space, dim=self._dim)
        index.init_index(max_elements=max(16, X.shape[0]), ef_construction=self.ef_construction, M=self.M)

        # For cosine space, L2-normalize inputs for numerical stability/consistency
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True) + self.eps
            Xn = X / norms
            self._normalized_for_cosine = True
        else:
            Xn = X

        labels = np.arange(Xn.shape[0], dtype=np.int64)
        index.add_items(Xn, labels)
        index.set_ef(max(self.ef_search, 64))

        self._index = index
        logger.info(
            f"HNSW index built: size={Xn.shape[0]}, dim={self._dim}, metric={self.metric}, M={self.M}, ef_construction={self.ef_construction}"
        )

    # @pyinstrument.profile()
    def estimate_density(self, query_vector: Sequence[float], k: int = 50) -> float:
        if self._index is None or self._dim is None:
            raise RuntimeError("HNSW index is not built. Call build(corpus_vectors) first.")

        import numpy as _np

        q = _ensure_1d_array(query_vector)
        if q.shape[0] != self._dim:
            raise ValueError(f"Dimension mismatch: query dim={q.shape[0]} vs index dim={self._dim}")

        if self.metric == "cosine":
            qn = q / (float(_np.linalg.norm(q)) + self.eps)
        else:
            qn = q

        # Ensure ef_search is sufficient for k
        try:
            self._index.set_ef(max(int(k) + 16, self.ef_search))  # type: ignore[attr-defined]
        except Exception:
            pass

        labels, dists = self._index.knn_query(qn, k=max(1, int(k)))  # type: ignore[attr-defined]
        # dists shape: (1, k)
        d = np.asarray(dists[0], dtype=np.float32)
        d = np.maximum(d, 0.0)

        # Optionally drop self neighbor (distance ~ 0)
        if self.self_exclude and d.size > 0 and d[0] <= (10.0 * self.eps):
            d = d[1:]
            if d.size == 0:
                return 0.0

        return _knn_density_from_dists(d, k=min(k, d.size), method=self.method, eps=self.eps)


from agents.ResearchAgent.arxiv_memory import ArxivMemory # type: ignore
import random

if __name__ == "__main__":
    mem = ArxivMemory()
    triples = mem.get_by_month("202406")
    vectors = [v for _, v, _ in triples if v]  # 取 index=1 的向量
    query = random.choice(vectors)

    # 1) 精确 KNN
    knn_est = KNNVectorDensityEstimator(metric="cosine", method="knn_average", self_exclude=True)
    density_knn = knn_est.estimate_density(query, vectors, k=100)
    print("KNN density:", density_knn)

    # 2) 近似 HNSW
    hnsw_est = HNSWVectorDensityEstimator(metric="cosine", method="knn_average", self_exclude=True)
    hnsw_est.build(vectors)
    density_hnsw = hnsw_est.estimate_density(query, k=100)
    print("HNSW density:", density_hnsw)