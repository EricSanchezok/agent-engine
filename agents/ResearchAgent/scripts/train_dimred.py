from __future__ import annotations
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

from agent_engine.agent_logger import AgentLogger
from agents.ResearchAgent.arxiv_memory import ArxivMemory


logger = AgentLogger(__name__)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / norms


def estimate_pca_components_for_variance(X: np.ndarray, target_variance: float, *, random_state: int = 42) -> Tuple[int, float]:
    """Return minimal k s.t. cumulative explained variance >= target_variance.

    Returns (k, total_explained), where total_explained is sum of all components' explained_variance_ratio_.
    """
    from sklearn.decomposition import PCA

    tv = float(max(0.0, min(1.0, target_variance)))
    # Use full SVD to get accurate spectrum on sampled data
    pca_full = PCA(n_components=None, svd_solver="full", random_state=random_state)
    XPCA = pca_full.fit_transform(X)
    evr = np.asarray(pca_full.explained_variance_ratio_, dtype=np.float64)
    csum = np.cumsum(evr)
    k = int(np.searchsorted(csum, tv) + 1)
    k = int(max(1, min(k, evr.size)))
    total = float(float(evr.sum()))
    logger.info(f"PCA variance analysis: target={tv:.3f}, k@target={k}, total_explained={total:.4f}")
    # Also log some reference points
    for thr in (0.80, 0.90, 0.95, 0.99):
        kk = int(np.searchsorted(csum, thr) + 1)
        if kk <= evr.size:
            logger.info(f"  k@{thr:.2f} = {kk}")
    return k, total


def _list_segments(base_dir: Path) -> List[str]:
    try:
        out = []
        for p in base_dir.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if len(name) == 6 and name[:4].isdigit() and name[4] == 'H' and name[5] in ('1', '2'):
                out.append(name)
        out.sort()
        return out
    except Exception:
        return []


def _open_memory_for_segment(base_dir: Path, seg_key: str):
    from agent_engine.memory import ScalableMemory

    seg_dir = base_dir / seg_key
    mem = ScalableMemory(
        name="arxiv_metadata",
        llm_client=None,
        enable_vectors=True,
        persist_dir=str(seg_dir),
    )
    return mem


def _sample_vectors_from_segment(base_dir: Path, seg_key: str, sample_ratio: float, max_per_segment: int) -> np.ndarray:
    mem = _open_memory_for_segment(base_dir, seg_key)
    try:
        total = mem.count()
    except Exception:
        total = 0
    if total <= 0:
        return np.empty((0, 0), dtype=np.float32)

    k = max(1, int(min(total, math.ceil(total * sample_ratio), max_per_segment)))

    # Use ORDER BY RANDOM() LIMIT k to fetch random subset; works on both DuckDB and SQLite
    try:
        cur = mem.db.execute("SELECT vector FROM items WHERE vector IS NOT NULL ORDER BY RANDOM() LIMIT ?", (int(k),))
        rows = mem.db.fetchall(cur)
    except Exception:
        # Fallback: sequential LIMIT
        cur = mem.db.execute("SELECT vector FROM items WHERE vector IS NOT NULL LIMIT ?", (int(k),))
        rows = mem.db.fetchall(cur)

    vecs: List[np.ndarray] = []
    for r in rows:
        vblob = r[0]
        if vblob is None:
            continue
        try:
            vec = np.frombuffer(vblob, dtype=np.float32)
            if vec.size == 0:
                continue
            vecs.append(vec)
        except Exception:
            continue

    if not vecs:
        return np.empty((0, 0), dtype=np.float32)

    # Ensure consistent dimension
    dim = int(vecs[0].size)
    out = np.vstack([v.astype(np.float32, copy=False) for v in vecs if v.size == dim])
    return out


def main() -> None:
    # ---- In-code configuration (edit here) ----
    SAMPLE_RATIO: float = 1.0
    MAX_PER_SEGMENT: int = 200000
    PCA_COMPONENTS: int = 50
    AUTO_TUNE_PCA: bool = True
    TARGET_VARIANCE: float = 0.95
    UMAP_COMPONENTS: int = 2   # 2 or 3
    UMAP_NEIGHBORS: int = 100
    UMAP_MIN_DIST: float = 0.1
    RANDOM_STATE: int = 42
    OVERWRITE: bool = True
    # -------------------------------------------

    if not (0.0 < SAMPLE_RATIO <= 1.0):
        raise ValueError("sample_ratio must be in (0,1]")
    if UMAP_COMPONENTS not in (2, 3):
        raise ValueError("umap_components must be 2 or 3")

    # Initialize to get base paths consistent with agent setup
    _ = ArxivMemory()  # ensure environment/logs are configured

    base_dir = Path(__file__).resolve().parent.parent / "database"
    model_dir = base_dir / "dimred_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    pca_path = model_dir / "pca.pkl"
    umap_path = model_dir / "umap.pkl"
    meta_path = model_dir / "meta.json"

    if (pca_path.exists() or umap_path.exists()) and not OVERWRITE:
        logger.info("Model files already exist and --overwrite not set; exiting.")
        return

    seg_keys = _list_segments(base_dir)
    if not seg_keys:
        logger.warning("No segment directories found under ResearchAgent/database/. Nothing to train.")
        return

    # Collect samples
    all_vecs: List[np.ndarray] = []
    total_picked = 0
    for seg in seg_keys:
        Xseg = _sample_vectors_from_segment(base_dir, seg, SAMPLE_RATIO, MAX_PER_SEGMENT)
        if Xseg.size == 0:
            logger.info(f"Segment {seg} yielded 0 vectors; skipping.")
            continue
        all_vecs.append(Xseg)
        total_picked += Xseg.shape[0]
        logger.info(f"Segment {seg}: picked {Xseg.shape[0]} vectors.")

    if not all_vecs:
        logger.warning("No vectors sampled from any segment; aborting.")
        return

    X = np.vstack(all_vecs).astype(np.float32, copy=False)
    X = l2_normalize_rows(X)
    dim = int(X.shape[1])
    logger.info(f"Training PCA+UMAP on {X.shape[0]} vectors with dim={dim}.")

    # PCA
    from sklearn.decomposition import PCA

    n_comp = int(min(PCA_COMPONENTS, X.shape[1]))
    if AUTO_TUNE_PCA:
        try:
            k_opt, total_evr = estimate_pca_components_for_variance(X, TARGET_VARIANCE, random_state=RANDOM_STATE)
            n_comp = int(min(max(2, k_opt), X.shape[1]))
            logger.info(f"AUTO_TUNE_PCA: using n_components={n_comp} to target variance>={TARGET_VARIANCE:.2f}")
        except Exception as e:
            logger.warning(f"AUTO_TUNE_PCA failed, fallback to PCA_COMPONENTS={PCA_COMPONENTS}: {e}")
            n_comp = int(min(PCA_COMPONENTS, X.shape[1]))
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    XPCA = pca.fit_transform(X)
    logger.info(f"PCA trained. Explained variance ratio sum={float(np.sum(pca.explained_variance_ratio_)):.4f}")

    # UMAP
    try:
        import umap  # type: ignore
    except Exception as e:
        logger.error("umap-learn is required. Please add 'umap-learn' to pyproject and run 'uv sync'.")
        return

    um = umap.UMAP(
        n_components=int(UMAP_COMPONENTS),
        n_neighbors=int(UMAP_NEIGHBORS),
        min_dist=float(UMAP_MIN_DIST),
        metric="euclidean",
        n_jobs=-1,
        verbose=True,
    )
    um.fit(XPCA)
    logger.info("UMAP trained.")

    # Persist models via pickle
    import pickle

    with pca_path.open("wb") as f:
        pickle.dump(pca, f)
    with umap_path.open("wb") as f:
        pickle.dump(um, f)
    meta = {
        "vector_dim": dim,
        "pca_components": int(n_comp),
        "umap_components": int(UMAP_COMPONENTS),
        "umap_neighbors": int(UMAP_NEIGHBORS),
        "umap_min_dist": float(UMAP_MIN_DIST),
        "random_state": int(RANDOM_STATE),
        "segments": seg_keys,
        "total_vectors": int(X.shape[0]),
        "sample_ratio": float(SAMPLE_RATIO),
        "max_per_segment": int(MAX_PER_SEGMENT),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"Saved PCA/UMAP models and meta under {str(model_dir)}")


if __name__ == "__main__":
    main()


