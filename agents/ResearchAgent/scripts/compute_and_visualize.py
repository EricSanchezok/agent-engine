from __future__ import annotations
import json
from pathlib import Path
from pickle import TRUE
from typing import Dict, List, Tuple

import numpy as np

from agent_engine.agent_logger import AgentLogger
from agents.ResearchAgent.arxiv_memory import ArxivMemory


logger = AgentLogger(__name__)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / norms


def load_models(base_dir: Path):
    model_dir = base_dir / "dimred_model"
    pca_path = model_dir / "pca.pkl"
    umap_path = model_dir / "umap.pkl"
    meta_path = model_dir / "meta.json"
    if not pca_path.exists() or not umap_path.exists():
        raise FileNotFoundError("PCA/UMAP models not found. Run scripts/train_dimred.py first.")
    import pickle

    with pca_path.open("rb") as f:
        pca = pickle.load(f)
    with umap_path.open("rb") as f:
        um = pickle.load(f)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return pca, um, meta


def compute_month_embedding(mem: ArxivMemory, month: str, *, include_vector: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    triples = mem.get_by_month(month, include_vector=include_vector)
    vectors = [v for _, v, _ in triples if v]
    metas = [md for _, v, md in triples if v]
    if not vectors:
        return np.empty((0, 0), dtype=np.float32), []
    X = np.asarray(vectors, dtype=np.float32)
    X = l2_normalize_rows(X)
    return X, metas


def _list_segments(base_dir: Path) -> List[str]:
    try:
        out: List[str] = []
        for p in (base_dir).iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if (len(name) == 6 and name[:4].isdigit() and name[4] == 'H' and name[5] in ('1', '2')) or name == "undated":
                out.append(name)
        out.sort()
        return out
    except Exception:
        return []


def _months_for_segment(seg_key: str) -> List[str]:
    if seg_key == "undated":
        return []
    year = int(seg_key[:4])
    half = seg_key[4:]
    return [f"{year}{m:02d}" for m in (range(1, 7) if half == "H1" else range(7, 13))]


def ensure_month_cache(base_dir: Path, month: str, *, overwrite: bool, pca, um, mem: ArxivMemory) -> Path:
    month_dir = base_dir / "monthly_embeddings"
    month_dir.mkdir(parents=True, exist_ok=True)
    out_path = month_dir / f"{month}.npz"
    if out_path.exists() and not overwrite:
        return out_path

    X, metas = compute_month_embedding(mem, month, include_vector=True)
    if X.size == 0:
        np.savez_compressed(out_path, Y=np.empty((0, um.n_components), dtype=np.float32), meta=np.array([], dtype=object))
        logger.info(f"Month {month}: no vectors. Saved empty embedding.")
        return out_path

    X50 = pca.transform(X)
    Y = um.transform(X50).astype(np.float32, copy=False)
    np.savez_compressed(out_path, Y=Y, meta=np.array(metas, dtype=object))
    logger.info(f"Month {month}: embedded {Y.shape[0]} items -> saved to {str(out_path)}")
    return out_path


def build_plot(month_to_path: Dict[str, Path], *, output_html: Path) -> None:
    # Lazy import plotly to avoid hard dependency at runtime
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception:
        logger.error("plotly is required. Add 'plotly' to pyproject optional deps and 'uv sync'.")
        return

    frames = []
    first_fig = None
    months_sorted = sorted(month_to_path.keys())
    for i, m in enumerate(months_sorted):
        data = np.load(month_to_path[m], allow_pickle=True)
        Y = data["Y"]
        meta = data["meta"].tolist() if "meta" in data else []
        if Y.size == 0:
            scatter = go.Scatter(x=[], y=[], mode="markers", marker=dict(size=4, color=[]))
        else:
            x = Y[:, 0]
            y = Y[:, 1] if Y.shape[1] >= 2 else np.zeros_like(Y[:, 0])
            color = None
            scatter = go.Scatter(x=x, y=y, mode="markers", marker=dict(size=4, color=color, colorscale="Viridis"), name=m)
        frame = go.Frame(data=[scatter], name=m)
        frames.append(frame)
        if first_fig is None:
            first_fig = go.Figure(data=[scatter])

    if first_fig is None:
        logger.warning("No frames to visualize.")
        return

    first_fig.update(frames=frames)
    first_fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 500, "redraw": False}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
            ]
        }],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Month: "},
            "steps": [{"label": m, "method": "animate", "args": [[m], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]} for m in months_sorted]
        }],
        title="Monthly UMAP Embeddings",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        width=1000,
        height=800,
    )
    first_fig.write_html(str(output_html), include_plotlyjs="cdn")
    logger.info(f"Saved interactive visualization to {str(output_html)}")


def main() -> None:
    # ---- In-code configuration (edit here) ----
    PROCESS_ALL_SEGMENTS: bool = True         # True: process all segments in database
    SEGMENT: str = "2024H1"                  # used only if PROCESS_ALL_SEGMENTS=False
    MONTHS: List[str] = []                    # manual months list; empty -> auto-discover per selection
    PROBE_MONTHS: bool = True                 # only include months that have at least 1 item
    OVERWRITE: bool = TRUE                   # recompute monthly cache if True
    OUTPUT_HTML: str = "monthly_embeddings.html"
    # -------------------------------------------

    mem = ArxivMemory()
    base_dir = Path(__file__).resolve().parent.parent / "database"

    # Load models
    pca, um, meta = load_models(base_dir)

    # Discover months if not provided
    months = MONTHS
    if not months:
        if PROCESS_ALL_SEGMENTS:
            seg_keys = _list_segments(base_dir)
            months_all: List[str] = []
            for sk in seg_keys:
                months_all.extend(_months_for_segment(sk))
            months = sorted(set(months_all))
        else:
            seg = SEGMENT
            year = int(seg[:4])
            half = seg[4:]
            months = [f"{year}{m:02d}" for m in (range(1,7) if half == "H1" else range(7,13))]

    if PROBE_MONTHS:
        filtered: List[str] = []
        for m in months:
            # fast probe: limit=1 and no vector
            try:
                triples = mem.get_by_month(m, include_vector=False, limit=1)
                if triples:
                    filtered.append(m)
            except Exception:
                # if error, keep month to avoid false negatives
                filtered.append(m)
        months = filtered

    # Compute/cache
    month_to_path: Dict[str, Path] = {}
    for m in months:
        path = ensure_month_cache(base_dir, m, overwrite=OVERWRITE, pca=pca, um=um, mem=mem)
        month_to_path[m] = path

    # Visualize
    output_html = (base_dir / OUTPUT_HTML)
    build_plot(month_to_path, output_html=output_html)


if __name__ == "__main__":
    main()
