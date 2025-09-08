from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    # Prefer project logger if available
    from agent_engine.agent_logger import logger as _LOGGER  # type: ignore
except Exception:  # pragma: no cover - fallback to std logging if project logger missing
    import logging

    _LOGGER = logging.getLogger("agent_engine.agent_logger_fallback")
    if not _LOGGER.handlers:
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        _handler.setFormatter(_formatter)
        _LOGGER.addHandler(_handler)
        _LOGGER.setLevel(logging.INFO)

try:
    from agent_engine.utils import get_relative_path_from_current_file
except Exception:  # pragma: no cover - keep runtime robust if utils not available
    get_relative_path_from_current_file = None  # type: ignore


class AmbiguousLabelError(ValueError):
    """Raised when a leaf risk name maps to multiple full paths."""


class LabelNotFoundError(KeyError):
    """Raised when a risk label cannot be found in the vocabulary."""


class RiskLabelIndexer:
    """
    Build a stable bidirectional mapping between ICU risks and indices.

    - Loads risks from a hierarchical JSON (systems -> categories -> [risks]).
    - Flattens to deterministic ordered labels using the JSON's insertion order.
    - Provides index_to_risk(index) and risk_to_index(risk) utilities.

    Label format:
        "System/Category/RiskName" (full path, case-sensitive)

    Leaf name lookup:
        If a caller passes a leaf name (e.g., "Sepsis"), we resolve it only if it
        is unique across the entire table; otherwise we raise AmbiguousLabelError
        with candidate suggestions and require the full path.
    """

    def __init__(self, risks_json_path: Optional[Path | str] = None) -> None:
        self._json_path: Path = self._resolve_json_path(risks_json_path)
        self._version_hash: str = self._compute_file_hash(self._json_path)

        labels: List[str] = self._flatten_table(self._json_path)
        self._labels: List[str] = labels
        self._risk_to_index: Dict[str, int] = {label: i for i, label in enumerate(labels)}

        leaf_to_full: Dict[str, List[str]] = {}
        for label in labels:
            leaf = label.split("/")[-1]
            leaf_to_full.setdefault(leaf, []).append(label)
        self._leaf_to_full: Dict[str, List[str]] = leaf_to_full

        _LOGGER.info(
            "RiskLabelIndexer loaded %d risks from %s (sha256=%s)",
            len(self._labels),
            str(self._json_path),
            self._version_hash[:12],
        )

    # -------------------- public API --------------------
    @property
    def size(self) -> int:
        """Total number of risks (vocabulary size)."""
        return len(self._labels)

    @property
    def version_hash(self) -> str:
        """SHA256 of the JSON file content for reproducibility."""
        return self._version_hash

    @property
    def json_path(self) -> Path:
        return self._json_path

    def get_all_labels(self) -> List[str]:
        """Return a copy of all labels in index order."""
        return list(self._labels)

    def get_index_mapping(self) -> Dict[str, int]:
        """Return a copy of risk->index mapping (full-path labels)."""
        return dict(self._risk_to_index)

    def index_to_risk(self, index: int) -> str:
        """Map index -> full-path risk label."""
        if index < 0 or index >= len(self._labels):
            raise IndexError(f"Index out of range: {index}")
        return self._labels[index]

    def risk_to_index(self, risk: str) -> int:
        """
        Map a risk string to index.

        Accepts either a full-path label ("System/Category/Risk") or a leaf risk
        name (e.g., "Sepsis") when it is globally unique.
        """
        # Full-path fast path
        if risk in self._risk_to_index:
            return self._risk_to_index[risk]

        # Leaf fallback if unique
        candidates = self._leaf_to_full.get(risk)
        if candidates is None:
            raise LabelNotFoundError(
                f"Risk label not found: '{risk}'. Provide a full path like 'System/Category/{risk}'."
            )
        if len(candidates) > 1:
            preview = ", ".join(candidates[:5]) + (" ..." if len(candidates) > 5 else "")
            raise AmbiguousLabelError(
                "Leaf risk name is ambiguous across systems/categories: "
                f"'{risk}'. Candidates: {preview}. Please use a full path."
            )
        full = candidates[0]
        return self._risk_to_index[full]

    def reload(self) -> None:
        """Reload risks_table.json and rebuild mappings."""
        self.__init__(self._json_path)

    # -------------------- internals --------------------
    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        data = path.read_bytes()
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _resolve_json_path(risks_json_path: Optional[Path | str]) -> Path:
        # Explicit path provided
        if risks_json_path is not None:
            p = Path(risks_json_path).resolve()
            if not p.exists():
                raise FileNotFoundError(f"risks_table.json not found at provided path: {p}")
            return p

        # Resolve relative to this file's directory via project utility
        if get_relative_path_from_current_file is not None:
            p = get_relative_path_from_current_file("risks_table.json").resolve()
            if p.exists():
                return p

        # Fallback: use __file__ directory
        here = Path(__file__).resolve().parent
        p = (here / "risks_table.json").resolve()
        if p.exists():
            return p

        raise FileNotFoundError(
            f"risks_table.json not found next to {__file__}. Expected at: {p}"
        )

    @staticmethod
    def _flatten_table(json_path: Path) -> List[str]:
        data = json.loads(json_path.read_text(encoding="utf-8"))

        labels: List[str] = []

        def walk(node, prefix: Tuple[str, ...]):
            if isinstance(node, dict):
                for key, value in node.items():  # preserve insertion order
                    walk(value, prefix + (key,))
            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, str):
                        labels.append("/".join(prefix + (item,)))
                    else:
                        walk(item, prefix)
            else:
                # Unexpected leaf; ignore silently to be robust
                pass

        walk(data, ())

        if not labels:
            raise ValueError("No risks found after flattening risks_table.json")

        # Deduplicate while preserving order
        seen: Dict[str, None] = {}
        unique_labels: List[str] = []
        for lbl in labels:
            if lbl not in seen:
                seen[lbl] = None
                unique_labels.append(lbl)

        return unique_labels


if __name__ == "__main__":
    indexer = RiskLabelIndexer()
    print(indexer.get_all_labels())
    print(indexer.get_index_mapping())
    print(indexer.index_to_risk(0))
    print(indexer.risk_to_index("Sepsis"))
    print(indexer.index_to_risk(1))
    print(indexer.risk_to_index("Cardiogenic Shock"))