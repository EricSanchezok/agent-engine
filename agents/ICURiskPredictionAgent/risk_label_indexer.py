from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from difflib import SequenceMatcher

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
            parts = label.split("/")
            # Always use the last part as the leaf name
            leaf = parts[-1]
            leaf_to_full.setdefault(leaf, []).append(label)
        self._leaf_to_full: Dict[str, List[str]] = leaf_to_full

        # Build additional lookup structures for robust matching
        self._normalized_risk_to_index: Dict[str, int] = {}
        self._risk_words_to_index: Dict[str, List[int]] = {}
        self._build_robust_lookups()

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
        Map a risk string to index with robust fallback strategies.

        Accepts:
        - Full-path label ("System/Category/Risk")
        - Leaf risk name (e.g., "Sepsis") when globally unique
        - Partial path matching
        - Fuzzy matching for similar names
        """
        # 1. Full-path fast path
        if risk in self._risk_to_index:
            return self._risk_to_index[risk]

        # 2. Normalized matching (case-insensitive, whitespace normalized)
        normalized_risk = self._normalize_text(risk)
        if normalized_risk in self._normalized_risk_to_index:
            return self._normalized_risk_to_index[normalized_risk]

        # 3. Extract last part of path and try leaf matching
        last_part = self._extract_last_path_component(risk)
        if last_part != risk:  # Only if we extracted something different
            try:
                return self._try_leaf_matching(last_part)
            except (AmbiguousLabelError, LabelNotFoundError):
                pass  # Continue to next strategy

        # 4. Try leaf matching with original risk
        try:
            return self._try_leaf_matching(risk)
        except (AmbiguousLabelError, LabelNotFoundError):
            pass  # Continue to next strategy

        # 5. Fuzzy matching
        fuzzy_match = self._fuzzy_match(risk)
        if fuzzy_match is not None:
            _LOGGER.warning(f"Fuzzy matched '{risk}' to '{self._labels[fuzzy_match]}'")
            return fuzzy_match

        # 6. Word-based partial matching
        word_match = self._word_based_match(risk)
        if word_match is not None:
            _LOGGER.warning(f"Word-based matched '{risk}' to '{self._labels[word_match]}'")
            return word_match

        # 7. Final fallback - raise error with suggestions
        suggestions = self._get_similar_risks(risk)
        suggestion_text = f" Similar risks: {', '.join(suggestions[:3])}" if suggestions else ""
        raise LabelNotFoundError(
            f"Risk label not found: '{risk}'.{suggestion_text} "
            f"Provide a full path like 'System/Category/{risk}'."
        )

    def reload(self) -> None:
        """Reload risks_table.json and rebuild mappings."""
        self.__init__(self._json_path)

    # -------------------- internals --------------------
    def _build_robust_lookups(self) -> None:
        """Build additional lookup structures for robust matching."""
        # Normalized lookup (case-insensitive, whitespace normalized)
        for i, label in enumerate(self._labels):
            normalized = self._normalize_text(label)
            if normalized not in self._normalized_risk_to_index:
                self._normalized_risk_to_index[normalized] = i
        
        # Word-based lookup for partial matching
        for i, label in enumerate(self._labels):
            words = self._extract_words(label)
            for word in words:
                if len(word) >= 3:  # Only index words with 3+ characters
                    self._risk_words_to_index.setdefault(word, []).append(i)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for case-insensitive matching."""
        return re.sub(r'\s+', ' ', text.strip().lower())

    def _extract_last_path_component(self, risk: str) -> str:
        """Extract the last component from a path-like string."""
        if '/' in risk:
            return risk.split('/')[-1].strip()
        return risk

    def _try_leaf_matching(self, risk: str) -> int:
        """Try to match using leaf name logic (original behavior)."""
        candidates = self._leaf_to_full.get(risk)
        if candidates is None:
            raise LabelNotFoundError(f"Risk label not found: '{risk}'")
        if len(candidates) > 1:
            preview = ", ".join(candidates[:5]) + (" ..." if len(candidates) > 5 else "")
            raise AmbiguousLabelError(
                "Leaf risk name is ambiguous across systems/categories: "
                f"'{risk}'. Candidates: {preview}. Please use a full path."
            )
        full = candidates[0]
        return self._risk_to_index[full]

    def _fuzzy_match(self, risk: str, threshold: float = 0.8) -> Optional[int]:
        """Find the best fuzzy match for a risk name."""
        best_match = None
        best_score = 0.0
        
        normalized_risk = self._normalize_text(risk)
        
        for i, label in enumerate(self._labels):
            # Try matching against full path
            full_score = SequenceMatcher(None, normalized_risk, self._normalize_text(label)).ratio()
            
            # Try matching against leaf name
            leaf_name = self._extract_last_path_component(label)
            leaf_score = SequenceMatcher(None, normalized_risk, self._normalize_text(leaf_name)).ratio()
            
            # Use the better score
            score = max(full_score, leaf_score)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = i
        
        return best_match

    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from text."""
        # Split on common delimiters and filter out short words
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if len(w) >= 2]

    def _word_based_match(self, risk: str) -> Optional[int]:
        """Find match based on word overlap."""
        risk_words = set(self._extract_words(risk))
        if not risk_words:
            return None
        
        best_match = None
        best_overlap = 0
        
        for i, label in enumerate(self._labels):
            label_words = set(self._extract_words(label))
            overlap = len(risk_words.intersection(label_words))
            
            # Require at least 50% word overlap
            if overlap > 0 and overlap >= len(risk_words) * 0.5:
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = i
        
        return best_match

    def _get_similar_risks(self, risk: str, limit: int = 5) -> List[str]:
        """Get similar risk names for error messages."""
        similarities = []
        normalized_risk = self._normalize_text(risk)
        
        for label in self._labels:
            # Calculate similarity with both full path and leaf name
            full_score = SequenceMatcher(None, normalized_risk, self._normalize_text(label)).ratio()
            leaf_name = self._extract_last_path_component(label)
            leaf_score = SequenceMatcher(None, normalized_risk, self._normalize_text(leaf_name)).ratio()
            
            score = max(full_score, leaf_score)
            if score > 0.3:  # Lower threshold for suggestions
                similarities.append((score, label))
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [label for _, label in similarities[:limit]]

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
    print(len(indexer.get_all_labels()))
    # print(indexer.get_index_mapping())
    # print(indexer.index_to_risk(0))
    # print(indexer.risk_to_index("Sepsis"))
    # print(indexer.index_to_risk(1))
    # print(indexer.risk_to_index("Cardiogenic Shock"))