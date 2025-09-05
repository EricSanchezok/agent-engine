from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent_engine.agent_logger.agent_logger import AgentLogger
from core.umls.umls_client import (
    UMLSClient,
    UMLSClientError,
)


_DEFAULT_SABS: Tuple[str, ...] = (
    "SNOMEDCT_US",
    "RXNORM",
    "LNC",
)


def _lower_ascii(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*|[A-Za-z]+\/[A-Za-z]+")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "with",
        "for",
        "by",
        "at",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "being",
        "been",
        "patient",
        "female",
        "male",
        "days",
        "day",
        "no",
        "yes",
        "normal",
        "abnormal",
    }
)


@dataclass(frozen=True)
class LinkedConcept:
    cui: str
    name: str
    semantic_types: List[Dict[str, Any]]
    root_source: Optional[str] = None


class UMLSConceptExtractor:
    """Lightweight UMLS concept extractor.

    Steps:
    - Generate phrase candidates with n-grams from text
    - Query UMLS search to link phrases to CUIs
    - Optionally fetch CUI details for semantic types (STYs)

    Notes:
    - This is a pragmatic extractor to bootstrap v2; you can later swap the
      candidate generator with scispaCy/QuickUMLS/SapBERT to improve accuracy.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        sabs: Iterable[str] = _DEFAULT_SABS,
        search_type: str = "words",
        allowed_sty_names: Optional[Iterable[str]] = None,
        logger: Optional[AgentLogger] = None,
    ) -> None:
        self._logger = logger or AgentLogger(self.__class__.__name__)
        self._api_key = api_key or os.getenv("UMLS_API_KEY", "").strip()
        if not self._api_key:
            raise ValueError("UMLS_API_KEY is required for concept extraction")

        self._client = UMLSClient(api_key=self._api_key)
        self._sabs = tuple(s.strip() for s in sabs if s and s.strip())
        self._search_type = search_type
        self._allowed_sty: Optional[frozenset[str]] = (
            frozenset(s.strip() for s in allowed_sty_names)
            if allowed_sty_names
            else frozenset(
                {
                    "Pharmacologic Substance",
                    "Clinical Drug",
                    "Organic Chemical",
                    "Inorganic Chemical",
                    "Therapeutic or Preventive Procedure",
                    "Health Care Activity",
                    "Laboratory Procedure",
                    "Diagnostic Procedure",
                }
            )
        )

    def extract(
        self,
        text: str,
        *,
        max_terms: int = 24,
        max_ngram: int = 5,
        min_char_len: int = 3,
        fetch_details: bool = True,
        max_cui_details: int = 64,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Extract UMLS concepts from free text.

        Returns a dict with two sections:
        - mentions: list of {text, span, linked{cui,name,semantic_types,root_source}}
        - concepts: aggregated bag of unique CUIs with weights and vocab sources
        """
        if not isinstance(text, str) or not text.strip():
            return {"mentions": [], "concepts": []}

        original = text
        lowered = _lower_ascii(text)
        tokens = [t for t in _TOKEN_RE.findall(lowered) if t not in _STOPWORDS]
        if not tokens:
            return {"mentions": [], "concepts": [], "debug": {"reason": "no_tokens"} if debug else {}}

        candidates = self._generate_candidates(tokens, max_ngram=max_ngram, min_char_len=min_char_len)
        ranked_phrases = self._rank_candidates(candidates, max_terms=max_terms)

        mentions: List[Dict[str, Any]] = []
        linked_cuis: Dict[str, Dict[str, Any]] = {}
        debug_items: List[Dict[str, Any]] = []

        for phrase in ranked_phrases:
            results = []
            tried: List[Dict[str, Any]] = []
            # Strategy: try searchType variations and root sources individually
            for st in (self._search_type, "exact", "approximate"):
                if results:
                    break
                # Try combined sabs first
                try:
                    r = self._client.search_all(phrase, sabs=self._sabs, search_type=st)
                    tried.append({"searchType": st, "sabs": list(self._sabs), "hits": len(r)})
                    if r:
                        results = r
                        break
                except UMLSClientError as exc:
                    tried.append({"searchType": st, "sabs": list(self._sabs), "error": str(exc)})
                # Then try per-source root_source if still empty
                for rs in self._sabs:
                    if results:
                        break
                    try:
                        r2 = self._client.search_all(phrase, root_source=rs, search_type=st)
                        tried.append({"searchType": st, "rootSource": rs, "hits": len(r2)})
                        if r2:
                            results = r2
                            break
                    except UMLSClientError as exc:
                        tried.append({"searchType": st, "rootSource": rs, "error": str(exc)})

            if debug:
                debug_items.append({"phrase": phrase, "tried": tried, "selected_hits": len(results)})

            if not results:
                continue

            top = results[0]
            cui = top.ui
            name = top.name
            root_source = top.root_source

            semantic_types: List[Dict[str, Any]] = []
            if fetch_details and len(linked_cuis) < max_cui_details:
                try:
                    details = self._client.get_cui_details(cui)
                    semantic_types = details.semantic_types or []
                    if not name:
                        name = details.name
                except UMLSClientError as exc:
                    self._logger.debug("get_cui_details failed for %s: %s", cui, exc)

            # Filter by allowed semantic types if configured
            if self._allowed_sty:
                if not any((isinstance(t, dict) and t.get("name") in self._allowed_sty) for t in semantic_types):
                    if debug:
                        debug_items.append({
                            "phrase": phrase,
                            "filtered": True,
                            "reason": "sty_not_allowed",
                            "sty": [t.get("name") for t in semantic_types if isinstance(t, dict)],
                        })
                    continue

            span = self._find_span(lowered, phrase)
            mention_item = {
                "text": phrase,
                "span": span,
                "linked": {
                    "cui": cui,
                    "name": name,
                    "semantic_types": semantic_types,
                    "root_source": root_source,
                },
            }
            mentions.append(mention_item)

            agg = linked_cuis.setdefault(
                cui,
                {
                    "cui": cui,
                    "name": name,
                    "sty": semantic_types,
                    "sabs": set(),
                    "weight": 0.0,
                },
            )
            if name and not agg.get("name"):
                agg["name"] = name
            if root_source:
                agg["sabs"].add(root_source)
            agg["weight"] = float(agg.get("weight", 0.0) + 1.0)

        # Normalize weights and finalize sabs lists
        concepts: List[Dict[str, Any]] = []
        if linked_cuis:
            max_w = max(v["weight"] for v in linked_cuis.values()) or 1.0
            for item in linked_cuis.values():
                item["weight"] = float(item["weight"]) / float(max_w)
                item["sabs"] = sorted(list(item["sabs"]))
                concepts.append(item)

        out: Dict[str, Any] = {"mentions": mentions, "concepts": concepts}
        if debug:
            out["debug"] = {"phrases": debug_items[: min(50, len(debug_items))]}
        return out

    def _generate_candidates(
        self,
        tokens: List[str],
        *,
        max_ngram: int,
        min_char_len: int,
    ) -> Dict[str, int]:
        candidates: Dict[str, int] = {}
        noise = {
            "ml",
            "mg",
            "mcg",
            "g",
            "kg",
            "dose",
            "doses",
            "volume",
            "vol",
            "single",
            "soft",
            "bag",
            "vial",
            "plastic",
            "injection",
            "tablet",
            "capsule",
            "iv",
            "drip",
            "route",
            "administration",
            "frequency",
            "qd",
            "bid",
            "tid",
            "qid",
        }
        n = len(tokens)
        for i in range(n):
            for l in range(1, max_ngram + 1):
                j = i + l
                if j > n:
                    break
                phrase_tokens = tokens[i:j]
                # Skip if all tokens are stopwords (already filtered) or too short
                phrase = " ".join(phrase_tokens)
                if len(phrase) < min_char_len:
                    continue
                # Prefer phrases containing at least one non-stopword token
                candidates[phrase] = candidates.get(phrase, 0) + 1
                # Also add a cleaned variant that drops dosing/units/noise tokens
                cleaned_tokens = [t for t in phrase_tokens if t not in noise and not t.isdigit()]
                if cleaned_tokens and cleaned_tokens != phrase_tokens:
                    cleaned = " ".join(cleaned_tokens)
                    if len(cleaned) >= min_char_len:
                        candidates[cleaned] = candidates.get(cleaned, 0) + 1
        return candidates

    def _rank_candidates(self, candidates: Dict[str, int], *, max_terms: int) -> List[str]:
        def score(item: Tuple[str, int]) -> float:
            phrase, freq = item
            tok_len = phrase.count(" ") + 1
            char_len = len(phrase)
            return 2.5 * freq + 1.2 * tok_len + 0.02 * char_len

        ranked = sorted(candidates.items(), key=score, reverse=True)
        out: List[str] = []
        seen: set[str] = set()
        for phrase, _ in ranked:
            # De-duplicate by substring containment to reduce near-duplicates
            if any(phrase in p or p in phrase for p in seen):
                continue
            seen.add(phrase)
            out.append(phrase)
            if len(out) >= max_terms:
                break
        return out

    def _find_span(self, lowered_text: str, phrase: str) -> List[int]:
        try:
            start = lowered_text.index(phrase)
            end = start + len(phrase)
            return [start, end]
        except ValueError:
            return [-1, -1]


__all__ = ["UMLSConceptExtractor", "LinkedConcept"]


