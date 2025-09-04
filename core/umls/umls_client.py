"""
UMLS client module for professional, reusable interactions with UTS REST APIs.

This module provides a typed, retry-enabled wrapper around core endpoints used in
ICU Memory concept extraction and normalization workflows:

- Search concepts/codes (paginate automatically)
- Fetch CUI details (semantic types, canonical name)
- Map a CUI to source vocabulary codes (e.g., SNOMEDCT_US, RXNORM)

Notes
- Authentication: this client follows the simplified apiKey query parameter
  pattern consistent with existing scripts in this repository. If you later move
  to the ticket-based authentication flow, you can extend the session builder
  without changing call sites.
- All functions return plain Python types (dicts/lists) and lightweight data
  classes to keep it dependency-light and easy to integrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from agent_engine.agent_logger.agent_logger import AgentLogger


# ---------------------------- Data models ----------------------------


@dataclass(frozen=True)
class UMLSResult:
    """Single search result item from UMLS /search API.

    Fields are kept close to UMLS payload for transparency.
    """

    ui: str
    name: str
    root_source: Optional[str] = None
    uri: Optional[str] = None


@dataclass(frozen=True)
class CUIDetails:
    """Details for a CUI from /content API."""

    ui: str
    name: str
    semantic_types: List[Dict[str, Any]]


class UMLSClientError(Exception):
    """Base error for UMLS client."""


class UMLSNotFoundError(UMLSClientError):
    """Raised when the requested resource is not found (HTTP 404)."""


# ---------------------------- Client ----------------------------


class UMLSClient:
    """Typed client for UMLS UTS APIs.

    Parameters
    - api_key: UTS profile API key
    - version: UMLS terminology version, default "current"
    - base_url: default "https://uts-ws.nlm.nih.gov"
    - timeout_sec: per-request timeout in seconds
    - max_retries: total retries for transient errors
    - backoff_factor: exponential backoff base factor

    Example
    -------
    client = UMLSClient(api_key="YOUR_KEY")
    # Search CUIs for a term
    cuis = client.search_cuis("sepsis", root_source="SNOMEDCT_US")
    # CUI details
    detail = client.get_cui_details(cuis[0])
    # Map CUI to source codes
    codes = client.map_cui_to_source_codes(cuis[0], sabs=["SNOMEDCT_US", "RXNORM"]).results
    """

    def __init__(
        self,
        api_key: str,
        version: str = "current",
        base_url: str = "https://uts-ws.nlm.nih.gov",
        timeout_sec: int = 15,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        logger: Optional[AgentLogger] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")

        self._api_key = api_key
        self._version = version
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_sec
        # Use project AgentLogger (singleton). Name parameter is best-effort.
        self._logger = logger or AgentLogger(self.__class__.__name__)

        retry = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)

        session = requests.Session()
        session.headers.update({"Accept": "application/json"})
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        self._session = session

    # ---------------------------- Public methods ----------------------------

    def search(
        self,
        query: str,
        *,
        sabs: Optional[Iterable[str]] = None,
        root_source: Optional[str] = None,
        search_type: Optional[str] = None,
        return_id_type: Optional[str] = None,
        page_number: int = 1,
    ) -> Dict[str, Any]:
        """Call UMLS /search API.

        Arguments
        - query: text or identifier for search (e.g., free text, CUI, etc.)
        - sabs: comma-separated list of vocabularies (e.g., ["MSH", "SNOMEDCT_US"]) -> 'sabs'
        - root_source: single source vocabulary name -> 'rootSource'
        - search_type: UMLS searchType (e.g., 'exact', 'words', 'phrase')
        - return_id_type: e.g., 'code' to return source codes
        - page_number: 1-based page index

        Returns raw JSON dict of the API response.
        """
        params: Dict[str, Any] = {
            "string": query,
            "apiKey": self._api_key,
            "pageNumber": page_number,
        }
        if sabs:
            params["sabs"] = ",".join(s.strip() for s in sabs if s)
        if root_source:
            params["rootSource"] = root_source
        if search_type:
            params["searchType"] = search_type
        if return_id_type:
            params["returnIdType"] = return_id_type

        url = f"{self._base_url}/search/{self._version}"
        self._logger.debug("UMLS search: %s", params)
        resp = self._session.get(url, params=params, timeout=self._timeout)
        if resp.status_code == 404:
            raise UMLSNotFoundError("search endpoint returned 404")
        if not resp.ok:
            raise UMLSClientError(f"search failed: {resp.status_code} {resp.text[:200]}")
        return resp.json()

    def iterate_search(
        self,
        query: str,
        *,
        sabs: Optional[Iterable[str]] = None,
        root_source: Optional[str] = None,
        search_type: Optional[str] = None,
        return_id_type: Optional[str] = None,
    ) -> Generator[List[UMLSResult], None, None]:
        """Iterate all pages of /search results, yielding lists of UMLSResult.

        Stops when an empty page is returned.
        """
        page = 0
        while True:
            page += 1
            data = self.search(
                query,
                sabs=sabs,
                root_source=root_source,
                search_type=search_type,
                return_id_type=return_id_type,
                page_number=page,
            )
            try:
                results = data["result"]["results"]
            except Exception as exc:
                raise UMLSClientError(f"unexpected search response shape: {data}") from exc

            if not results:
                break

            normalized: List[UMLSResult] = [
                UMLSResult(
                    ui=item.get("ui", ""),
                    name=item.get("name", ""),
                    root_source=item.get("rootSource"),
                    uri=item.get("uri"),
                )
                for item in results
            ]
            yield normalized

    def search_all(
        self,
        query: str,
        *,
        sabs: Optional[Iterable[str]] = None,
        root_source: Optional[str] = None,
        search_type: Optional[str] = None,
        return_id_type: Optional[str] = None,
    ) -> List[UMLSResult]:
        """Collect all search results across pages into a single list."""
        out: List[UMLSResult] = []
        for page_items in self.iterate_search(
            query,
            sabs=sabs,
            root_source=root_source,
            search_type=search_type,
            return_id_type=return_id_type,
        ):
            out.extend(page_items)
        return out

    def search_cuis(
        self,
        term: str,
        *,
        root_source: Optional[str] = None,
        search_type: Optional[str] = None,
    ) -> List[str]:
        """Return CUIs for a term using /search (UI will be CUIs for default return type)."""
        results = self.search_all(
            term,
            root_source=root_source,
            search_type=search_type,
        )
        # When not using returnIdType=code, UI is typically a CUI
        return [r.ui for r in results if r.ui]

    def get_cui_details(self, cui: str) -> CUIDetails:
        """Fetch CUI details including semantic types via /content/{version}/CUI/{cui}."""
        if not cui:
            raise ValueError("cui is required")
        url = f"{self._base_url}/content/{self._version}/CUI/{cui}"
        params = {"apiKey": self._api_key}
        resp = self._session.get(url, params=params, timeout=self._timeout)
        if resp.status_code == 404:
            raise UMLSNotFoundError(f"CUI not found: {cui}")
        if not resp.ok:
            raise UMLSClientError(f"get_cui_details failed: {resp.status_code} {resp.text[:200]}")

        data = resp.json().get("result", {})
        return CUIDetails(
            ui=data.get("ui", cui),
            name=data.get("name", ""),
            semantic_types=data.get("semanticTypes", []) or [],
        )

    def map_cui_to_source_codes(
        self,
        cui: str,
        *,
        sabs: Iterable[str],
    ) -> "CodeSearchResult":
        """Map a CUI to source vocabulary codes by using /search with returnIdType=code.

        This mirrors the behavior of the legacy script (UMLS-concept.py) but returns
        structured results suitable for programmatic use.
        """
        if not cui:
            raise ValueError("cui is required")
        sabs_list = list(sabs)
        if not sabs_list:
            raise ValueError("sabs must not be empty")

        results: List[UMLSResult] = self.search_all(
            cui,
            sabs=sabs_list,
            return_id_type="code",
        )
        # Here, result.ui is the source vocabulary code (not a CUI) when returnIdType=code
        return CodeSearchResult(
            cui=cui,
            sabs=sabs_list,
            results=results,
        )


@dataclass(frozen=True)
class CodeSearchResult:
    """Structured mapping from a CUI to source vocabulary codes.

    Each result item has:
    - ui: source code identifier in that vocabulary
    - name: label from the vocabulary
    - root_source: the vocabulary name (e.g., SNOMEDCT_US, RXNORM)
    - uri: the vocabulary item URI
    """

    cui: str
    sabs: List[str]
    results: List[UMLSResult]


__all__ = [
    "UMLSClient",
    "UMLSClientError",
    "UMLSNotFoundError",
    "UMLSResult",
    "CUIDetails",
    "CodeSearchResult",
]


