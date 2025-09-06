import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Agent Engine imports
from agent_engine.agent_logger.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.memory import ScalableMemory
from agent_engine.utils import get_current_file_dir


class ICUMemoryAgent:
    """ICU Memory Agent for patient-specific and global vector memories.

    Capabilities:
    - For each patient, create a dedicated scalable vector DB named by patient_id.
    - Maintain a global vector cache DB to avoid repeated embedding cost per event_id.
    - Add events with custom item_id = event_id for easy lookup.
    - Provide time-based utilities: events within N hours, and latest N events.

    Notes:
    - Embedding service: AzureClient with model 'text-embedding-3-large'.
    - API key is read from environment variable 'AZURE_API_KEY'.
    - We try to reuse vectors from the global cache by event_id before calling embeddings.
    - Metadata stored for each event includes at least: patient_id, timestamp, event_type, sub_type.
    """

    def __init__(self) -> None:
        load_dotenv()
        self.logger = AgentLogger(self.__class__.__name__)

        # Azure client configuration via environment variables
        api_key = os.getenv("AZURE_API_KEY", "")
        base_url = os.getenv("AZURE_BASE_URL", "https://gpt.yunstorm.com/")
        api_version = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
        if not api_key:
            self.logger.error("AZURE_API_KEY not found in environment variables")
            raise ValueError("AZURE_API_KEY is required")

        self.llm_client = AzureClient(api_key=api_key, base_url=base_url, api_version=api_version)
        # Explicitly choose embedding model
        self.embed_model: str = os.getenv("AGENT_ENGINE_EMBED_MODEL", "text-embedding-3-large")
        # Accept both Path and str for persist_dir
        self.persist_dir: str | Path = get_current_file_dir() / 'database'

        # Global vector cache: isolate storage under its own subdirectory
        self._vector_cache = ScalableMemory(
            name="icu_vector_cache",
            llm_client=self.llm_client,
            embed_model=self.embed_model,
            persist_dir=Path(self.persist_dir),
        )

        # Patient memory cache in process
        self._patient_memories: Dict[str, ScalableMemory] = {}
        self._lock = threading.Lock()

        self.logger.info("ICUMemoryAgent initialized with Azure embeddings and ScalableMemory backends")

    def delete_patient_memory(self, patient_id: str) -> bool:
        try:
            with self._lock:
                mem = self._patient_memories.pop(patient_id, None)
            if mem is not None:
                try:
                    mem.db.close()
                except Exception:
                    pass

            import shutil
            from pathlib import Path

            targets: list[Path] = []
            try:
                targets.append(Path(self.persist_dir))
            except Exception:
                pass

            try:
                from agent_engine.utils.project_root import get_project_root
                targets.append(Path(get_project_root()) / ".memory")
            except Exception:
                pass

            for d in targets:
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)

            return True
        except Exception as e:
            self.logger.error(f"Failed to delete patient memory '{patient_id}': {e}")
            return False

    def delete_vector_cache(self) -> bool:
        try:
            try:
                self._vector_cache.clear()
            except Exception as e:
                self.logger.warning(f"Failed to clear vector cache items: {e}")
            try:
                self._vector_cache.db.close()
            except Exception:
                pass
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete vector cache: {e}")
            return False

    def close_all(self) -> None:
        """Close all underlying memory backends (patient memories and vector cache)."""
        # Close patient memories
        try:
            with self._lock:
                mems = list(self._patient_memories.values())
                self._patient_memories.clear()
            for mem in mems:
                try:
                    mem.db.close()
                except Exception:
                    pass
        except Exception:
            pass

        # Close vector cache
        try:
            self._vector_cache.db.close()
        except Exception:
            pass

    async def add_event(self, patient_id: str, event: Dict[str, Any]) -> str:
        """Add a single ICU event into the patient's memory and the global vector cache.

        The item's id is the event's id.
        We reuse vector from the global cache by event_id when available to avoid embedding cost.

        Args:
            patient_id: Unique patient identifier (also used as memory name).
            event: Event dict or envelope. Prefer keys: 'event_id', 'timestamp', 'event_type', 'sub_type',
                   'event_content', 'raw'. If not enveloped, supports 'id' as event id.

        Returns:
            The event_id used as item id.
        """
        event_id = self._extract_event_id(event)
        if not event_id:
            raise ValueError("Event is missing 'event_id' or 'id'")

        content = self._event_to_text(event)

        # Try reuse vector from global cache
        _, vector, _ = self._vector_cache.get_by_id(event_id)
        if vector is None:
            # Embed once via Azure
            vector = await self._embed_text_async(content)
            await self._vector_cache.add(
                content=f"ICU_EVENT_VECTOR::{event_id}",
                vector=vector,
                metadata={"patient_id": patient_id},
                item_id=event_id,
            )

        # Upsert into patient's memory using provided vector and custom id
        md = self._build_metadata(patient_id, event)
        patient_mem = self._get_patient_memory(patient_id)
        await patient_mem.add(content=content, vector=vector, metadata=md, item_id=event_id)
        return event_id

    async def add_events(self, patient_id: str, events: List[Dict[str, Any]]) -> List[str]:
        """Batch add events. Reuses cached vectors when available and embeds only uncached ones.

        Args:
            patient_id: Unique patient identifier.
            events: List of event dicts.

        Returns:
            List of event_ids added.
        """
        if not events:
            return []

        # Prepare items with vectors (reuse cache when present)
        items: List[Dict[str, Any]] = []
        to_embed_indices: List[int] = []
        texts_to_embed: List[str] = []

        for idx, ev in enumerate(events):
            event_id = self._extract_event_id(ev)
            if not event_id:
                continue
            content = self._event_to_text(ev)
            _, cached_vec, _ = self._vector_cache.get_by_id(event_id)
            if cached_vec is not None:
                items.append({
                    "id": event_id,
                    "content": content,
                    "vector": cached_vec,
                    "metadata": self._build_metadata(patient_id, ev),
                })
            else:
                items.append({
                    "id": event_id,
                    "content": content,
                    "vector": None,
                    "metadata": self._build_metadata(patient_id, ev),
                })
                to_embed_indices.append(idx)
                texts_to_embed.append(content)

        # Embed missing vectors once (batch)
        if texts_to_embed:
            vectors = await self._embed_batch_async(texts_to_embed)
            # Write them into cache and into items
            v_i = 0
            for idx in to_embed_indices:
                ev = events[idx]
                event_id = self._extract_event_id(ev)
                vec = vectors[v_i]
                v_i += 1
                # Cache
                await self._vector_cache.add(
                    content=f"ICU_EVENT_VECTOR::{event_id}",
                    vector=vec,
                    metadata={"patient_id": patient_id},
                    item_id=event_id,
                )
                # Assign to items (find the corresponding placeholder)
                # items list is aligned with events order, find by id
                for it in items:
                    if it["id"] == event_id and it["vector"] is None:
                        it["vector"] = vec
                        break

        # Persist to patient's memory in batch
        patient_mem = self._get_patient_memory(patient_id)
        ids = await patient_mem.add_many(items)
        return ids

    def get_event_by_id(self, patient_id: str, event_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single stored event from the patient's memory by id."""
        mem = self._get_patient_memory(patient_id)
        content, vector, metadata = mem.get_by_id(event_id)
        if content is None:
            return None
        out = {"id": event_id, "content": content, "metadata": metadata}
        return out

    def get_events_within_hours(
        self,
        patient_id: str,
        ref_time: Optional[str | datetime],
        hours: int,
        include_vectors: bool = False,
        sub_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return events within [ref_time - hours, ref_time].

        Args:
            patient_id: Unique patient id.
            ref_time: Reference time (ISO string or datetime). If None, uses now (UTC).
            hours: Window size in hours.
            include_vectors: Whether to include vectors in results.

        Returns:
            List of events sorted by timestamp ascending within the window.
        """
        if hours <= 0:
            return []
        ref_dt = self._to_datetime(ref_time) if ref_time is not None else datetime.now(timezone.utc)
        start_dt = ref_dt - timedelta(hours=hours)

        mem = self._get_patient_memory(patient_id)
        sub_types_set = set(sub_types) if sub_types else None
        results = []
        for content, vector, md in mem.get_all():
            ts_str = md.get("timestamp")
            ts = self._to_datetime(ts_str)
            if ts is None:
                continue
            # filter by sub_type if provided
            if sub_types_set is not None:
                st = md.get("sub_type")
                if st not in sub_types_set:
                    continue
            if start_dt <= ts <= ref_dt:
                item = {
                    "id": md.get("id"),
                    "timestamp": ts_str,
                    "event_type": md.get("event_type"),
                    "sub_type": md.get("sub_type"),
                    "content": content,
                    "metadata": md,
                }
                if include_vectors:
                    item["vector"] = vector
                results.append(item)

        results.sort(key=lambda x: self._to_datetime(x.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))
        return results

    def get_recent_events(
        self,
        patient_id: str,
        n: int,
        include_vectors: bool = False,
        sub_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the most recent N events by timestamp (descending)."""
        if n <= 0:
            return []
        mem = self._get_patient_memory(patient_id)
        sub_types_set = set(sub_types) if sub_types else None
        items = []
        for content, vector, md in mem.get_all():
            ts_str = md.get("timestamp")
            ts = self._to_datetime(ts_str)
            if ts is None:
                continue
            # filter by sub_type if provided
            if sub_types_set is not None:
                st = md.get("sub_type")
                if st not in sub_types_set:
                    continue
            item = {
                "id": md.get("id"),
                "timestamp": ts_str,
                "event_type": md.get("event_type"),
                "sub_type": md.get("sub_type"),
                "content": content,
                "metadata": md,
            }
            if include_vectors:
                item["vector"] = vector
            items.append((ts, item))

        items.sort(key=lambda x: x[0], reverse=True)
        return [it[1] for it in items[:n]]

    def get_patient_memory_info(self, patient_id: str) -> Dict[str, Any]:
        """Return basic info of the patient's memory (backend/index/stats)."""
        return self._get_patient_memory(patient_id).get_info()

    def get_vector_from_cache(self, event_id: str) -> Optional[List[float]]:
        """Get a cached vector by event_id from the global cache."""
        _, vector, _ = self._vector_cache.get_by_id(event_id)
        return vector

    def get_events_between(
        self,
        patient_id: str,
        start_time: Optional[str | datetime],
        end_time: Optional[str | datetime],
        include_vectors: bool = False,
        sub_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return events within [start_time, end_time].

        If start_time is None, use the earliest event timestamp.
        If end_time is None, use the latest event timestamp.

        Args:
            patient_id: Unique patient id.
            start_time: ISO string or datetime for range start (inclusive), or None.
            end_time: ISO string or datetime for range end (inclusive), or None.
            include_vectors: Whether to include vectors in results.
            sub_types: Optional list of sub_type filters. If provided, only events whose
                metadata.sub_type is in this list are included.

        Returns:
            List of events sorted by timestamp ascending within the window.
        """
        mem = self._get_patient_memory(patient_id)
        sub_types_set = set(sub_types) if sub_types else None

        # First pass: collect items and determine earliest/latest timestamps
        collected: List[Tuple[Optional[datetime], str, str, str, str, Dict[str, Any], List[float], str]] = []
        earliest: Optional[datetime] = None
        latest: Optional[datetime] = None
        for content, vector, md in mem.get_all():
            ts_str = md.get("timestamp")
            ts = self._to_datetime(ts_str)
            if ts is None:
                continue
            # filter by sub_type if provided
            if sub_types_set is not None:
                st = md.get("sub_type")
                if st not in sub_types_set:
                    continue
            collected.append((ts, md.get("id"), ts_str or "", md.get("event_type"), md.get("sub_type"), md, vector, content))
            if earliest is None or ts < earliest:
                earliest = ts
            if latest is None or ts > latest:
                latest = ts

        if not collected:
            return []

        # Determine range boundaries
        s_dt = self._to_datetime(start_time) if start_time is not None else earliest
        e_dt = self._to_datetime(end_time) if end_time is not None else latest
        if s_dt is None and earliest is not None:
            s_dt = earliest
        if e_dt is None and latest is not None:
            e_dt = latest
        if s_dt is None or e_dt is None:
            return []
        # Normalize order
        if s_dt > e_dt:
            s_dt, e_dt = e_dt, s_dt

        results: List[Dict[str, Any]] = []
        for ts, iid, ts_str, ev_type, sub_t, md, vector, content in collected:
            assert ts is not None
            if s_dt <= ts <= e_dt:
                item: Dict[str, Any] = {
                    "id": iid,
                    "timestamp": ts_str,
                    "event_type": ev_type,
                    "sub_type": sub_t,
                    "content": content,
                    "metadata": md,
                }
                if include_vectors:
                    item["vector"] = vector
                results.append(item)

        results.sort(key=lambda x: self._to_datetime(x.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))
        return results

    async def query_search(
        self,
        patient_id: str,
        query: Any,
        top_k: int = 5,
        threshold: float = 0.0,
        include_vectors: bool = False,
        sub_types: Optional[List[str]] = None,
        ef_search: Optional[int] = None,
        near_duplicate_delta: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Vector search within a patient's memory using ScalableMemory.search.

        Args:
            patient_id: Unique patient id.
            query: Text or vector.
            top_k: Max results to return.
            threshold: Minimum cosine similarity.
            include_vectors: Whether to include vectors in results.
            sub_types: Optional list of sub_type filters applied after search.
            ef_search: Optional search breadth parameter for HNSW.
            near_duplicate_delta: Exclude near-duplicates if > 0.0.

        Returns:
            List of results with fields: id, timestamp, event_type, sub_type, content, metadata, similarity,
            and optional vector when include_vectors=True.
        """
        mem = self._get_patient_memory(patient_id)
        raw = await mem.search(
            query,
            top_k=top_k,
            threshold=threshold,
            metadata_filter=None,
            ef_search=ef_search,
            near_duplicate_delta=near_duplicate_delta,
        )

        sub_types_set = set(sub_types) if sub_types else None
        out: List[Dict[str, Any]] = []
        for content, sim, md in raw:
            if sub_types_set is not None:
                st = md.get("sub_type")
                if st not in sub_types_set:
                    continue
            item: Dict[str, Any] = {
                "id": md.get("id"),
                "timestamp": md.get("timestamp"),
                "event_type": md.get("event_type"),
                "sub_type": md.get("sub_type"),
                "content": content,
                "metadata": md,
                "similarity": sim,
            }
            if include_vectors:
                iid = md.get("id")
                if iid:
                    _, vec, _ = mem.get_by_id(iid)
                    if vec is not None:
                        item["vector"] = vec
            out.append(item)
        return out

    # ----------------------- Internal helpers -----------------------
    def _get_patient_memory(self, patient_id: str) -> ScalableMemory:
        """Get or create a ScalableMemory instance for a patient.

        Memory name is exactly the patient_id, as required.
        """
        with self._lock:
            if patient_id in self._patient_memories:
                return self._patient_memories[patient_id]

            mem = ScalableMemory(
                name=str(patient_id),
                llm_client=self.llm_client,
                embed_model=self.embed_model,
                persist_dir=Path(self.persist_dir),
            )
            self._patient_memories[patient_id] = mem
            return mem

    def _extract_event_id(self, event: Dict[str, Any]) -> Optional[str]:
        return (
            self._get_nested(event, ["event_id"]) or
            self._get_nested(event, ["id"]) or
            self._get_nested(event, ["raw", "id"]) or
            None
        )

    def _extract_timestamp(self, event: Dict[str, Any]) -> Optional[str]:
        return (
            self._get_nested(event, ["timestamp"]) or
            self._get_nested(event, ["raw", "timestamp"]) or
            None
        )

    def _build_metadata(self, patient_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "patient_id": patient_id,
            "timestamp": self._extract_timestamp(event),
            "event_type": self._get_nested(event, ["event_type"]) or self._get_nested(event, ["raw", "event_type"]) or "",
            "sub_type": self._get_nested(event, ["sub_type"]) or self._get_nested(event, ["raw", "sub_type"]) or "",
            "raw_content": self._get_nested(event, ["raw", "event_content"]),
        }

    def _event_to_text(self, event: Dict[str, Any]) -> str:
        """Create a concise, deterministic text representation for embedding and storage."""
        content = self._get_nested(event, ["event_content"]) or self._get_nested(event, ["raw", "event_content"]) or ""
        try:
            import json

            content_str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, sort_keys=True)
        except Exception:
            content_str = str(content)
        return content_str

    def _get_nested(self, d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
        cur: Any = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    def _to_datetime(self, ts: Optional[str | datetime]) -> Optional[datetime]:
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        s = str(ts).strip()
        # Normalize 'Z' suffix to +00:00
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            # Best-effort parse: try without timezone
            try:
                dt = datetime.fromisoformat(s.split(".")[0])
            except Exception:
                return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    async def _embed_text_async(self, text: str) -> List[float]:
        vec = await self.llm_client.embedding(text, model_name=self.embed_model)
        if not isinstance(vec, list):
            raise RuntimeError("Embedding returned unexpected type")
        if vec and isinstance(vec[0], list):
            vec = vec[0]
        return [float(x) for x in vec]

    async def _embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        res = await self.llm_client.embedding(texts, model_name=self.embed_model)
        if isinstance(res, list) and res and isinstance(res[0], list):
            if len(res) != len(texts):
                out: List[List[float]] = []
                for t in texts:
                    out.append(await self._embed_text_async(t))
                return out
            return [[float(x) for x in vec] for vec in res]
        return [await self._embed_text_async(t) for t in texts]


