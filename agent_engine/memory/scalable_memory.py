"""
High-performance, scalable vector memory for agents.

This module provides a scalable memory implementation with:
- Optional approximate nearest neighbor index (HNSW via hnswlib; fallback to Annoy; fallback to brute force)
- Persistent storage (DuckDB preferred; fallback to SQLite with WAL)
- Custom ID support and upsert semantics
- Batch ingestion and efficient retrieval
- Thread-safe read/write with a simple RWLock

Design goals:
- Backwards-friendly APIs inspired by the existing Memory class, while adding advanced capabilities
- RAG-friendly ANN search with metadata filtering and similarity threshold
"""

from __future__ import annotations

import os
import json
import math
import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..agent_logger.agent_logger import AgentLogger
from ..utils.project_root import get_project_root
from .embedder import Embedder


logger = AgentLogger(__name__)


class _RWLock:
    """A simple reader-writer lock.

    Multiple readers can hold the lock simultaneously, but writes are exclusive.
    """

    def __init__(self) -> None:
        self._readers: int = 0
        self._readers_lock = threading.Lock()
        self._resource_lock = threading.Lock()

    def acquire_read(self) -> None:
        with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                self._resource_lock.acquire()

    def release_read(self) -> None:
        with self._readers_lock:
            self._readers -= 1
            if self._readers == 0:
                self._resource_lock.release()

    def acquire_write(self) -> None:
        self._resource_lock.acquire()

    def release_write(self) -> None:
        self._resource_lock.release()

    def read_locked(self):
        class _Ctx:
            def __init__(self, parent: _RWLock) -> None:
                self._p = parent

            def __enter__(self):
                self._p.acquire_read()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._p.release_read()

        return _Ctx(self)

    def write_locked(self):
        class _Ctx:
            def __init__(self, parent: _RWLock) -> None:
                self._p = parent

            def __enter__(self):
                self._p.acquire_write()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._p.release_write()

        return _Ctx(self)


def _to_blob(vector: List[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> List[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()


# NOTE: Async runner removed. Async methods will be used directly.


class _DB:
    """Simple DB adapter that prefers DuckDB and falls back to SQLite.

    For SQLite we enable WAL and performance pragmas.
    """

    def __init__(self, db_path: Path, prefer_duckdb: bool = True) -> None:
        self.db_path = db_path
        self.prefer_duckdb = prefer_duckdb
        self.backend: str
        self._conn: Any
        self._connect()

    def _connect(self) -> None:
        if self.prefer_duckdb:
            try:
                import duckdb  # type: ignore

                self._conn = duckdb.connect(str(self.db_path))
                self.backend = "duckdb"
                return
            except Exception as e:
                # Provide rich diagnostics for decoding-related issues
                try:
                    import sys
                    db_path_str = str(self.db_path)
                    fsenc = sys.getfilesystemencoding() or "unknown"
                    if isinstance(e, UnicodeDecodeError):
                        obj = getattr(e, "object", None)
                        start = getattr(e, "start", None)
                        end = getattr(e, "end", None)
                        reason = getattr(e, "reason", None)
                        around_hex = None
                        if isinstance(obj, (bytes, bytearray)) and isinstance(start, int):
                            lo = max(0, start - 8)
                            hi = min(len(obj), start + 8)
                            slice_bytes = obj[lo:hi]
                            around_hex = " ".join(f"{b:02x}" for b in slice_bytes)
                        logger.warning(
                            "DuckDB failed to open (UnicodeDecodeError): encoding=%s start=%s end=%s reason=%s path=%r fsenc=%s around_hex=%s. Falling back to SQLite.",
                            getattr(e, "encoding", None), start, end, reason, db_path_str, fsenc, around_hex,
                        )
                    else:
                        logger.warning(
                            "DuckDB failed to open: %s: %r path=%r fsenc=%s. Falling back to SQLite.",
                            type(e).__name__, e, db_path_str, fsenc,
                        )
                except Exception:
                    logger.warning("DuckDB failed to open and detailed logging also failed. Falling back to SQLite.")

        # Fallback to SQLite
        try:
            # If DuckDB was preferred but failed, switch to a clear .sqlite3 file name
            if self.prefer_duckdb and str(self.db_path).lower().endswith(".duckdb"):
                try:
                    self.db_path = self.db_path.with_suffix(".sqlite3")
                except Exception:
                    # Fallback if suffix replacement fails for any reason
                    self.db_path = Path(str(self.db_path) + ".sqlite3")
        except Exception:
            pass

        # Use a connect timeout for better multi-process WAL behavior
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=5.0,
        )
        self.backend = "sqlite"

        # Pragmas tuned for read-heavy, multi-process WAL usage
        # WAL provides readers-writer concurrency; NORMAL sync balances safety/perf
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA mmap_size=30000000000;")  # 30GB hint
        # New pragmas for stability and throughput
        self._conn.execute("PRAGMA busy_timeout=5000;")  # ms
        self._conn.execute("PRAGMA cache_size=-16384;")    # ~16MB cache
        self._conn.execute("PRAGMA wal_autocheckpoint=2000;")  # ~8MB @ 4KB pages
        self._conn.execute("PRAGMA journal_size_limit=67108864;")  # 64MB cap
        self._conn.execute("PRAGMA locking_mode=NORMAL;")
        self._conn.commit()
        # Track last WAL checkpoint time
        try:
            self._last_wal_checkpoint_ts = time.time()
        except Exception:
            self._last_wal_checkpoint_ts = 0.0

    def execute(self, sql: str, params: Tuple[Any, ...] = ()):
        if self.backend == "duckdb":
            return self._conn.execute(sql, params)
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return cur

    def executemany(self, sql: str, seq_of_params: Iterable[Tuple[Any, ...]]):
        if self.backend == "duckdb":
            return self._conn.execute(sql, seq_of_params)
        cur = self._conn.cursor()
        cur.executemany(sql, seq_of_params)
        return cur

    def fetchall(self, cursor) -> List[Tuple[Any, ...]]:
        if self.backend == "duckdb":
            return cursor.fetchall()
        return cursor.fetchall()

    def fetchone(self, cursor) -> Optional[Tuple[Any, ...]]:
        if self.backend == "duckdb":
            return cursor.fetchone()
        return cursor.fetchone()

    def commit(self) -> None:
        self._conn.commit()
        # Periodically checkpoint WAL to control file growth without blocking writers
        try:
            if self.backend == "sqlite":
                now = time.time()
                last = getattr(self, "_last_wal_checkpoint_ts", 0.0)
                if (now - last) > 10.0:
                    try:
                        self._conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
                    finally:
                        self._last_wal_checkpoint_ts = now
        except Exception:
            # Non-fatal; continue
            pass

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class _BaseANNIndex:
    """Interface for ANN index backends."""

    def add_or_update(self, label: int, vector: List[float]) -> None:
        raise NotImplementedError

    def add_many(self, labels: List[int], vectors: List[List[float]]) -> None:
        raise NotImplementedError

    def search(self, vector: List[float], top_k: int, ef_search: Optional[int] = None) -> Tuple[List[int], List[float]]:
        raise NotImplementedError

    def delete(self, label: int) -> None:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path, dim: int) -> bool:
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        raise NotImplementedError


class _HNSWIndex(_BaseANNIndex):
    def __init__(self, dim: int, space: str = "cosine", init_capacity: int = 1024, m: int = 16, ef_construction: int = 200) -> None:
        self.dim = dim
        self.space = space
        self.m = m
        self.ef_construction = ef_construction
        self.capacity = max(16, int(init_capacity))
        self._index = None
        self._init_index()

    def _init_index(self) -> None:
        try:
            import hnswlib  # type: ignore

            self._index = hnswlib.Index(space=self.space, dim=self.dim)
            # allow_replace_deleted enables re-adding the same label after deletion
            self._index.init_index(max_elements=self.capacity, ef_construction=self.ef_construction, M=self.m, allow_replace_deleted=True)
            self._index.set_ef(64)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize hnswlib index: {e}")

    def _ensure_capacity(self, need: int) -> None:
        if self._index is None:
            return
        try:
            current = self.capacity
            if need <= current:
                return
            new_cap = 1 << (int(math.log2(max(need, current))) + 1)
            self._index.resize_index(new_cap)
            self.capacity = new_cap
        except Exception as e:
            raise RuntimeError(f"Failed to resize index: {e}")

    def add_or_update(self, label: int, vector: List[float]) -> None:
        if self._index is None:
            raise RuntimeError("Index not initialized")
        self._ensure_capacity(label + 1)
        import hnswlib  # type: ignore
        try:
            self._index.add_items(np.asarray([vector], dtype=np.float32), np.asarray([label], dtype=np.int64))
        except RuntimeError as e:
            # If already exists, try delete then add
            try:
                self._index.mark_deleted(label)
                self._index.add_items(np.asarray([vector], dtype=np.float32), np.asarray([label], dtype=np.int64))
            except Exception as e2:
                raise RuntimeError(f"Failed to update label {label}: {e} / {e2}")

    def add_many(self, labels: List[int], vectors: List[List[float]]) -> None:
        if self._index is None:
            raise RuntimeError("Index not initialized")
        max_label = max(labels) if labels else 0
        self._ensure_capacity(max_label + 1)
        self._index.add_items(np.asarray(vectors, dtype=np.float32), np.asarray(labels, dtype=np.int64))

    def search(self, vector: List[float], top_k: int, ef_search: Optional[int] = None) -> Tuple[List[int], List[float]]:
        if self._index is None:
            raise RuntimeError("Index not initialized")
        if ef_search is not None:
            self._index.set_ef(ef_search)
        labels, distances = self._index.knn_query(np.asarray([vector], dtype=np.float32), k=max(1, top_k))
        # For cosine, distances are (1 - cosine_similarity)
        return labels[0].tolist(), distances[0].tolist()

    def delete(self, label: int) -> None:
        if self._index is None:
            raise RuntimeError("Index not initialized")
        self._index.mark_deleted(label)

    def save(self, path: Path) -> None:
        if self._index is None:
            return
        self._index.save_index(str(path))

    def load(self, path: Path, dim: int) -> bool:
        try:
            import hnswlib  # type: ignore

            if not path.exists():
                return False
            self._index = hnswlib.Index(space=self.space, dim=dim)
            self._index.load_index(str(path))
            # capacity unknown after load; approximate with current elements * 2
            try:
                self.capacity = max(1024, int(self._index.get_current_count() * 2))
            except Exception:
                self.capacity = 1024
            return True
        except Exception as e:
            logger.warning(f"Failed to load hnsw index: {e}")
            return False

    def info(self) -> Dict[str, Any]:
        size = None
        try:
            size = int(self._index.get_current_count()) if self._index is not None else 0
        except Exception:
            size = None
        return {
            "backend": "hnswlib",
            "size": size,
            "capacity": self.capacity,
            "space": self.space,
            "m": self.m,
            "ef_construction": self.ef_construction,
        }


class _AnnoyIndex(_BaseANNIndex):
    def __init__(self, dim: int, metric: str = "angular", n_trees: int = 10) -> None:
        self.dim = dim
        self.metric = metric
        self.n_trees = n_trees
        self._index = None
        self._built = False
        try:
            from annoy import AnnoyIndex  # type: ignore

            self._index = AnnoyIndex(self.dim, self.metric)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Annoy index: {e}")

    def add_or_update(self, label: int, vector: List[float]) -> None:
        # Annoy has no true update; we set and mark rebuild
        if self._index is None:
            raise RuntimeError("Index not initialized")
        self._index.add_item(label, vector)
        self._built = False

    def add_many(self, labels: List[int], vectors: List[List[float]]) -> None:
        if self._index is None:
            raise RuntimeError("Index not initialized")
        for lb, vec in zip(labels, vectors):
            self._index.add_item(lb, vec)
        self._built = False

    def _ensure_built(self) -> None:
        if self._index is None:
            return
        if not self._built:
            self._index.build(self.n_trees)
            self._built = True

    def search(self, vector: List[float], top_k: int, ef_search: Optional[int] = None) -> Tuple[List[int], List[float]]:
        if self._index is None:
            raise RuntimeError("Index not initialized")
        self._ensure_built()
        labels = self._index.get_nns_by_vector(vector, top_k, include_distances=True)
        return labels[0], labels[1]

    def delete(self, label: int) -> None:
        # Annoy does not support deletion; caller should rebuild periodically
        pass

    def save(self, path: Path) -> None:
        if self._index is None:
            return
        self._ensure_built()
        self._index.save(str(path))

    def load(self, path: Path, dim: int) -> bool:
        try:
            from annoy import AnnoyIndex  # type: ignore

            if not path.exists():
                return False
            self._index = AnnoyIndex(dim, self.metric)
            ok = self._index.load(str(path))
            self._built = True
            return bool(ok)
        except Exception as e:
            logger.warning(f"Failed to load Annoy index: {e}")
            return False

    def info(self) -> Dict[str, Any]:
        return {
            "backend": "annoy",
            "metric": self.metric,
            "n_trees": self.n_trees,
        }


class _BruteForceIndex(_BaseANNIndex):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._labels: List[int] = []
        self._vectors: List[List[float]] = []

    def add_or_update(self, label: int, vector: List[float]) -> None:
        try:
            idx = self._labels.index(label)
            self._vectors[idx] = vector
        except ValueError:
            self._labels.append(label)
            self._vectors.append(vector)

    def add_many(self, labels: List[int], vectors: List[List[float]]) -> None:
        for lb, vec in zip(labels, vectors):
            self.add_or_update(lb, vec)

    def search(self, vector: List[float], top_k: int, ef_search: Optional[int] = None) -> Tuple[List[int], List[float]]:
        if not self._vectors:
            return [], []
        arr = np.asarray(self._vectors, dtype=np.float32)
        q = np.asarray(vector, dtype=np.float32)
        # cosine distance = 1 - cosine_similarity
        v_norm = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        sims = v_norm @ q_norm
        dists = 1.0 - sims
        idxs = np.argsort(dists)[:top_k].tolist()
        labels = [self._labels[i] for i in idxs]
        distances = [float(dists[i]) for i in idxs]
        return labels, distances

    def delete(self, label: int) -> None:
        try:
            idx = self._labels.index(label)
            self._labels.pop(idx)
            self._vectors.pop(idx)
        except ValueError:
            pass

    def save(self, path: Path) -> None:
        try:
            data = {"labels": self._labels, "vectors": self._vectors}
            path.write_text(json.dumps(data))
        except Exception:
            pass

    def load(self, path: Path, dim: int) -> bool:
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            self._labels = data.get("labels", [])
            self._vectors = data.get("vectors", [])
            return True
        except Exception:
            return False

    def info(self) -> Dict[str, Any]:
        return {"backend": "bruteforce", "size": len(self._labels)}


class ScalableMemory:
    """Scalable vector memory using ANN index and persistent DB.

    Storage layout under project_root/.memory/{name}/:
    - db: {name}.duckdb or {name}.sqlite3
    - index: index_hnsw.bin / index_annoy.ann / index_bruteforce.json
    - meta table in DB keeps vector_dimension and schema version
    - labels table in DB maps numeric labels used by the ANN index to user IDs
    - items table stores id, content, metadata, vector blob
    """

    SCHEMA_VERSION: int = 1
    _initialized_paths: Dict[str, bool] = {}

    def __init__(
        self,
        name: str,
        model_name: str = "all-MiniLM-L6-v2",
        index_backend: Optional[str] = None,
        db_backend: Optional[str] = None,
        index_params: Optional[Dict[str, Any]] = None,
        persist_dir: Optional[str] = None,
        llm_client: Optional[Any] = None,
        embed_model: Optional[str] = None,
        enable_vectors: bool = True,
    ) -> None:
        if not name:
            raise ValueError("Memory name is required")

        self.name = name
        self.model_name = model_name
        self.embedder: Optional[Embedder] = None
        # Optional external LLM client embedding
        self._llm_client: Optional[Any] = llm_client
        self._embed_model: Optional[str] = embed_model or os.getenv("AGENT_ENGINE_EMBED_MODEL")
        # Whether vector storage and ANN search are enabled
        self._enable_vectors: bool = bool(enable_vectors)

        # Paths
        project_root = get_project_root()
        base_dir = Path(persist_dir) if persist_dir else (project_root / ".memory" / name)
        base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir

        # DB
        prefer_duckdb = (db_backend or os.getenv("AGENT_ENGINE_MEMORY_DB", "duckdb")).lower() == "duckdb"
        db_filename = f"{name}.duckdb" if prefer_duckdb else f"{name}.sqlite3"
        self.db = _DB(base_dir / db_filename, prefer_duckdb=prefer_duckdb)

        # Index backend preference: hnswlib -> annoy -> brute
        ib = (index_backend or os.getenv("AGENT_ENGINE_MEMORY_INDEX", "hnswlib")).lower()
        self._preferred_index_backend = ib
        self._index: Optional[_BaseANNIndex] = None
        self._index_path: Optional[Path] = None
        self._index_backend: str = "disabled" if not self._enable_vectors else "unknown"
        self._index_params = index_params or {}

        # Locks
        self._rw = _RWLock()

        # Init schema and index
        self._ensure_schema_initialized()
        # Defer index initialization until we know the vector dimension
        # (from meta or first added vector)

    # ---------- Initialization ----------
    def _get_embedder(self) -> Embedder:
        if self.embedder is None:
            self.embedder = Embedder(model_name=self.model_name)
            # Preload in background to reduce first-use latency
            try:
                self.embedder.preload_async()
            except Exception:
                pass
        return self.embedder

    def _use_llm(self) -> bool:
        return self._llm_client is not None

    async def _embed_text_async(self, text: str) -> List[float]:
        if self._use_llm():
            vec = await self._llm_client.embedding(text, model_name=self._embed_model)
            if vec is None:
                raise RuntimeError("LLM client returned None embedding")
            if isinstance(vec, list) and vec and isinstance(vec[0], (float, int)):
                return [float(x) for x in vec]
            raise RuntimeError("LLM client returned unexpected embedding shape for single text")
        else:
            return self._get_embedder().embed(text)

    async def _embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._use_llm():
            res = await self._llm_client.embedding(texts, model_name=self._embed_model)
            if res is None:
                raise RuntimeError("LLM client returned None embeddings")
            if isinstance(res, list) and res and isinstance(res[0], list):
                if len(res) != len(texts):
                    out: List[List[float]] = []
                    for t in texts:
                        out.append(await self._embed_text_async(t))
                    return out
                return [[float(x) for x in vec] for vec in res]
            elif isinstance(res, list) and (not res or isinstance(res[0], (float, int))):
                out: List[List[float]] = []
                for t in texts:
                    out.append(await self._embed_text_async(t))
                return out
            else:
                raise RuntimeError("LLM client returned unexpected embeddings shape for batch")
        else:
            return self._get_embedder().embed_batch(texts)

    def _get_vector_dim(self) -> Optional[int]:
        # Try DB meta first to avoid loading models unnecessarily
        cur = self.db.execute("SELECT value FROM meta WHERE key = ?", ("vector_dimension",))
        row = self.db.fetchone(cur)
        if row and row[0]:
            try:
                return int(row[0])
            except Exception:
                pass
        return None

    def _ensure_schema_initialized(self) -> None:
        # Use the absolute DB file path as cache key to avoid collisions between
        # different memory files under the same directory.
        try:
            db_key = str(self.db.db_path)
        except Exception:
            db_key = str(self.base_dir)
        cache_key = db_key
        if ScalableMemory._initialized_paths.get(cache_key):
            return

        # tables: meta, labels(label INTEGER PRIMARY KEY, id TEXT UNIQUE), items(id TEXT PRIMARY KEY, content TEXT, metadata TEXT, vector BLOB)
        if self.db.backend == "duckdb":
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR
                )
                """
            )
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS labels (
                    label BIGINT PRIMARY KEY,
                    id VARCHAR UNIQUE
                )
                """
            )
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id VARCHAR PRIMARY KEY,
                    content VARCHAR NOT NULL,
                    metadata VARCHAR,
                    vector BLOB
                )
                """
            )
        else:
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS labels (
                    label INTEGER PRIMARY KEY,
                    id TEXT UNIQUE
                )
                """
            )
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    vector BLOB
                )
                """
            )
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_items_content ON items(content)")

        # schema version
        self._upsert_meta("schema_version", str(self.SCHEMA_VERSION))
        self.db.commit()
        ScalableMemory._initialized_paths[cache_key] = True

    def _sanitize_name_for_fs(self, name: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in str(name))

    def _compute_index_path(self, backend: str) -> Path:
        safe = self._sanitize_name_for_fs(self.name)
        if backend == "hnswlib":
            return self.base_dir / f"index_hnsw_{safe}.bin"
        if backend == "annoy":
            return self.base_dir / f"index_annoy_{safe}.ann"
        return self.base_dir / f"index_bruteforce_{safe}.json"

    def _ensure_index_initialized(self) -> None:
        dim_opt = self._get_vector_dim()
        if dim_opt is None:
            raise RuntimeError("Vector dimension is unknown; cannot initialize index before first add")
        dim = int(dim_opt)

        # Try preferred index backend; fallback chain
        tried: List[str] = []
        for backend in [self._preferred_index_backend, "hnswlib", "annoy", "bruteforce"]:
            if backend in tried:
                continue
            tried.append(backend)
            try:
                if backend == "hnswlib":
                    init_cap = max(1024, self._get_count_estimate() * 2)
                    m = int(self._index_params.get("M", 16))
                    ef_c = int(self._index_params.get("ef_construction", 200))
                    self._index = _HNSWIndex(dim=dim, space="cosine", init_capacity=init_cap, m=m, ef_construction=ef_c)
                    self._index_backend = "hnswlib"
                    new_path = self._compute_index_path(self._index_backend)
                    legacy_path = self.base_dir / "index_hnsw.bin"
                elif backend == "annoy":
                    n_trees = int(self._index_params.get("n_trees", 10))
                    self._index = _AnnoyIndex(dim=dim, metric="angular", n_trees=n_trees)
                    self._index_backend = "annoy"
                    new_path = self._compute_index_path(self._index_backend)
                    legacy_path = self.base_dir / "index_annoy.ann"
                else:
                    self._index = _BruteForceIndex(dim=dim)
                    self._index_backend = "bruteforce"
                    new_path = self._compute_index_path(self._index_backend)
                    legacy_path = self.base_dir / "index_bruteforce.json"

                # Prefer new per-name path; fallback to legacy shared filename
                self._index_path = new_path
                if not self._index.load(new_path, dim):
                    if not self._index.load(legacy_path, dim):
                        # Rebuild from DB if vectors exist (and then save to new path)
                        self._rebuild_index_from_db()
                    else:
                        self._index_path = legacy_path
                logger.info(f"ScalableMemory index ready: backend={self._index_backend}")
                return
            except Exception as e:
                logger.warning(f"Index backend {backend} unavailable: {e}")
                continue

        # Should never reach here
        raise RuntimeError("Failed to initialize any ANN index backend")

    def _init_index_for_dim(self, dim: int) -> None:
        """Initialize index for a known vector dimension and persist meta."""
        # Write meta if not set
        current_dim = self._get_vector_dim()
        if current_dim is None:
            self._upsert_meta("vector_dimension", str(dim))
            self.db.commit()
        elif int(current_dim) != int(dim):
            # If no items stored yet, allow updating the dimension
            if self._get_count_estimate() == 0:
                self._upsert_meta("vector_dimension", str(dim))
                self.db.commit()
            else:
                raise RuntimeError(f"Vector dimension mismatch: existing={current_dim} new={dim}")
        # Initialize
        tried: List[str] = []
        for backend in [self._preferred_index_backend, "hnswlib", "annoy", "bruteforce"]:
            if backend in tried:
                continue
            tried.append(backend)
            try:
                if backend == "hnswlib":
                    init_cap = max(1024, self._get_count_estimate() * 2)
                    m = int(self._index_params.get("M", 16))
                    ef_c = int(self._index_params.get("ef_construction", 200))
                    self._index = _HNSWIndex(dim=dim, space="cosine", init_capacity=init_cap, m=m, ef_construction=ef_c)
                    self._index_backend = "hnswlib"
                    new_path = self._compute_index_path(self._index_backend)
                    legacy_path = self.base_dir / "index_hnsw.bin"
                elif backend == "annoy":
                    n_trees = int(self._index_params.get("n_trees", 10))
                    self._index = _AnnoyIndex(dim=dim, metric="angular", n_trees=n_trees)
                    self._index_backend = "annoy"
                    new_path = self._compute_index_path(self._index_backend)
                    legacy_path = self.base_dir / "index_annoy.ann"
                else:
                    self._index = _BruteForceIndex(dim=dim)
                    self._index_backend = "bruteforce"
                    new_path = self._compute_index_path(self._index_backend)
                    legacy_path = self.base_dir / "index_bruteforce.json"

                # Prefer new per-name path; fallback to legacy shared filename
                self._index_path = new_path
                if not self._index.load(new_path, dim):
                    if not self._index.load(legacy_path, dim):
                        # Rebuild from DB if vectors exist (and then save to new path)
                        self._rebuild_index_from_db()
                    else:
                        self._index_path = legacy_path
                logger.info(f"ScalableMemory index ready: backend={self._index_backend}")
                return
            except Exception as e:
                logger.warning(f"Index backend {backend} unavailable: {e}")
                continue
        raise RuntimeError("Failed to initialize any ANN index backend")

    def _get_count_estimate(self) -> int:
        cur = self.db.execute("SELECT COUNT(*) FROM items")
        row = self.db.fetchone(cur)
        return int(row[0]) if row else 0

    def _rebuild_index_from_db(self) -> None:
        # Load all labels and vectors
        cur = self.db.execute("SELECT labels.label, items.vector FROM labels JOIN items ON labels.id = items.id ORDER BY labels.label ASC")
        rows = self.db.fetchall(cur)
        if not rows:
            return
        labels = [int(r[0]) for r in rows]
        vectors = [_from_blob(r[1]) for r in rows]
        # batch add
        batch = 4096
        for i in range(0, len(labels), batch):
            self._index.add_many(labels[i : i + batch], vectors[i : i + batch])
        # Persist index
        self._index.save(self._index_path)

    # ---------- ID/Label helpers ----------
    def _get_or_create_label_for_id(self, item_id: str) -> int:
        cur = self.db.execute("SELECT label FROM labels WHERE id = ?", (item_id,))
        row = self.db.fetchone(cur)
        if row:
            return int(row[0])
        # assign next label: max(label)+1
        cur = self.db.execute("SELECT COALESCE(MAX(label), -1) + 1 FROM labels")
        row = self.db.fetchone(cur)
        next_label = int(row[0]) if row else 0
        self._upsert_label(next_label, item_id)
        self.db.commit()
        return next_label

    def _label_for_id(self, item_id: str) -> Optional[int]:
        cur = self.db.execute("SELECT label FROM labels WHERE id = ?", (item_id,))
        row = self.db.fetchone(cur)
        return int(row[0]) if row else None

    # ---------- Public APIs ----------
    async def add(
        self,
        content: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
    ) -> str:
        """Add a new item. If item_id exists, it will be upserted.

        Returns the item_id.
        """
        if not content and vector is None:
            raise ValueError("Either content or vector must be provided")

        if item_id is None:
            # stable id by hashing content+metadata to avoid duplicates by default
            base = (content or "") + "\n" + json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
            import hashlib

            item_id = hashlib.md5(base.encode("utf-8")).hexdigest()

        if self._enable_vectors:
            if vector is None:
                vector = await self._embed_text_async(content)
            if not vector:
                raise ValueError("Vector generation failed")
            # Ensure index initialized with the correct dimension
            if self._index is None:
                self._init_index_for_dim(len(vector))
        else:
            # In non-vector mode, ignore provided vectors and store None
            vector = None

        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

        with self._rw.write_locked():
            # Upsert DB
            self._upsert_item(
                item_id=item_id,
                content=content,
                metadata_json=meta_json,
                vector_blob=_to_blob(vector) if (self._enable_vectors and vector is not None) else None,
            )
            self.db.commit()

            # Update index (vector mode only)
            if self._enable_vectors:
                label = self._get_or_create_label_for_id(item_id)
                assert self._index is not None
                self._index.add_or_update(label, vector)  # type: ignore[arg-type]
                self._maybe_persist_index()

        return item_id

    async def add_many(self, items: List[Dict[str, Any]]) -> List[str]:
        """Batch add items.

        Each item: {"content": str, "vector": Optional[List[float]], "metadata": Optional[dict], "id": Optional[str]}
        Returns list of item_ids.
        """
        if not items:
            return []

        contents = [it.get("content") for it in items]
        need_embed_idx = [i for i, it in enumerate(items) if it.get("vector") is None]

        if need_embed_idx and self._enable_vectors:
            texts = [contents[i] or "" for i in need_embed_idx]
            vectors_batch = await self._embed_batch_async(texts)
            for idx, vec in zip(need_embed_idx, vectors_batch):
                items[idx]["vector"] = vec

        # Prepare insert
        ids: List[str] = []
        rows: List[Tuple[Any, ...]] = []
        labels: List[int] = []
        vectors: List[List[float]] = []

        for it in items:
            content = it.get("content")
            vector = it.get("vector")
            metadata = it.get("metadata")
            item_id = it.get("id")
            if item_id is None:
                base = (content or "") + "\n" + json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
                import hashlib

                item_id = hashlib.md5(base.encode("utf-8")).hexdigest()

            if self._enable_vectors:
                if vector is None:
                    raise ValueError("Vector missing after embedding phase")

            meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
            ids.append(item_id)
            rows.append((item_id, content, meta_json, _to_blob(vector) if (self._enable_vectors and vector is not None) else None))
            if self._enable_vectors:
                labels.append(self._get_or_create_label_for_id(item_id))
                assert vector is not None
                vectors.append(vector)

        with self._rw.write_locked():
            # DB upserts
            for (iid, content, meta_json, vec_blob) in rows:
                self._upsert_item(
                    item_id=iid,
                    content=content,
                    metadata_json=meta_json,
                    vector_blob=vec_blob,
                )
            self.db.commit()

            if self._enable_vectors:
                # Initialize index if needed
                if self._index is None and vectors:
                    self._init_index_for_dim(len(vectors[0]))
                assert self._index is not None
                self._index.add_many(labels, vectors)
                self._maybe_persist_index()

        return ids

    async def upsert(self, item_id: str, content: Optional[str] = None, vector: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Upsert by id. If vector is None and content provided, compute embedding."""
        if self._enable_vectors:
            if vector is None and content is not None:
                vector = await self._embed_text_async(content)
            if vector is None:
                # Try to reuse stored vector if exists
                old = self.get_by_id(item_id)
                if old[1]:
                    vector = old[1]
            if vector is None:
                raise ValueError("Vector is required when neither content nor stored vector exist")

        if content is None:
            # keep existing content if any
            cur = self.db.execute("SELECT content FROM items WHERE id = ?", (item_id,))
            row = self.db.fetchone(cur)
            content = row[0] if row else ""

        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

        with self._rw.write_locked():
            self._upsert_item(
                item_id=item_id,
                content=content,
                metadata_json=meta_json,
                vector_blob=_to_blob(vector) if (self._enable_vectors and vector is not None) else None,
            )
            self.db.commit()

            if self._enable_vectors:
                label = self._get_or_create_label_for_id(item_id)
                assert self._index is not None
                self._index.add_or_update(label, vector)  # type: ignore[arg-type]
                self._maybe_persist_index()
        return item_id

    def get_by_id(self, item_id: str) -> Tuple[Optional[str], Optional[List[float]], Dict[str, Any]]:
        cur = self.db.execute("SELECT content, vector, metadata FROM items WHERE id = ?", (item_id,))
        row = self.db.fetchone(cur)
        if not row:
            return None, None, {}
        content = row[0]
        vector = _from_blob(row[1]) if row[1] is not None else None
        metadata = json.loads(row[2]) if row[2] else {}
        return content, vector, metadata

    def get_by_content(self, content: str) -> Tuple[Optional[List[float]], Dict[str, Any]]:
        # ID is hash(content) consistent with legacy behavior
        import hashlib

        content_id = hashlib.md5(content.encode("utf-8")).hexdigest()
        _, vector, metadata = self.get_by_id(content_id)
        return vector, metadata

    async def get_by_vector(self, vector: List[float], threshold: float = 0.9) -> Tuple[Optional[str], Dict[str, Any]]:
        if not self._enable_vectors:
            logger.warning("get_by_vector is not supported: vectors disabled")
            return None, {}
        res = await self.search(vector, top_k=1, threshold=threshold)
        if not res:
            return None, {}
        return res[0][0], res[0][2]

    def delete_by_id(self, item_id: str) -> bool:
        with self._rw.write_locked():
            if self._enable_vectors:
                label = self._label_for_id(item_id)
                if label is not None:
                    try:
                        assert self._index is not None
                        self._index.delete(label)
                    except Exception:
                        pass
            cur = self.db.execute("DELETE FROM items WHERE id = ?", (item_id,))
            self.db.commit()
            # Keep labels mapping for potential re-use; do not remove to avoid label collisions
            return cur.rowcount > 0 if hasattr(cur, "rowcount") else True

    def delete_by_content(self, content: str) -> bool:
        import hashlib

        item_id = hashlib.md5(content.encode("utf-8")).hexdigest()
        return self.delete_by_id(item_id)

    async def delete_by_vector(self, vector: List[float], threshold: float = 0.9) -> bool:
        if not self._enable_vectors:
            logger.warning("delete_by_vector is not supported: vectors disabled")
            return False
        res = await self.search(vector, top_k=1, threshold=threshold)
        if not res:
            return False
        item_id = res[0][3]["id"] if "id" in res[0][3] else None
        if not item_id:
            return False
        return self.delete_by_id(item_id)

    async def search(
        self,
        query: Any,
        top_k: int = 5,
        threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
        ef_search: Optional[int] = None,
        near_duplicate_delta: float = 0.0,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search by text or vector.

        Returns list of (content, similarity, metadata). Metadata contains "id" as well.

        Args:
            query: Text or vector to search by.
            top_k: Max number of results to return.
            threshold: Minimum cosine similarity required to include a result.
            metadata_filter: Optional exact-match filter on metadata.
            ef_search: Optional index-specific search breadth parameter.
            near_duplicate_delta: If > 0, exclude results whose similarity to the query
                is within `near_duplicate_delta` of 1.0 (i.e., treat as near-duplicates of the query).
                For example, 0.01 will drop items with sim >= 0.99.
        """
        if not self._enable_vectors:
            logger.warning("search is not supported: vectors disabled")
            return []
        if isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            q_vec = query  # type: ignore
        else:
            q_vec = await self._embed_text_async(str(query))

        # Clamp k to number of items to avoid hnswlib contiguous array error when k > size
        total = self.count()
        if total <= 0:
            return []
        k = int(max(1, min(int(top_k), total)))

        with self._rw.read_locked():
            assert self._index is not None
            labels, distances = self._index.search(q_vec, top_k=k, ef_search=(max(k, int(ef_search)) if ef_search is not None else None))

            results: List[Tuple[str, float, Dict[str, Any]]] = []
            for lb, dist in zip(labels, distances):
                # cosine distance -> similarity
                sim = float(max(-1.0, min(1.0, 1.0 - float(dist))))
                if sim < threshold:
                    continue

                # Exclude near-duplicates that are almost identical to the query vector
                # If near_duplicate_delta == 0.0 (default), do nothing
                if near_duplicate_delta > 0.0:
                    # If similarity is within delta of 1.0, skip this result
                    if (1.0 - sim) <= near_duplicate_delta:
                        continue

                # map label -> id
                cur = self.db.execute("SELECT id FROM labels WHERE label = ?", (int(lb),))
                row = self.db.fetchone(cur)
                if not row:
                    continue
                item_id = row[0]

                cur = self.db.execute("SELECT content, metadata FROM items WHERE id = ?", (item_id,))
                row2 = self.db.fetchone(cur)
                if not row2:
                    continue
                content = row2[0]
                md = json.loads(row2[1]) if row2[1] else {}
                md["id"] = item_id

                if metadata_filter and not _match_metadata(md, metadata_filter):
                    continue

                results.append((content, sim, md))
            return results

    # Utilities
    def count(self) -> int:
        cur = self.db.execute("SELECT COUNT(*) FROM items")
        row = self.db.fetchone(cur)
        return int(row[0]) if row else 0

    def clear(self) -> None:
        with self._rw.write_locked():
            self.db.execute("DELETE FROM items")
            self.db.commit()
            # do not clear labels to preserve label continuity; optional: vacuum index by rebuilding
            self._rebuild_index_from_db()
            self._maybe_persist_index(force=True)

    def get_all(self) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        items: List[Tuple[str, List[float], Dict[str, Any]]] = []
        cur = self.db.execute("SELECT id, content, vector, metadata FROM items")
        rows = self.db.fetchall(cur)
        for row in rows:
            content = row[1]
            vector = _from_blob(row[2]) if (self._enable_vectors and row[2] is not None) else []
            md = json.loads(row[3]) if row[3] else {}
            md["id"] = row[0]
            items.append((content, vector, md))
        return items

    def get_all_contents(self) -> List[str]:
        cur = self.db.execute("SELECT content FROM items")
        return [r[0] for r in self.db.fetchall(cur)]

    def get_all_vectors(self) -> Dict[str, List[float]]:
        if not self._enable_vectors:
            return {}
        cur = self.db.execute("SELECT id, vector FROM items WHERE vector IS NOT NULL")
        rows = self.db.fetchall(cur)
        return {r[0]: _from_blob(r[1]) for r in rows}

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        cur = self.db.execute("SELECT id, metadata FROM items")
        rows = self.db.fetchall(cur)
        out: List[Dict[str, Any]] = []
        for r in rows:
            md = json.loads(r[1]) if r[1] else {}
            md["id"] = r[0]
            out.append(md)
        return out

    def get_info(self) -> Dict[str, Any]:
        info = {
            "name": self.name,
            "path": str(self.base_dir),
            "count": self.count(),
            "vectors_enabled": self._enable_vectors,
            "embedder_method": (self._get_embedder().method if self._enable_vectors else None),
            "vector_dimension": (self._get_vector_dim() if self._enable_vectors else None),
            "index": (self._index.info() if self._enable_vectors and self._index is not None else {"backend": "disabled"}),
            "db_backend": self.db.backend,
        }
        return info

    def _maybe_persist_index(self, force: bool = False) -> None:
        # Persist index no more often than every 2 seconds unless forced
        now = time.time()
        last = getattr(self, "_last_index_persist", 0.0)
        if not self._enable_vectors:
            return
        if force or (now - last) > 2.0:
            try:
                assert self._index is not None
                assert self._index_path is not None
                self._index.save(self._index_path)
                self._last_index_persist = now
            except Exception as e:
                logger.warning(f"Index persistence failed: {e}")

    # ---------- Upsert helpers (handle DuckDB vs SQLite) ----------
    def _upsert_meta(self, key: str, value: str) -> None:
        if self.db.backend == "duckdb":
            self.db.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
        else:
            self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))

    def _upsert_label(self, label: int, item_id: str) -> None:
        if self.db.backend == "duckdb":
            self.db.execute(
                "INSERT INTO labels(label, id) VALUES(?, ?) ON CONFLICT(label) DO UPDATE SET id=excluded.id",
                (label, item_id),
            )
        else:
            self.db.execute("INSERT OR REPLACE INTO labels(label, id) VALUES(?, ?)", (label, item_id))

    def _upsert_item(self, *, item_id: str, content: str, metadata_json: Optional[str], vector_blob: Optional[bytes]) -> None:
        if self.db.backend == "duckdb":
            self.db.execute(
                """
                INSERT INTO items(id, content, metadata, vector)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  content=excluded.content,
                  metadata=excluded.metadata,
                  vector=excluded.vector
                """,
                (item_id, content, metadata_json, vector_blob),
            )
        else:
            self.db.execute(
                "INSERT OR REPLACE INTO items(id, content, metadata, vector) VALUES(?, ?, ?, ?)",
                (item_id, content, metadata_json, vector_blob),
            )


def _match_metadata(md: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    """Simple equality-based metadata filter with dotted key support.

    Example: {"source": "doc", "author.name": "alice"}
    """

    def _get(d: Dict[str, Any], dotted: str) -> Any:
        parts = dotted.split(".")
        cur: Any = d
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur

    for key, val in flt.items():
        if _get(md, key) != val:
            return False
    return True


