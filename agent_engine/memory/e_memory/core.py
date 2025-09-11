"""
EMemory core implementation using SQLite and HNSWIndex.
"""

from __future__ import annotations

import os
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import hnswlib
import numpy as np

from ...agent_logger.agent_logger import AgentLogger
from ...utils.project_root import get_project_root
from .models import Record


logger = AgentLogger(__name__)


class EMemory:
    """
    A lightweight vector memory implementation using SQLite and HNSWIndex.
    
    Features:
    - SQLite database for persistent storage
    - HNSW indexing for efficient vector similarity search
    - Basic CRUD operations
    - Thread-safe operations
    """

    def __init__(
        self,
        name: str,
        persist_dir: Optional[str] = None,
        dimension: int = 1536,  # Default OpenAI embedding dimension
        max_elements: int = 10000,
        ef_construction: int = 300,
        M: int = 24,
        space: str = "cosine"
    ):
        """
        Initialize EMemory.
        
        Args:
            name: Memory name
            persist_dir: Storage directory (optional, defaults to root/.memory/name)
            dimension: Vector dimension
            max_elements: Maximum number of elements in HNSW index
            ef_construction: HNSW construction parameter
            M: HNSW M parameter
            space: Distance metric ('cosine', 'l2', 'ip')
        """
        self.name = name
        self.dimension = dimension
        self.max_elements = max_elements
        self.space = space
        
        # Determine storage directory [[memory:8183017]]
        if persist_dir is None:
            root = get_project_root()
            self.persist_dir = root / ".memory" / name
        else:
            self.persist_dir = Path(persist_dir)
        
        # Create directory if it doesn't exist
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database and index
        self.db_path = self.persist_dir / "memory.sqlite"
        self.index_path = self.persist_dir / "index.bin"
        
        # Thread safety
        self._lock = threading.Lock()
        
        # ID mapping for HNSW (string ID -> integer ID)
        self._id_to_int = {}
        self._int_to_id = {}
        self._next_int_id = 0
        
        # Initialize components
        self._init_database()
        self._init_hnsw_index(ef_construction, M)
        self._load_id_mappings()
        
        logger.info(f"EMemory '{name}' initialized at {self.persist_dir}")

    def _init_database(self) -> None:
        """Initialize SQLite database with required schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    attributes TEXT,
                    content TEXT,
                    vector BLOB,
                    timestamp TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON records(timestamp)")
            
            # Table for ID mappings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS id_mappings (
                    string_id TEXT PRIMARY KEY,
                    int_id INTEGER UNIQUE
                )
            """)
            conn.commit()

    def _init_hnsw_index(self, ef_construction: int, M: int) -> None:
        """Initialize or load HNSW index."""
        self.hnsw_index = hnswlib.Index(space=self.space, dim=self.dimension)
        
        if self.index_path.exists():
            # Load existing index
            try:
                self.hnsw_index.load_index(str(self.index_path))
                logger.info(f"Loaded existing HNSW index with {self.hnsw_index.get_current_count()} elements")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new index.")
                self.hnsw_index.init_index(max_elements=self.max_elements, ef_construction=ef_construction, M=M)
        else:
            # Create new index
            self.hnsw_index.init_index(max_elements=self.max_elements, ef_construction=ef_construction, M=M)
            logger.info("Created new HNSW index")

    def _vector_to_blob(self, vector: List[float]) -> bytes:
        """Convert vector to binary blob for storage."""
        return np.asarray(vector, dtype=np.float32).tobytes()

    def _blob_to_vector(self, blob: bytes) -> List[float]:
        """Convert binary blob back to vector."""
        return np.frombuffer(blob, dtype=np.float32).tolist()

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO8601 format."""
        return datetime.utcnow().isoformat() + "Z"

    def _load_id_mappings(self) -> None:
        """Load existing ID mappings from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT string_id, int_id FROM id_mappings")
            for string_id, int_id in cursor.fetchall():
                self._id_to_int[string_id] = int_id
                self._int_to_id[int_id] = string_id
                self._next_int_id = max(self._next_int_id, int_id + 1)

    def _get_or_create_int_id(self, string_id: str) -> int:
        """Get or create integer ID for string ID."""
        if string_id in self._id_to_int:
            return self._id_to_int[string_id]
        
        int_id = self._next_int_id
        self._next_int_id += 1
        
        self._id_to_int[string_id] = int_id
        self._int_to_id[int_id] = string_id
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO id_mappings (string_id, int_id) VALUES (?, ?)", (string_id, int_id))
            conn.commit()
        
        return int_id

    def _remove_int_id(self, string_id: str) -> None:
        """Remove integer ID mapping for string ID."""
        if string_id in self._id_to_int:
            int_id = self._id_to_int[string_id]
            del self._id_to_int[string_id]
            del self._int_to_id[int_id]
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM id_mappings WHERE string_id = ?", (string_id,))
                conn.commit()

    def add(self, record: Record) -> str:
        """
        Add a record to the memory.
        
        Args:
            record: Record object to add
            
        Returns:
            Record ID
        """
        with self._lock:
            # Generate ID and timestamp if not provided
            record_id = record.id or self._generate_id()
            record_timestamp = record.timestamp or self._get_current_timestamp()
            record_attributes = record.attributes or {}
            
            # Insert into database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO records (id, attributes, content, vector, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    record_id,
                    json.dumps(record_attributes),
                    record.content,
                    self._vector_to_blob(record.vector) if record.vector else None,
                    record_timestamp
                ))
                conn.commit()
            
            # Add to HNSW index if vector is provided
            if record.vector:
                try:
                    int_id = self._get_or_create_int_id(record_id)
                    
                    # Check if ID already exists in index
                    try:
                        self.hnsw_index.mark_deleted(int_id)
                    except RuntimeError:
                        pass  # ID doesn't exist, which is fine
                    
                    self.hnsw_index.add_items([record.vector], [int_id])
                    self.hnsw_index.save_index(str(self.index_path))
                except Exception as e:
                    logger.error(f"Failed to add vector to HNSW index: {e}")
            
            logger.debug(f"Added record {record_id} to EMemory")
            return record_id

    def get(self, id: str) -> Optional[Record]:
        """
        Get a record by ID.
        
        Args:
            id: Record ID
            
        Returns:
            Record if found, None otherwise
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, attributes, content, vector, timestamp
                    FROM records WHERE id = ?
                """, (id,))
                row = cursor.fetchone()
                
                if row:
                    record_id, attributes_json, content, vector_blob, timestamp = row
                    return Record(
                        id=record_id,
                        attributes=json.loads(attributes_json) if attributes_json else {},
                        content=content,
                        vector=self._blob_to_vector(vector_blob) if vector_blob else None,
                        timestamp=timestamp
                    )
                return None

    def update(self, record: Record) -> bool:
        """
        Update an existing record.
        
        Args:
            record: Record object with updated data (must have id)
            
        Returns:
            True if updated, False if record not found
        """
        if not record.id:
            raise ValueError("Record must have an ID for update")
        
        # Check if record exists
        existing = self.get(record.id)
        if not existing:
            return False
        
        # Update the record (reuse add method which handles INSERT OR REPLACE)
        self.add(record)
        return True

    def delete(self, id: str) -> bool:
        """
        Delete a record by ID.
        
        Args:
            id: Record ID
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM records WHERE id = ?", (id,))
                deleted = cursor.rowcount > 0
                conn.commit()
            
            # Mark as deleted in HNSW index and remove ID mapping
            if deleted:
                try:
                    if id in self._id_to_int:
                        int_id = self._id_to_int[id]
                        self.hnsw_index.mark_deleted(int_id)
                        self.hnsw_index.save_index(str(self.index_path))
                        self._remove_int_id(id)
                except RuntimeError:
                    pass  # ID might not exist in index
                
                logger.debug(f"Deleted record {id} from EMemory")
            
            return deleted

    def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Record]:
        """
        List all records.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of records
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT id, attributes, content, vector, timestamp
                    FROM records ORDER BY timestamp DESC
                """
                params = []
                
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                
                if offset > 0:
                    query += " OFFSET ?"
                    params.append(offset)
                
                cursor = conn.execute(query, params)
                records = []
                
                for row in cursor.fetchall():
                    record_id, attributes_json, content, vector_blob, timestamp = row
                    records.append(Record(
                        id=record_id,
                        attributes=json.loads(attributes_json) if attributes_json else {},
                        content=content,
                        vector=self._blob_to_vector(vector_blob) if vector_blob else None,
                        timestamp=timestamp
                    ))
                
                return records

    def count(self) -> int:
        """
        Get total number of records.
        
        Returns:
            Number of records
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM records")
                return cursor.fetchone()[0]

    def clear(self) -> None:
        """Clear all records from memory."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM records")
                conn.commit()
            
            # Clear ID mappings
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM id_mappings")
                conn.commit()
            
            self._id_to_int.clear()
            self._int_to_id.clear()
            self._next_int_id = 0
            
            # Recreate HNSW index
            self.hnsw_index = hnswlib.Index(space=self.space, dim=self.dimension)
            self.hnsw_index.init_index(max_elements=self.max_elements, ef_construction=200, M=16)
            
            # Remove index file
            if self.index_path.exists():
                self.index_path.unlink()
            
            logger.info("Cleared all records from EMemory")

    def search_similar(
        self,
        query_vector: List[float],
        k: int = 10,
        ef: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors using HNSW index.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            ef: Search parameter (optional)
            
        Returns:
            List of (record_id, distance) tuples
        """
        with self._lock:
            if self.hnsw_index.get_current_count() == 0:
                return []
            
            # Set ef parameter if provided
            if ef is not None:
                self.hnsw_index.set_ef(ef)
            
            try:
                labels, distances = self.hnsw_index.knn_query([query_vector], k=k)
                # Convert integer IDs back to string IDs
                results = []
                for label, dist in zip(labels[0], distances[0]):
                    if label in self._int_to_id:
                        results.append((self._int_to_id[label], float(dist)))
                return results
            except Exception as e:
                logger.error(f"Failed to search similar vectors: {e}")
                return []

    def search_similar_records(
        self,
        query_vector: List[float],
        k: int = 10,
        ef: Optional[int] = None
    ) -> List[Tuple[Record, float]]:
        """
        Search for similar records using HNSW index.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            ef: Search parameter (optional)
            
        Returns:
            List of (Record, distance) tuples
        """
        similar_ids = self.search_similar(query_vector, k, ef)
        results = []
        
        for record_id, distance in similar_ids:
            record = self.get(record_id)
            if record:
                results.append((record, distance))
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_records = self.count()
            index_count = self.hnsw_index.get_current_count() if hasattr(self.hnsw_index, 'get_current_count') else 0
            
            return {
                "name": self.name,
                "total_records": total_records,
                "indexed_vectors": index_count,
                "dimension": self.dimension,
                "persist_dir": str(self.persist_dir),
                "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
                "index_size_bytes": self.index_path.stat().st_size if self.index_path.exists() else 0
            }
