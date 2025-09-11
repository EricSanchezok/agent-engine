"""
EMemory core implementation using SQLite and ChromaDB.
"""

from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np
import chromadb
from chromadb.config import Settings

from ...agent_logger.agent_logger import AgentLogger
from ...utils.project_root import get_project_root
from .models import Record


logger = AgentLogger(__name__)


class EMemory:
    """
    A lightweight vector memory implementation using SQLite and ChromaDB.
    
    Features:
    - SQLite database for metadata storage
    - ChromaDB for efficient vector similarity search
    - File-based storage with concurrent read/write support
    - Basic CRUD operations
    - One EMemory instance = one SQLite file + one ChromaDB collection
    """

    def __init__(
        self,
        name: str,
        persist_dir: Optional[str] = None,
        dimension: int = 1536,  # Default OpenAI embedding dimension
        distance_metric: str = "cosine"
    ):
        """
        Initialize EMemory.
        
        Args:
            name: Memory name (used for file naming)
            persist_dir: Storage directory (optional, defaults to root/.memory/name)
            dimension: Vector dimension
            distance_metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.name = name
        self.dimension = dimension
        self.distance_metric = distance_metric
        
        # Determine storage directory [[memory:8183017]]
        if persist_dir is None:
            root = get_project_root()
            self.persist_dir = root / ".memory" / name
        else:
            self.persist_dir = Path(persist_dir)
        
        # Create directory if it doesn't exist
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths - one EMemory = one SQLite file + one ChromaDB collection
        self.sqlite_path = self.persist_dir / f"{name}.sqlite"
        self.chroma_path = self.persist_dir / f"{name}_chroma"
        
        # Initialize SQLite database
        self._init_sqlite()
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        logger.info(f"EMemory '{name}' initialized at {self.persist_dir}")
        logger.info(f"SQLite: {self.sqlite_path}")
        logger.info(f"ChromaDB: {self.chroma_path}")

    def _init_sqlite(self) -> None:
        """Initialize SQLite database with required schema."""
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    attributes TEXT,
                    content TEXT,
                    timestamp TEXT,
                    has_vector INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON records(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_has_vector ON records(has_vector)")
            conn.commit()

    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client with file storage
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.name,
                metadata={"hnsw:space": self.distance_metric}
            )
            
            logger.info(f"ChromaDB collection '{self.name}' ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise


    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO8601 format."""
        return datetime.utcnow().isoformat() + "Z"


    def add(self, record: Record) -> str:
        """
        Add a record to the memory.
        
        Args:
            record: Record object to add
            
        Returns:
            Record ID
        """
        # Generate ID and timestamp if not provided
        record_id = record.id or self._generate_id()
        record_timestamp = record.timestamp or self._get_current_timestamp()
        record_attributes = record.attributes or {}
        has_vector = 1 if record.vector else 0
        
        # Insert into SQLite
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO records (id, attributes, content, timestamp, has_vector)
                VALUES (?, ?, ?, ?, ?)
            """, (
                record_id,
                json.dumps(record_attributes),
                record.content,
                record_timestamp,
                has_vector
            ))
            conn.commit()
        
        # Add to ChromaDB if vector is provided
        if record.vector:
            try:
                self.collection.add(
                    embeddings=[record.vector],
                    ids=[record_id],
                    metadatas=[{
                        "content": record.content,
                        "attributes": json.dumps(record_attributes),
                        "timestamp": record_timestamp
                    }]
                )
            except Exception as e:
                logger.error(f"Failed to add vector to ChromaDB: {e}")
        
        logger.debug(f"Added record {record_id} to EMemory")
        return record_id

    def add_batch(self, records: List[Record]) -> List[str]:
        """
        Add multiple records to the memory in batch.
        
        Args:
            records: List of Record objects to add
            
        Returns:
            List of record IDs
        """
        if not records:
            return []
        
        # ChromaDB has a maximum batch size limit
        MAX_CHROMADB_BATCH_SIZE = 5000
        
        all_record_ids = []
        
        # Process records in chunks to respect ChromaDB batch size limits
        for i in range(0, len(records), MAX_CHROMADB_BATCH_SIZE):
            chunk = records[i:i + MAX_CHROMADB_BATCH_SIZE]
            chunk_ids = self._add_batch_chunk(chunk)
            all_record_ids.extend(chunk_ids)
        
        logger.debug(f"Added {len(records)} records to EMemory in batch")
        return all_record_ids
    
    def _add_batch_chunk(self, records: List[Record]) -> List[str]:
        """Add a chunk of records to the memory."""
        record_ids = []
        vectors_to_add = []
        metadatas_to_add = []
        ids_for_vectors = []
        
        # Prepare all records for database insertion
        db_data = []
        for record in records:
            # Generate ID and timestamp if not provided
            record_id = record.id or self._generate_id()
            record_timestamp = record.timestamp or self._get_current_timestamp()
            record_attributes = record.attributes or {}
            has_vector = 1 if record.vector else 0
            
            record_ids.append(record_id)
            
            # Prepare database data
            db_data.append((
                record_id,
                json.dumps(record_attributes),
                record.content,
                record_timestamp,
                has_vector
            ))
            
            # Prepare vector data for ChromaDB
            if record.vector:
                vectors_to_add.append(record.vector)
                metadatas_to_add.append({
                    "content": record.content,
                    "attributes": json.dumps(record_attributes),
                    "timestamp": record_timestamp
                })
                ids_for_vectors.append(record_id)
        
        # Batch insert into SQLite
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO records (id, attributes, content, timestamp, has_vector)
                VALUES (?, ?, ?, ?, ?)
            """, db_data)
            conn.commit()
        
        # Batch add to ChromaDB if vectors are provided
        if vectors_to_add:
            try:
                self.collection.add(
                    embeddings=vectors_to_add,
                    ids=ids_for_vectors,
                    metadatas=metadatas_to_add
                )
            except Exception as e:
                logger.error(f"Failed to add vectors to ChromaDB: {e}")
        
        return record_ids

    def get(self, id: str) -> Optional[Record]:
        """
        Get a record by ID.
        
        Args:
            id: Record ID
            
        Returns:
            Record if found, None otherwise
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.execute("""
                SELECT id, attributes, content, timestamp, has_vector
                FROM records WHERE id = ?
            """, (id,))
            row = cursor.fetchone()
            
            if row:
                record_id, attributes_json, content, timestamp, has_vector = row
                return Record(
                    id=record_id,
                    attributes=json.loads(attributes_json) if attributes_json else {},
                    content=content,
                    vector=None,  # We don't return the vector in this method
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
        # Delete from SQLite
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.execute("DELETE FROM records WHERE id = ?", (id,))
            deleted = cursor.rowcount > 0
            conn.commit()
        
        # Delete from ChromaDB if it exists
        if deleted:
            try:
                self.collection.delete(ids=[id])
            except Exception as e:
                logger.warning(f"Failed to delete vector from ChromaDB: {e}")
        
        if deleted:
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
        with sqlite3.connect(self.sqlite_path) as conn:
            query = """
                SELECT id, attributes, content, timestamp, has_vector
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
                record_id, attributes_json, content, timestamp, has_vector = row
                records.append(Record(
                    id=record_id,
                    attributes=json.loads(attributes_json) if attributes_json else {},
                    content=content,
                    vector=None,  # We don't return the vector in this method
                    timestamp=timestamp
                ))
            
            return records

    def count(self) -> int:
        """
        Get total number of records.
        
        Returns:
            Number of records
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM records")
            return cursor.fetchone()[0]

    def clear(self) -> None:
        """Clear all records from memory."""
        # Clear SQLite
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("DELETE FROM records")
            conn.commit()
        
        # Clear ChromaDB collection
        try:
            # Get all IDs and delete them
            all_docs = self.collection.get()
            if all_docs and all_docs.get('ids'):
                self.collection.delete(ids=all_docs['ids'])
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB collection: {e}")
        
        logger.info("Cleared all records from EMemory")

    def search_similar(
        self,
        query_vector: List[float],
        k: int = 10,
        ef: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors using ChromaDB.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            ef: Search parameter (ignored for ChromaDB)
            
        Returns:
            List of (record_id, distance) tuples
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k
            )
            
            # ChromaDB returns distances, convert to similarity scores
            record_ids = results['ids'][0]
            distances = results['distances'][0]
            
            return [(record_id, float(distance)) for record_id, distance in zip(record_ids, distances)]
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
        Search for similar records using ChromaDB.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            ef: Search parameter (ignored for ChromaDB)
            
        Returns:
            List of (Record, distance) tuples
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=['metadatas']
            )
            
            if not results or not results.get('ids') or not results['ids']:
                return []
            
            record_ids = results['ids'][0] if results['ids'] else []
            distances = results['distances'][0] if results['distances'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            
            if not record_ids:
                return []
            
            results_list = []
            for record_id, distance, metadata in zip(record_ids, distances, metadatas):
                # Create Record from ChromaDB metadata
                record = Record(
                    id=record_id,
                    attributes=json.loads(metadata.get("attributes", "{}")),
                    content=metadata.get("content", ""),
                    vector=None,  # We don't return the vector in this method
                    timestamp=metadata.get("timestamp", "")
                )
                results_list.append((record, float(distance)))
            
            return results_list
        except Exception as e:
            logger.error(f"Failed to search similar records: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Statistics dictionary
        """
        total_records = self.count()
        
        # Get ChromaDB collection info
        try:
            collection_count = self.collection.count()
        except Exception:
            collection_count = 0
        
        # Calculate file sizes
        sqlite_size = self.sqlite_path.stat().st_size if self.sqlite_path.exists() else 0
        chroma_size = sum(f.stat().st_size for f in self.chroma_path.rglob('*') if f.is_file()) if self.chroma_path.exists() else 0
        
        return {
            "name": self.name,
            "total_records": total_records,
            "indexed_vectors": collection_count,
            "dimension": self.dimension,
            "persist_dir": str(self.persist_dir),
            "sqlite_size_bytes": sqlite_size,
            "chroma_size_bytes": chroma_size,
            "sqlite_path": str(self.sqlite_path),
            "chroma_path": str(self.chroma_path)
        }
