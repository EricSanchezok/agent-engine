"""
from __future__ import annotations
EMemory core implementation using SQLite and ChromaDB.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

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
        distance_metric: str = "cosine"
    ):
        """
        Initialize EMemory.
        
        Args:
            name: Memory name (used for file naming)
            persist_dir: Storage directory (optional, defaults to root/.memory/name)
            distance_metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.name = name
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


    def add(self, record: Record) -> bool:
        """
        Add a record to the memory.
        
        Args:
            record: Record object to add
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Generate ID and timestamp if not provided
            record_id = record.id or self._generate_id()
            record_timestamp = record.timestamp or self._get_current_timestamp()
            record_attributes = record.attributes or {}
            has_vector = 1 if record.vector else 0
            
            # Try to add to ChromaDB if vector is provided
            if has_vector:
                try:
                    # ChromaDB only stores ID and vector - no metadata to avoid duplication
                    self.collection.add(
                        embeddings=[record.vector],
                        ids=[record_id]
                    )
                    logger.debug(f"Successfully added vector for record {record_id} to ChromaDB")
                except Exception as e:
                    logger.error(f"Failed to add vector to ChromaDB for record {record_id}: {e}")
                    # Set has_vector to 0 since ChromaDB storage failed
                    has_vector = 0
            
            # Always insert into SQLite (contains all metadata)
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
            
            logger.debug(f"Added record {record_id} to EMemory (has_vector={has_vector})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add record to EMemory: {e}")
            return False

    def update(self, record: Record) -> bool:
        """
        Update an existing record in the memory.
        
        This method properly handles vector updates, including removing old vectors
        when updating a record to have no vector, preventing orphaned vectors.
        
        Args:
            record: Record object to update (must have an ID)
            
        Returns:
            True if successfully updated, False otherwise
        """
        record_id = record.id
        if not record_id or not self.exists(record_id):
            logger.warning(f"Record {record_id} not found, cannot update.")
            return False

        try:
            # 1. Check if old record has vector
            old_has_vector = self.has_vector(record_id)
            
            # 2. Prepare new record data
            new_has_vector = 1 if record.vector else 0
            
            # 3. Handle vector storage logic
            if new_has_vector and record.vector:
                # Add or update vector in ChromaDB
                try:
                    self.collection.add(embeddings=[record.vector], ids=[record_id])
                    logger.debug(f"Successfully updated vector for record {record_id} in ChromaDB")
                except Exception as e:
                    logger.error(f"Failed to update vector in ChromaDB for record {record_id}: {e}")
                    new_has_vector = 0  # Set to 0 if ChromaDB update failed
            elif old_has_vector and not new_has_vector:
                # Key: If old record has vector but new record doesn't, delete from ChromaDB
                try:
                    self.collection.delete(ids=[record_id])
                    logger.debug(f"Successfully deleted old vector for record {record_id} from ChromaDB")
                except Exception as e:
                    logger.warning(f"Failed to delete old vector from ChromaDB for record {record_id}: {e}")

            # 4. Update SQLite metadata
            record_timestamp = record.timestamp or self._get_current_timestamp()
            record_attributes = record.attributes or {}
            
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute("""
                    UPDATE records 
                    SET attributes = ?, content = ?, timestamp = ?, has_vector = ?
                    WHERE id = ?
                """, (
                    json.dumps(record_attributes),
                    record.content,
                    record_timestamp,
                    new_has_vector,
                    record_id
                ))
                conn.commit()
                
            logger.debug(f"Updated record {record_id} (has_vector={old_has_vector} -> {new_has_vector})")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update record {record_id}: {e}")
            return False

    def update_batch(self, records: List[Record]) -> bool:
        """
        Update multiple existing records in the memory in batch.
        
        This method efficiently handles batch updates with proper vector management,
        including removing old vectors when updating records to have no vector.
        
        Args:
            records: List of Record objects to update (all must have IDs)
            
        Returns:
            True if all records were successfully updated, False otherwise
        """
        if not records:
            return True
        
        try:
            # ChromaDB has a maximum batch size limit
            MAX_CHROMADB_BATCH_SIZE = 5000
            
            # Process records in chunks to respect ChromaDB batch size limits
            for i in range(0, len(records), MAX_CHROMADB_BATCH_SIZE):
                chunk = records[i:i + MAX_CHROMADB_BATCH_SIZE]
                success = self._update_batch_chunk(chunk)
                if not success:
                    logger.error(f"Failed to update batch chunk starting at index {i}")
                    return False
            
            logger.debug(f"Updated {len(records)} records in EMemory batch")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update batch records in EMemory: {e}")
            return False
    
    def _update_batch_chunk(self, records: List[Record]) -> bool:
        """Update a chunk of records in the memory."""
        try:
            # Step 1: Validate all records have IDs and exist
            record_ids = [record.id for record in records if record.id]
            if len(record_ids) != len(records):
                logger.error("All records must have IDs for batch update")
                return False
            
            # Check existence of all records
            existing_map = self.exists_batch(record_ids)
            missing_ids = [rid for rid, exists in existing_map.items() if not exists]
            if missing_ids:
                logger.error(f"Records not found for batch update: {missing_ids}")
                return False
            
            # Step 2: Get current vector status for all records
            old_vector_map = self.has_vector_batch(record_ids)
            
            # Step 3: Prepare data for batch operations
            vectors_to_add = []
            ids_for_vectors = []
            ids_to_delete_from_chroma = []
            db_updates = []
            
            for record in records:
                record_id = record.id
                old_has_vector = old_vector_map.get(record_id, False)
                new_has_vector = 1 if record.vector else 0
                
                # Prepare timestamp and attributes
                record_timestamp = record.timestamp or self._get_current_timestamp()
                record_attributes = record.attributes or {}
                
                # Handle vector operations
                if new_has_vector and record.vector:
                    # Record has vector - add/update in ChromaDB
                    vectors_to_add.append(record.vector)
                    ids_for_vectors.append(record_id)
                elif old_has_vector and not new_has_vector:
                    # Record had vector but now doesn't - delete from ChromaDB
                    ids_to_delete_from_chroma.append(record_id)
                
                # Prepare SQLite update data
                db_updates.append((
                    json.dumps(record_attributes),
                    record.content,
                    record_timestamp,
                    new_has_vector,
                    record_id
                ))
            
            # Step 4: Batch operations on ChromaDB
            successful_vector_ids = set()
            
            # Add/update vectors in ChromaDB
            if vectors_to_add:
                try:
                    self.collection.add(
                        embeddings=vectors_to_add,
                        ids=ids_for_vectors
                    )
                    successful_vector_ids = set(ids_for_vectors)
                    logger.debug(f"Successfully updated {len(vectors_to_add)} vectors in ChromaDB")
                except Exception as e:
                    logger.error(f"Failed to update vectors in ChromaDB: {e}")
                    # Continue with SQLite updates even if ChromaDB fails
            
            # Delete vectors from ChromaDB
            if ids_to_delete_from_chroma:
                try:
                    self.collection.delete(ids=ids_to_delete_from_chroma)
                    logger.debug(f"Successfully deleted {len(ids_to_delete_from_chroma)} vectors from ChromaDB")
                except Exception as e:
                    logger.warning(f"Failed to delete vectors from ChromaDB: {e}")
            
            # Step 5: Update has_vector status based on ChromaDB success
            final_db_updates = []
            for i, (attributes_json, content, timestamp, original_has_vector, record_id) in enumerate(db_updates):
                # Set has_vector to 1 only if ChromaDB storage was successful
                final_has_vector = 1 if record_id in successful_vector_ids else 0
                final_db_updates.append((
                    attributes_json,
                    content,
                    timestamp,
                    final_has_vector,
                    record_id
                ))
            
            # Step 6: Batch update SQLite
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.executemany("""
                    UPDATE records 
                    SET attributes = ?, content = ?, timestamp = ?, has_vector = ?
                    WHERE id = ?
                """, final_db_updates)
                conn.commit()
            
            logger.debug(f"Updated {len(records)} records in batch chunk")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update batch chunk: {e}")
            return False

    def add_batch(self, records: List[Record]) -> bool:
        """
        Add multiple records to the memory in batch.
        
        Args:
            records: List of Record objects to add
            
        Returns:
            True if all records were successfully added, False otherwise
        """
        if not records:
            return True
        
        try:
            # ChromaDB has a maximum batch size limit
            MAX_CHROMADB_BATCH_SIZE = 5000
            
            # Process records in chunks to respect ChromaDB batch size limits
            for i in range(0, len(records), MAX_CHROMADB_BATCH_SIZE):
                chunk = records[i:i + MAX_CHROMADB_BATCH_SIZE]
                success = self._add_batch_chunk(chunk)
                if not success:
                    logger.error(f"Failed to add batch chunk starting at index {i}")
                    return False
            
            logger.debug(f"Added {len(records)} records to EMemory in batch")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add batch records to EMemory: {e}")
            return False
    
    def _add_batch_chunk(self, records: List[Record]) -> bool:
        """Add a chunk of records to the memory."""
        try:
            vectors_to_add = []
            ids_for_vectors = []
            successful_vector_ids = set()
            
            # Prepare all records for database insertion
            db_data = []
            for record in records:
                # Generate ID and timestamp if not provided
                record_id = record.id or self._generate_id()
                record_timestamp = record.timestamp or self._get_current_timestamp()
                record_attributes = record.attributes or {}
                has_vector = 1 if record.vector else 0
                
                # Prepare vector data for ChromaDB (only ID and vector)
                if has_vector:
                    vectors_to_add.append(record.vector)
                    ids_for_vectors.append(record_id)
                
                # Prepare database data (will be updated after ChromaDB attempt)
                db_data.append((
                    record_id,
                    json.dumps(record_attributes),
                    record.content,
                    record_timestamp,
                    has_vector
                ))
            
            # Try to batch add to ChromaDB if vectors are provided
            if vectors_to_add:
                try:
                    # ChromaDB only stores ID and vector - no metadata to avoid duplication
                    self.collection.add(
                        embeddings=vectors_to_add,
                        ids=ids_for_vectors
                    )
                    # Mark all vectors as successfully added
                    successful_vector_ids = set(ids_for_vectors)
                    logger.debug(f"Successfully added {len(vectors_to_add)} vectors to ChromaDB")
                except Exception as e:
                    logger.error(f"Failed to add vectors to ChromaDB: {e}")
                    # No vectors were successfully added, all has_vector will be 0
            
            # Update has_vector status based on ChromaDB success
            final_db_data = []
            for i, (record_id, attributes_json, content, timestamp, original_has_vector) in enumerate(db_data):
                # Set has_vector to 1 only if ChromaDB storage was successful
                final_has_vector = 1 if record_id in successful_vector_ids else 0
                final_db_data.append((
                    record_id,
                    attributes_json,
                    content,
                    timestamp,
                    final_has_vector
                ))
            
            # Batch insert into SQLite (contains all metadata)
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO records (id, attributes, content, timestamp, has_vector)
                    VALUES (?, ?, ?, ?, ?)
                """, final_db_data)
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add batch chunk: {e}")
            return False

    def get(self, id: str) -> Optional[Record]:
        """
        Get a record by ID.
        
        Args:
            id: Record ID
            
        Returns:
            Record if found, None otherwise
        """
        # Check if record exists first
        if not self.exists(id):
            return None
        
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.execute("""
                SELECT id, attributes, content, timestamp, has_vector
                FROM records WHERE id = ?
            """, (id,))
            row = cursor.fetchone()
            
            if row:
                record_id, attributes_json, content, timestamp, has_vector = row
                vector = self.get_vector(id) if has_vector else None
                return Record(
                    id=record_id,
                    attributes=json.loads(attributes_json) if attributes_json else {},
                    content=content,
                    vector=vector,
                    timestamp=timestamp
                )
            return None

    def get_batch(self, ids: List[str]) -> Dict[str, Record]:
        """
        Get multiple records by ID in a single query.
        
        This method solves the N+1 query problem by fetching all records
        in a single database operation and then batch-loading vectors.
        
        Args:
            ids: List of record IDs
            
        Returns:
            Dictionary mapping ID to Record object (only includes existing records)
        """
        if not ids:
            return {}
        
        # Create placeholders for SQL query
        placeholders = ','.join(['?' for _ in ids])
        
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.execute(f"""
                SELECT id, attributes, content, timestamp, has_vector
                FROM records WHERE id IN ({placeholders})
            """, ids)
            rows = cursor.fetchall()

        if not rows:
            return {}

        # Separate records with and without vectors for batch processing
        records_with_vectors = []
        records_map = {}
        
        for row in rows:
            record_id, attributes_json, content, timestamp, has_vector = row
            
            # Parse attributes
            attributes = json.loads(attributes_json) if attributes_json else {}
            
            # Store record without vector first
            records_map[record_id] = Record(
                id=record_id,
                attributes=attributes,
                content=content,
                vector=None,  # Will be filled later for records with vectors
                timestamp=timestamp
            )
            
            if has_vector:
                records_with_vectors.append(record_id)

        # Batch get vectors for records that have them
        if records_with_vectors:
            try:
                vector_data = self.collection.get(ids=records_with_vectors, include=["embeddings"])
                if vector_data and vector_data.get("embeddings") is not None:
                    embeddings = vector_data["embeddings"]
                    vector_ids = vector_data.get("ids", [])
                    
                    # Handle both list and numpy array cases
                    if hasattr(embeddings, '__len__') and len(embeddings) > 0:
                        # Create mapping from id to vector
                        for i, record_id in enumerate(vector_ids):
                            if i < len(embeddings):
                                vector = embeddings[i]
                                # Handle numpy array or list
                                if hasattr(vector, 'tolist'):
                                    vector = vector.tolist()
                                elif hasattr(vector, '__iter__'):
                                    vector = list(vector)
                                else:
                                    vector = [vector]
                                
                                # Update the record with its vector
                                if record_id in records_map:
                                    records_map[record_id].vector = vector
            except Exception as e:
                logger.warning(f"Failed to get vectors in batch: {e}")
        
        return records_map

    def get_vector(self, id: str) -> Optional[List[float]]:
        """
        Get the vector embedding for a record by ID.
        
        Args:
            id: Record ID
            
        Returns:
            Vector embedding as List[float] if found and has vector, None otherwise
        """
        # Check if record exists and has vector
        if not self.exists(id) or not self.has_vector(id):
            return None
        
        try:
            # Get vector from ChromaDB
            vector_data = self.collection.get(ids=[id], include=["embeddings"])
            if vector_data and vector_data.get("embeddings") is not None:
                embeddings = vector_data["embeddings"]
                if embeddings is not None and len(embeddings) > 0:
                    # Handle numpy array or list
                    vector = embeddings[0]
                    if hasattr(vector, 'tolist'):
                        return vector.tolist()
                    elif hasattr(vector, '__iter__'):
                        return list(vector)
                    else:
                        return [vector]
        except Exception as e:
            logger.warning(f"Failed to get vector for record {id}: {e}")
        
        return None

    def delete(self, id: str) -> bool:
        """
        Delete a record by ID.
        
        Args:
            id: Record ID
            
        Returns:
            True if deleted, False if not found
        """
        # Check if record exists first
        if not self.exists(id):
            logger.debug(f"Record {id} not found, nothing to delete")
            return False
        
        # Check if record has vector before deletion
        has_vector = self.has_vector(id)
        
        # Delete from SQLite
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.execute("DELETE FROM records WHERE id = ?", (id,))
            conn.commit()
        
        # Delete from ChromaDB if it has vector
        if has_vector:
            try:
                self.collection.delete(ids=[id])
            except Exception as e:
                logger.warning(f"Failed to delete vector from ChromaDB: {e}")
        
        logger.debug(f"Deleted record {id} from EMemory")
        return True

    def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Record]:
        """
        List all records.
        
        ⚠️  WARNING: This method loads ALL records into memory at once!
        For large databases (millions of records), this can cause memory explosion.
        Consider using more specific query methods or pagination with small limits.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of records
        """
        import warnings
        warnings.warn(
            "list_all() loads ALL records into memory. For large databases, "
            "this can cause memory explosion. Consider using specific query methods "
            "or pagination with small limits.",
            UserWarning,
            stacklevel=2
        )
        
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

    def clear(self, confirm: bool = True) -> None:
        """
        Clear all records from memory.
        
        Args:
            confirm: If True, requires user confirmation before clearing.
                    If False, clears immediately without confirmation.
        """
        # Get count before clearing for logging
        total_records = self.count()
        
        if confirm:
            if total_records == 0:
                logger.info("No records to clear in EMemory")
                return
            
            print(f"\n⚠️  WARNING: You are about to delete ALL {total_records} records from EMemory '{self.name}'!")
            print("This action cannot be undone!")
            print("\nTo confirm deletion, type 'DELETE' and press Enter.")
            print("To cancel, press Enter or type anything else.")
            
            try:
                user_input = input("Confirmation: ").strip()
                if user_input != "DELETE":
                    print("Deletion cancelled.")
                    logger.info("EMemory clear operation cancelled by user")
                    return
            except KeyboardInterrupt:
                print("\nDeletion cancelled.")
                logger.info("EMemory clear operation cancelled by user (Ctrl+C)")
                return
            except EOFError:
                print("\nDeletion cancelled.")
                logger.info("EMemory clear operation cancelled by user (EOF)")
                return
        
        # Proceed with clearing
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
        
        logger.info(f"Cleared {total_records} records from EMemory")

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
            # ChromaDB only returns IDs and distances (no metadata since we removed it)
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k
            )
            
            if not results or not results.get('ids') or not results['ids']:
                return []
            
            record_ids = results['ids'][0] if results['ids'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            if not record_ids:
                return []
            
            # Verify that all returned records still exist (data consistency check)
            existing_records = self.exists_batch(record_ids)
            
            # Filter to only existing records and their corresponding distances
            valid_record_ids = []
            valid_distances = []
            for record_id, distance in zip(record_ids, distances):
                if existing_records.get(record_id, False):
                    valid_record_ids.append(record_id)
                    valid_distances.append(distance)
            
            if not valid_record_ids:
                return []
            
            # Batch get all records to solve N+1 query problem
            records_map = self.get_batch(valid_record_ids)
            
            # Build results list
            results_list = []
            for record_id, distance in zip(valid_record_ids, valid_distances):
                record = records_map.get(record_id)
                if record:
                    results_list.append((record, float(distance)))
            
            return results_list
        except Exception as e:
            logger.error(f"Failed to search similar records: {e}")
            return []

    def exists(self, id: str) -> bool:
        """
        Check if a record exists by ID.
        
        Args:
            id: Record ID
            
        Returns:
            True if record exists, False otherwise
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.execute("SELECT 1 FROM records WHERE id = ?", (id,))
            return cursor.fetchone() is not None
    
    def has_vector(self, id: str) -> bool:
        """
        Check if a record has a vector embedding.
        
        Args:
            id: Record ID
            
        Returns:
            True if record has vector, False otherwise
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.execute("SELECT has_vector FROM records WHERE id = ?", (id,))
            row = cursor.fetchone()
            return row is not None and row[0] == 1
    
    def exists_batch(self, ids: List[str]) -> Dict[str, bool]:
        """
        Check existence of multiple records by IDs.
        
        Args:
            ids: List of record IDs
            
        Returns:
            Dictionary mapping ID to existence status
        """
        if not ids:
            return {}
        
        # Create placeholders for SQL query
        placeholders = ','.join(['?' for _ in ids])
        
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.execute(f"""
                SELECT id FROM records WHERE id IN ({placeholders})
            """, ids)
            existing_ids = {row[0] for row in cursor.fetchall()}
        
        return {id: id in existing_ids for id in ids}
    
    def has_vector_batch(self, ids: List[str]) -> Dict[str, bool]:
        """
        Check if multiple records have vector embeddings.
        
        Args:
            ids: List of record IDs
            
        Returns:
            Dictionary mapping ID to vector existence status
        """
        if not ids:
            return {}
        
        # Create placeholders for SQL query
        placeholders = ','.join(['?' for _ in ids])
        
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.execute(f"""
                SELECT id, has_vector FROM records WHERE id IN ({placeholders})
            """, ids)
            results = {row[0]: row[1] == 1 for row in cursor.fetchall()}
        
        # Ensure all requested IDs are in the result
        return {id: results.get(id, False) for id in ids}

    def query_by_date_range(
        self, 
        start_date: str, 
        end_date: str, 
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Record]:
        """
        Query records by date range using SQLite for efficient filtering.
        
        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD) 
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of records within the date range
        """
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                query = """
                SELECT id, attributes, content, timestamp, has_vector
                FROM records 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            """
                params = [start_date, end_date]
                
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                
                if offset > 0:
                    query += " OFFSET ?"
                    params.append(offset)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                if not rows:
                    return []
                
                # Separate records with and without vectors for batch processing
                records_with_vectors = []
                records_without_vectors = []
                id_to_row_mapping = {}
                
                for row in rows:
                    id, attributes_json, content, timestamp, has_vector = row
                    id_to_row_mapping[id] = row
                    
                    if has_vector:
                        records_with_vectors.append(id)
                    else:
                        records_without_vectors.append(id)
                
                # Batch get vectors for records that have them
                vectors_data = {}
                if records_with_vectors:
                    try:
                        vector_data = self.collection.get(ids=records_with_vectors, include=["embeddings"])
                        if vector_data and vector_data.get("embeddings") is not None:
                            embeddings = vector_data["embeddings"]
                            ids = vector_data.get("ids", [])
                            
                            # Handle both list and numpy array cases
                            if hasattr(embeddings, '__len__') and len(embeddings) > 0:
                                # Create mapping from id to vector
                                for i, record_id in enumerate(ids):
                                    if i < len(embeddings):
                                        vectors_data[record_id] = embeddings[i]
                    except Exception as e:
                        logger.warning(f"Failed to get vectors in batch: {e}")
                
                # Build final records list
                records = []
                for row in rows:
                    id, attributes_json, content, timestamp, has_vector = row
                    
                    # Parse attributes
                    attributes = json.loads(attributes_json) if attributes_json else {}
                    
                    # Get vector if exists
                    vector = vectors_data.get(id) if has_vector else None
                    
                    record = Record(
                        id=id,
                        content=content,
                        vector=vector,
                        attributes=attributes,
                        timestamp=timestamp
                    )
                    records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to query records by date range ({start_date} to {end_date}): {e}")
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
            "persist_dir": str(self.persist_dir),
            "sqlite_size_bytes": sqlite_size,
            "chroma_size_bytes": chroma_size,
            "sqlite_path": str(self.sqlite_path),
            "chroma_path": str(self.chroma_path)
        }
    
    def close(self):
        """
        Properly close database connections and release file handles.
        
        This method should be called when the EMemory instance is no longer needed
        to ensure proper cleanup of database connections and file handles.
        """
        logger.info(f"Closing EMemory '{self.name}'")
        
        try:
            # Close ChromaDB client
            if hasattr(self, 'chroma_client') and self.chroma_client:
                try:
                    self.chroma_client = None
                    logger.debug(f"Closed ChromaDB client for EMemory '{self.name}'")
                except Exception as e:
                    logger.warning(f"Error closing ChromaDB client: {e}")
            
            # Force garbage collection to release file handles
            import gc
            gc.collect()
            
            logger.info(f"EMemory '{self.name}' closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing EMemory '{self.name}': {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.close()