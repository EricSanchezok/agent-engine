"""
Vector-based memory storage for agents.

This module provides a SQLite-based vector database for storing and retrieving
text content with associated vectors and metadata.
"""

import sqlite3
import json
import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set
from pathlib import Path
import hashlib

from ..agent_logger.agent_logger import AgentLogger
from ..utils.project_root import get_project_root
from .embedder import Embedder

logger = AgentLogger(__name__)


class Memory:
    """Vector-based memory storage using SQLite database"""
    
    # Schema version for SQLite PRAGMA user_version
    SCHEMA_VERSION: int = 1
    # In-process cache of initialized database paths to avoid repeated checks
    _initialized_paths: Set[str] = set()

    def __init__(self, name: str, db_path: Optional[str] = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the memory storage
        
        Args:
            name: Name of the memory database (required)
            db_path: Path to the database file (optional, will use .memory/name.db if not provided)
            model_name: Model name for sentence transformers (e.g., "all-MiniLM-L6-v2")
        """
        if not name:
            raise ValueError("Memory name is required")
        
        self.name = name
        self.model_name = model_name
        self.embedder = None  # Initialize embedder lazily when needed
        
        # Set database path
        if db_path:
            self.db_path = db_path
        else:
            # Use project root/.memory/name.db
            project_root = get_project_root()
            memory_dir = project_root / ".memory"
            memory_dir.mkdir(exist_ok=True)
            self.db_path = str(memory_dir / f"{name}.db")
        
        self._ensure_db_initialized()

    def _ensure_db_initialized(self):
        """Ensure the database is initialized only once per process and path.

        This checks if the DB file exists and has the required schema and indexes.
        If already initialized (either persisted on disk or cached in-process), it skips work.
        """
        # Fast path: in-process cache
        if self.db_path in Memory._initialized_paths:
            return

        # Check on-disk state; if not initialized, run init
        if not self._is_db_initialized():
            self._init_db()

        # Mark as initialized for this process
        Memory._initialized_paths.add(self.db_path)
    
    def _init_db(self):
        """Initialize the SQLite database with required tables"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create vectors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_id ON vectors (id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content ON vectors (content)")

            # Set schema version to mark successful initialization
            cursor.execute(f"PRAGMA user_version = {Memory.SCHEMA_VERSION}")
            
            conn.commit()

    def _is_db_initialized(self) -> bool:
        """Check whether the database exists and has the expected schema and indexes."""
        if not os.path.exists(self.db_path):
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check table existence
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'")
                if cursor.fetchone() is None:
                    return False

                # Check required columns
                cursor.execute("PRAGMA table_info(vectors)")
                columns = {row[1] for row in cursor.fetchall()}  # row[1] is column name
                required_columns = {"id", "vector", "content", "metadata"}
                if not required_columns.issubset(columns):
                    return False

                # Check indexes
                cursor.execute("PRAGMA index_list('vectors')")
                index_names = {row[1] for row in cursor.fetchall()}  # row[1] is index name
                if "idx_id" not in index_names or "idx_content" not in index_names:
                    return False

                # Check schema version
                cursor.execute("PRAGMA user_version")
                version_row = cursor.fetchone()
                version = version_row[0] if version_row else 0
                if version < Memory.SCHEMA_VERSION:
                    return False

                return True
        except sqlite3.Error as e:
            logger.error(f"Failed to check database initialization: {e}")
            return False
    
    def _vector_to_blob(self, vector: List[float]) -> bytes:
        """Convert vector list to binary blob for storage"""
        return np.array(vector, dtype=np.float32).tobytes()
    
    def _blob_to_vector(self, blob: bytes) -> List[float]:
        """Convert binary blob back to vector list"""
        return np.frombuffer(blob, dtype=np.float32).tolist()
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content using hash"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_embedder(self) -> Embedder:
        """Get or create embedder instance"""
        if self.embedder is None:
            self.embedder = Embedder(model_name=self.model_name)
        return self.embedder
    
    def _calculate_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Similarity score between 0 and 1
        """
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # Check for zero vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        
        # Normalize vectors
        v1_normalized = v1 / v1_norm
        v2_normalized = v2 / v2_norm
        
        # Calculate cosine similarity
        similarity = np.dot(v1_normalized, v2_normalized)
        return float(np.clip(similarity, -1.0, 1.0))
    
    def _find_most_similar(self, query_vector: List[float], vectors: List[List[float]], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find the most similar vectors to a query vector using independent similarity calculation
        
        Args:
            query_vector: Query vector to compare against
            vectors: List of vectors to compare with
            top_k: Number of top similar vectors to return
            
        Returns:
            List of tuples (index, similarity_score) sorted by similarity
        """
        similarities = []
        for i, vector in enumerate(vectors):
            sim = self._calculate_similarity(query_vector, vector)
            similarities.append((i, sim))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def add(self, content: str, vector: Optional[List[float]] = None, metadata: Optional[Dict] = None):
        """
        Add content to memory
        
        Args:
            content: Text content to store (required)
            vector: Pre-computed vector (optional, will be computed if not provided)
            metadata: Additional metadata to store (optional)
        """
        if not content:
            raise ValueError("Content is required")
        
        # Generate ID from content
        content_id = self._generate_id(content)
        
        # Check if content already exists
        if self.get_by_content(content)[0] is not None:
            # logger.info(f"Content already exists: {content_id}")
            return
        
        # Compute vector if not provided
        if vector is None:
            logger.info(f"Computing vector for content: {content_id}")
            embedder = self._get_embedder()
            # Ensure embedder is fitted with current content for consistent dimensions
            if not embedder._fitted:
                embedder.fit([content])
            vector = embedder.embed(content)
            
            # Validate vector
            if not vector or len(vector) == 0:
                raise ValueError(f"Generated vector is empty for content: {content}")
            if np.linalg.norm(vector) == 0:
                logger.warning(f"Generated zero vector for content: {content}")
        
        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO vectors (id, vector, content, metadata)
                VALUES (?, ?, ?, ?)
            """, (content_id, self._vector_to_blob(vector), content, metadata_json))
            conn.commit()
    
    def get_by_vector(self, vector: List[float]) -> Optional[Tuple[str, Dict]]:
        """
        Get content and metadata by vector
        
        Args:
            vector: Query vector
            
        Returns:
            Tuple of (content, metadata) or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vector, content, metadata FROM vectors")
            rows = cursor.fetchall()
            
            if not rows:
                return None, {}
            
            # Find most similar vector using independent similarity calculation
            vectors = [self._blob_to_vector(row[0]) for row in rows]
            similarities = self._find_most_similar(vector, vectors, top_k=1)
            
            if similarities and similarities[0][1] > 0.9:  # High similarity threshold
                idx = similarities[0][0]
                content = rows[idx][1]
                metadata = json.loads(rows[idx][2]) if rows[idx][2] else {}
                return content, metadata
        
        return None, {}
    
    def get_by_content(self, content: str) -> Optional[Tuple[List[float], Dict]]:
        """
        Get vector and metadata by content
        
        Args:
            content: Query content
            
        Returns:
            Tuple of (vector, metadata) or None if not found
        """
        content_id = self._generate_id(content)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vector, metadata FROM vectors WHERE id = ?", (content_id,))
            row = cursor.fetchone()
            
            if row:
                vector = self._blob_to_vector(row[0])
                metadata = json.loads(row[1]) if row[1] else {}
                return vector, metadata
        
        return None, {}
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar content
        
        Args:
            query: Query text or vector
            top_k: Number of top results to return
            
        Returns:
            List of tuples (content, similarity_score, metadata)
        """
        # If query is a list of floats, treat as vector
        if isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            query_vector = query
        else:
            # Treat as text and embed using embedder
            embedder = self._get_embedder()
            if not embedder._fitted:
                embedder.fit([query])
            query_vector = embedder.embed(query)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vector, content, metadata FROM vectors")
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Calculate similarities using independent similarity calculation
            vectors = [self._blob_to_vector(row[0]) for row in rows]
            similarities = self._find_most_similar(query_vector, vectors, top_k=top_k)
            
            # Return results with content and metadata
            results = []
            for idx, score in similarities:
                content = rows[idx][1]
                metadata = json.loads(rows[idx][2]) if rows[idx][2] else {}
                results.append((content, score, metadata))
            
            return results
    
    def delete_by_content(self, content: str) -> bool:
        """
        Delete content by text
        
        Args:
            content: Content to delete
            
        Returns:
            True if deleted, False if not found
        """
        content_id = self._generate_id(content)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM vectors WHERE id = ?", (content_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_by_vector(self, vector: List[float]) -> bool:
        """
        Delete content by vector (finds most similar and deletes it)
        
        Args:
            vector: Vector to match for deletion
            
        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, vector FROM vectors")
            rows = cursor.fetchall()
            
            if not rows:
                return False
            
            # Find most similar vector using independent similarity calculation
            vectors = [self._blob_to_vector(row[1]) for row in rows]
            similarities = self._find_most_similar(vector, vectors, top_k=1)
            
            if similarities and similarities[0][1] > 0.9:  # High similarity threshold
                idx = similarities[0][0]
                content_id = rows[idx][0]
                
                cursor.execute("DELETE FROM vectors WHERE id = ?", (content_id,))
                conn.commit()
                return cursor.rowcount > 0
        
        return False
    
    def get_all_vectors(self) -> Dict[str, List[float]]:
        """
        Get all stored vectors
        
        Returns:
            Dictionary mapping content IDs to vectors
        """
        vectors = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, vector FROM vectors")
            for row in cursor.fetchall():
                vectors[row[0]] = self._blob_to_vector(row[1])
        return vectors
    
    def get_all_contents(self) -> List[str]:
        """
        Get all stored content
        
        Returns:
            List of all content strings
        """
        contents = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM vectors")
            for row in cursor.fetchall():
                contents.append(row[0])
        return contents
    
    def get_all_metadata(self) -> List[Dict]:
        """
        Get all stored metadata
        
        Returns:
            List of all metadata dictionaries
        """
        metadata_list = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT metadata FROM vectors")
            for row in cursor.fetchall():
                if row[0]:
                    metadata_list.append(json.loads(row[0]))
                else:
                    metadata_list.append({})
        return metadata_list
    
    def get_all(self) -> List[Tuple[str, List[float], Dict]]:
        """
        Get all stored items with their content, vector, and metadata
        
        Returns:
            List of tuples (content, vector, metadata) for all stored items
        """
        items = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content, vector, metadata FROM vectors")
            for row in cursor.fetchall():
                content = row[0]
                vector = self._blob_to_vector(row[1])
                metadata = json.loads(row[2]) if row[2] else {}
                items.append((content, vector, metadata))
        return items
    
    def count(self) -> int:
        """
        Get the number of stored items
        
        Returns:
            Number of items in memory
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM vectors")
            return cursor.fetchone()[0]
    
    def clear(self):
        """Clear all stored data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM vectors")
            conn.commit()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the memory
        
        Returns:
            Dictionary with memory information
        """
        embedder = self._get_embedder()
        return {
            "name": self.name,
            "db_path": self.db_path,
            "count": self.count(),
            "embedder_method": embedder.method,
            "vector_dimension": embedder.get_vector_dimension()
        }
