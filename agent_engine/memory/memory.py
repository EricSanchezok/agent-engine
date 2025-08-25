"""
Vector-based memory storage for agents.

This module provides a SQLite-based vector database for storing and retrieving
text content with associated vectors and metadata.
"""

import sqlite3
import json
import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import hashlib

from ..agent_logger.agent_logger import AgentLogger
from ..utils.project_root import get_project_root
from .embedder import Embedder

logger = AgentLogger(__name__)


class Memory:
    """Vector-based memory storage using SQLite database"""
    
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
        
        # Create embedder with specified model
        self.embedder = Embedder(model_name=model_name)
        
        # Initialize the embedder
        self.embedder.fit()
        
        # Set database path
        if db_path:
            self.db_path = db_path
        else:
            # Use project root/.memory/name.db
            project_root = get_project_root()
            memory_dir = project_root / ".memory"
            memory_dir.mkdir(exist_ok=True)
            self.db_path = str(memory_dir / f"{name}.db")
        
        self._init_db()
    
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
            
            conn.commit()
    
    def _vector_to_blob(self, vector: List[float]) -> bytes:
        """Convert vector list to binary blob for storage"""
        return np.array(vector, dtype=np.float32).tobytes()
    
    def _blob_to_vector(self, blob: bytes) -> List[float]:
        """Convert binary blob back to vector list"""
        return np.frombuffer(blob, dtype=np.float32).tolist()
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content using hash"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
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
            # Ensure embedder is fitted with current content for consistent dimensions
            if not self.embedder._fitted:
                self.embedder.fit([content])
            vector = self.embedder.embed(content)
        
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
            
            # Find most similar vector
            vectors = [self._blob_to_vector(row[0]) for row in rows]
            similarities = self.embedder.find_most_similar(vector, vectors, top_k=1)
            
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
            # Treat as text and embed
            query_vector = self.embedder.embed(query)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vector, content, metadata FROM vectors")
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Calculate similarities
            vectors = [self._blob_to_vector(row[0]) for row in rows]
            similarities = self.embedder.find_most_similar(query_vector, vectors, top_k=top_k)
            
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
            
            # Find most similar vector
            vectors = [self._blob_to_vector(row[1]) for row in rows]
            similarities = self.embedder.find_most_similar(vector, vectors, top_k=1)
            
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
        return {
            "name": self.name,
            "db_path": self.db_path,
            "count": self.count(),
            "embedder_method": self.embedder.method,
            "vector_dimension": self.embedder.get_vector_dimension()
        }
