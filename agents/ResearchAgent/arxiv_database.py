"""
ArxivDatabase - A PodEMemory-based database for arXiv papers.

This module provides storage and retrieval functionality for arXiv papers
using PodEMemory as the underlying storage engine.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_engine.agent_logger import AgentLogger
from agent_engine.memory.e_memory.pod_ememory import PodEMemory
from agent_engine.memory.e_memory.models import Record
from agent_engine.utils import get_current_file_dir

from core.arxiv_fetcher import ArxivPaper

MAX_ELEMENTS_PER_SHARD = 200000

class ArxivDatabase:
    """
    A PodEMemory-based database for arXiv papers.
    
    This class provides storage and retrieval functionality for arXiv papers
    without handling embedding generation. Embeddings should be provided
    externally when adding papers.
    """
    
    def __init__(
        self,
        name: str = "arxiv_papers",
        persist_dir: Optional[str] = None,
        distance_metric: str = "cosine"
    ):
        """
        Initialize ArxivDatabase.
        
        Args:
            name: Database name (used for PodEMemory)
            persist_dir: Storage directory (optional, defaults to ResearchAgent/database)
            distance_metric: Distance metric for vector search
        """
        self.logger = AgentLogger(self.__class__.__name__)
        
        # Determine storage directory
        if persist_dir is None:
            self.persist_dir = get_current_file_dir() / 'database' / name
        else:
            self.persist_dir = Path(persist_dir) / name
        
        # Initialize PodEMemory
        self.pod_memory = PodEMemory(
            name=name,
            persist_dir=str(self.persist_dir.parent),
            max_elements_per_shard=MAX_ELEMENTS_PER_SHARD,
            distance_metric=distance_metric
        )
        
        self.logger.info(f"ArxivDatabase '{name}' initialized at {self.persist_dir}")
        self.logger.info(f"Max elements per shard: {MAX_ELEMENTS_PER_SHARD}")
    
    def add_paper(self, paper: ArxivPaper, embedding: Optional[List[float]] = None) -> str:
        """
        Add a single paper to the database.
        
        Args:
            paper: ArxivPaper object to add
            embedding: Optional embedding vector for the paper
            
        Returns:
            Record ID
        """
        if not isinstance(paper, ArxivPaper):
            raise TypeError("paper must be an ArxivPaper instance")
        
        # Convert ArxivPaper to Record
        record = self._paper_to_record(paper, embedding)
        
        # Add to PodEMemory
        record_id = self.pod_memory.add(record)
        
        self.logger.info(f"Added paper {paper.full_id} to database")
        return record_id
    
    def add_papers(
        self, 
        papers: List[ArxivPaper], 
        embeddings: Optional[List[Optional[List[float]]]] = None
    ) -> List[str]:
        """
        Add multiple papers to the database in batch.
        
        Args:
            papers: List of ArxivPaper objects
            embeddings: Optional list of embedding vectors (must match papers length)
            
        Returns:
            List of record IDs
        """
        if not papers:
            return []
        
        if embeddings is not None and len(embeddings) != len(papers):
            raise ValueError("embeddings length must match papers length")
        
        # Convert papers to records
        records = []
        for i, paper in enumerate(papers):
            embedding = embeddings[i] if embeddings else None
            record = self._paper_to_record(paper, embedding)
            records.append(record)
        
        # Add to PodEMemory in batch
        record_ids = self.pod_memory.add_batch(records)
        
        self.logger.info(f"Added {len(papers)} papers to database in batch")
        return record_ids
    
    def get_paper_by_id(self, paper_id: str) -> Optional[ArxivPaper]:
        """
        Get a paper by its arXiv ID.
        
        Args:
            paper_id: arXiv paper ID
            
        Returns:
            ArxivPaper object if found, None otherwise
        """
        record = self.pod_memory.get(paper_id)
        if record is None:
            return None
        
        return self._record_to_paper(record)
    
    def update_paper(self, paper: ArxivPaper, embedding: Optional[List[float]] = None) -> bool:
        """
        Update an existing paper in the database.
        
        Args:
            paper: ArxivPaper object with updated data
            embedding: Optional updated embedding vector
            
        Returns:
            True if updated successfully, False otherwise
        """
        if not isinstance(paper, ArxivPaper):
            raise TypeError("paper must be an ArxivPaper instance")
        
        # Convert to record
        record = self._paper_to_record(paper, embedding)
        
        # Update in PodEMemory
        success = self.pod_memory.update(record)
        
        if success:
            self.logger.info(f"Updated paper {paper.full_id} in database")
        else:
            self.logger.warning(f"Failed to update paper {paper.full_id} - not found")
        
        return success
    
    def delete_paper(self, paper_id: str) -> bool:
        """
        Delete a paper from the database.
        
        Args:
            paper_id: arXiv paper ID
            
        Returns:
            True if deleted successfully, False otherwise
        """
        success = self.pod_memory.delete(paper_id)
        
        if success:
            self.logger.info(f"Deleted paper {paper_id} from database")
        else:
            self.logger.warning(f"Failed to delete paper {paper_id} - not found")
        
        return success
    
    def search_similar_papers(
        self, 
        query_vector: List[float], 
        k: int = 10
    ) -> List[ArxivPaper]:
        """
        Search for papers similar to the query vector.
        
        Args:
            query_vector: Query embedding vector
            k: Number of similar papers to return
            
        Returns:
            List of ArxivPaper objects
        """
        # Search for similar records
        similar_records = self.pod_memory.search_similar_records(query_vector, k)
        
        # Convert records to papers
        papers = []
        for record, distance in similar_records:
            paper = self._record_to_paper(record)
            papers.append(paper)
        
        self.logger.info(f"Found {len(papers)} similar papers")
        return papers
    
    def count(self) -> int:
        """
        Get total number of papers in the database.
        
        Returns:
            Total number of papers
        """
        return self.pod_memory.count()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.pod_memory.get_stats()
    
    def exists(self, paper: ArxivPaper) -> bool:
        """
        Check if a paper exists by its arXiv ID.
        
        Args:
            paper: ArxivPaper object
            
        Returns:
            True if paper exists, False otherwise
        """
        return self.pod_memory.exists(paper.full_id)
    
    def has_vector(self, paper: ArxivPaper) -> bool:
        """
        Check if a paper has a vector embedding.
        
        Args:
            paper: ArxivPaper object
            
        Returns:
            True if paper has vector, False otherwise
        """
        return self.pod_memory.has_vector(paper.full_id)
    
    def exists_batch(self, papers: List[ArxivPaper]) -> Dict[str, bool]:
        """
        Check existence of multiple papers.
        
        Args:
            papers: List of ArxivPaper objects
            
        Returns:
            Dictionary mapping paper full_id to existence status
        """
        paper_ids = [paper.full_id for paper in papers]
        return self.pod_memory.exists_batch(paper_ids)
    
    def has_vector_batch(self, papers: List[ArxivPaper]) -> Dict[str, bool]:
        """
        Check if multiple papers have vector embeddings.
        
        Args:
            papers: List of ArxivPaper objects
            
        Returns:
            Dictionary mapping paper full_id to vector existence status
        """
        paper_ids = [paper.full_id for paper in papers]
        return self.pod_memory.has_vector_batch(paper_ids)

    def clear(self) -> None:
        """Clear all papers from the database."""
        self.pod_memory.clear()
        self.logger.info("Cleared all papers from ArxivDatabase")
    
    def _paper_to_record(self, paper: ArxivPaper, embedding: Optional[List[float]] = None) -> Record:
        """
        Convert ArxivPaper to PodEMemory Record.
        
        Args:
            paper: ArxivPaper object
            embedding: Optional embedding vector
            
        Returns:
            Record object
        """
        # Prepare attributes dictionary
        attributes = {
            "title": paper.title,
            "authors": paper.authors,
            "categories": paper.categories,
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "pdf_file_path": paper.pdf_file_path,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
            "comment": paper.comment,
            "metadata": paper.metadata,
            "version": paper.version,
        }
        
        # Add dates as ISO strings
        if paper.published_date:
            attributes["published_date"] = paper.published_date.isoformat()
        if paper.submitted_date:
            attributes["submitted_date"] = paper.submitted_date.isoformat()
        
        # Create record
        record = Record(
            id=paper.full_id,
            content=paper.summary,
            vector=embedding,
            attributes=attributes,
            timestamp=paper.published_date.isoformat() if paper.published_date else None
        )
        
        return record
    
    def _record_to_paper(self, record: Record) -> ArxivPaper:
        """
        Convert PodEMemory Record to ArxivPaper.
        
        Args:
            record: Record object
            
        Returns:
            ArxivPaper object
        """
        # Extract basic fields
        attrs = record.attributes
        
        # Parse dates
        published_date = None
        if attrs.get("published_date"):
            published_date = datetime.fromisoformat(attrs["published_date"])
        
        submitted_date = None
        if attrs.get("submitted_date"):
            submitted_date = datetime.fromisoformat(attrs["submitted_date"])
        
        # Create ArxivPaper
        paper = ArxivPaper(
            id=record.id.split('v')[0] if 'v' in record.id else record.id,  # Remove version for id
            version=attrs.get("version"),
            title=attrs.get("title", ""),
            authors=attrs.get("authors", []),
            categories=attrs.get("categories", []),
            summary=attrs.get("summary", ""),
            published_date=published_date,
            submitted_date=submitted_date,
            pdf_url=attrs.get("pdf_url", ""),
            doi=attrs.get("doi"),
            journal_ref=attrs.get("journal_ref"),
            comment=attrs.get("comment"),
            pdf_file_path=attrs.get("pdf_file_path"),
            metadata=attrs.get("metadata", {})
        )
        
        return paper
