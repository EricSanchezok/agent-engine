"""
ArxivDatabase - A PodEMemory-based database for arXiv papers.

This module provides storage and retrieval functionality for arXiv papers
using PodEMemory as the underlying storage engine.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pyinstrument

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
        self.pod_ememory = PodEMemory(
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
        success = self.pod_ememory.add(record)
        
        if success:
            self.logger.info(f"Added paper {paper.full_id} to database")
            return record.id
        else:
            self.logger.error(f"Failed to add paper {paper.full_id} to database")
            return ""
    
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
        success = self.pod_ememory.add_batch(records)
        
        if success:
            # Extract record IDs from the records
            record_ids = [record.id for record in records]
            self.logger.info(f"Added {len(papers)} papers to database in batch")
            return record_ids
        else:
            self.logger.error("Failed to add papers to database in batch")
            return []
    
    def get_paper_by_id(self, paper_id: str) -> Optional[ArxivPaper]:
        """
        Get a paper by its arXiv ID.
        
        Args:
            paper_id: arXiv paper ID
            
        Returns:
            ArxivPaper object if found, None otherwise
        """
        record = self.pod_ememory.get(paper_id)
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
        success = self.pod_ememory.update(record)
        
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
        success = self.pod_ememory.delete(paper_id)
        
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
        similar_records = self.pod_ememory.search_similar_records(query_vector, k)
        
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
        return self.pod_ememory.count()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.pod_ememory.get_stats()
    
    def exists(self, paper_or_id: Union[ArxivPaper, str]) -> bool:
        """
        Check if a paper exists by its arXiv ID.
        
        Args:
            paper_or_id: ArxivPaper object or paper_full_id string
            
        Returns:
            True if paper exists, False otherwise
        """
        if isinstance(paper_or_id, ArxivPaper):
            paper_id = paper_or_id.full_id
        else:
            paper_id = paper_or_id
        return self.pod_ememory.exists(paper_id)
    
    def has_vector(self, paper_or_id: Union[ArxivPaper, str]) -> bool:
        """
        Check if a paper has a vector embedding.
        
        Args:
            paper_or_id: ArxivPaper object or paper_full_id string
            
        Returns:
            True if paper has vector, False otherwise
        """
        if isinstance(paper_or_id, ArxivPaper):
            paper_id = paper_or_id.full_id
        else:
            paper_id = paper_or_id
        return self.pod_ememory.has_vector(paper_id)
    
    def exists_batch(self, papers_or_ids: List[Union[ArxivPaper, str]]) -> Dict[str, bool]:
        """
        Check existence of multiple papers.
        
        Args:
            papers_or_ids: List of ArxivPaper objects or paper_full_id strings
            
        Returns:
            Dictionary mapping paper full_id to existence status
        """
        paper_ids = []
        for item in papers_or_ids:
            if isinstance(item, ArxivPaper):
                paper_ids.append(item.full_id)
            else:
                paper_ids.append(item)
        return self.pod_ememory.exists_batch(paper_ids)
    
    def has_vector_batch(self, papers_or_ids: List[Union[ArxivPaper, str]]) -> Dict[str, bool]:
        """
        Check if multiple papers have vector embeddings.
        
        Args:
            papers_or_ids: List of ArxivPaper objects or paper_full_id strings
            
        Returns:
            Dictionary mapping paper full_id to vector existence status
        """
        paper_ids = []
        for item in papers_or_ids:
            if isinstance(item, ArxivPaper):
                paper_ids.append(item.full_id)
            else:
                paper_ids.append(item)
        return self.pod_ememory.has_vector_batch(paper_ids)

    def get_vector(self, paper: ArxivPaper) -> Optional[List[float]]:
        """
        Get the vector embedding for a specific paper.
        
        Args:
            paper: ArxivPaper object
            
        Returns:
            Vector embedding as List[float] if found, None otherwise
        """
        if not isinstance(paper, ArxivPaper):
            raise TypeError("paper must be an ArxivPaper instance")
        
        record = self.pod_ememory.get(paper.full_id)
        if record is None:
            self.logger.warning(f"Paper {paper.full_id} not found in database")
            return None
        
        if record.vector is None:
            self.logger.warning(f"Paper {paper.full_id} has no vector embedding")
            return None
        
        return record.vector
    
    def get_vectors(self, papers: List[ArxivPaper]) -> List[Optional[List[float]]]:
        """
        Get vector embeddings for multiple papers.
        
        Args:
            papers: List of ArxivPaper objects
            
        Returns:
            List of vector embeddings (List[float]) or None for papers without vectors
        """
        if not papers:
            return []
        
        vectors = []
        for paper in papers:
            vector = self.get_vector(paper)
            vectors.append(vector)
        
        self.logger.info(f"Retrieved vectors for {len(papers)} papers")
        return vectors

    def get_papers_by_date(
        self, 
        target_date: datetime, 
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[ArxivPaper]:
        """
        Get all papers published on a specific date, optionally filtered by categories.
        
        This method uses efficient SQL-based date filtering instead of loading all records
        into memory, making it suitable for large databases with millions of records.
        
        Args:
            target_date: Target date to filter papers (datetime object)
            categories: Optional list of arXiv categories to filter by. 
                       If None, returns papers from all categories.
            limit: Optional limit on number of papers to return
                       
        Returns:
            List of ArxivPaper objects matching the criteria
        """
        # Use efficient date range query instead of list_all()
        target_date_str = target_date.date().isoformat()
        
        # Query records for the specific date using efficient SQL filtering
        # Note: timestamp stores full ISO datetime, so we need to query the full day range
        start_datetime = f"{target_date_str}T00:00:00"
        end_datetime = f"{target_date_str}T23:59:59"
        
        records = self.pod_ememory.query_by_date_range(
            start_date=start_datetime,
            end_date=end_datetime,
            limit=limit
        )
        
        # Filter by categories if provided
        filtered_papers = []
        for record in records:
            paper = self._record_to_paper(record)
            
            # Check category filter if provided
            if categories is not None:
                # Check if any of the paper's categories match the filter categories
                if not any(cat in paper.categories for cat in categories):
                    continue
            
            filtered_papers.append(paper)
        
        self.logger.info(f"Found {len(filtered_papers)} papers for date {target_date_str}")
        if categories:
            self.logger.info(f"Filtered by categories: {categories}")
        
        return filtered_papers

    # @pyinstrument.profile()
    def get_vectors_by_date(
        self, 
        target_date: datetime, 
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Optional[List[float]]]:
        """
        Get vector embeddings for all papers published on a specific date, optionally filtered by categories.
        
        This method uses efficient SQL-based date filtering and returns only vectors,
        making it suitable for large databases with millions of records.
        
        Args:
            target_date: Target date to filter papers (datetime object)
            categories: Optional list of arXiv categories to filter by. 
                       If None, returns vectors from all categories.
            limit: Optional limit on number of vectors to return
                       
        Returns:
            List of vector embeddings (List[float]) or None for papers without vectors
        """
        # Use efficient date range query instead of list_all()
        target_date_str = target_date.date().isoformat()
        
        # Query records for the specific date using efficient SQL filtering
        # Note: timestamp stores full ISO datetime, so we need to query the full day range
        start_datetime = f"{target_date_str}T00:00:00"
        end_datetime = f"{target_date_str}T23:59:59"
        
        records = self.pod_ememory.query_by_date_range(
            start_date=start_datetime,
            end_date=end_datetime,
            limit=limit
        )
        
        # Extract vectors and filter by categories if provided
        vectors = []
        for record in records:
            # Check category filter if provided
            if categories is not None:
                # Convert record to paper to check categories
                paper = self._record_to_paper(record)
                # Check if any of the paper's categories match the filter categories
                if not any(cat in paper.categories for cat in categories):
                    continue
            
            # Add vector (can be None if no embedding)
            vectors.append(record.vector)
        
        self.logger.info(f"Found {len(vectors)} vectors for date {target_date_str}")
        if categories:
            self.logger.info(f"Filtered by categories: {categories}")
        
        return vectors

    def clear(self, confirm: bool = True) -> None:
        """
        Clear all papers from the database.
        
        Args:
            confirm: If True, requires user confirmation before clearing.
                    If False, clears immediately without confirmation.
        """
        self.pod_ememory.clear(confirm=confirm)
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
            metadata=attrs.get("metadata", {})
        )
        
        return paper
