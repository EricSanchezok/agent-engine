"""
ArxivPaper - A clean, structured representation of arXiv papers.

This module provides a dataclass-based Paper class that separates concerns
and provides type safety for arXiv paper data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
import arxiv
from typing import Any, Dict, List, Optional


def _clean_arxiv_id(raw_id: str) -> str:
    """
    Clean arXiv ID by converting underscores to dots and removing version suffix.
    
    Args:
        raw_id: Raw arXiv ID (e.g., "2507_18009v3")
        
    Returns:
        Cleaned arXiv ID (e.g., "2507.18009")
        
    Examples:
        >>> _clean_arxiv_id("2507_18009v3")
        '2507.18009'
        >>> _clean_arxiv_id("2507.18009")
        '2507.18009'
    """
    if not raw_id:
        raise ValueError("arXiv ID cannot be empty")
    
    # Convert underscores to dots
    fixed = raw_id.replace("_", ".")
    
    # Remove version suffix (v1, v2, etc.)
    fixed = re.sub(r"v\d+$", "", fixed)
    
    return fixed


def _extract_version(raw_id: str) -> Optional[str]:
    """
    Extract version number from arXiv ID.
    
    Args:
        raw_id: Raw arXiv ID
        
    Returns:
        Version string (e.g., "v2") or None if no version
    """
    match = re.search(r"v(\d+)$", raw_id)
    return f"v{match.group(1)}" if match else None


@dataclass
class ArxivPaper:
    """
    A clean, structured representation of an arXiv paper.
    
    This class provides type-safe access to paper metadata and separates
    concerns from storage and database operations.
    """
    
    # Core identification
    id: str  # Clean arXiv ID (e.g., "2507.18009")
    version: Optional[str] = None  # Version suffix (e.g., "v2")
    
    # Basic information
    title: str = ""
    authors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    summary: str = ""
    
    # Date information
    published_date: Optional[datetime] = None
    submitted_date: Optional[datetime] = None
    
    # Links and references
    pdf_url: str = ""
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    comment: Optional[str] = None
    
    
    # Extended metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and clean the paper data after initialization."""
        # Clean the arXiv ID
        self.id = _clean_arxiv_id(self.id)
        
        # Extract version if not provided
        if self.version is None:
            # We need the original ID to extract version, but we've already cleaned it
            # This is a limitation of the current design - we'd need the original ID
            pass
    
    @property
    def full_id(self) -> str:
        """Get the full arXiv ID including version."""
        if self.version:
            return f"{self.id}{self.version}"
        return self.id
    
    @property
    def clean_title(self) -> str:
        """Get a cleaned version of the title for file naming."""
        # Remove special characters and normalize whitespace
        clean = re.sub(r"[^\w\s]", "", self.title.lower())
        clean = re.sub(r"\s+", "_", clean).strip("_")
        return clean.capitalize() if clean else "unknown_title"
    
    @property
    def year(self) -> Optional[int]:
        """Extract year from published or submitted date."""
        if self.published_date:
            return self.published_date.year
        if self.submitted_date:
            return self.submitted_date.year
        return None
    
    @property
    def primary_category(self) -> Optional[str]:
        """Get the primary arXiv category."""
        return self.categories[0] if self.categories else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "version": self.version,
            "title": self.title,
            "authors": self.authors,
            "categories": self.categories,
            "summary": self.summary,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "submitted_date": self.submitted_date.isoformat() if self.submitted_date else None,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "journal_ref": self.journal_ref,
            "comment": self.comment,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_arxiv_result(cls, result: "arxiv.Result") -> "ArxivPaper":
        """
        Create ArxivPaper from arxiv.Result object.
        
        Args:
            result: arxiv.Result object from arxiv package
            
        Returns:
            ArxivPaper instance
        """
        import arxiv  # Import here to avoid circular imports
        
        if not isinstance(result, arxiv.Result):
            raise TypeError("result must be an arxiv.Result instance")
        
        # Extract arXiv ID and version
        raw_id = result.entry_id.split("/")[-1]
        clean_id = _clean_arxiv_id(raw_id)
        version = _extract_version(raw_id)
        
        # Extract authors
        authors = [author.name for author in result.authors] if result.authors else []
        
        # Extract categories
        categories = result.categories if result.categories else []
        
        # Generate PDF URL
        pdf_url = getattr(result, "pdf_url", "") or result.entry_id.replace("/abs/", "/pdf/")
        
        # Clean title
        title = (result.title or "").strip()
        
        return cls(
            id=clean_id,
            version=version,
            title=title,
            authors=authors,
            categories=categories,
            summary=result.summary or "",
            published_date=result.published,
            submitted_date=getattr(result, "submitted", None),
            pdf_url=pdf_url,
            doi=result.doi,
            journal_ref=result.journal_ref,
            comment=result.comment,
            metadata={
                "links": [str(link.href) for link in (result.links or []) if hasattr(link, "href")],
                "raw_id": raw_id,
            }
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArxivPaper":
        """
        Create ArxivPaper from dictionary.
        
        Args:
            data: Dictionary containing paper data
            
        Returns:
            ArxivPaper instance
        """
        # Parse dates if they exist
        published_date = None
        if data.get("published_date"):
            if isinstance(data["published_date"], str):
                published_date = datetime.fromisoformat(data["published_date"])
            elif isinstance(data["published_date"], datetime):
                published_date = data["published_date"]
        
        submitted_date = None
        if data.get("submitted_date"):
            if isinstance(data["submitted_date"], str):
                submitted_date = datetime.fromisoformat(data["submitted_date"])
            elif isinstance(data["submitted_date"], datetime):
                submitted_date = data["submitted_date"]
        
        return cls(
            id=data["id"],
            version=data.get("version"),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            categories=data.get("categories", []),
            summary=data.get("summary", ""),
            published_date=published_date,
            submitted_date=submitted_date,
            pdf_url=data.get("pdf_url", ""),
            doi=data.get("doi"),
            journal_ref=data.get("journal_ref"),
            comment=data.get("comment"),
            metadata=data.get("metadata", {}),
        )
    
    def __str__(self) -> str:
        """String representation of the paper."""
        return f"ArxivPaper(id='{self.full_id}', title='{self.title[:50]}...')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ArxivPaper(id='{self.full_id}', title='{self.title}', "
            f"authors={len(self.authors)}, categories={self.categories})"
        )
