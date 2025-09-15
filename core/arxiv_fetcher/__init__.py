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


arXiv Fetcher Module

This module provides clean interfaces for fetching and working with arXiv papers.
"""

from .arxiv_paper import ArxivPaper
from .arxiv_fetcher import ArxivFetcher, normalize_arxiv_id

__all__ = ["ArxivPaper", "ArxivFetcher", "normalize_arxiv_id"]
