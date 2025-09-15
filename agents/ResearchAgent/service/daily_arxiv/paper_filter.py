"""
Daily arXiv Paper Filter and Download

This module provides a simple class for filtering and downloading papers
from a specific date based on similarity to qiji library.
"""

import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_current_file_dir

from agents.ResearchAgent.arxiv_database import ArxivDatabase
from agents.ResearchAgent.qiji_library import QijiLibrary
from core.arxiv_fetcher import ArxivFetcher
from .config import DailyArxivConfig


class DailyArxivPaperFilter:
    """
    Daily arXiv paper filter and download service.
    
    This class filters papers from a specific date based on similarity to qiji library
    and downloads the most relevant papers.
    """
    
    def __init__(self, pdf_storage_dir: Optional[str] = None):
        """
        Initialize the paper filter.
        
        Args:
            pdf_storage_dir: Directory to store downloaded PDFs (optional, uses config if None)
        """
        self.logger = AgentLogger(self.__class__.__name__)
        
        # Initialize components
        self.arxiv_database = ArxivDatabase()
        self.qiji_library = QijiLibrary()
        
        # Set up PDF storage directory
        if pdf_storage_dir is None:
            self.pdf_storage_dir = Path(DailyArxivConfig.get_pdf_storage_dir())
        else:
            self.pdf_storage_dir = Path(pdf_storage_dir)
        
        # Ensure PDF storage directory exists
        self.pdf_storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ArxivFetcher
        self.arxiv_fetcher = ArxivFetcher(pdf_storage_dir=str(self.pdf_storage_dir))
        
        self.logger.info(f"DailyArxivPaperFilter initialized with PDF storage: {self.pdf_storage_dir}")
    
    async def filter_and_download_papers(
        self, 
        target_date: date,
        top_k: Optional[int] = None,
        max_concurrent_downloads: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Filter and download papers for a specific date.
        
        Args:
            target_date: Date to process papers for
            top_k: Number of top papers to select and download (uses config if None)
            max_concurrent_downloads: Maximum concurrent downloads (uses config if None)
            
        Returns:
            Dictionary with results and statistics
        """
        if top_k is None:
            top_k = DailyArxivConfig.TOP_K_PAPERS
            
        if max_concurrent_downloads is None:
            max_concurrent_downloads = DailyArxivConfig.MAX_CONCURRENT_DOWNLOADS
        
        self.logger.info(f"Starting paper filtering for date: {target_date}")
        
        try:
            # Step 1: Get papers for the target date
            papers = self._get_papers_for_date(target_date)
            if not papers:
                return {
                    "success": False,
                    "reason": f"No papers found for date {target_date}",
                    "target_date": target_date.isoformat(),
                    "papers_processed": 0,
                    "papers_downloaded": 0
                }
            
            self.logger.info(f"Found {len(papers)} papers for {target_date}")
            
            # Step 2: Get vectors for the papers
            vectors = self._get_vectors_for_papers(papers)
            valid_vectors = [v for v in vectors if v is not None]
            
            if not valid_vectors:
                return {
                    "success": False,
                    "reason": "No papers with valid vectors found",
                    "target_date": target_date.isoformat(),
                    "papers_processed": len(papers),
                    "papers_downloaded": 0
                }
            
            self.logger.info(f"Found {len(valid_vectors)} papers with valid vectors")
            
            # Step 3: Calculate minimum distances using qiji_library
            distances = await self._calculate_minimum_distances(valid_vectors)
            
            # Step 4: Select top K papers
            selected_papers = self._select_top_papers(papers, vectors, distances, top_k)
            if not selected_papers:
                return {
                    "success": False,
                    "reason": "No papers selected for download",
                    "target_date": target_date.isoformat(),
                    "papers_processed": len(papers),
                    "papers_downloaded": 0
                }
            
            self.logger.info(f"Selected {len(selected_papers)} papers for download")
            
            # Step 5: Download selected papers
            download_results = await self._download_papers(selected_papers, max_concurrent_downloads)
            
            # Calculate statistics
            successful_downloads = sum(1 for success, _ in download_results if success)
            
            result = {
                "success": True,
                "target_date": target_date.isoformat(),
                "papers_processed": len(papers),
                "papers_with_vectors": len(valid_vectors),
                "papers_selected": len(selected_papers),
                "papers_downloaded": successful_downloads,
                "download_failures": len(selected_papers) - successful_downloads,
                "selected_papers": [
                    {
                        "full_id": paper.full_id,
                        "title": paper.title,
                        "distance": distances[i] if i < len(distances) else None
                    }
                    for i, paper in enumerate(selected_papers)
                ]
            }
            
            self.logger.info(f"Filtering completed: {successful_downloads}/{len(selected_papers)} papers downloaded")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in paper filtering: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "target_date": target_date.isoformat(),
                "papers_processed": 0,
                "papers_downloaded": 0
            }
    
    def _get_papers_for_date(self, target_date: date) -> List[Any]:
        """Get papers for the target date from ArxivDatabase."""
        target_datetime = datetime.combine(target_date, datetime.min.time())
        papers = self.arxiv_database.get_papers_by_date(target_datetime)
        return papers
    
    def _get_vectors_for_papers(self, papers: List[Any]) -> List[Optional[List[float]]]:
        """Get vectors for the papers from ArxivDatabase."""
        return self.arxiv_database.get_vectors(papers)
    
    async def _calculate_minimum_distances(self, vectors: List[List[float]]) -> List[float]:
        """Calculate minimum distances using qiji_library."""
        self.logger.info(f"Calculating minimum distances for {len(vectors)} vectors")
        distances = await self.qiji_library.find_minimum_distances_batch(vectors)
        self.logger.info(f"Distance calculation completed for {len(distances)} vectors")
        return distances
    
    def _select_top_papers(
        self, 
        papers: List[Any], 
        vectors: List[Optional[List[float]]], 
        distances: List[float], 
        top_k: int
    ) -> List[Any]:
        """Select top K papers based on minimum distances."""
        # Create list of (paper, distance) tuples for papers with valid vectors
        paper_distance_pairs = []
        distance_index = 0
        
        for i, (paper, vector) in enumerate(zip(papers, vectors)):
            if vector is not None:
                if distance_index < len(distances):
                    paper_distance_pairs.append((paper, distances[distance_index]))
                    distance_index += 1
        
        if not paper_distance_pairs:
            self.logger.warning("No papers with valid distances found")
            return []
        
        # Sort by distance (ascending - smaller distance means more similar)
        paper_distance_pairs.sort(key=lambda x: x[1])
        
        # Select top K papers
        selected_papers = [paper for paper, _ in paper_distance_pairs[:top_k]]
        
        self.logger.info(f"Selected {len(selected_papers)} papers out of {len(paper_distance_pairs)} with valid distances")
        
        # Log the selected papers
        for i, (paper, distance) in enumerate(paper_distance_pairs[:top_k]):
            self.logger.info(f"Top {i+1}: {paper.full_id} (distance: {distance:.4f})")
        
        return selected_papers
    
    async def _download_papers(
        self, 
        papers: List[Any], 
        max_concurrent: int
    ) -> List[Tuple[bool, Any]]:
        """Download PDFs for the selected papers."""
        self.logger.info(f"Starting download of {len(papers)} papers")
        
        download_results = await self.arxiv_fetcher.download_papers(
            papers=papers,
            max_concurrent=max_concurrent
        )
        
        successful_count = sum(1 for success, _ in download_results if success)
        self.logger.info(f"Download completed: {successful_count}/{len(papers)} successful")
        
        # Log download results
        for success, paper in download_results:
            status = "SUCCESS" if success else "FAILED"
            self.logger.info(f"Download {status}: {paper.full_id}")
        
        return download_results


async def main():
    """Test the paper filter."""
    filter_service = DailyArxivPaperFilter()
    
    # Test with today's date
    today = date.today()
    result = await filter_service.filter_and_download_papers(today)
    
    print("Paper Filter Results:")
    print(f"Success: {result['success']}")
    print(f"Target Date: {result['target_date']}")
    print(f"Papers Processed: {result['papers_processed']}")
    print(f"Papers Downloaded: {result['papers_downloaded']}")
    
    if result['success']:
        print(f"Selected Papers: {len(result['selected_papers'])}")
        for paper_info in result['selected_papers'][:5]:  # Show first 5
            print(f"  - {paper_info['full_id']}: {paper_info['title'][:50]}...")
    else:
        print(f"Error: {result.get('error', result.get('reason', 'Unknown error'))}")


if __name__ == "__main__":
    asyncio.run(main())
