"""
Daily arXiv Filter and Download Service - Step 1

This module handles the first step of the daily arXiv service:
1. Check if ArxivDatabase has updated papers for today
2. Get vectors for today's papers
3. Calculate minimum distances using qiji_library
4. Select top 16 papers and download them
"""

import asyncio
from datetime import datetime, date
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_current_file_dir

from agents.ResearchAgent.arxiv_database import ArxivDatabase
from agents.ResearchAgent.qiji_library import QijiLibrary
from core.arxiv_fetcher import ArxivFetcher
from .config import DailyArxivConfig


class DailyArxivFilterAndDownload:
    """
    Daily arXiv filter and download service for Step 1.
    
    This class handles filtering today's papers based on similarity to qiji library
    and downloading the most relevant papers.
    """
    
    def __init__(self, pdf_storage_dir: Optional[str] = None):
        """
        Initialize the daily arXiv filter and download service.
        
        Args:
            pdf_storage_dir: Directory to store downloaded PDFs (optional, uses config if None)
        """
        self.logger = AgentLogger(self.__class__.__name__)
        
        # Initialize components
        self.arxiv_database = ArxivDatabase()
        self.qiji_library = QijiLibrary()
        
        # Set up PDF storage directory from config
        if pdf_storage_dir is None:
            self.pdf_storage_dir = Path(DailyArxivConfig.get_pdf_storage_dir())
        else:
            self.pdf_storage_dir = Path(pdf_storage_dir)
        
        # Initialize ArxivFetcher with PDF storage
        self.arxiv_fetcher = ArxivFetcher(pdf_storage_dir=str(self.pdf_storage_dir))
        
        self.logger.info(f"DailyArxivFilterAndDownload initialized")
        self.logger.info(f"PDF storage directory: {self.pdf_storage_dir}")
    
    async def check_database_update(self, target_date: Optional[date] = None) -> bool:
        """
        Check if ArxivDatabase has papers for the target date.
        
        Args:
            target_date: Date to check (defaults to today)
            
        Returns:
            True if database has papers for the date, False otherwise
        """
        if target_date is None:
            target_date = date.today()
        
        self.logger.info(f"Checking database for papers on {target_date}")
        
        # Get papers for the target date
        papers = self.arxiv_database.get_papers_by_date(
            target_date=datetime.combine(target_date, datetime.min.time()),
            limit=1  # Just check if any papers exist
        )
        
        has_papers = len(papers) > 0
        self.logger.info(f"Database check result: {len(papers)} papers found for {target_date}")
        
        return has_papers
    
    async def get_today_papers_and_vectors(
        self, 
        target_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> Tuple[List[Any], List[Optional[List[float]]]]:
        """
        Get today's papers and their vectors from ArxivDatabase.
        
        Args:
            target_date: Date to get papers for (defaults to today)
            limit: Optional limit on number of papers
            
        Returns:
            Tuple of (papers, vectors) where vectors can be None for papers without embeddings
        """
        if target_date is None:
            target_date = date.today()
        
        self.logger.info(f"Getting papers and vectors for {target_date}")
        
        # Get papers for the target date
        papers = self.arxiv_database.get_papers_by_date(
            target_date=datetime.combine(target_date, datetime.min.time()),
            limit=limit
        )
        
        if not papers:
            self.logger.warning(f"No papers found for {target_date}")
            return [], []
        
        # Get vectors for the papers
        vectors = self.arxiv_database.get_vectors(papers)
        
        self.logger.info(f"Retrieved {len(papers)} papers with {sum(1 for v in vectors if v is not None)} vectors")
        
        return papers, vectors
    
    async def calculate_minimum_distances(
        self, 
        vectors: List[Optional[List[float]]]
    ) -> List[Optional[float]]:
        """
        Calculate minimum distances for all vectors using qiji_library.
        
        Args:
            vectors: List of vectors (can contain None values)
            
        Returns:
            List of minimum distances (None for failed calculations)
        """
        if not vectors:
            self.logger.warning("No vectors provided for distance calculation")
            return []
        
        # Filter out None vectors for distance calculation
        valid_vectors = [v for v in vectors if v is not None]
        valid_indices = [i for i, v in enumerate(vectors) if v is not None]
        
        if not valid_vectors:
            self.logger.warning("No valid vectors found for distance calculation")
            return [None] * len(vectors)
        
        self.logger.info(f"Calculating minimum distances for {len(valid_vectors)} valid vectors")
        
        # Calculate minimum distances using qiji_library
        distances = await self.qiji_library.find_minimum_distances_batch(valid_vectors)
        
        # Map distances back to original vector positions
        result_distances = [None] * len(vectors)
        for i, distance in enumerate(distances):
            original_index = valid_indices[i]
            result_distances[original_index] = distance
        
        successful_count = sum(1 for d in result_distances if d is not None)
        self.logger.info(f"Distance calculation completed: {successful_count}/{len(vectors)} successful")
        
        return result_distances
    
    def select_top_papers(
        self, 
        papers: List[Any], 
        distances: List[Optional[float]], 
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        Select top K papers with minimum distances.
        
        Args:
            papers: List of papers
            distances: List of minimum distances (None for invalid distances)
            top_k: Number of top papers to select (uses config if None)
            
        Returns:
            List of selected papers
        """
        if top_k is None:
            top_k = DailyArxivConfig.TOP_K_PAPERS
            
        if len(papers) != len(distances):
            raise ValueError("Papers and distances lists must have the same length")
        
        # Create list of (paper, distance) tuples, filtering out None distances
        paper_distance_pairs = []
        for i, (paper, distance) in enumerate(zip(papers, distances)):
            if distance is not None:
                paper_distance_pairs.append((paper, distance))
        
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
    
    async def download_selected_papers(
        self, 
        papers: List[Any], 
        max_concurrent: Optional[int] = None
    ) -> List[Tuple[bool, Any]]:
        """
        Download PDFs for selected papers.
        
        Args:
            papers: List of papers to download
            max_concurrent: Maximum concurrent downloads (uses config if None)
            
        Returns:
            List of (success, paper) tuples
        """
        if max_concurrent is None:
            max_concurrent = DailyArxivConfig.MAX_CONCURRENT_DOWNLOADS
            
        if not papers:
            self.logger.warning("No papers provided for download")
            return []
        
        self.logger.info(f"Starting download of {len(papers)} papers")
        
        # Download papers using ArxivFetcher
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
    
    async def run_step1(
        self, 
        target_date: Optional[date] = None,
        top_k: Optional[int] = None,
        max_concurrent_downloads: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the complete Step 1 process.
        
        Args:
            target_date: Date to process (defaults to config or today)
            top_k: Number of top papers to select and download (uses config if None)
            max_concurrent_downloads: Maximum concurrent downloads (uses config if None)
            
        Returns:
            Dictionary with results and statistics
        """
        if target_date is None:
            if DailyArxivConfig.TARGET_DATE:
                try:
                    target_date = datetime.strptime(DailyArxivConfig.TARGET_DATE, "%Y-%m-%d").date()
                except ValueError:
                    self.logger.warning(f"Invalid TARGET_DATE in config: {DailyArxivConfig.TARGET_DATE}, using today")
                    target_date = date.today()
            else:
                target_date = date.today()
        
        if top_k is None:
            top_k = DailyArxivConfig.TOP_K_PAPERS
            
        if max_concurrent_downloads is None:
            max_concurrent_downloads = DailyArxivConfig.MAX_CONCURRENT_DOWNLOADS
        
        self.logger.info(f"Starting Step 1 for date: {target_date}")
        
        try:
            # Step 1: Check if database has updated papers
            has_papers = await self.check_database_update(target_date)
            if not has_papers:
                self.logger.warning(f"No papers found for {target_date}, skipping Step 1")
                return {
                    "success": False,
                    "reason": "No papers found for target date",
                    "target_date": target_date.isoformat(),
                    "papers_processed": 0,
                    "papers_downloaded": 0
                }
            
            # Step 2: Get today's papers and vectors
            papers, vectors = await self.get_today_papers_and_vectors(target_date)
            if not papers:
                self.logger.warning(f"No papers retrieved for {target_date}")
                return {
                    "success": False,
                    "reason": "No papers retrieved",
                    "target_date": target_date.isoformat(),
                    "papers_processed": 0,
                    "papers_downloaded": 0
                }
            
            # Step 3: Calculate minimum distances
            distances = await self.calculate_minimum_distances(vectors)
            
            # Step 4: Select top papers
            selected_papers = self.select_top_papers(papers, distances, top_k)
            if not selected_papers:
                self.logger.warning("No papers selected for download")
                return {
                    "success": False,
                    "reason": "No papers selected",
                    "target_date": target_date.isoformat(),
                    "papers_processed": len(papers),
                    "papers_downloaded": 0
                }
            
            # Step 5: Download selected papers
            download_results = await self.download_selected_papers(
                selected_papers, 
                max_concurrent_downloads
            )
            
            # Calculate statistics
            successful_downloads = sum(1 for success, _ in download_results if success)
            
            result = {
                "success": True,
                "target_date": target_date.isoformat(),
                "papers_processed": len(papers),
                "papers_with_vectors": sum(1 for v in vectors if v is not None),
                "papers_selected": len(selected_papers),
                "papers_downloaded": successful_downloads,
                "download_failures": len(selected_papers) - successful_downloads,
                "selected_papers": [
                    {
                        "full_id": paper.full_id,
                        "title": paper.title,
                        "distance": distances[papers.index(paper)] if paper in papers else None
                    }
                    for paper in selected_papers
                ]
            }
            
            self.logger.info(f"Step 1 completed successfully: {successful_downloads}/{len(selected_papers)} papers downloaded")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Step 1: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "target_date": target_date.isoformat(),
                "papers_processed": 0,
                "papers_downloaded": 0
            }


async def main():
    """Test the daily arXiv filter and download service."""
    service = DailyArxivFilterAndDownload()
    
    # Test with today's date
    result = await service.run_step1()
    
    print("Step 1 Results:")
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
