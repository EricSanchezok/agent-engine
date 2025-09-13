"""
Arxiv Database Preloader

This script preloads arXiv papers into the database by:
1. Rolling back week by week from today
2. Fetching all papers for each week using ArxivFetcher
3. Generating embeddings for paper summaries using QzClient
4. Saving successful papers to ArxivDatabase
5. Logging failed paper IDs to JSON files
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client.qz_client import QzClient

from core.arxiv_fetcher.arxiv_fetcher import ArxivFetcher
from core.arxiv_fetcher.arxiv_paper import ArxivPaper
from agents.ResearchAgent.arxiv_database import ArxivDatabase


class ArxivPreloader:
    """Arxiv database preloader with weekly rolling and concurrent processing."""
    
    def __init__(
        self,
        qz_api_key: str,
        qz_base_url: str,
        embedding_model: str = "text-embedding-3-small",
        max_concurrent_embeddings: int = 10,
        database_name: str = "arxiv_papers",
        database_dir: Optional[str] = None
    ):
        """
        Initialize the Arxiv preloader.
        
        Args:
            qz_api_key: Qz API key for embedding generation
            qz_base_url: Qz API base URL
            embedding_model: Model name for embeddings
            max_concurrent_embeddings: Maximum concurrent embedding requests
            database_name: Name for the ArxivDatabase
            database_dir: Directory for database storage
        """
        self.logger = AgentLogger(self.__class__.__name__)
        
        # Initialize components
        self.arxiv_fetcher = ArxivFetcher()
        self.qz_client = QzClient(api_key=qz_api_key, base_url=qz_base_url)
        self.arxiv_database = ArxivDatabase(
            name=database_name,
            persist_dir=database_dir
        )
        
        # Configuration
        self.embedding_model = embedding_model
        self.max_concurrent_embeddings = max_concurrent_embeddings
        
        # Create failed IDs directory
        self.failed_ids_dir = Path(__file__).parent.parent / "database" / "preload_failed_ids"
        self.failed_ids_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ArxivPreloader initialized with embedding model: {embedding_model}")
        self.logger.info(f"Failed IDs will be saved to: {self.failed_ids_dir}")
    
    async def close(self):
        """Close all connections."""
        await self.qz_client.close()
        self.logger.info("ArxivPreloader connections closed")
    
    def _get_week_range(self, start_date: datetime) -> Tuple[datetime, datetime]:
        """
        Get the week range for a given start date.
        
        Args:
            start_date: Starting date
            
        Returns:
            Tuple of (week_start, week_end) dates
        """
        # Get the start of the week (Monday)
        days_since_monday = start_date.weekday()
        week_start = start_date - timedelta(days=days_since_monday)
        week_end = week_start + timedelta(days=6)  # Sunday
        
        return week_start, week_end
    
    def _format_date_for_query(self, date: datetime) -> str:
        """Format date for arXiv query (YYYYMMDD)."""
        return date.strftime("%Y%m%d")
    
    def _format_date_range_name(self, start_date: datetime, end_date: datetime) -> str:
        """Format date range for file naming."""
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return f"{start_str}_{end_str}"
    
    async def _fetch_papers_for_week(self, start_date: datetime, end_date: datetime) -> List[ArxivPaper]:
        """
        Fetch all papers for a given week.
        
        Args:
            start_date: Week start date
            end_date: Week end date
            
        Returns:
            List of ArxivPaper objects
        """
        start_str = self._format_date_for_query(start_date)
        end_str = self._format_date_for_query(end_date)
        
        # Build query for the date range
        query = f"submittedDate:[{start_str} TO {end_str}]"
        
        self.logger.info(f"Fetching papers for week {start_str} to {end_str}")
        
        try:
            papers = await self.arxiv_fetcher.search_papers(
                query=query,
                max_results=10000  # Large number to get all papers
            )
            
            self.logger.info(f"Found {len(papers)} papers for week {start_str} to {end_str}")
            return papers
            
        except Exception as e:
            self.logger.error(f"Failed to fetch papers for week {start_str} to {end_str}: {e}")
            return []
    
    async def _generate_embedding(self, paper: ArxivPaper) -> Optional[List[float]]:
        """
        Generate embedding for a single paper's summary.
        
        Args:
            paper: ArxivPaper object
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            if not paper.summary.strip():
                self.logger.warning(f"Paper {paper.full_id} has empty summary")
                return None
            
            embedding = await self.qz_client.get_embeddings(
                model_name=self.embedding_model,
                text=paper.summary
            )
            
            if embedding is None:
                self.logger.warning(f"Failed to generate embedding for paper {paper.full_id}")
                return None
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding for paper {paper.full_id}: {e}")
            return None
    
    async def _generate_embeddings_concurrent(
        self, 
        papers: List[ArxivPaper]
    ) -> Tuple[List[Tuple[ArxivPaper, List[float]]], List[ArxivPaper]]:
        """
        Generate embeddings for multiple papers concurrently.
        
        Args:
            papers: List of ArxivPaper objects
            
        Returns:
            Tuple of (successful_papers_with_embeddings, failed_papers)
        """
        if not papers:
            return [], []
        
        self.logger.info(f"Generating embeddings for {len(papers)} papers")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_embeddings)
        
        async def process_paper(paper: ArxivPaper) -> Tuple[ArxivPaper, Optional[List[float]]]:
            async with semaphore:
                embedding = await self._generate_embedding(paper)
                return paper, embedding
        
        # Process all papers concurrently
        tasks = [process_paper(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful and failed results
        successful_papers = []
        failed_papers = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Exception processing paper {papers[i].full_id}: {result}")
                failed_papers.append(papers[i])
            else:
                paper, embedding = result
                if embedding is not None:
                    successful_papers.append((paper, embedding))
                else:
                    failed_papers.append(paper)
        
        self.logger.info(f"Embedding generation completed: {len(successful_papers)} successful, {len(failed_papers)} failed")
        return successful_papers, failed_papers
    
    async def _save_papers_to_database(
        self, 
        papers_with_embeddings: List[Tuple[ArxivPaper, List[float]]]
    ) -> List[str]:
        """
        Save papers with embeddings to the database.
        
        Args:
            papers_with_embeddings: List of (paper, embedding) tuples
            
        Returns:
            List of successfully saved paper IDs
        """
        if not papers_with_embeddings:
            return []
        
        self.logger.info(f"Saving {len(papers_with_embeddings)} papers to database")
        
        try:
            # Extract papers and embeddings
            papers = [item[0] for item in papers_with_embeddings]
            embeddings = [item[1] for item in papers_with_embeddings]
            
            # Save to database
            record_ids = self.arxiv_database.add_papers(papers, embeddings)
            
            self.logger.info(f"Successfully saved {len(record_ids)} papers to database")
            return [paper.full_id for paper in papers]
            
        except Exception as e:
            self.logger.error(f"Failed to save papers to database: {e}")
            return []
    
    def _save_failed_ids(self, failed_papers: List[ArxivPaper], week_name: str):
        """
        Save failed paper IDs to JSON file.
        
        Args:
            failed_papers: List of failed ArxivPaper objects
            week_name: Week identifier for filename
        """
        if not failed_papers:
            return
        
        failed_ids = [paper.full_id for paper in failed_papers]
        
        # Create filename
        filename = f"{week_name}.json"
        filepath = self.failed_ids_dir / filename
        
        # Save to JSON
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "week_range": week_name,
                    "failed_count": len(failed_ids),
                    "failed_ids": failed_ids,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(failed_ids)} failed IDs to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save failed IDs to {filepath}: {e}")
    
    async def _process_week(self, start_date: datetime, end_date: datetime) -> dict:
        """
        Process a single week of papers.
        
        Args:
            start_date: Week start date
            end_date: Week end date
            
        Returns:
            Dictionary with processing results
        """
        week_name = self._format_date_range_name(start_date, end_date)
        
        self.logger.info(f"Processing week: {week_name}")
        
        # Fetch papers for the week
        papers = await self._fetch_papers_for_week(start_date, end_date)
        
        if not papers:
            self.logger.info(f"No papers found for week {week_name}")
            return {
                "week": week_name,
                "total_papers": 0,
                "successful_papers": 0,
                "failed_papers": 0,
                "saved_papers": 0
            }
        
        # Generate embeddings concurrently
        successful_papers, failed_papers = await self._generate_embeddings_concurrent(papers)
        
        # Save successful papers to database
        saved_paper_ids = await self._save_papers_to_database(successful_papers)
        
        # Save failed IDs to file
        self._save_failed_ids(failed_papers, week_name)
        
        result = {
            "week": week_name,
            "total_papers": len(papers),
            "successful_papers": len(successful_papers),
            "failed_papers": len(failed_papers),
            "saved_papers": len(saved_paper_ids)
        }
        
        self.logger.info(f"Week {week_name} processed: {result}")
        return result
    
    async def preload_weeks(self, num_weeks: int = 10) -> List[dict]:
        """
        Preload papers for the specified number of weeks, rolling back from today.
        
        Args:
            num_weeks: Number of weeks to process
            
        Returns:
            List of processing results for each week
        """
        self.logger.info(f"Starting preload for {num_weeks} weeks")
        
        results = []
        current_date = datetime.now()
        
        for week_offset in range(num_weeks):
            # Calculate the week range
            week_start_date = current_date - timedelta(weeks=week_offset)
            week_start, week_end = self._get_week_range(week_start_date)
            
            # Process the week
            result = await self._process_week(week_start, week_end)
            results.append(result)
            
            # Small delay between weeks to be respectful to APIs
            await asyncio.sleep(2)
        
        # Summary
        total_papers = sum(r["total_papers"] for r in results)
        total_successful = sum(r["successful_papers"] for r in results)
        total_failed = sum(r["failed_papers"] for r in results)
        total_saved = sum(r["saved_papers"] for r in results)
        
        self.logger.info(f"Preload completed:")
        self.logger.info(f"  Total papers processed: {total_papers}")
        self.logger.info(f"  Successful embeddings: {total_successful}")
        self.logger.info(f"  Failed embeddings: {total_failed}")
        self.logger.info(f"  Saved to database: {total_saved}")
        
        return results


async def main():
    """Main execution function."""
    from preload_config import PreloadConfig
    
    # Validate configuration
    if not PreloadConfig.validate():
        return
    
    PreloadConfig.print_config()
    
    # Initialize preloader
    preloader = ArxivPreloader(
        qz_api_key=PreloadConfig.QZ_API_KEY,
        qz_base_url=PreloadConfig.QZ_BASE_URL,
        embedding_model=PreloadConfig.EMBEDDING_MODEL,
        max_concurrent_embeddings=PreloadConfig.MAX_CONCURRENT_EMBEDDINGS,
        database_name=PreloadConfig.DATABASE_NAME,
        database_dir=PreloadConfig.DATABASE_DIR
    )
    
    try:
        # Preload papers for specified number of weeks
        results = await preloader.preload_weeks(num_weeks=PreloadConfig.DEFAULT_NUM_WEEKS)
        
        # Print summary
        print("\n=== Preload Summary ===")
        for result in results:
            print(f"Week {result['week']}: {result['total_papers']} papers, "
                  f"{result['successful_papers']} successful, "
                  f"{result['failed_papers']} failed, "
                  f"{result['saved_papers']} saved")
        
    except Exception as e:
        print(f"Preload failed: {e}")
        
    finally:
        await preloader.close()


if __name__ == "__main__":
    asyncio.run(main())
