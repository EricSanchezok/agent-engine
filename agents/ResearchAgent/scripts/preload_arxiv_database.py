"""
Arxiv Database Preloader

This script preloads arXiv papers into the database by:
1. Rolling back day by day from today
2. Fetching all papers for each day using ArxivFetcher
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
    """Arxiv database preloader with daily rolling and concurrent processing."""
    
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
    
    def _get_day_range(self, target_date: datetime) -> Tuple[datetime, datetime]:
        """
        Get the day range for a given target date.
        For arXiv search, we use [target_date, target_date + 1 day) range.
        
        Args:
            target_date: Target date to search papers for
            
        Returns:
            Tuple of (day_start, day_end) dates
        """
        # Start of the target day
        day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        # End of the target day (start of next day)
        day_end = day_start + timedelta(days=1)
        
        return day_start, day_end
    
    def _format_date_for_query(self, date: datetime) -> str:
        """Format date for arXiv query (YYYYMMDD)."""
        return date.strftime("%Y%m%d")
    
    def _format_date_range_name(self, start_date: datetime, end_date: datetime) -> str:
        """Format date range for file naming."""
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return f"{start_str}_{end_str}"
    
    async def _fetch_papers_for_day(self, start_date: datetime, end_date: datetime) -> List[ArxivPaper]:
        """
        Fetch all papers for a given day.
        
        Args:
            start_date: Day start date
            end_date: Day end date (next day)
            
        Returns:
            List of ArxivPaper objects
        """
        start_str = self._format_date_for_query(start_date)
        end_str = self._format_date_for_query(end_date)
        
        # Build query for the date range
        query = f"submittedDate:[{start_str} TO {end_str}]"
        
        self.logger.info(f"Fetching papers for day {start_str} to {end_str}")
        
        try:
            papers = await self.arxiv_fetcher.search_papers(
                query=query,
                max_results=10000  # Large number to get all papers
            )
            
            self.logger.info(f"Found {len(papers)} papers for day {start_str} to {end_str}")
            return papers
            
        except Exception as e:
            self.logger.error(f"Failed to fetch papers for day {start_str} to {end_str}: {e}")
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
    
    async def _process_day(self, start_date: datetime, end_date: datetime) -> dict:
        """
        Process a single day of papers.
        
        Args:
            start_date: Day start date
            end_date: Day end date (next day)
            
        Returns:
            Dictionary with processing results
        """
        day_name = self._format_date_range_name(start_date, end_date)
        
        self.logger.info(f"Processing day: {day_name}")
        
        # Fetch papers for the day
        papers = await self._fetch_papers_for_day(start_date, end_date)
        
        if not papers:
            self.logger.info(f"No papers found for day {day_name}")
            return {
                "day": day_name,
                "total_papers": 0,
                "filtered_papers": 0,
                "successful_papers": 0,
                "failed_papers": 0,
                "saved_papers": 0
            }
        
        # Check which papers already exist and have vectors
        self.logger.info(f"Checking existence of {len(papers)} papers in database")
        existence_map = self.arxiv_database.exists_batch(papers)
        vector_map = self.arxiv_database.has_vector_batch(papers)
        
        # Filter out papers that already exist and have vectors
        papers_to_process = []
        filtered_count = 0
        
        for paper in papers:
            paper_id = paper.full_id
            exists = existence_map.get(paper_id, False)
            has_vector = vector_map.get(paper_id, False)
            
            if exists and has_vector:
                filtered_count += 1
                self.logger.debug(f"Paper {paper_id} already exists with vector, skipping")
            else:
                papers_to_process.append(paper)
        
        self.logger.info(f"Filtered out {filtered_count} papers that already exist with vectors")
        self.logger.info(f"Processing {len(papers_to_process)} new papers")
        
        if not papers_to_process:
            self.logger.info(f"All papers for day {day_name} already exist with vectors")
            return {
                "day": day_name,
                "total_papers": len(papers),
                "filtered_papers": filtered_count,
                "successful_papers": 0,
                "failed_papers": 0,
                "saved_papers": 0
            }
        
        # Generate embeddings concurrently for new papers only
        successful_papers, failed_papers = await self._generate_embeddings_concurrent(papers_to_process)
        
        # Save successful papers to database
        saved_paper_ids = await self._save_papers_to_database(successful_papers)
        
        # Save failed IDs to file
        self._save_failed_ids(failed_papers, day_name)
        
        result = {
            "day": day_name,
            "total_papers": len(papers),
            "filtered_papers": filtered_count,
            "successful_papers": len(successful_papers),
            "failed_papers": len(failed_papers),
            "saved_papers": len(saved_paper_ids)
        }
        
        self.logger.info(f"Day {day_name} processed: {result}")
        return result
    
    async def preload_days(self, num_days: int = 10) -> List[dict]:
        """
        Preload papers for the specified number of days, rolling back from today.
        
        Args:
            num_days: Number of days to process
            
        Returns:
            List of processing results for each day
        """
        from preload_config import PreloadConfig
        
        self.logger.info(f"Starting preload for {num_days} days")
        
        results = []
        current_date = datetime.now()
        
        for day_offset in range(num_days):
            # Calculate the day range
            target_date = current_date - timedelta(days=day_offset)
            day_start, day_end = self._get_day_range(target_date)
            
            # Process the day
            result = await self._process_day(day_start, day_end)
            results.append(result)
            
            # Small delay between days to be respectful to APIs
            await asyncio.sleep(PreloadConfig.DELAY_BETWEEN_DAYS)
        
        # Summary
        total_papers = sum(r["total_papers"] for r in results)
        total_filtered = sum(r["filtered_papers"] for r in results)
        total_successful = sum(r["successful_papers"] for r in results)
        total_failed = sum(r["failed_papers"] for r in results)
        total_saved = sum(r["saved_papers"] for r in results)
        
        self.logger.info(f"Preload completed:")
        self.logger.info(f"  Total papers fetched: {total_papers}")
        self.logger.info(f"  Papers already exist with vectors: {total_filtered}")
        self.logger.info(f"  New papers processed: {total_papers - total_filtered}")
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
        # Preload papers for specified number of days
        results = await preloader.preload_days(num_days=PreloadConfig.DEFAULT_NUM_DAYS)
        
        # Print summary
        print("\n=== Preload Summary ===")
        for result in results:
            print(f"Day {result['day']}: {result['total_papers']} papers fetched, "
                  f"{result['filtered_papers']} already exist, "
                  f"{result['successful_papers']} successful, "
                  f"{result['failed_papers']} failed, "
                  f"{result['saved_papers']} saved")
        
    except Exception as e:
        print(f"Preload failed: {e}")
        
    finally:
        await preloader.close()


if __name__ == "__main__":
    asyncio.run(main())
