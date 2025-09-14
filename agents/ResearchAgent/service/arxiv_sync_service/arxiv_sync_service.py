"""
Arxiv Sync Service

A robust service for synchronizing arXiv papers to the database.
Provides daily sync functionality with weekly rollback and embedding generation.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import traceback

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client.qz_client import QzClient

from core.arxiv_fetcher.arxiv_fetcher import ArxivFetcher
from core.arxiv_fetcher.arxiv_paper import ArxivPaper
from agents.ResearchAgent.arxiv_database import ArxivDatabase

from .config import ArxivSyncConfig


class ArxivSyncService:
    """
    Arxiv synchronization service that:
    1. Runs continuously and checks for new papers every 15 minutes
    2. Syncs the current week's papers (Monday to Sunday)
    3. Generates embeddings for new papers
    4. Handles errors gracefully with retries
    """
    
    def __init__(self):
        """Initialize the Arxiv sync service."""
        self.logger = AgentLogger(self.__class__.__name__)
        self.config = ArxivSyncConfig()
        
        # Initialize components
        self.arxiv_fetcher = ArxivFetcher()
        self.qz_client = QzClient(
            api_key=self.config.QZ_API_KEY,
            base_url=self.config.QZ_BASE_URL
        )
        self.arxiv_database = ArxivDatabase(
            name=self.config.DATABASE_NAME,
            persist_dir=self.config.DATABASE_DIR
        )
        
        # Service state
        self.is_running = False
        self.last_sync_date = None
        self.sync_stats = {
            "total_syncs": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "total_papers_processed": 0,
            "total_papers_added": 0
        }
        
        self.logger.info(f"{self.config.SERVICE_NAME} initialized")
        self.logger.info(f"Sync interval: {self.config.SYNC_INTERVAL_MINUTES} minutes")
        self.logger.info(f"Max concurrent embeddings: {self.config.MAX_CONCURRENT_EMBEDDINGS}")
    
    async def close(self):
        """Close all connections."""
        await self.qz_client.close()
        self.logger.info(f"{self.config.SERVICE_NAME} connections closed")
    
    def _get_current_week_dates(self) -> List[datetime]:
        """
        Get all dates for the current week (Monday to Sunday).
        
        Returns:
            List of datetime objects for the current week
        """
        today = datetime.now()
        
        # Find Monday of current week
        days_since_monday = today.weekday()  # Monday = 0, Sunday = 6
        monday = today - timedelta(days=days_since_monday)
        
        # Generate all 7 days of the week
        week_dates = []
        for i in range(7):
            week_dates.append(monday + timedelta(days=i))
        
        return week_dates
    
    def _format_date_for_query(self, date: datetime) -> str:
        """Format date for arXiv query (YYYYMMDD)."""
        return date.strftime("%Y%m%d")
    
    def _get_day_range(self, target_date: datetime) -> tuple[datetime, datetime]:
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
        
        self.logger.info(f"Fetching papers for day {start_str}")
        
        try:
            papers = await self.arxiv_fetcher.search_papers(
                query=query,
                max_results=10000  # Large number to get all papers
            )
            
            self.logger.info(f"Found {len(papers)} papers for day {start_str}")
            return papers
            
        except Exception as e:
            self.logger.error(f"Failed to fetch papers for day {start_str}: {e}")
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
                model_name=self.config.EMBEDDING_MODEL,
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
    ) -> tuple[List[tuple[ArxivPaper, List[float]]], List[ArxivPaper]]:
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
        semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_EMBEDDINGS)
        
        async def process_paper(paper: ArxivPaper) -> tuple[ArxivPaper, Optional[List[float]]]:
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
        papers_with_embeddings: List[tuple[ArxivPaper, List[float]]]
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
    
    def _filter_new_papers(self, papers: List[ArxivPaper]) -> List[ArxivPaper]:
        """
        Filter out papers that already exist in the database with vectors.
        
        Args:
            papers: List of ArxivPaper objects
            
        Returns:
            List of papers that need to be processed (don't exist or don't have vectors)
        """
        if not papers:
            return []
        
        # Check which papers already exist with vectors
        new_papers = []
        for paper in papers:
            if not self.arxiv_database.exists(paper) or not self.arxiv_database.has_vector(paper):
                new_papers.append(paper)
        
        self.logger.info(f"Filtered {len(papers)} papers to {len(new_papers)} new papers")
        return new_papers
    
    async def _process_day(self, date: datetime) -> Dict[str, Any]:
        """
        Process a single day of papers.
        
        Args:
            date: Date to process
            
        Returns:
            Dictionary with processing results
        """
        day_start, day_end = self._get_day_range(date)
        day_str = self._format_date_for_query(date)
        
        self.logger.info(f"Processing day: {day_str}")
        
        # Fetch papers for the day
        papers = await self._fetch_papers_for_day(day_start, day_end)
        
        if not papers:
            self.logger.info(f"No papers found for day {day_str}")
            return {
                "date": day_str,
                "total_papers": 0,
                "new_papers": 0,
                "successful_embeddings": 0,
                "failed_embeddings": 0,
                "saved_papers": 0
            }
        
        # Filter out papers that already exist with vectors
        new_papers = self._filter_new_papers(papers)
        
        if not new_papers:
            self.logger.info(f"All papers for day {day_str} already exist with vectors")
            return {
                "date": day_str,
                "total_papers": len(papers),
                "new_papers": 0,
                "successful_embeddings": 0,
                "failed_embeddings": 0,
                "saved_papers": 0
            }
        
        # Generate embeddings concurrently
        successful_papers, failed_papers = await self._generate_embeddings_concurrent(new_papers)
        
        # Save successful papers to database
        saved_paper_ids = await self._save_papers_to_database(successful_papers)
        
        result = {
            "date": day_str,
            "total_papers": len(papers),
            "new_papers": len(new_papers),
            "successful_embeddings": len(successful_papers),
            "failed_embeddings": len(failed_papers),
            "saved_papers": len(saved_paper_ids)
        }
        
        self.logger.info(f"Day {day_str} processed: {result}")
        return result
    
    async def _sync_current_week(self) -> Dict[str, Any]:
        """
        Sync papers for the current week (Monday to Sunday).
        
        Returns:
            Dictionary with sync results
        """
        self.logger.info("Starting current week sync")
        
        week_dates = self._get_current_week_dates()
        results = []
        
        for date in week_dates:
            try:
                result = await self._process_day(date)
                results.append(result)
                
                # Small delay between days
                await asyncio.sleep(self.config.DELAY_BETWEEN_DAYS)
                
            except Exception as e:
                self.logger.error(f"Error processing day {self._format_date_for_query(date)}: {e}")
                results.append({
                    "date": self._format_date_for_query(date),
                    "error": str(e),
                    "total_papers": 0,
                    "new_papers": 0,
                    "successful_embeddings": 0,
                    "failed_embeddings": 0,
                    "saved_papers": 0
                })
        
        # Calculate summary
        total_papers = sum(r.get("total_papers", 0) for r in results)
        total_new_papers = sum(r.get("new_papers", 0) for r in results)
        total_successful = sum(r.get("successful_embeddings", 0) for r in results)
        total_failed = sum(r.get("failed_embeddings", 0) for r in results)
        total_saved = sum(r.get("saved_papers", 0) for r in results)
        
        summary = {
            "sync_type": "current_week",
            "dates_processed": len(results),
            "total_papers": total_papers,
            "new_papers": total_new_papers,
            "successful_embeddings": total_successful,
            "failed_embeddings": total_failed,
            "saved_papers": total_saved,
            "day_results": results
        }
        
        self.logger.info(f"Week sync completed: {summary}")
        return summary
    
    async def _sync_with_retry(self) -> bool:
        """
        Perform sync with retry logic.
        
        Returns:
            True if sync was successful, False otherwise
        """
        for attempt in range(1, self.config.MAX_RETRY_ATTEMPTS + 1):
            try:
                self.logger.info(f"Sync attempt {attempt}/{self.config.MAX_RETRY_ATTEMPTS}")
                
                result = await self._sync_current_week()
                
                # Update stats
                self.sync_stats["total_syncs"] += 1
                self.sync_stats["successful_syncs"] += 1
                self.sync_stats["total_papers_processed"] += result["total_papers"]
                self.sync_stats["total_papers_added"] += result["saved_papers"]
                
                self.last_sync_date = datetime.now()
                self.logger.info(f"Sync successful on attempt {attempt}")
                return True
                
            except Exception as e:
                self.logger.error(f"Sync attempt {attempt} failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                
                if attempt < self.config.MAX_RETRY_ATTEMPTS:
                    self.logger.info(f"Retrying in {self.config.RETRY_DELAY_SECONDS} seconds...")
                    await asyncio.sleep(self.config.RETRY_DELAY_SECONDS)
                else:
                    self.logger.error("All sync attempts failed")
                    self.sync_stats["total_syncs"] += 1
                    self.sync_stats["failed_syncs"] += 1
                    return False
        
        return False
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get current service statistics."""
        return {
            "service_name": self.config.SERVICE_NAME,
            "is_running": self.is_running,
            "last_sync_date": self.last_sync_date.isoformat() if self.last_sync_date else None,
            "database_stats": self.arxiv_database.get_stats(),
            "sync_stats": self.sync_stats.copy()
        }
    
    async def start(self):
        """Start the sync service."""
        if not self.config.validate():
            self.logger.error("Configuration validation failed. Service cannot start.")
            return
        
        self.config.print_config()
        self.is_running = True
        self.logger.info(f"{self.config.SERVICE_NAME} started")
        
        try:
            while self.is_running:
                self.logger.info("Starting sync cycle...")
                
                success = await self._sync_with_retry()
                
                if success:
                    self.logger.info("Sync cycle completed successfully")
                else:
                    self.logger.error("Sync cycle failed")
                
                # Wait for next sync cycle
                self.logger.info(f"Waiting {self.config.SYNC_INTERVAL_MINUTES} minutes for next sync...")
                await asyncio.sleep(self.config.SYNC_INTERVAL_MINUTES * 60)
                
        except KeyboardInterrupt:
            self.logger.info("Service interrupted by user")
        except Exception as e:
            self.logger.error(f"Service error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the sync service."""
        self.is_running = False
        await self.close()
        self.logger.info(f"{self.config.SERVICE_NAME} stopped")
    
    async def run_once(self) -> bool:
        """Run sync once (for testing or manual execution)."""
        if not self.config.validate():
            self.logger.error("Configuration validation failed.")
            return False
        
        self.config.print_config()
        self.logger.info("Running single sync...")
        
        success = await self._sync_with_retry()
        
        if success:
            self.logger.info("Single sync completed successfully")
        else:
            self.logger.error("Single sync failed")
        
        await self.close()
        return success


async def main():
    """Main execution function."""
    service = ArxivSyncService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
