"""
Arxiv Sync Service

A robust service for synchronizing arXiv papers to the database.
Provides daily sync functionality with weekly rollback and embedding generation.
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

import asyncio
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
from agents.ResearchAgent.arxiv_database_health_monitor import SafeArxivDatabaseOperations, ArxivRepairConfig
from agents.ResearchAgent.service.arxiv_sync_service.config import ArxivSyncConfig


class ArxivSyncService:
    """
    Arxiv synchronization service that:
    1. Runs continuously and checks for new papers every 15 minutes
    2. Syncs papers from today going back 7 days (rolling 7-day window)
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
        
        # Initialize safe ArxivDatabase operations with health monitoring
        self.safe_db = SafeArxivDatabaseOperations(
            self.arxiv_database, 
            enable_monitoring=True
        )
        
        # Service state
        self.is_running = False
        self.last_sync_date = None
        self.last_health_report_date = None
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
        self.safe_db.close()
        self.logger.info(f"{self.config.SERVICE_NAME} connections closed")
    
    def get_health_status(self):
        """Get current database health status."""
        return self.safe_db.get_health_status()
    
    def _get_rolling_week_dates(self) -> List[datetime]:
        """
        Get dates for the rolling 7-day window (today going back 7 days).
        
        Returns:
            List of datetime objects from today going back 7 days
        """
        today = datetime.now()
        
        # Generate dates going back 7 days from today
        rolling_dates = []
        for i in range(7):
            date = today - timedelta(days=i)
            rolling_dates.append(date)
        
        return rolling_dates
    
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
        Save papers with embeddings to the database in smaller batches.
        
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
            
            # Save to database with detailed logging and error handling
            self.logger.info(f"Starting database save operation for {len(papers)} papers...")
            
            # Validate embeddings before saving
            valid_embeddings = []
            invalid_count = 0
            for i, embedding in enumerate(embeddings):
                if embedding is None or len(embedding) == 0:
                    invalid_count += 1
                    self.logger.warning(f"Invalid embedding for paper {i}: {embedding}")
                else:
                    valid_embeddings.append(embedding)
            
            if invalid_count > 0:
                self.logger.warning(f"Found {invalid_count} invalid embeddings out of {len(embeddings)}")
            
            # Save to database in small batches to avoid blocking
            self.logger.info(f"Starting batch save operation for {len(papers)} papers...")
            
            # Process in small batches to avoid database blocking
            batch_size = 10  # Very small batch size to avoid issues
            saved_paper_ids = []
            
            for i in range(0, len(papers), batch_size):
                batch_papers = papers[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size] if embeddings else None
                
                batch_num = i // batch_size + 1
                total_batches = (len(papers) + batch_size - 1) // batch_size
                
                self.logger.info(f"Saving batch {batch_num}/{total_batches} ({len(batch_papers)} papers)...")
                
                try:
                    # Validate batch data before saving
                    valid_batch_papers = []
                    valid_batch_embeddings = []
                    
                    for j, paper in enumerate(batch_papers):
                        try:
                            # Check if paper data is valid
                            if not paper.id or not paper.title:
                                self.logger.warning(f"Skipping invalid paper in batch {batch_num}: missing id or title")
                                continue
                                
                            # Check if embedding is valid
                            embedding = batch_embeddings[j] if batch_embeddings and j < len(batch_embeddings) else None
                            if embedding is not None and (not embedding or len(embedding) == 0):
                                self.logger.warning(f"Skipping paper {paper.id} in batch {batch_num}: invalid embedding")
                                continue
                                
                            valid_batch_papers.append(paper)
                            if batch_embeddings:
                                valid_batch_embeddings.append(embedding)
                                
                        except Exception as e:
                            self.logger.warning(f"Skipping paper in batch {batch_num} due to validation error: {e}")
                            continue
                    
                    if not valid_batch_papers:
                        self.logger.warning(f"No valid papers in batch {batch_num}, skipping...")
                        continue
                    
                    # Save valid batch
                    batch_record_ids = self.safe_db.add_papers_safe(valid_batch_papers, valid_batch_embeddings if valid_batch_embeddings else None)
                    saved_paper_ids.extend(batch_record_ids)
                    self.logger.info(f"Successfully saved batch {batch_num}: {len(batch_record_ids)} papers")
                    
                except Exception as e:
                    self.logger.error(f"Failed to save batch {batch_num}: {e}")
                    self.logger.error(f"Exception type: {type(e).__name__}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Try to save batch without embeddings as fallback
                    try:
                        self.logger.info(f"Attempting to save batch {batch_num} without embeddings...")
                        batch_record_ids = self.safe_db.add_papers_safe(valid_batch_papers, None)
                        saved_paper_ids.extend(batch_record_ids)
                        self.logger.info(f"Successfully saved batch {batch_num} without embeddings: {len(batch_record_ids)} papers")
                    except Exception as e2:
                        self.logger.error(f"Failed to save batch {batch_num} without embeddings: {e2}")
                        continue
            
            self.logger.info(f"Batch save completed: {len(saved_paper_ids)} papers saved successfully")
            
            return saved_paper_ids
            
        except Exception as e:
            self.logger.error(f"Failed to save papers to database: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _filter_new_papers(self, papers: List[ArxivPaper]) -> List[ArxivPaper]:
        """
        Filter out papers that already exist in the database with vectors.
        """
        if not papers:
            return []
        
        self.logger.info(f"Checking existence for {len(papers)} papers in batch...")
        
        # 🚀 步骤1: 使用两次批量查询，代替数千次单独查询
        existence_map = self.arxiv_database.exists_batch(papers)
        vector_map = self.arxiv_database.has_vector_batch(papers)
        
        # 步骤2: 在内存中进行高效过滤
        new_papers = []
        for paper in papers:
            paper_id = paper.full_id
            
            # 直接从字典中获取结果，无数据库交互
            if not existence_map.get(paper_id, False) or not vector_map.get(paper_id, False):
                new_papers.append(paper)
                
        self.logger.info(f"Filtered {len(papers)} papers down to {len(new_papers)} new papers to process.")
        return new_papers
    
    async def _check_database_health_on_error(self, operation: str, day_str: str) -> bool:
        """
        Check database health only when errors occur.
        
        Args:
            operation: Description of the operation that failed
            day_str: Day identifier for logging
            
        Returns:
            True if database is healthy, False if critical issues found
        """
        if not self.safe_db.health_monitor:
            return True
            
        self.logger.warning(f"Error in {operation} for day {day_str}, performing health check...")
        health_result = await self.safe_db.health_monitor.perform_health_check()
        
        if health_result.overall_health == "critical":
            self.logger.error(f"ArxivDatabase health is critical after {operation} error")
            # Save critical health report
            report_path = self.safe_db.health_monitor.save_health_report()
            self.logger.info(f"Critical health report saved to: {report_path}")
            return False
        elif health_result.overall_health == "degraded":
            self.logger.warning(f"ArxivDatabase health is degraded after {operation} error")
            # Save degraded health report
            report_path = self.safe_db.health_monitor.save_health_report()
            self.logger.info(f"Degraded health report saved to: {report_path}")
        
        return True

    async def _process_day(self, date: datetime) -> Dict[str, Any]:
        """
        Process a single day of papers with enhanced error checking.
        
        Args:
            date: Date to process
            
        Returns:
            Dictionary with processing results
        """
        day_start, day_end = self._get_day_range(date)
        day_str = self._format_date_for_query(date)
        
        self.logger.info(f"Processing day: {day_str}")
        
        try:
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
            
            # Checkpoint 1: Verify papers were fetched successfully
            if len(papers) == 0:
                self.logger.warning(f"Unexpected: 0 papers fetched for day {day_str}")
                return {
                    "date": day_str,
                    "total_papers": 0,
                    "new_papers": 0,
                    "successful_embeddings": 0,
                    "failed_embeddings": 0,
                    "saved_papers": 0,
                    "warning": "No papers fetched"
                }
            
            # Filter out papers that already exist with vectors
            try:
                new_papers = self._filter_new_papers(papers)
            except Exception as e:
                self.logger.error(f"Error filtering papers for day {day_str}: {e}")
                if not await self._check_database_health_on_error("paper filtering", day_str):
                    return {
                        "date": day_str,
                        "total_papers": len(papers),
                        "new_papers": 0,
                        "successful_embeddings": 0,
                        "failed_embeddings": 0,
                        "saved_papers": 0,
                        "error": "Database health critical after filtering error"
                    }
                # If health check passed, continue with empty list
                new_papers = []
            
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
            
            # Checkpoint 2: Verify new papers
            if len(new_papers) == 0:
                self.logger.warning(f"Unexpected: 0 new papers for day {day_str}")
                return {
                    "date": day_str,
                    "total_papers": len(papers),
                    "new_papers": 0,
                    "successful_embeddings": 0,
                    "failed_embeddings": 0,
                    "saved_papers": 0,
                    "warning": "No new papers"
                }
            
            # Generate embeddings concurrently
            try:
                successful_papers, failed_papers = await self._generate_embeddings_concurrent(new_papers)
            except Exception as e:
                self.logger.error(f"Error generating embeddings for day {day_str}: {e}")
                if not await self._check_database_health_on_error("embedding generation", day_str):
                    return {
                        "date": day_str,
                        "total_papers": len(papers),
                        "new_papers": len(new_papers),
                        "successful_embeddings": 0,
                        "failed_embeddings": len(new_papers),
                        "saved_papers": 0,
                        "error": "Database health critical after embedding generation error"
                    }
                # If health check passed, continue with empty results
                successful_papers = []
                failed_papers = new_papers
            
            # Checkpoint 3: Verify embedding generation results
            if len(successful_papers) == 0 and len(new_papers) > 0:
                self.logger.warning(f"No embeddings were generated for day {day_str} despite {len(new_papers)} new papers")
                if not await self._check_database_health_on_error("embedding verification", day_str):
                    return {
                        "date": day_str,
                        "total_papers": len(papers),
                        "new_papers": len(new_papers),
                        "successful_embeddings": 0,
                        "failed_embeddings": len(new_papers),
                        "saved_papers": 0,
                        "error": "Database health critical after embedding verification"
                    }
            
            # Save successful papers to database
            try:
                saved_paper_ids = await self._save_papers_to_database(successful_papers)
            except Exception as e:
                self.logger.error(f"Error saving papers for day {day_str}: {e}")
                if not await self._check_database_health_on_error("paper saving", day_str):
                    return {
                        "date": day_str,
                        "total_papers": len(papers),
                        "new_papers": len(new_papers),
                        "successful_embeddings": len(successful_papers),
                        "failed_embeddings": len(failed_papers),
                        "saved_papers": 0,
                        "error": "Database health critical after saving error"
                    }
                # If health check passed, continue with empty list
                saved_paper_ids = []
            
            # Checkpoint 4: Verify save results
            if len(saved_paper_ids) == 0 and len(successful_papers) > 0:
                self.logger.warning(f"No papers were saved for day {day_str} despite {len(successful_papers)} successful embeddings")
                if not await self._check_database_health_on_error("save verification", day_str):
                    return {
                        "date": day_str,
                        "total_papers": len(papers),
                        "new_papers": len(new_papers),
                        "successful_embeddings": len(successful_papers),
                        "failed_embeddings": len(failed_papers),
                        "saved_papers": 0,
                        "error": "Database health critical after save verification"
                    }
            
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
            
        except Exception as e:
            self.logger.error(f"Unexpected error processing day {day_str}: {e}")
            if not await self._check_database_health_on_error("day processing", day_str):
                return {
                    "date": day_str,
                    "total_papers": 0,
                    "new_papers": 0,
                    "successful_embeddings": 0,
                    "failed_embeddings": 0,
                    "saved_papers": 0,
                    "error": f"Critical database health issue: {str(e)}"
                }
            return {
                "date": day_str,
                "total_papers": 0,
                "new_papers": 0,
                "successful_embeddings": 0,
                "failed_embeddings": 0,
                "saved_papers": 0,
                "error": str(e)
            }
    
    async def _sync_rolling_week(self) -> Dict[str, Any]:
        """
        Sync papers for the rolling 7-day window (today going back 7 days).
        
        Returns:
            Dictionary with sync results
        """
        self.logger.info("Starting rolling 7-day sync")
        
        rolling_dates = self._get_rolling_week_dates()
        results = []
        
        for date in rolling_dates:
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
            "sync_type": "rolling_7_days",
            "dates_processed": len(results),
            "total_papers": total_papers,
            "new_papers": total_new_papers,
            "successful_embeddings": total_successful,
            "failed_embeddings": total_failed,
            "saved_papers": total_saved,
            "day_results": results
        }
        
        self.logger.info(f"Rolling 7-day sync completed: {summary}")
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
                
                result = await self._sync_rolling_week()
                
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
                
                try:
                    success = await self._sync_with_retry()
                    
                    if success:
                        self.logger.info("Sync cycle completed successfully")
                    else:
                        self.logger.error("Sync cycle failed")
                        # Only perform health check after sync failure
                        if self.safe_db.health_monitor:
                            self.logger.warning("Sync cycle failed, performing health check...")
                            health_result = await self.safe_db.health_monitor.perform_health_check()
                            if health_result.overall_health == "critical":
                                self.logger.error("ArxivDatabase health is critical after sync failure")
                                self.logger.info("Attempting automatic repair...")
                                await self.safe_db.health_monitor.repair_corrupted_shards(health_result)
                                # Save health report after critical issue
                                report_path = self.safe_db.health_monitor.save_health_report()
                                self.logger.info(f"Critical health report saved to: {report_path}")
                                # Wait shorter time before retry
                                await asyncio.sleep(300)  # 5 minutes
                                continue
                            elif health_result.overall_health == "degraded":
                                self.logger.warning("ArxivDatabase health is degraded after sync failure")
                                # Save health report for degraded status
                                report_path = self.safe_db.health_monitor.save_health_report()
                                self.logger.info(f"Degraded health report saved to: {report_path}")
                    
                    # Save daily health report (once per day)
                    current_date = datetime.now().date()
                    if (self.last_health_report_date is None or 
                        current_date > self.last_health_report_date):
                        if self.safe_db.health_monitor:
                            report_path = self.safe_db.health_monitor.save_health_report()
                            self.logger.info(f"Daily health report saved to: {report_path}")
                            self.last_health_report_date = current_date
                    
                    # Wait for next sync cycle
                    self.logger.info(f"Waiting {self.config.SYNC_INTERVAL_MINUTES} minutes for next sync...")
                    await asyncio.sleep(self.config.SYNC_INTERVAL_MINUTES * 60)
                    
                except Exception as e:
                    self.logger.error(f"Error in sync cycle: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    # Continue to next cycle instead of stopping the service
                    self.logger.info("Continuing to next sync cycle...")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
                
        except KeyboardInterrupt:
            self.logger.info("Service interrupted by user")
        except Exception as e:
            self.logger.error(f"Service error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.logger.info("Service is stopping...")
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
