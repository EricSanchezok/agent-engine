"""
Arxiv Metadata Fetcher

This script fetches arXiv paper metadata by:
1. Rolling back day by day from today
2. Fetching all papers for each day using ArxivFetcher
3. Saving paper metadata to database (without embeddings)
4. Logging failed paper IDs to JSON files

This is the first step of the two-step preloading process.
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
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from agent_engine.agent_logger import AgentLogger

from core.arxiv_fetcher.arxiv_fetcher import ArxivFetcher
from core.arxiv_fetcher.arxiv_paper import ArxivPaper
from agents.ResearchAgent.arxiv_database import ArxivDatabase
from agents.ResearchAgent.arxiv_database_health_monitor import SafeArxivDatabaseOperations, ArxivRepairConfig


class ArxivMetadataFetcher:
    """Arxiv metadata fetcher with daily rolling and metadata-only storage."""
    
    def __init__(
        self,
        database_name: str = "arxiv_papers",
        database_dir: Optional[str] = None,
        enable_health_monitoring: bool = True
    ):
        """
        Initialize the Arxiv metadata fetcher.
        
        Args:
            database_name: Name for the ArxivDatabase
            database_dir: Directory for database storage
            enable_health_monitoring: Whether to enable database health monitoring
        """
        self.logger = AgentLogger(self.__class__.__name__)
        
        # Initialize components
        self.arxiv_fetcher = ArxivFetcher()
        self.arxiv_database = ArxivDatabase(
            name=database_name,
            persist_dir=database_dir
        )
        
        # Initialize safe ArxivDatabase operations with health monitoring
        self.safe_db = SafeArxivDatabaseOperations(
            self.arxiv_database, 
            enable_monitoring=enable_health_monitoring
        )
        
        # Create failed IDs directory
        self.failed_ids_dir = Path(__file__).parent.parent / "database" / "fetch_failed_ids"
        self.failed_ids_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("ArxivMetadataFetcher initialized with health monitoring")
        self.logger.info(f"Failed IDs will be saved to: {self.failed_ids_dir}")
    
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
    
    def _filter_new_papers(self, papers: List[ArxivPaper]) -> List[ArxivPaper]:
        """
        Filter out papers that already exist in the database.
        
        Args:
            papers: List of ArxivPaper objects
            
        Returns:
            List of papers that need to be saved (don't exist in database)
        """
        if not papers:
            return []
        
        # Use batch method for efficient checking
        self.logger.info(f"Checking existence for {len(papers)} papers in batch")
        existence_map = self.arxiv_database.exists_batch(papers)
        
        # Filter out papers that already exist
        new_papers = []
        for paper in papers:
            paper_id = paper.full_id
            exists = existence_map.get(paper_id, False)
            
            if not exists:
                new_papers.append(paper)
        
        self.logger.info(f"Filtered {len(papers)} papers to {len(new_papers)} new papers")
        return new_papers
    
    async def _save_papers_metadata(self, papers: List[ArxivPaper]) -> List[str]:
        """
        Save paper metadata to the database (without embeddings).
        
        Args:
            papers: List of ArxivPaper objects
            
        Returns:
            List of successfully saved paper IDs
        """
        if not papers:
            return []
        
        self.logger.info(f"Saving {len(papers)} paper metadata to database")
        
        try:
            # Save to database without embeddings (None for embeddings) using safe operations
            embeddings = [None] * len(papers)
            record_ids = self.safe_db.add_papers_safe(papers, embeddings)
            
            self.logger.info(f"Successfully saved {len(record_ids)} paper metadata to database")
            return [paper.full_id for paper in papers]
            
        except Exception as e:
            self.logger.error(f"Failed to save paper metadata to database: {e}")
            return []
    
    def _save_failed_ids(self, failed_papers: List[ArxivPaper], day_name: str):
        """
        Save failed paper IDs to JSON file.
        
        Args:
            failed_papers: List of failed ArxivPaper objects
            day_name: Day identifier for filename
        """
        if not failed_papers:
            return
        
        failed_ids = [paper.full_id for paper in failed_papers]
        
        # Create filename
        filename = f"{day_name}.json"
        filepath = self.failed_ids_dir / filename
        
        # Save to JSON
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "day_range": day_name,
                    "failed_count": len(failed_ids),
                    "failed_ids": failed_ids,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(failed_ids)} failed IDs to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save failed IDs to {filepath}: {e}")
    
    async def _check_database_health_on_error(self, operation: str, day_name: str) -> bool:
        """
        Check database health only when errors occur.
        
        Args:
            operation: Description of the operation that failed
            day_name: Day identifier for logging
            
        Returns:
            True if database is healthy, False if critical issues found
        """
        if not self.safe_db.health_monitor:
            return True
            
        self.logger.warning(f"Error in {operation} for day {day_name}, performing health check...")
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

    async def _process_day(self, start_date: datetime, end_date: datetime) -> dict:
        """
        Process a single day of papers with enhanced error checking.
        
        Args:
            start_date: Day start date
            end_date: Day end date (next day)
            
        Returns:
            Dictionary with processing results
        """
        day_name = self._format_date_range_name(start_date, end_date)
        
        self.logger.info(f"Processing day: {day_name}")
        
        try:
            # Fetch papers for the day
            papers = await self._fetch_papers_for_day(start_date, end_date)
            
            if not papers:
                self.logger.info(f"No papers found for day {day_name}")
                return {
                    "day": day_name,
                    "total_papers": 0,
                    "filtered_papers": 0,
                    "saved_papers": 0,
                    "failed_papers": 0
                }
            
            # Checkpoint 1: Verify papers were fetched successfully
            if len(papers) == 0:
                self.logger.warning(f"Unexpected: 0 papers fetched for day {day_name}")
                return {
                    "day": day_name,
                    "total_papers": 0,
                    "filtered_papers": 0,
                    "saved_papers": 0,
                    "failed_papers": 0,
                    "warning": "No papers fetched"
                }
            
            # Filter out papers that already exist
            try:
                papers_to_save = self._filter_new_papers(papers)
                filtered_count = len(papers) - len(papers_to_save)
            except Exception as e:
                self.logger.error(f"Error filtering papers for day {day_name}: {e}")
                if not await self._check_database_health_on_error("paper filtering", day_name):
                    return {
                        "day": day_name,
                        "total_papers": len(papers),
                        "filtered_papers": 0,
                        "saved_papers": 0,
                        "failed_papers": len(papers),
                        "error": "Database health critical after filtering error"
                    }
                # If health check passed, continue with empty list
                papers_to_save = []
                filtered_count = len(papers)
            
            if not papers_to_save:
                self.logger.info(f"All papers for day {day_name} already exist in database")
                return {
                    "day": day_name,
                    "total_papers": len(papers),
                    "filtered_papers": filtered_count,
                    "saved_papers": 0,
                    "failed_papers": 0
                }
            
            # Checkpoint 2: Verify papers to save
            if len(papers_to_save) == 0:
                self.logger.warning(f"Unexpected: 0 papers to save for day {day_name}")
                return {
                    "day": day_name,
                    "total_papers": len(papers),
                    "filtered_papers": filtered_count,
                    "saved_papers": 0,
                    "failed_papers": 0,
                    "warning": "No papers to save"
                }
            
            # Save paper metadata to database
            try:
                saved_paper_ids = await self._save_papers_metadata(papers_to_save)
            except Exception as e:
                self.logger.error(f"Error saving papers for day {day_name}: {e}")
                if not await self._check_database_health_on_error("paper saving", day_name):
                    return {
                        "day": day_name,
                        "total_papers": len(papers),
                        "filtered_papers": filtered_count,
                        "saved_papers": 0,
                        "failed_papers": len(papers_to_save),
                        "error": "Database health critical after saving error"
                    }
                # If health check passed, continue with empty list
                saved_paper_ids = []
            
            # Checkpoint 3: Verify save results
            if len(saved_paper_ids) == 0 and len(papers_to_save) > 0:
                self.logger.warning(f"No papers were saved for day {day_name} despite {len(papers_to_save)} papers to save")
                if not await self._check_database_health_on_error("paper save verification", day_name):
                    return {
                        "day": day_name,
                        "total_papers": len(papers),
                        "filtered_papers": filtered_count,
                        "saved_papers": 0,
                        "failed_papers": len(papers_to_save),
                        "error": "Database health critical after save verification"
                    }
            
            # Calculate failed papers (papers that couldn't be saved)
            failed_papers = []
            if len(saved_paper_ids) < len(papers_to_save):
                saved_ids_set = set(saved_paper_ids)
                failed_papers = [paper for paper in papers_to_save if paper.full_id not in saved_ids_set]
                self.logger.warning(f"{len(failed_papers)} papers failed to save for day {day_name}")
            
            # Save failed IDs to file
            if failed_papers:
                try:
                    self._save_failed_ids(failed_papers, day_name)
                except Exception as e:
                    self.logger.error(f"Error saving failed IDs for day {day_name}: {e}")
            
            result = {
                "day": day_name,
                "total_papers": len(papers),
                "filtered_papers": filtered_count,
                "saved_papers": len(saved_paper_ids),
                "failed_papers": len(failed_papers)
            }
            
            self.logger.info(f"Day {day_name} processed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Unexpected error processing day {day_name}: {e}")
            if not await self._check_database_health_on_error("day processing", day_name):
                return {
                    "day": day_name,
                    "total_papers": 0,
                    "filtered_papers": 0,
                    "saved_papers": 0,
                    "failed_papers": 0,
                    "error": f"Critical database health issue: {str(e)}"
                }
            return {
                "day": day_name,
                "total_papers": 0,
                "filtered_papers": 0,
                "saved_papers": 0,
                "failed_papers": 0,
                "error": str(e)
            }
    
    async def fetch_days(self, num_days: int = 10) -> List[dict]:
        """
        Fetch papers for the specified number of days, rolling back from today.
        
        Args:
            num_days: Number of days to process
            
        Returns:
            List of processing results for each day
        """
        from preload_config import PreloadConfig
        
        self.logger.info(f"Starting metadata fetch for {num_days} days")
        
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
        total_saved = sum(r["saved_papers"] for r in results)
        total_failed = sum(r["failed_papers"] for r in results)
        
        self.logger.info(f"Metadata fetch completed:")
        self.logger.info(f"  Total papers fetched: {total_papers}")
        self.logger.info(f"  Papers already exist: {total_filtered}")
        self.logger.info(f"  New papers saved: {total_saved}")
        self.logger.info(f"  Failed to save: {total_failed}")
        
        return results
    
    def get_health_status(self):
        """Get current database health status."""
        return self.safe_db.get_health_status()
    
    def close(self):
        """Close health monitoring and cleanup resources."""
        self.safe_db.close()
        self.logger.info("ArxivMetadataFetcher closed")


async def main():
    """Main execution function."""
    from preload_config import PreloadConfig
    
    # Validate configuration
    if not PreloadConfig.validate():
        return
    
    PreloadConfig.print_config()
    
    # Initialize fetcher
    fetcher = ArxivMetadataFetcher(
        database_name=PreloadConfig.DATABASE_NAME,
        database_dir=PreloadConfig.DATABASE_DIR
    )
    
    try:
        # Fetch papers for specified number of days
        results = await fetcher.fetch_days(num_days=PreloadConfig.DEFAULT_NUM_DAYS)
        
        # Print summary
        print("\n=== Metadata Fetch Summary ===")
        for result in results:
            print(f"Day {result['day']}: {result['total_papers']} papers fetched, "
                  f"{result['filtered_papers']} already exist, "
                  f"{result['saved_papers']} saved, "
                  f"{result['failed_papers']} failed")
        
        # Print health status
        health_status = fetcher.get_health_status()
        if health_status:
            print("\n=== ArxivDatabase Health Status ===")
            print(f"Monitoring Active: {health_status['monitoring_active']}")
            if health_status['last_check']:
                last_check = health_status['last_check']
                print(f"Overall Health: {last_check['overall_health']}")
                print(f"Healthy Shards: {last_check['healthy_shards']}/{last_check['total_shards']}")
                print(f"Total Papers: {last_check['total_papers']:,}")
                print(f"Papers with Vectors: {last_check['papers_with_vectors']:,}")
            
            # Save health report to file
            if fetcher.safe_db.health_monitor:
                report_path = fetcher.safe_db.health_monitor.save_health_report()
                print(f"Health report saved to: {report_path}")
        
    except Exception as e:
        print(f"Metadata fetch failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always close the fetcher
        fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
