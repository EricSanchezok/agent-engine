from pathlib import Path
from agent_engine.agent_logger import set_agent_log_directory

current_file_dir = Path(__file__).parent
log_dir = current_file_dir / 'logs'
set_agent_log_directory(str(log_dir))

"""
Daily arXiv Scheduler

This module provides automatic daily scheduling for the arXiv paper processing pipeline.
It runs the complete pipeline daily and handles retries, failures, and database update delays.
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

from agent_engine.agent_logger import set_agent_log_directory

current_file_dir = Path(__file__).parent
log_dir = current_file_dir / 'logs'
set_agent_log_directory(str(log_dir))


import asyncio
import time
from datetime import date, datetime, timezone, timedelta
from typing import Dict, Any, Optional

from agent_engine.agent_logger import AgentLogger

from agents.ResearchAgent.service.daily_arxiv.paper_filter import DailyArxivPaperFilter
from agents.ResearchAgent.service.daily_arxiv.swiss_tournament import SwissTournamentRanker
from agents.ResearchAgent.service.daily_arxiv.paper_condenser import PaperCondenser
from agents.ResearchAgent.service.daily_arxiv.result_manager import ResultManager
from agents.ResearchAgent.service.daily_arxiv.config import DailyArxivConfig


class DailyArxivScheduler:
    """
    Daily arXiv scheduler that automatically runs the complete pipeline.
    
    This class handles:
    1. Daily automatic execution
    2. Database update delays (wait 15 minutes and retry)
    3. Pipeline failures and retries
    4. Paper supplementation for incomplete downloads
    """
    
    def __init__(self):
        """Initialize the daily scheduler."""
        self.logger = AgentLogger(self.__class__.__name__)
        
        # Initialize components
        self.paper_filter = DailyArxivPaperFilter()
        self.swiss_ranker = SwissTournamentRanker()
        self.paper_condenser = PaperCondenser()
        self.result_manager = ResultManager()
        
        # Configuration
        self.retry_delay_minutes = 15  # Wait 15 minutes before retry
        self.max_retry_attempts = 48   # Maximum retry attempts per day
        self.check_interval_hours = 1  # Check every hour for new data
        
        self.logger.info("DailyArxivScheduler initialized")
    
    async def run_daily_pipeline(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Run the complete daily pipeline for a specific date.
        
        Args:
            target_date: Date to process (defaults to today)
            
        Returns:
            Dictionary with pipeline results
        """
        if target_date is None:
            target_date = date.today()
        
        self.logger.info(f"Starting daily pipeline for date: {target_date}")
        
        # Check if already processed
        if self.result_manager.is_date_processed(target_date):
            self.logger.info(f"Date {target_date} has already been processed")
            return {
                "success": True,
                "message": f"Date {target_date} already processed",
                "target_date": target_date.isoformat()
            }
        
        # Mark as in progress
        if not self.result_manager.mark_date_in_progress(target_date):
            return {
                "success": False,
                "error": f"Could not mark date {target_date} as in progress",
                "target_date": target_date.isoformat()
            }
        
        attempt = 0
        while attempt < self.max_retry_attempts:
            attempt += 1
            self.logger.info(f"Pipeline attempt {attempt}/{self.max_retry_attempts} for {target_date}")
            
            try:
                # Run the complete pipeline
                result = await self._run_complete_pipeline(target_date)
                
                if result["success"]:
                    self.logger.info(f"Pipeline completed successfully for {target_date}")
                    return result
                else:
                    self.logger.warning(f"Pipeline failed for {target_date}: {result.get('error', 'Unknown error')}")
                    
                    # Check if it's a database update issue
                    if self._is_database_update_issue(result):
                        if attempt < self.max_retry_attempts:
                            self.logger.info(f"Database not updated, waiting {self.retry_delay_minutes} minutes before retry")
                            await asyncio.sleep(self.retry_delay_minutes * 60)
                            continue
                        else:
                            self.logger.error(f"Max retry attempts reached for {target_date}")
                            self.result_manager.mark_date_failed(target_date, "Max retry attempts reached")
                            return result
                    else:
                        # Other types of failures - retry immediately
                        if attempt < self.max_retry_attempts:
                            self.logger.info("Retrying immediately due to non-database issue")
                            await asyncio.sleep(30)  # Short delay before retry
                            continue
                        else:
                            self.logger.error(f"Max retry attempts reached for {target_date}")
                            self.result_manager.mark_date_failed(target_date, result.get('error', 'Unknown error'))
                            return result
            
            except Exception as e:
                self.logger.error(f"Unexpected error in pipeline attempt {attempt}: {e}", exc_info=True)
                
                if attempt < self.max_retry_attempts:
                    self.logger.info(f"Retrying after error, attempt {attempt + 1}")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
                    continue
                else:
                    self.logger.error(f"Max retry attempts reached after error for {target_date}")
                    self.result_manager.mark_date_failed(target_date, str(e))
                    return {
                        "success": False,
                        "error": str(e),
                        "target_date": target_date.isoformat()
                    }
        
        # Should not reach here, but just in case
        error_msg = f"Pipeline failed after {self.max_retry_attempts} attempts"
        self.result_manager.mark_date_failed(target_date, error_msg)
        return {
            "success": False,
            "error": error_msg,
            "target_date": target_date.isoformat()
        }
    
    async def _run_complete_pipeline(self, target_date: date) -> Dict[str, Any]:
        """Run the complete pipeline for a specific date."""
        try:
            # Step 1: Filter and download papers (with supplementation)
            self.logger.info("Step 1: Filtering and downloading papers...")
            filter_result = await self.paper_filter.filter_and_download_papers(
                target_date, 
                ensure_complete_download=True
            )
            
            if not filter_result["success"]:
                return {
                    "success": False,
                    "step": "filter",
                    "error": filter_result.get("error", filter_result.get("reason", "Unknown error")),
                    "target_date": target_date.isoformat()
                }
            
            self.logger.info(f"Filter step completed: {filter_result['papers_downloaded']} papers downloaded")
            
            # Step 2: Swiss tournament ranking
            if not filter_result.get("successful_pdf_paths"):
                return {
                    "success": False,
                    "step": "rank",
                    "error": "No PDF files available for ranking",
                    "target_date": target_date.isoformat(),
                    "filter_result": filter_result
                }
            
            self.logger.info("Step 2: Running Swiss tournament ranking...")
            rank_result = await self.swiss_ranker.rank_papers_from_pdf_paths(
                filter_result["successful_pdf_paths"]
            )
            
            if not rank_result["success"]:
                return {
                    "success": False,
                    "step": "rank",
                    "error": rank_result.get("error", rank_result.get("reason", "Unknown error")),
                    "target_date": target_date.isoformat(),
                    "filter_result": filter_result
                }
            
            self.logger.info(f"Ranking step completed: {len(rank_result['top_papers'])} top papers selected")
            
            # Step 3: Condense papers
            self.logger.info("Step 3: Condensing papers...")
            condensed_reports = await self.paper_condenser.condense_papers(
                rank_result["top_papers"],
                filter_result["successful_pdf_paths"]
            )
            
            if not condensed_reports:
                return {
                    "success": False,
                    "step": "condense",
                    "error": "Failed to generate any condensed reports",
                    "target_date": target_date.isoformat(),
                    "filter_result": filter_result,
                    "rank_result": rank_result
                }
            
            self.logger.info(f"Condensation step completed: {len(condensed_reports)} reports generated")
            
            # Step 4: Save condensed reports to files
            self.logger.info("Step 4: Saving condensed reports...")
            result_dir = self._get_result_directory(target_date)
            saved_files = self.paper_condenser.save_reports_to_files(condensed_reports, str(result_dir))
            
            # Step 5: Save complete result
            self.logger.info("Step 5: Saving complete result...")
            result_file_path = self.result_manager.save_daily_result(
                target_date,
                filter_result,
                rank_result,
                saved_files,
                filter_result.get("paper_metadata", {})
            )
            
            # Mark as completed
            self.result_manager.mark_date_completed(target_date, result_file_path)
            
            # Combine results
            complete_result = {
                "success": True,
                "target_date": target_date.isoformat(),
                "filter_result": filter_result,
                "rank_result": rank_result,
                "condensed_reports": saved_files,
                "result_file_path": result_file_path,
                "summary": {
                    "total_papers_processed": filter_result["papers_processed"],
                    "papers_downloaded": filter_result["papers_downloaded"],
                    "total_comparisons": rank_result["total_comparisons"],
                    "top_papers_selected": len(rank_result["top_papers"]),
                    "condensed_reports_generated": len(saved_files)
                }
            }
            
            return complete_result
            
        except Exception as e:
            self.logger.error(f"Error in complete pipeline: {e}", exc_info=True)
            return {
                "success": False,
                "step": "unknown",
                "error": str(e),
                "target_date": target_date.isoformat()
            }
    
    def _is_database_update_issue(self, result: Dict[str, Any]) -> bool:
        """Check if the failure is due to database not being updated."""
        error = result.get("error", "").lower()
        reason = result.get("reason", "").lower()
        
        # Common indicators of database update issues
        database_indicators = [
            "no papers found",
            "no papers with valid vectors",
            "no papers selected",
            "database not updated",
            "no data available"
        ]
        
        return any(indicator in error or indicator in reason for indicator in database_indicators)
    
    def _get_result_directory(self, target_date: date) -> Path:
        """Get the result directory for a specific date."""
        year_dir = str(target_date.year)
        month_dir = f"{target_date.month:02d}"
        day_dir = f"{target_date.day:02d}"
        
        result_storage_dir = Path(DailyArxivConfig.get_result_storage_dir())
        return result_storage_dir / year_dir / month_dir / day_dir
    
    async def run_continuous_monitoring(self):
        """Run continuous monitoring for daily processing."""
        self.logger.info("Starting continuous monitoring for daily arXiv processing")
        
        while True:
            try:
                # Check if today's processing is needed
                today = date.today()
                
                if not self.result_manager.is_date_processed(today):
                    self.logger.info(f"Processing needed for today: {today}")
                    result = await self.run_daily_pipeline(today)
                    
                    if result["success"]:
                        self.logger.info(f"Successfully processed {today}")
                    else:
                        self.logger.error(f"Failed to process {today}: {result.get('error', 'Unknown error')}")
                else:
                    self.logger.info(f"Today ({today}) already processed")
                
                # Wait before next check
                self.logger.info(f"Waiting {self.check_interval_hours} hours before next check")
                await asyncio.sleep(self.check_interval_hours * 3600)
                
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt, stopping monitoring")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous monitoring: {e}", exc_info=True)
                self.logger.info("Waiting 1 hour before retrying monitoring")
                await asyncio.sleep(3600)
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        today = date.today()
        is_processed = self.result_manager.is_date_processed(today)
        status_summary = self.result_manager.get_processing_status_summary()
        
        return {
            "today": today.isoformat(),
            "is_today_processed": is_processed,
            "status_summary": status_summary
        }


async def main():
    """Main entry point for the daily scheduler."""
    print("=== Daily arXiv Scheduler ===")
    
    # Print configuration
    DailyArxivConfig.print_config()
    print()
    
    # Validate configuration
    if not DailyArxivConfig.validate():
        print("❌ Configuration validation failed")
        return
    
    print("✅ Configuration validation passed")
    print()
    
    # Initialize scheduler
    scheduler = DailyArxivScheduler()
    
    # Check current status
    status = scheduler.get_processing_status()
    print(f"Today: {status['today']}")
    print(f"Is Today Processed: {status['is_today_processed']}")
    print(f"Status Summary: {status['status_summary']}")
    print()
    
    # Run today's pipeline
    print("Running today's pipeline...")
    result = await scheduler.run_daily_pipeline(date(2025, 9, 16))
    
    print(f"Pipeline Result: {result['success']}")
    if result['success']:
        if 'summary' in result:
            summary = result['summary']
            print(f"Papers Processed: {summary['total_papers_processed']}")
            print(f"Papers Downloaded: {summary['papers_downloaded']}")
            print(f"Total Comparisons: {summary['total_comparisons']}")
            print(f"Top Papers Selected: {summary['top_papers_selected']}")
            print(f"Condensed Reports Generated: {summary['condensed_reports_generated']}")
        else:
            print(f"Message: {result.get('message', 'N/A')}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print()
    print("=== Daily Scheduler Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
