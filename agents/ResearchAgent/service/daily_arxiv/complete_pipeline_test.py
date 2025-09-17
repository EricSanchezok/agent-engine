"""
Complete Pipeline Test for Daily arXiv Service

This module tests the complete pipeline: filter -> download -> rank -> condense -> save.
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
from datetime import date

from agent_engine.agent_logger import AgentLogger

from agents.ResearchAgent.service.daily_arxiv.paper_filter import DailyArxivPaperFilter
from agents.ResearchAgent.service.daily_arxiv.swiss_tournament import SwissTournamentRanker
from agents.ResearchAgent.service.daily_arxiv.paper_condenser import PaperCondenser
from agents.ResearchAgent.service.daily_arxiv.result_manager import ResultManager
from agents.ResearchAgent.service.daily_arxiv.config import DailyArxivConfig


class CompletePipelineTest:
    """Complete pipeline test for the daily arXiv service."""
    
    def __init__(self):
        self.logger = AgentLogger(self.__class__.__name__)
        self.paper_filter = DailyArxivPaperFilter()
        self.swiss_ranker = SwissTournamentRanker()
        self.paper_condenser = PaperCondenser()
        self.result_manager = ResultManager()
    
    async def test_complete_pipeline(self, target_date: date) -> dict:
        """
        Test the complete pipeline: filter -> download -> rank -> condense -> save.
        
        Args:
            target_date: Date to process papers for
            
        Returns:
            Dictionary with complete pipeline results
        """
        self.logger.info(f"Starting complete pipeline test for date: {target_date}")
        
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
        
        try:
            # Step 1: Filter and download papers
            self.logger.info("Step 1: Filtering and downloading papers...")
            filter_result = await self.paper_filter.filter_and_download_papers(target_date)
            
            if not filter_result["success"]:
                error_msg = filter_result.get("error", filter_result.get("reason", "Unknown error"))
                self.result_manager.mark_date_failed(target_date, error_msg)
                return {
                    "success": False,
                    "step": "filter",
                    "error": error_msg,
                    "target_date": target_date.isoformat()
                }
            
            self.logger.info(f"Filter step completed: {filter_result['papers_downloaded']} papers downloaded")
            
            # Step 2: Swiss tournament ranking
            if not filter_result.get("successful_pdf_paths"):
                error_msg = "No PDF files available for ranking"
                self.result_manager.mark_date_failed(target_date, error_msg)
                return {
                    "success": False,
                    "step": "rank",
                    "error": error_msg,
                    "target_date": target_date.isoformat(),
                    "filter_result": filter_result
                }
            
            self.logger.info("Step 2: Running Swiss tournament ranking...")
            rank_result = await self.swiss_ranker.rank_papers_from_pdf_paths(
                filter_result["successful_pdf_paths"]
            )
            
            if not rank_result["success"]:
                error_msg = rank_result.get("error", rank_result.get("reason", "Unknown error"))
                self.result_manager.mark_date_failed(target_date, error_msg)
                return {
                    "success": False,
                    "step": "rank",
                    "error": error_msg,
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
                error_msg = "Failed to generate any condensed reports"
                self.result_manager.mark_date_failed(target_date, error_msg)
                return {
                    "success": False,
                    "step": "condense",
                    "error": error_msg,
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
                saved_files
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
            
            self.logger.info("Complete pipeline test successful")
            return complete_result
            
        except Exception as e:
            self.logger.error(f"Pipeline test failed: {e}", exc_info=True)
            self.result_manager.mark_date_failed(target_date, str(e))
            return {
                "success": False,
                "step": "unknown",
                "error": str(e),
                "target_date": target_date.isoformat()
            }
    
    def _get_result_directory(self, target_date: date) -> Path:
        """Get the result directory for a specific date."""
        year_dir = str(target_date.year)
        month_dir = f"{target_date.month:02d}"
        day_dir = f"{target_date.day:02d}"
        
        result_storage_dir = Path(DailyArxivConfig.get_result_storage_dir())
        return result_storage_dir / year_dir / month_dir / day_dir
    
    def test_status_tracking(self, target_date: date) -> dict:
        """Test the status tracking functionality."""
        self.logger.info(f"Testing status tracking for date: {target_date}")
        
        # Check initial status
        is_processed = self.result_manager.is_date_processed(target_date)
        
        # Get status summary
        summary = self.result_manager.get_processing_status_summary()
        
        return {
            "target_date": target_date.isoformat(),
            "is_processed": is_processed,
            "status_summary": summary
        }


async def main():
    """Run complete pipeline test."""
    print("=== Complete Daily arXiv Pipeline Test ===")
    
    # Print configuration
    DailyArxivConfig.print_config()
    print()
    
    # Validate configuration
    if not DailyArxivConfig.validate():
        print("❌ Configuration validation failed")
        return
    
    print("✅ Configuration validation passed")
    print()
    
    # Initialize test
    test = CompletePipelineTest()
    
    # Test date
    test_date = date(2025, 9, 16)
    
    print(f"Testing with date: {test_date}")
    print()
    
    # Test status tracking first
    print("Status Tracking Test:")
    print("-" * 40)
    status_result = test.test_status_tracking(test_date)
    print(f"Is Processed: {status_result['is_processed']}")
    print(f"Status Summary: {status_result['status_summary']}")
    print()
    
    # Test complete pipeline
    print("Complete Pipeline Test:")
    print("-" * 40)
    complete_result = await test.test_complete_pipeline(test_date)
    
    print(f"Success: {complete_result['success']}")
    print(f"Target Date: {complete_result['target_date']}")
    
    if complete_result['success']:
        if 'summary' in complete_result:
            summary = complete_result['summary']
            print(f"Total Papers Processed: {summary['total_papers_processed']}")
            print(f"Papers Downloaded: {summary['papers_downloaded']}")
            print(f"Total Comparisons: {summary['total_comparisons']}")
            print(f"Top Papers Selected: {summary['top_papers_selected']}")
            print(f"Condensed Reports Generated: {summary['condensed_reports_generated']}")
            
            print(f"\nResult File: {complete_result.get('result_file_path', 'N/A')}")
            
            print("\nFinal Top Papers:")
            for paper_info in complete_result['rank_result']['top_papers'][:5]:
                print(f"  {paper_info['rank']}. {paper_info['paper_id']}: {paper_info['title'][:50]}... (Score: {paper_info['score']:.1f})")
        else:
            print(f"Message: {complete_result.get('message', 'N/A')}")
    else:
        print(f"Error at step {complete_result.get('step', 'unknown')}: {complete_result['error']}")
    
    print()
    print("=== Complete Pipeline Test Finished ===")


if __name__ == "__main__":
    asyncio.run(main())
