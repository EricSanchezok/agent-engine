"""
Integration Test for Daily arXiv Service

This module tests the complete pipeline from paper filtering to Swiss tournament ranking.
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
from agents.ResearchAgent.service.daily_arxiv.config import DailyArxivConfig


class DailyArxivIntegrationTest:
    """Integration test for the complete daily arXiv pipeline."""
    
    def __init__(self):
        self.logger = AgentLogger(self.__class__.__name__)
        self.paper_filter = DailyArxivPaperFilter()
        self.swiss_ranker = SwissTournamentRanker()
    
    async def test_complete_pipeline(self, target_date: date) -> dict:
        """
        Test the complete pipeline: filter -> download -> rank.
        
        Args:
            target_date: Date to process papers for
            
        Returns:
            Dictionary with complete pipeline results
        """
        self.logger.info(f"Starting complete pipeline test for date: {target_date}")
        
        try:
            # Step 1: Filter and download papers
            self.logger.info("Step 1: Filtering and downloading papers...")
            filter_result = await self.paper_filter.filter_and_download_papers(target_date)
            
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
            
            # Combine results
            complete_result = {
                "success": True,
                "target_date": target_date.isoformat(),
                "filter_result": filter_result,
                "rank_result": rank_result,
                "summary": {
                    "total_papers_processed": filter_result["papers_processed"],
                    "papers_downloaded": filter_result["papers_downloaded"],
                    "total_comparisons": rank_result["total_comparisons"],
                    "top_papers_selected": len(rank_result["top_papers"])
                }
            }
            
            self.logger.info("Complete pipeline test successful")
            return complete_result
            
        except Exception as e:
            self.logger.error(f"Pipeline test failed: {e}", exc_info=True)
            return {
                "success": False,
                "step": "unknown",
                "error": str(e),
                "target_date": target_date.isoformat()
            }
    
    async def test_swiss_tournament_only(self, target_date: date) -> dict:
        """
        Test only the Swiss tournament ranking using existing PDFs.
        
        Args:
            target_date: Date to load papers for
            
        Returns:
            Dictionary with ranking results
        """
        self.logger.info(f"Testing Swiss tournament only for date: {target_date}")
        
        try:
            result = await self.swiss_ranker.rank_papers_from_date(target_date)
            return result
            
        except Exception as e:
            self.logger.error(f"Swiss tournament test failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "target_date": target_date.isoformat()
            }


async def main():
    """Run integration tests."""
    print("=== Daily arXiv Integration Test ===")
    
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
    test = DailyArxivIntegrationTest()
    
    # Test date
    test_date = date(2025, 9, 16)
    
    print(f"Testing with date: {test_date}")
    print()
    
    # Test 1: Swiss tournament only (using existing PDFs)
    print("Test 1: Swiss Tournament Only")
    print("-" * 40)
    rank_result = await test.test_swiss_tournament_only(test_date)
    
    print(f"Success: {rank_result['success']}")
    print(f"Papers Processed: {rank_result.get('papers_processed', 0)}")
    print(f"Total Comparisons: {rank_result.get('total_comparisons', 0)}")
    
    if rank_result['success']:
        print("Top Papers:")
        for paper_info in rank_result.get('top_papers', [])[:5]:  # Show first 5
            print(f"  {paper_info['rank']}. {paper_info['paper_id']}: {paper_info['title'][:50]}... (Score: {paper_info['score']:.1f})")
    else:
        print(f"Error: {rank_result.get('error', rank_result.get('reason', 'Unknown error'))}")
    
    print()
    
    # Test 2: Complete pipeline (if Swiss tournament test was successful)
    if rank_result['success'] and rank_result.get('papers_processed', 0) > 0:
        print("Test 2: Complete Pipeline")
        print("-" * 40)
        complete_result = await test.test_complete_pipeline(test_date)
        
        print(f"Success: {complete_result['success']}")
        print(f"Target Date: {complete_result['target_date']}")
        
        if complete_result['success']:
            summary = complete_result['summary']
            print(f"Total Papers Processed: {summary['total_papers_processed']}")
            print(f"Papers Downloaded: {summary['papers_downloaded']}")
            print(f"Total Comparisons: {summary['total_comparisons']}")
            print(f"Top Papers Selected: {summary['top_papers_selected']}")
            
            print("\nFinal Top Papers:")
            for paper_info in complete_result['rank_result']['top_papers'][:5]:
                print(f"  {paper_info['rank']}. {paper_info['paper_id']}: {paper_info['title'][:50]}... (Score: {paper_info['score']:.1f})")
        else:
            print(f"Error at step {complete_result['step']}: {complete_result['error']}")
    else:
        print("Test 2: Skipped (Swiss tournament test failed or no papers found)")
    
    print()
    print("=== Integration Test Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
