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


Test script for DailyArxivPaperFilter
"""

import asyncio
from datetime import date, timedelta

from agent_engine.agent_logger import AgentLogger
from agents.ResearchAgent.service.daily_arxiv.config import DailyArxivConfig
from agents.ResearchAgent.service.daily_arxiv.paper_filter import DailyArxivPaperFilter


async def test_paper_filter():
    """Test the paper filter with different dates."""
    logger = AgentLogger("PaperFilterTest")
    
    # Validate configuration first
    if not DailyArxivConfig.validate():
        logger.error("Configuration validation failed")
        return False
    
    logger.info("Configuration validation passed")
    
    # Initialize filter
    filter_service = DailyArxivPaperFilter()
    
    # Test with different dates
    test_dates = [
        date(2025, 9, 11)
    ]
    
    for test_date in test_dates:
        logger.info(f"Testing with date: {test_date}")
        
        try:
            result = await filter_service.filter_and_download_papers(test_date)
            
            print(f"\n=== Results for {test_date} ===")
            print(f"Success: {result['success']}")
            print(f"Papers Processed: {result['papers_processed']}")
            print(f"Papers with Vectors: {result.get('papers_with_vectors', 0)}")
            print(f"Papers Selected: {result.get('papers_selected', 0)}")
            print(f"Papers Downloaded: {result['papers_downloaded']}")
            
            if result['success'] and result['selected_papers']:
                print(f"\nTop 3 Selected Papers:")
                for i, paper_info in enumerate(result['selected_papers'][:3], 1):
                    distance = paper_info.get('distance')
                    distance_str = f" (distance: {distance:.4f})" if distance else ""
                    print(f"{i}. {paper_info['full_id']}{distance_str}")
                    print(f"   {paper_info['title'][:60]}...")
            elif not result['success']:
                print(f"Error: {result.get('error', result.get('reason', 'Unknown error'))}")
            
        except Exception as e:
            logger.error(f"Error testing date {test_date}: {e}", exc_info=True)
            print(f"Error testing date {test_date}: {e}")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_paper_filter())
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test failed")
