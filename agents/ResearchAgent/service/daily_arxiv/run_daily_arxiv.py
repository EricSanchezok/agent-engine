"""
Daily arXiv Service Runner

Simple script to run the daily arXiv service using configuration.
"""

import asyncio
from datetime import date

from agent_engine.agent_logger import AgentLogger

from agents.ResearchAgent.service.daily_arxiv.daily_arxiv_service import DailyArxivService
from agents.ResearchAgent.service.daily_arxiv.config import DailyArxivConfig


async def main():
    """Main entry point."""
    logger = AgentLogger("DailyArxivRunner")
    
    # Validate configuration first
    if not DailyArxivConfig.validate():
        logger.error("Configuration validation failed")
        return 1
    
    logger.info("Configuration validation passed")
    
    # Initialize service
    service = DailyArxivService()
    
    try:
        # Check service status first
        logger.info("Checking service status...")
        status = await service.check_service_status()
        
        print("\n=== Service Status ===")
        print(f"Target Date: {status['target_date']}")
        print(f"Database Has Papers: {status.get('database_has_papers', False)}")
        print(f"Total Papers: {status.get('total_papers', 0)}")
        print(f"Papers with Vectors: {status.get('papers_with_vectors', 0)}")
        print(f"Service Ready: {status.get('service_ready', False)}")
        
        if 'error' in status:
            print(f"Error: {status['error']}")
            return 1
        
        if not status.get('service_ready', False):
            print("Service not ready - no papers available for processing")
            return 0
        
        # Run full service
        logger.info("Running daily arXiv service...")
        result = await service.run_daily_service()
        
        print("\n=== Daily Service Results ===")
        print(f"Success: {result['service_success']}")
        print(f"Target Date: {result['target_date']}")
        print(f"Duration: {result['service_duration_seconds']:.2f} seconds")
        print(f"Steps Completed: {', '.join(result['steps_completed'])}")
        
        if result['service_success']:
            step1 = result['step1_result']
            print(f"\n=== Step 1 Results ===")
            print(f"Papers Processed: {step1['papers_processed']}")
            print(f"Papers with Vectors: {step1['papers_with_vectors']}")
            print(f"Papers Selected: {step1['papers_selected']}")
            print(f"Papers Downloaded: {step1['papers_downloaded']}")
            print(f"Download Failures: {step1['download_failures']}")
            
            if step1['selected_papers']:
                print(f"\n=== Selected Papers ===")
                for i, paper_info in enumerate(step1['selected_papers'][:10], 1):  # Show first 10
                    distance = paper_info.get('distance')
                    distance_str = f" (distance: {distance:.4f})" if distance else ""
                    print(f"{i:2d}. {paper_info['full_id']}{distance_str}")
                    print(f"    {paper_info['title'][:80]}...")
            
            return 0
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    
    except Exception as e:
        logger.error(f"Service execution failed: {e}", exc_info=True)
        print(f"Service execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)