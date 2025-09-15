"""
Daily arXiv Service Runner

Simple script to run the daily arXiv service with command-line interface.
"""

import asyncio
import argparse
from datetime import date, datetime
from pathlib import Path

from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_current_file_dir

from agents.ResearchAgent.service.daily_arxiv.daily_arxiv_service import DailyArxivService
from agents.ResearchAgent.service.daily_arxiv.config import DailyArxivConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Daily arXiv Service")
    
    parser.add_argument(
        "--date", 
        type=str, 
        help="Target date in YYYY-MM-DD format (overrides config)"
    )
    
    parser.add_argument(
        "--top-k", 
        type=int, 
        help="Number of top papers to select and download (overrides config)"
    )
    
    parser.add_argument(
        "--max-concurrent", 
        type=int, 
        help="Maximum concurrent downloads (overrides config)"
    )
    
    parser.add_argument(
        "--pdf-dir", 
        type=str,
        help="PDF storage directory (overrides config)"
    )
    
    parser.add_argument(
        "--status-only", 
        action="store_true",
        help="Only check service status, don't run the full service"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--show-config", 
        action="store_true",
        help="Show current configuration and exit"
    )
    
    parser.add_argument(
        "--validate-config", 
        action="store_true",
        help="Validate configuration and exit"
    )
    
    return parser.parse_args()


def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.")


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set up logging
    logger = AgentLogger("DailyArxivRunner")
    
    if args.verbose:
        logger.info("Verbose logging enabled")
    
    # Handle config-related commands
    if args.show_config:
        DailyArxivConfig.print_config()
        return 0
    
    if args.validate_config:
        if DailyArxivConfig.validate():
            print("✅ Configuration is valid")
            return 0
        else:
            print("❌ Configuration validation failed")
            return 1
    
    # Parse target date (command line overrides config)
    target_date = None
    if args.date:
        try:
            target_date = parse_date(args.date)
        except ValueError as e:
            logger.error(f"Date parsing error: {e}")
            return 1
    
    # Set up PDF storage directory (command line overrides config)
    pdf_storage_dir = args.pdf_dir
    
    logger.info(f"PDF storage directory: {pdf_storage_dir or DailyArxivConfig.get_pdf_storage_dir()}")
    
    # Initialize service
    service = DailyArxivService(pdf_storage_dir=pdf_storage_dir)
    
    try:
        if args.status_only:
            # Only check status
            logger.info("Checking service status...")
            status = await service.check_service_status(target_date)
            
            print("\n=== Service Status ===")
            print(f"Target Date: {status['target_date']}")
            print(f"Database Has Papers: {status.get('database_has_papers', False)}")
            print(f"Total Papers: {status.get('total_papers', 0)}")
            print(f"Papers with Vectors: {status.get('papers_with_vectors', 0)}")
            print(f"Service Ready: {status.get('service_ready', False)}")
            
            if 'error' in status:
                print(f"Error: {status['error']}")
                return 1
            
            return 0
        
        else:
            # Run full service
            logger.info(f"Running daily arXiv service for {target_date or 'config/today'}")
            
            # Use command line args if provided, otherwise use config defaults
            top_k = args.top_k if args.top_k is not None else None
            max_concurrent = args.max_concurrent if args.max_concurrent is not None else None
            
            logger.info(f"Top K: {top_k or DailyArxivConfig.TOP_K_PAPERS}, Max Concurrent: {max_concurrent or DailyArxivConfig.MAX_CONCURRENT_DOWNLOADS}")
            
            result = await service.run_daily_service(
                target_date=target_date,
                top_k=top_k,
                max_concurrent_downloads=max_concurrent
            )
            
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
