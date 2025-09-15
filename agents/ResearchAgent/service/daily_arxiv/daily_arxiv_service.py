"""
Daily arXiv Service - Main Entry Point

This module provides the main entry point for the daily arXiv service,
coordinating all steps of the process.
"""

import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, Optional

from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_current_file_dir

from .filter_and_download import DailyArxivFilterAndDownload
from .config import DailyArxivConfig


class DailyArxivService:
    """
    Main daily arXiv service coordinator.
    
    This class coordinates all steps of the daily arXiv processing pipeline.
    """
    
    def __init__(self, pdf_storage_dir: Optional[str] = None):
        """
        Initialize the daily arXiv service.
        
        Args:
            pdf_storage_dir: Directory to store downloaded PDFs (optional, uses config if None)
        """
        self.logger = AgentLogger(self.__class__.__name__)
        
        # Initialize Step 1 service
        self.filter_and_download = DailyArxivFilterAndDownload(pdf_storage_dir)
        
        self.logger.info("DailyArxivService initialized")
    
    async def run_daily_service(
        self, 
        target_date: Optional[date] = None,
        top_k: Optional[int] = None,
        max_concurrent_downloads: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the complete daily arXiv service.
        
        Args:
            target_date: Date to process (defaults to config or today)
            top_k: Number of top papers to select and download (uses config if None)
            max_concurrent_downloads: Maximum concurrent downloads (uses config if None)
            
        Returns:
            Dictionary with complete service results
        """
        if target_date is None:
            target_date = date.today()
        
        if top_k is None:
            top_k = DailyArxivConfig.TOP_K_PAPERS
            
        if max_concurrent_downloads is None:
            max_concurrent_downloads = DailyArxivConfig.MAX_CONCURRENT_DOWNLOADS
        
        self.logger.info(f"Starting daily arXiv service for {target_date}")
        
        service_start_time = datetime.now()
        
        try:
            # Step 1: Filter and Download
            self.logger.info("Executing Step 1: Filter and Download")
            step1_result = await self.filter_and_download.run_step1(
                target_date=target_date,
                top_k=top_k,
                max_concurrent_downloads=max_concurrent_downloads
            )
            
            # TODO: Add additional steps here as they are implemented
            # Step 2: Analysis and Processing
            # Step 3: Report Generation
            # etc.
            
            service_end_time = datetime.now()
            service_duration = (service_end_time - service_start_time).total_seconds()
            
            # Compile final results
            final_result = {
                "service_success": step1_result["success"],
                "target_date": target_date.isoformat(),
                "service_duration_seconds": service_duration,
                "step1_result": step1_result,
                "steps_completed": ["filter_and_download"],
                "steps_pending": ["analysis", "report_generation"]  # Placeholder for future steps
            }
            
            if step1_result["success"]:
                self.logger.info(f"Daily service completed successfully in {service_duration:.2f} seconds")
                self.logger.info(f"Downloaded {step1_result['papers_downloaded']} papers")
            else:
                self.logger.warning(f"Daily service completed with issues: {step1_result.get('reason', 'Unknown error')}")
            
            return final_result
            
        except Exception as e:
            service_end_time = datetime.now()
            service_duration = (service_end_time - service_start_time).total_seconds()
            
            self.logger.error(f"Daily service failed: {e}", exc_info=True)
            
            return {
                "service_success": False,
                "target_date": target_date.isoformat(),
                "service_duration_seconds": service_duration,
                "error": str(e),
                "steps_completed": [],
                "steps_pending": ["filter_and_download", "analysis", "report_generation"]
            }
    
    async def check_service_status(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Check the status of the daily service for a specific date.
        
        Args:
            target_date: Date to check (defaults to today)
            
        Returns:
            Dictionary with service status information
        """
        if target_date is None:
            target_date = date.today()
        
        self.logger.info(f"Checking service status for {target_date}")
        
        try:
            # Check if database has papers for the date
            has_papers = await self.filter_and_download.check_database_update(target_date)
            
            # Get basic statistics
            papers, vectors = await self.filter_and_download.get_today_papers_and_vectors(target_date)
            
            status = {
                "target_date": target_date.isoformat(),
                "database_has_papers": has_papers,
                "total_papers": len(papers),
                "papers_with_vectors": sum(1 for v in vectors if v is not None),
                "service_ready": has_papers and len(papers) > 0
            }
            
            self.logger.info(f"Service status: {status['total_papers']} papers, {status['papers_with_vectors']} with vectors")
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error checking service status: {e}", exc_info=True)
            return {
                "target_date": target_date.isoformat(),
                "error": str(e),
                "service_ready": False
            }


async def main():
    """Test the daily arXiv service."""
    service = DailyArxivService()
    
    # Check service status first
    print("Checking service status...")
    status = await service.check_service_status()
    print(f"Service Status: {status}")
    
    if status.get("service_ready", False):
        print("\nRunning daily service...")
        result = await service.run_daily_service()
        
        print("\nDaily Service Results:")
        print(f"Success: {result['service_success']}")
        print(f"Duration: {result['service_duration_seconds']:.2f} seconds")
        print(f"Steps Completed: {result['steps_completed']}")
        
        if result['service_success']:
            step1 = result['step1_result']
            print(f"Papers Processed: {step1['papers_processed']}")
            print(f"Papers Downloaded: {step1['papers_downloaded']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        print("Service not ready - no papers available for processing")


if __name__ == "__main__":
    asyncio.run(main())
