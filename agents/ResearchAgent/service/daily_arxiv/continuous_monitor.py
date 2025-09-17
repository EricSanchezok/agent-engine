from pathlib import Path
from agent_engine.agent_logger import set_agent_log_directory

current_file_dir = Path(__file__).parent
log_dir = current_file_dir / 'logs'
set_agent_log_directory(str(log_dir))

"""
Continuous Monitor for Daily arXiv Service

This module provides continuous monitoring for the daily arXiv processing pipeline.
It runs continuously and processes papers daily with automatic retries and error handling.
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
import signal
import sys
from datetime import date, datetime, timezone
from typing import Dict, Any

from agent_engine.agent_logger import AgentLogger

from agents.ResearchAgent.service.daily_arxiv.daily_scheduler import DailyArxivScheduler
from agents.ResearchAgent.service.daily_arxiv.config import DailyArxivConfig


class ContinuousMonitor:
    """
    Continuous monitor for daily arXiv processing.
    
    This class runs continuously and monitors for daily processing needs,
    handling automatic retries and error recovery.
    """
    
    def __init__(self):
        """Initialize the continuous monitor."""
        self.logger = AgentLogger(self.__class__.__name__)
        self.scheduler = DailyArxivScheduler()
        self.running = True
        
        # Set up signal handlers for graceful shutdown (Windows compatible)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("ContinuousMonitor initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.running = False
    
    async def run(self):
        """Run the continuous monitoring loop."""
        self.logger.info("Starting continuous monitoring for daily arXiv processing")
        
        try:
            while self.running:
                try:
                    # Check if today's processing is needed
                    today = date.today()
                    
                    if not self.scheduler.result_manager.is_date_processed(today):
                        self.logger.info(f"Processing needed for today: {today}")
                        result = await self.scheduler.run_daily_pipeline(today)
                        
                        if result["success"]:
                            self.logger.info(f"Successfully processed {today}")
                            if 'summary' in result:
                                summary = result['summary']
                                self.logger.info(f"Summary: {summary['papers_downloaded']} papers downloaded, "
                                                f"{summary['total_comparisons']} comparisons, "
                                                f"{summary['condensed_reports_generated']} reports generated")
                        else:
                            self.logger.error(f"Failed to process {today}: {result.get('error', 'Unknown error')}")
                    else:
                        self.logger.info(f"Today ({today}) already processed")
                    
                    # Wait before next check (1 hour) with periodic running check
                    if self.running:
                        self.logger.info("Waiting 1 hour before next check")
                        for _ in range(3600):  # 3600 seconds = 1 hour
                            if not self.running:
                                break
                            await asyncio.sleep(1)
                
                except KeyboardInterrupt:
                    self.logger.info("Received keyboard interrupt, stopping monitoring")
                    self.running = False
                    break
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                    if self.running:
                        self.logger.info("Waiting 1 hour before retrying")
                        for _ in range(3600):
                            if not self.running:
                                break
                            await asyncio.sleep(1)
        
        except Exception as e:
            self.logger.error(f"Fatal error in continuous monitoring: {e}", exc_info=True)
        finally:
            self.logger.info("Continuous monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return self.scheduler.get_processing_status()


async def main():
    """Main entry point for continuous monitoring."""
    print("=== Continuous Daily arXiv Monitor ===")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    # Print configuration
    DailyArxivConfig.print_config()
    print()
    
    # Validate configuration
    if not DailyArxivConfig.validate():
        print("❌ Configuration validation failed")
        return
    
    print("✅ Configuration validation passed")
    print()
    
    # Initialize monitor
    monitor = ContinuousMonitor()
    
    # Show initial status
    status = monitor.get_status()
    print(f"Today: {status['today']}")
    print(f"Is Today Processed: {status['is_today_processed']}")
    print(f"Status Summary: {status['status_summary']}")
    print()
    
    # Start monitoring
    try:
        await monitor.run()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
    
    print("=== Continuous Monitor Stopped ===")


if __name__ == "__main__":
    asyncio.run(main())
