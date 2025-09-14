#!/usr/bin/env python3
"""
Run Arxiv Sync Service Once

This script runs the Arxiv sync service once for testing or manual execution.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.agent_logger import set_agent_log_directory

# Set up logging directory
current_file_dir = Path(__file__).parent
log_dir = current_file_dir / 'logs'
log_dir.mkdir(exist_ok=True)
set_agent_log_directory(str(log_dir))

from arxiv_sync_service import ArxivSyncService


async def main():
    """Main execution function."""
    print("Running Arxiv Sync Service once...")
    print(f"Logs will be saved to: {log_dir}")
    
    service = ArxivSyncService()
    
    try:
        success = await service.run_once()
        if success:
            print("Sync completed successfully!")
            sys.exit(0)
        else:
            print("Sync failed!")
            sys.exit(1)
    except Exception as e:
        print(f"Service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
