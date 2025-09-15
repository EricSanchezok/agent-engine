#!/usr/bin/env python3
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


Start script for Arxiv Sync Service

This script starts the Arxiv sync service for continuous operation.
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
    print("Starting Arxiv Sync Service...")
    print(f"Logs will be saved to: {log_dir}")
    
    service = ArxivSyncService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        print("\nService interrupted by user")
        await service.stop()
    except Exception as e:
        print(f"Service failed: {e}")
        await service.stop()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
