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
    """Main execution function with automatic restart capability."""
    print("Starting Arxiv Sync Service...")
    print(f"Logs will be saved to: {log_dir}")
    
    restart_count = 0
    max_restarts = 5
    
    while restart_count < max_restarts:
        service = None
        try:
            print(f"Starting service (attempt {restart_count + 1}/{max_restarts})...")
            service = ArxivSyncService()
            await service.start()
            # If we reach here, the service stopped normally
            break
            
        except KeyboardInterrupt:
            print("\nService interrupted by user")
            if service:
                await service.stop()
            break
            
        except Exception as e:
            restart_count += 1
            print(f"Service failed (attempt {restart_count}): {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            if service:
                try:
                    await service.stop()
                except:
                    pass
            
            if restart_count < max_restarts:
                print(f"Restarting service in 30 seconds... (attempt {restart_count + 1}/{max_restarts})")
                await asyncio.sleep(30)
            else:
                print(f"Maximum restart attempts ({max_restarts}) reached. Service will not restart.")
                break


if __name__ == "__main__":
    asyncio.run(main())
