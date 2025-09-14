#!/usr/bin/env python3
"""
Example usage of Arxiv Sync Service

This script demonstrates how to use the Arxiv sync service programmatically.
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


async def example_usage():
    """Demonstrate service usage."""
    print("=== Arxiv Sync Service Example Usage ===")
    
    # Initialize service
    service = ArxivSyncService()
    
    try:
        # Get service statistics
        print("\n1. Getting service statistics:")
        stats = service.get_service_stats()
        print(f"   Service name: {stats['service_name']}")
        print(f"   Is running: {stats['is_running']}")
        print(f"   Database stats: {stats['database_stats']}")
        
        # Run sync once
        print("\n2. Running sync once:")
        success = await service.run_once()
        print(f"   Sync result: {'Success' if success else 'Failed'}")
        
        # Get updated statistics
        print("\n3. Getting updated statistics:")
        stats = service.get_service_stats()
        print(f"   Total syncs: {stats['sync_stats']['total_syncs']}")
        print(f"   Successful syncs: {stats['sync_stats']['successful_syncs']}")
        print(f"   Total papers processed: {stats['sync_stats']['total_papers_processed']}")
        print(f"   Total papers added: {stats['sync_stats']['total_papers_added']}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await service.close()


async def main():
    """Main function."""
    await example_usage()


if __name__ == "__main__":
    asyncio.run(main())
