#!/usr/bin/env python3
"""
Test script for Arxiv Sync Service

This script tests the service configuration and basic functionality.
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
from config import ArxivSyncConfig


async def test_service():
    """Test the Arxiv sync service."""
    print("Testing Arxiv Sync Service...")
    
    # Test configuration
    
    print("\n=== Configuration Test ===")
    if not ArxivSyncConfig.validate():
        print("❌ Configuration validation failed")
        return False
    else:
        print("✅ Configuration validation passed")
    
    ArxivSyncConfig.print_config()
    
    # Test service initialization
    print("\n=== Service Initialization Test ===")
    try:
        service = ArxivSyncService()
        print("✅ Service initialized successfully")
        
        # Test service stats
        stats = service.get_service_stats()
        print(f"✅ Service stats retrieved: {stats}")
        
        # Test database connection
        db_stats = service.arxiv_database.get_stats()
        print(f"✅ Database connection successful: {db_stats}")
        
        await service.close()
        print("✅ Service closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False


async def main():
    """Main test function."""
    success = await test_service()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
