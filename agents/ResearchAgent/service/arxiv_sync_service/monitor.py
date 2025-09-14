#!/usr/bin/env python3
"""
Monitor script for Arxiv Sync Service

This script provides monitoring and status information for the service.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agents.ResearchAgent.arxiv_database import ArxivDatabase
from config import ArxivSyncConfig


def monitor_service():
    """Monitor service status and database statistics."""
    print("=== Arxiv Sync Service Monitor ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Check configuration
    print("\n1. Configuration Status:")
    if ArxivSyncConfig.validate():
        print("   ✅ Configuration valid")
        print(f"   📊 Sync interval: {ArxivSyncConfig.SYNC_INTERVAL_MINUTES} minutes")
        print(f"   🔄 Max concurrent embeddings: {ArxivSyncConfig.MAX_CONCURRENT_EMBEDDINGS}")
        print(f"   🔁 Max retry attempts: {ArxivSyncConfig.MAX_RETRY_ATTEMPTS}")
    else:
        print("   ❌ Configuration invalid")
        return
    
    # Check database
    print("\n2. Database Status:")
    try:
        db = ArxivDatabase(
            name=ArxivSyncConfig.DATABASE_NAME,
            persist_dir=ArxivSyncConfig.DATABASE_DIR
        )
        
        db_stats = db.get_stats()
        paper_count = db.count()
        
        print(f"   ✅ Database connected")
        print(f"   📚 Total papers: {paper_count}")
        print(f"   📊 Database stats: {db_stats}")
        
    except Exception as e:
        print(f"   ❌ Database error: {e}")
    
    # Check logs
    print("\n3. Log Status:")
    log_dir = Path(__file__).parent / 'logs'
    if log_dir.exists():
        log_files = list(log_dir.glob('*.log'))
        print(f"   📝 Log directory: {log_dir}")
        print(f"   📄 Log files: {len(log_files)}")
        
        if log_files:
            # Get the most recent log file
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"   📅 Latest log: {latest_log.name}")
            print(f"   🕒 Last modified: {datetime.fromtimestamp(latest_log.stat().st_mtime)}")
    else:
        print("   ⚠️  Log directory not found")
    
    print("\n=== Monitor Complete ===")


def main():
    """Main function."""
    monitor_service()


if __name__ == "__main__":
    main()
