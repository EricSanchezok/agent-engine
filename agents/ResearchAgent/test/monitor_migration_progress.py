from __future__ import annotations

import time
from typing import Dict, List

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ResearchAgent.config import PAPER_DSN_TEMPLATE
from agents.ResearchAgent.paper_memory import PaperMemory, PaperMemoryConfig


logger = AgentLogger("MigrationProgressMonitor")


def check_migration_progress() -> Dict[str, int]:
    """Check current migration progress for all segments."""
    logger.info("Checking migration progress...")
    
    # Initialize PaperMemory
    pm = PaperMemory(PaperMemoryConfig(
        dsn_template=PAPER_DSN_TEMPLATE,
        collection_name="papers",
        vector_field="text_vec",
        vector_dim=3072,
        metric="cosine",
        index_params={"lists": 100},
    ))
    
    # Segments to check
    segments = [
        "2022H1", "2022H2",
        "2023H1", "2023H2", 
        "2024H1", "2024H2",
        "2025H1", "2025H2",
        "undated"
    ]
    
    progress = {}
    total_records = 0
    
    for seg in segments:
        try:
            um = pm._get_segment_um(seg)
            from agent_engine.memory.ultra_memory import Filter
            records = um.query("papers", Filter())
            count = len(records)
            progress[seg] = count
            total_records += count
            
        except Exception as e:
            logger.error(f"Error checking segment {seg}: {e}")
            progress[seg] = 0
    
    return progress


def print_progress_report(progress: Dict[str, int], previous_progress: Dict[str, int] = None) -> None:
    """Print a formatted progress report."""
    logger.info("=" * 60)
    logger.info("MIGRATION PROGRESS REPORT")
    logger.info("=" * 60)
    
    total_records = sum(progress.values())
    segments_with_data = sum(1 for count in progress.values() if count > 0)
    
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total records migrated: {total_records:,}")
    logger.info(f"Segments with data: {segments_with_data}/{len(progress)}")
    logger.info("")
    
    logger.info("Segment Details:")
    logger.info("-" * 40)
    
    for seg, count in progress.items():
        status = "✓" if count > 0 else "○"
        logger.info(f"  {seg:>8}: {status} {count:>8,} records")
        
        # Show progress if we have previous data
        if previous_progress and seg in previous_progress:
            prev_count = previous_progress[seg]
            if count > prev_count:
                new_records = count - prev_count
                logger.info(f"           +{new_records:,} new records")
    
    logger.info("-" * 40)
    
    # Migration status assessment
    if total_records > 0:
        logger.info("✓ Migration is active and progressing")
        if segments_with_data == len(progress):
            logger.info("✓ All segments have been processed")
        else:
            remaining = len(progress) - segments_with_data
            logger.info(f"○ {remaining} segments still pending")
    else:
        logger.info("○ No data found - migration may not have started yet")
    
    logger.info("=" * 60)


def monitor_migration(interval_minutes: int = 5, max_checks: int = None) -> None:
    """Monitor migration progress at regular intervals."""
    logger.info(f"Starting migration monitoring (check every {interval_minutes} minutes)")
    
    previous_progress = None
    check_count = 0
    
    try:
        while True:
            check_count += 1
            logger.info(f"Progress check #{check_count}")
            
            current_progress = check_migration_progress()
            print_progress_report(current_progress, previous_progress)
            
            previous_progress = current_progress
            
            # Check if we should stop
            if max_checks and check_count >= max_checks:
                logger.info(f"Reached maximum checks ({max_checks}), stopping monitoring")
                break
            
            # Check if migration is complete
            total_records = sum(current_progress.values())
            segments_with_data = sum(1 for count in current_progress.values() if count > 0)
            
            if segments_with_data == len(current_progress) and total_records > 0:
                logger.info("✓ Migration appears to be complete!")
                break
            
            # Wait for next check
            logger.info(f"Waiting {interval_minutes} minutes for next check...")
            time.sleep(interval_minutes * 60)
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")


def main():
    """Main function."""
    logger.info("Migration Progress Monitor")
    
    # Check configuration
    if not PAPER_DSN_TEMPLATE or "USER:PASS@HOST:PORT" in PAPER_DSN_TEMPLATE:
        logger.error("PAPER_DSN_TEMPLATE is not properly configured")
        return
    
    # Single check mode
    logger.info("Running single progress check...")
    progress = check_migration_progress()
    print_progress_report(progress)
    
    # Ask user if they want continuous monitoring
    logger.info("")
    logger.info("Options:")
    logger.info("1. Run continuous monitoring (check every 5 minutes)")
    logger.info("2. Exit")
    
    try:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            monitor_migration(interval_minutes=5)
        else:
            logger.info("Exiting...")
    except KeyboardInterrupt:
        logger.info("Exiting...")


if __name__ == "__main__":
    main()
