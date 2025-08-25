#!/usr/bin/env python3
"""
Demo script for AgentLogger class showcasing all features
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_engine.agent_logger import AgentLogger


def demo_agent_logger():
    """Demonstrate all features of AgentLogger"""
    
    print("🚀 AgentLogger Demo - Showcasing All Features")
    print("=" * 60)
    
    # Feature 1: Basic initialization with default settings
    print("\n📁 Feature 1: Basic Initialization")
    print("-" * 40)
    logger = AgentLogger("DemoLogger")
    print(f"✓ Logger created: {logger.name}")
    print(f"✓ Log directory: {logger.log_dir}")
    print(f"✓ Log file: {logger.log_filename}")
    
    # Feature 2: Colored output for different log levels
    print("\n🎨 Feature 2: Colored Output")
    print("-" * 40)
    logger.debug("This is a DEBUG message (cyan)")
    logger.info("This is an INFO message (green)")
    logger.warning("This is a WARNING message (yellow)")
    logger.error("This is an ERROR message (red)")
    logger.critical("This is a CRITICAL message (magenta)")
    print("✓ All log levels displayed with different colors")
    
    # Feature 3: Exception logging
    print("\n⚠️  Feature 3: Exception Logging")
    print("-" * 40)
    try:
        result = 10 / 0
    except Exception as e:
        logger.exception("Division by zero error occurred")
        print("✓ Exception logged with full traceback")
    
    # Feature 4: Custom log directory
    print("\n📂 Feature 4: Custom Log Directory")
    print("-" * 40)
    custom_logger = AgentLogger("CustomDemo", log_dir="./demo_logs")
    custom_logger.info("This log goes to custom directory")
    print("✓ Custom log directory created and used")
    
    # Feature 5: Get latest logs
    print("\n📖 Feature 5: Get Latest Logs")
    print("-" * 40)
    latest_logs = logger.get_latest_logs(3)
    print("Latest 3 log entries:")
    print(latest_logs)
    
    # Feature 6: Multiple messages to demonstrate logging
    print("\n📝 Feature 6: Multiple Log Messages")
    print("-" * 40)
    for i in range(5):
        logger.info(f"Demo message {i+1}: Testing various features of AgentLogger")
        time.sleep(0.1)
    print("✓ Multiple messages logged")
    
    # Feature 7: Cleanup old logs
    print("\n🧹 Feature 7: Cleanup Old Logs")
    print("-" * 40)
    logger.cleanup_old_logs(keep_days=1)
    print("✓ Old log cleanup completed")
    
    # Feature 8: Final summary
    print("\n📊 Feature 8: Final Summary")
    print("-" * 40)
    print(f"✓ Total log files in {logger.log_dir}:")
    
    log_files = list(logger.log_dir.glob("agent_log_*.log"))
    for log_file in log_files:
        size = log_file.stat().st_size
        print(f"  • {log_file.name}: {size} bytes")
    
    print(f"\n✓ Current active log: {logger.log_filename}")
    print(f"✓ Link file: {logger.log_dir / 'current.log'}")
    
    # Feature 9: Get final logs
    print("\n🔍 Feature 9: Final Log Retrieval")
    print("-" * 40)
    final_logs = logger.get_latest_logs(5)
    print("Final 5 log entries:")
    print(final_logs)
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 60)
    print("AgentLogger features demonstrated:")
    print("• ✅ Colored console output")
    print("• ✅ File-based logging")
    print("• ✅ Automatic file rotation")
    print("• ✅ Link file management")
    print("• ✅ Latest log retrieval")
    print("• ✅ Exception logging")
    print("• ✅ Custom log directories")
    print("• ✅ Old log cleanup")


if __name__ == "__main__":
    demo_agent_logger()
