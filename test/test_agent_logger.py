#!/usr/bin/env python3
"""
Test script for AgentLogger class
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_engine.agent_logger import AgentLogger


def test_agent_logger():
    """Test various features of AgentLogger"""
    
    print("Starting AgentLogger test...")
    
    # Test 1: Basic initialization with default settings
    print("\n=== Test 1: Basic initialization ===")
    logger = AgentLogger("TestLogger")
    
    # Test 2: Different log levels with colors
    print("\n=== Test 2: Different log levels ===")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test 3: Custom log directory
    print("\n=== Test 3: Custom log directory ===")
    custom_logger = AgentLogger("CustomLogger", log_dir="./custom_logs")
    custom_logger.info("This log goes to custom directory")
    
    # Test 4: Get latest logs
    print("\n=== Test 4: Get latest logs ===")
    latest_logs = logger.get_latest_logs(5)
    print("Latest 5 log lines:")
    print(latest_logs)
    
    # Test 5: Exception logging
    print("\n=== Test 5: Exception logging ===")
    try:
        raise ValueError("This is a test exception")
    except Exception as e:
        logger.exception("Caught an exception")
    
    # Test 6: Multiple log messages to test file rotation
    print("\n=== Test 6: Multiple log messages ===")
    for i in range(10):
        logger.info(f"Test message {i+1}: This is a test log message to fill up the log file")
        time.sleep(0.1)  # Small delay to see timestamps
    
    # Test 7: Cleanup old logs
    print("\n=== Test 7: Cleanup old logs ===")
    logger.cleanup_old_logs(keep_days=1)
    
    # Test 8: Final log retrieval
    print("\n=== Test 8: Final log retrieval ===")
    final_logs = logger.get_latest_logs(10)
    print("Final 10 log lines:")
    print(final_logs)
    
    print("\n=== Test completed ===")
    print(f"Log files created in: {logger.log_dir}")
    print(f"Current log file: {logger.log_filename}")
    print(f"Link file: {logger.log_dir / 'current.log'}")


if __name__ == "__main__":
    test_agent_logger()
