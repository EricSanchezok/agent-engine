#!/usr/bin/env python3
"""
Test script for AgentLogger file rotation functionality
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_engine.agent_logger import AgentLogger


def test_log_rotation():
    """Test log file rotation functionality"""
    
    print("Starting log rotation test...")
    
    # Create logger with small max file size (1KB for testing)
    logger = AgentLogger("RotationTestLogger", max_file_size=1024)
    
    print(f"Created logger with max file size: {logger.max_file_size} bytes")
    print(f"Log file: {logger.log_filepath}")
    
    # Fill up the log file to trigger rotation
    print("\n=== Filling log file to trigger rotation ===")
    message = "This is a test message to fill up the log file. " * 10
    
    for i in range(50):
        logger.info(f"Message {i+1}: {message}")
        print(f"Logged message {i+1}")
        
        # Check if rotation occurred
        if logger.log_filename != f"agent_log_{logger.log_filename.split('_', 2)[1]}_{logger.log_filename.split('_', 2)[2]}":
            print(f"\nLog rotation occurred! New file: {logger.log_filename}")
            break
    
    # Get latest logs from current file
    print("\n=== Latest logs from current file ===")
    latest_logs = logger.get_latest_logs(5)
    print(latest_logs)
    
    print("\n=== Test completed ===")
    print(f"Final log file: {logger.log_filename}")
    print(f"Log directory contents:")
    
    # List all log files
    for log_file in logger.log_dir.glob("agent_log_*.log"):
        size = log_file.stat().st_size
        print(f"  {log_file.name}: {size} bytes")


if __name__ == "__main__":
    test_log_rotation()
