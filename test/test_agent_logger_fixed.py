#!/usr/bin/env python3
"""
Test script for the fixed AgentLogger
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agent_engine.agent_logger.agent_logger import AgentLogger
import time

def test_agent_logger():
    """Test the fixed AgentLogger functionality"""
    print("Testing AgentLogger...")
    
    # Create logger instance
    logger = AgentLogger("TestLogger")
    
    # Test different log levels
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
    
    # Wait a bit to see the effect
    time.sleep(1)
    
    print("Test completed. Check the logs folder for results.")
    print("The agent_logger_000_link.log should now contain the actual log content.")

if __name__ == "__main__":
    test_agent_logger()
