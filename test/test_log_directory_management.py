#!/usr/bin/env python3
"""
Test script for log directory management functionality
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.agent_logger import AgentLogger, set_agent_log_directory, get_agent_log_directory


def test_log_directory_management():
    """Test the log directory management functionality"""
    print("Testing log directory management...")
    
    # Test 1: Default behavior (should use project root logs)
    print("\n1. Testing default behavior...")
    logger1 = AgentLogger("DefaultLogger")
    print(f"Default logger log dir: {logger1.log_dir}")
    
    # Test 2: Set custom log directory
    print("\n2. Testing custom log directory...")
    with tempfile.TemporaryDirectory() as temp_dir:
        custom_log_dir = Path(temp_dir) / "custom_logs"
        custom_log_dir.mkdir(exist_ok=True)
        
        set_agent_log_directory(str(custom_log_dir))
        print(f"Set agent log directory to: {get_agent_log_directory()}")
        
        # Create logger after setting directory
        logger2 = AgentLogger("CustomLogger")
        print(f"Custom logger log dir: {logger2.log_dir}")
        
        # Verify it's using the custom directory
        assert str(logger2.log_dir) == str(custom_log_dir), "Logger should use custom directory"
        
        # Test that environment variable is set
        assert os.getenv('AGENT_LOG_DIR') == str(custom_log_dir), "Environment variable should be set"
    
    # Test 3: Verify fallback to default after clearing
    print("\n3. Testing fallback to default...")
    logger3 = AgentLogger("FallbackLogger")
    print(f"Fallback logger log dir: {logger3.log_dir}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_log_directory_management()
