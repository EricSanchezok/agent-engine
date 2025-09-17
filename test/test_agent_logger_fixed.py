#!/usr/bin/env python3
"""
Test script for FIXED AgentLogger and LogDirectoryManager
Tests various scenarios to verify the fixes work correctly
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from agent_engine.agent_logger.agent_logger import AgentLogger
from agent_engine.agent_logger.log_directory_manager import (
    set_agent_log_directory, 
    get_agent_log_directory, 
    clear_agent_log_directory,
    update_existing_loggers_directory
)


def test_fixed_logger_names():
    """Test 1: Check if logger names are now properly handled"""
    print("=" * 60)
    print("Test 1: Fixed Logger Names")
    print("=" * 60)
    
    # Create temporary directory for this test
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    try:
        set_agent_log_directory(temp_dir)
        
        # Test with different logger names
        logger1 = AgentLogger(name="TestLogger1")
        logger2 = AgentLogger(name="TestLogger2")
        logger3 = AgentLogger(name="TestLogger3")
        
        print(f"Logger1 name: {logger1.name}")
        print(f"Logger2 name: {logger2.name}")
        print(f"Logger3 name: {logger3.name}")
        
        # Log messages with different loggers
        logger1.info("This is a message from TestLogger1")
        logger2.warning("This is a warning from TestLogger2")
        logger3.error("This is an error from TestLogger3")
        
        # Check if they are different instances
        print(f"Are logger1 and logger2 the same instance? {logger1 is logger2}")
        print(f"Are logger2 and logger3 the same instance? {logger2 is logger3}")
        
        # Check log files in the directory
        print(f"\nLog files in {temp_dir}:")
        log_files = list(Path(temp_dir).glob("*.log"))
        for log_file in log_files:
            print(f"  Found log file: {log_file.name}")
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"  Content preview: {content[:200]}...")
                
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fixed_log_directory_manager():
    """Test 2: Check if log directory manager now works properly"""
    print("\n" + "=" * 60)
    print("Test 2: Fixed Log Directory Manager")
    print("=" * 60)
    
    # Create multiple temporary directories
    temp_dir1 = tempfile.mkdtemp()
    temp_dir2 = tempfile.mkdtemp()
    temp_dir3 = tempfile.mkdtemp()
    
    print(f"Temp dir 1: {temp_dir1}")
    print(f"Temp dir 2: {temp_dir2}")
    print(f"Temp dir 3: {temp_dir3}")
    
    try:
        # Test 1: Set log directory and create logger
        print("\n--- Test 2.1: Set log directory and create logger ---")
        set_agent_log_directory(temp_dir1)
        print(f"Set log directory to: {get_agent_log_directory()}")
        
        logger1 = AgentLogger(name="DirTestLogger1")
        logger1.info("Message 1 - should be in temp_dir1")
        
        # Test 2: Change log directory and create new logger
        print("\n--- Test 2.2: Change log directory and create new logger ---")
        set_agent_log_directory(temp_dir2)
        print(f"Changed log directory to: {get_agent_log_directory()}")
        
        logger2 = AgentLogger(name="DirTestLogger2")
        logger2.info("Message 2 - should be in temp_dir2")
        
        # Test 3: Change log directory again
        print("\n--- Test 2.3: Change log directory again ---")
        set_agent_log_directory(temp_dir3)
        print(f"Changed log directory to: {get_agent_log_directory()}")
        
        logger3 = AgentLogger(name="DirTestLogger3")
        logger3.info("Message 3 - should be in temp_dir3")
        
        # Check log files in each directory
        print("\n--- Checking log files in each directory ---")
        for i, temp_dir in enumerate([temp_dir1, temp_dir2, temp_dir3], 1):
            print(f"\nDirectory {i} ({temp_dir}):")
            log_files = list(Path(temp_dir).glob("*.log"))
            if log_files:
                for log_file in log_files:
                    print(f"  Found log file: {log_file.name}")
                    with open(log_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"  Content preview: {content[:200]}...")
            else:
                print("  No log files found")
                
    finally:
        # Cleanup
        for temp_dir in [temp_dir1, temp_dir2, temp_dir3]:
            shutil.rmtree(temp_dir, ignore_errors=True)


def test_logger_after_directory_change_fixed():
    """Test 3: Test logger behavior after directory change (fixed)"""
    print("\n" + "=" * 60)
    print("Test 3: Logger After Directory Change (Fixed)")
    print("=" * 60)
    
    temp_dir1 = tempfile.mkdtemp()
    temp_dir2 = tempfile.mkdtemp()
    
    print(f"Temp dir 1: {temp_dir1}")
    print(f"Temp dir 2: {temp_dir2}")
    
    try:
        # Set initial directory and create logger
        set_agent_log_directory(temp_dir1)
        logger = AgentLogger(name="ChangeTestLogger")
        logger.info("Initial message in temp_dir1")
        
        # Change directory
        set_agent_log_directory(temp_dir2)
        logger.info("Message after directory change - should now go to temp_dir2")
        
        # Check both directories
        print(f"\nDirectory 1 ({temp_dir1}):")
        log_files1 = list(Path(temp_dir1).glob("*.log"))
        for log_file in log_files1:
            print(f"  {log_file.name}")
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"  Content: {content}")
        
        print(f"\nDirectory 2 ({temp_dir2}):")
        log_files2 = list(Path(temp_dir2).glob("*.log"))
        for log_file in log_files2:
            print(f"  {log_file.name}")
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"  Content: {content}")
                
    finally:
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)


def test_multiple_loggers_same_name_fixed():
    """Test 4: Test multiple loggers with same name (should work now)"""
    print("\n" + "=" * 60)
    print("Test 4: Multiple Loggers with Same Name (Fixed)")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    try:
        set_agent_log_directory(temp_dir)
        
        # Create multiple loggers with same name - should return same instance
        logger1 = AgentLogger(name="SameNameLogger")
        logger2 = AgentLogger(name="SameNameLogger")
        logger3 = AgentLogger(name="SameNameLogger")
        
        print(f"Logger1 name: {logger1.name}")
        print(f"Logger2 name: {logger2.name}")
        print(f"Logger3 name: {logger3.name}")
        print(f"Are they the same instance? {logger1 is logger2 and logger2 is logger3}")
        
        # Log messages
        logger1.info("Message from logger1")
        logger2.warning("Message from logger2")
        logger3.error("Message from logger3")
        
        # Check log file
        log_file = logger1.log_filepath
        if log_file.exists():
            print(f"\nLog file content:")
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)
                
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_environment_variable_behavior_fixed():
    """Test 5: Test environment variable behavior (fixed)"""
    print("\n" + "=" * 60)
    print("Test 5: Environment Variable Behavior (Fixed)")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Test setting environment variable directly
        os.environ['AGENT_LOG_DIR'] = temp_dir
        print(f"Set AGENT_LOG_DIR environment variable to: {os.environ['AGENT_LOG_DIR']}")
        
        # Create logger without using log_directory_manager
        logger = AgentLogger(name="EnvVarTestLogger")
        logger.info("Message using environment variable")
        
        # Check if log file was created in the right place
        print(f"\nLog file path: {logger.log_filepath}")
        print(f"Expected directory: {temp_dir}")
        print(f"Actual directory: {logger.log_filepath.parent}")
        print(f"Match: {logger.log_filepath.parent == Path(temp_dir)}")
        
        # Check log file content
        if logger.log_filepath.exists():
            with open(logger.log_filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"\nLog content: {content}")
                
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if 'AGENT_LOG_DIR' in os.environ:
            del os.environ['AGENT_LOG_DIR']


def test_proxy_server_scenario():
    """Test 6: Test the proxy_server.py scenario"""
    print("\n" + "=" * 60)
    print("Test 6: Proxy Server Scenario")
    print("=" * 60)
    
    # Simulate the proxy_server.py scenario
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Simulate the proxy_server.py code
        current_file_dir = Path(tempfile.mkdtemp())  # Simulate proxy_server directory
        log_dir = current_file_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Setting log directory to: {log_dir}")
        set_agent_log_directory(str(log_dir))
        
        # Create logger (simulating proxy_server usage)
        logger = AgentLogger(name="ProxyServerLogger")
        logger.info("Proxy server started")
        logger.warning("This is a warning from proxy server")
        logger.error("This is an error from proxy server")
        
        # Check log files
        print(f"\nLog files in {log_dir}:")
        log_files = list(log_dir.glob("*.log"))
        for log_file in log_files:
            print(f"  Found log file: {log_file.name}")
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"  Content: {content}")
                
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(current_file_dir, ignore_errors=True)


def main():
    """Run all fixed tests"""
    print("Starting FIXED AgentLogger and LogDirectoryManager Tests")
    print("=" * 80)
    
    test_fixed_logger_names()
    test_fixed_log_directory_manager()
    test_logger_after_directory_change_fixed()
    test_multiple_loggers_same_name_fixed()
    test_environment_variable_behavior_fixed()
    test_proxy_server_scenario()
    
    print("\n" + "=" * 80)
    print("All fixed tests completed!")


if __name__ == "__main__":
    main()
