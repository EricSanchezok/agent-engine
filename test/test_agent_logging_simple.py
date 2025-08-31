#!/usr/bin/env python3
"""
Simple test script to verify agent logging directory setup
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.agent_logger import set_agent_log_directory, get_agent_log_directory, AgentLogger


def test_arxiv_agent_logging():
    """Test ArxivSearchAgent logging directory setup"""
    print("Testing ArxivSearchAgent logging directory setup...")
    
    # Import the config to get LOG_DIR
    from agents.ArxivSearchAgent.config import LOG_DIR
    print(f"ArxivSearchAgent LOG_DIR: {LOG_DIR}")
    
    # Set the log directory
    set_agent_log_directory(str(LOG_DIR))
    print(f"Set agent log directory to: {get_agent_log_directory()}")
    
    # Create various loggers
    agent_logger = AgentLogger("ArxivSearchAgent.Main")
    skill_logger = AgentLogger("ArxivSearchAgent.SkillIdentifier")
    llm_logger = AgentLogger("ArxivSearchAgent.LLMClient")
    
    # Log some messages
    agent_logger.info("ArxivSearchAgent initialized")
    skill_logger.info("Skill identification started")
    llm_logger.info("LLM client connected")
    
    print(f"✅ Logs should be written to: {LOG_DIR}")
    
    # Check if log files were created
    log_files = list(LOG_DIR.glob("*.log"))
    print(f"📁 Found {len(log_files)} log files:")
    for log_file in log_files:
        print(f"   └─ {log_file.name}")


if __name__ == "__main__":
    test_arxiv_agent_logging()
