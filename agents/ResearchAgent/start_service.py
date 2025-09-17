#!/usr/bin/env python3
"""
Start Inno-Researcher Service
Simple script to start the Inno-Researcher service.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.agent_logger import agent_logger

if __name__ == "__main__":
    try:
        agent_logger.info("Starting Inno-Researcher service...")
        from agents.ResearchAgent.main import main
        main()
    except KeyboardInterrupt:
        agent_logger.info("Service stopped by user")
    except Exception as e:
        agent_logger.error(f"Failed to start service: {e}")
        sys.exit(1)
