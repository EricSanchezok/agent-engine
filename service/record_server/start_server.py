#!/usr/bin/env python3
"""
Start script for Record Memory Server

This script starts the FastAPI server on port 5050.
"""
from pathlib import Path
from agent_engine.agent_logger import set_agent_log_directory

current_file_dir = Path(__file__).parent
log_dir = current_file_dir / 'logs'
set_agent_log_directory(str(log_dir))

import uvicorn
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import app

if __name__ == "__main__":
    print("Starting Record Memory Server on port 5050...")
    print("Server will be available at: http://localhost:5050")
    print("API documentation will be available at: http://localhost:5050/docs")
    
    uvicorn.run(
        "server:app",  # Use import string for reload support
        host="0.0.0.0",
        port=5050,
        reload=False,  # Enable auto-reload for development
        log_level="info"
    )
