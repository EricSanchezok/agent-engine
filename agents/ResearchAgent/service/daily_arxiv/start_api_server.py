#!/usr/bin/env python3
"""
Start the Daily arXiv Paper Reports API Server
"""

import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.agent_logger import set_agent_log_directory

current_file_dir = Path(__file__).parent
log_dir = current_file_dir / 'logs'
set_agent_log_directory(str(log_dir))


from agents.ResearchAgent.service.daily_arxiv.api_server import start_server

def main():
    parser = argparse.ArgumentParser(description='Start Daily arXiv Paper Reports API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("Starting Daily arXiv Paper Reports API Server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print("=" * 50)
    
    try:
        start_server(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
