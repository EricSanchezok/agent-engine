#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ICU Agent Server Startup Script
Easy way to start the ICU Agent backend server with proper configuration.
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import flask_cors
        print("Dependencies check: OK")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def start_server():
    """Start the ICU Agent backend server"""
    if not check_dependencies():
        return False
    
    print("Starting ICU Agent Backend Server...")
    print("Server URL: http://localhost:5000")
    print("Frontend URL: file://" + str(Path("index.html").absolute()))
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("ICU Agent - Intelligent ICU Assistant")
    print("====================================")
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("Error: app.py not found. Please run this script from the web/icu_agent directory.")
        sys.exit(1)
    
    # Start the server
    success = start_server()
    
    if not success:
        sys.exit(1)
