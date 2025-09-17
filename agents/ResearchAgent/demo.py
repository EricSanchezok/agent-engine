#!/usr/bin/env python3
"""
Inno-Researcher Demo Script
Demonstrates how to use the Inno-Researcher service and client.
"""

import asyncio
import subprocess
import time
import sys
from pathlib import Path

from agent_engine.agent_logger import agent_logger


def run_command(cmd: str, cwd: str = None) -> subprocess.Popen:
    """Run a command in a subprocess"""
    agent_logger.info(f"Running command: {cmd}")
    return subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


def check_service_health(url: str = "http://localhost:8000") -> bool:
    """Check if service is running"""
    try:
        import requests
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Main demo function"""
    print("🚀 Inno-Researcher Demo")
    print("=" * 50)
    
    # Check if service is already running
    if check_service_health():
        print("✅ Service is already running!")
        print("You can now use the client:")
        print("  uv run agents/ResearchAgent/client.py")
        print("\nOr test concurrency:")
        print("  uv run agents/ResearchAgent/test_concurrency.py")
        return
    
    print("🔧 Starting Inno-Researcher service...")
    print("This will start the service in the background.")
    print("You can then use the client to interact with it.")
    
    # Start the service
    service_process = run_command("uv run agents/ResearchAgent/start_service.py")
    
    # Wait a bit for service to start
    print("⏳ Waiting for service to start...")
    time.sleep(5)
    
    # Check if service started successfully
    if check_service_health():
        print("✅ Service started successfully!")
        print("\n📋 Next steps:")
        print("1. Open a new terminal")
        print("2. Run the client: uv run agents/ResearchAgent/client.py")
        print("3. Or test concurrency: uv run agents/ResearchAgent/test_concurrency.py")
        print("\n🛑 To stop the service, press Ctrl+C")
        
        try:
            # Keep the service running
            service_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping service...")
            service_process.terminate()
            service_process.wait()
            print("✅ Service stopped")
    else:
        print("❌ Service failed to start")
        print("Check the logs for errors")
        service_process.terminate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted")
    except Exception as e:
        agent_logger.error(f"Demo error: {e}")
        print(f"❌ Demo failed: {e}")
