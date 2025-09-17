"""
Start Research Agent Server

This script starts the Research Agent Server with proper configuration.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

import uvicorn
from agent_engine.agent_logger.agent_logger import AgentLogger

logger = AgentLogger(__name__)


def main():
    """Start the Research Agent Server"""
    
    # Check required environment variables
    azure_api_key = os.getenv("AZURE_API_KEY")
    if not azure_api_key:
        print("❌ Error: AZURE_API_KEY environment variable is required")
        print("Please set your Azure OpenAI API key:")
        print("  export AZURE_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    # Optional configuration
    azure_base_url = os.getenv("AZURE_BASE_URL", "https://gpt.yunstorm.com/")
    azure_api_version = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
    model_name = os.getenv("MODEL_NAME", "gpt-4.1")
    
    logger.info("🚀 Starting Research Agent Server")
    logger.info(f"📡 Host: {host}")
    logger.info(f"🔌 Port: {port}")
    logger.info(f"🔄 Reload: {reload}")
    logger.info(f"🤖 Model: {model_name}")
    logger.info(f"🌐 Azure endpoint: {azure_base_url}")
    
    print("=" * 50)
    print("🤖 Research Agent Server")
    print("=" * 50)
    print(f"📡 Server will start on: http://{host}:{port}")
    print(f"🤖 Model: {model_name}")
    print(f"🌐 Azure endpoint: {azure_base_url}")
    print("=" * 50)
    print("📋 Available endpoints:")
    print(f"  GET  http://{host}:{port}/health")
    print(f"  POST http://{host}:{port}/chat")
    print(f"  POST http://{host}:{port}/chat/stream")
    print(f"  GET  http://{host}:{port}/sessions")
    print("=" * 50)
    print("💡 To test the server, run:")
    print(f"  python client.py --base-url http://{host}:{port}")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
