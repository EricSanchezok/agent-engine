"""
Configuration file for Arxiv Database Preloader

Set your Qz API credentials here or use environment variables.
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

from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class PreloadConfig:
    """Configuration for the Arxiv preloader."""
    
    USE_ERIC_VPN = os.getenv("USE_ERIC_VPN", "false").lower() == "true" 
    ERIC_VPN_URL = os.getenv("ERIC_VPN_URL", "http://eric-vpn.cpolar.top/r/")
    MODEL_PROXY_ROUTE = os.getenv("QWEN3_EMBEDDING_8B_H100_PROXY_ROUTE", "eric_qwen3_embedding_8b_h100")
    MODEL_URL = os.getenv("QWEN3_EMBEDDING_8B_H100_URL", "https://jpep8ehg8opgckcqkcc5e5eg9b8ecbcm.openapi-qb.sii.edu.cn")

    # Qz API Configuration
    QZ_API_KEY: str = os.getenv("QZ_API_KEY", "")
    QZ_BASE_URL: str = ERIC_VPN_URL + MODEL_PROXY_ROUTE if USE_ERIC_VPN else MODEL_URL
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "eric-qwen3-embedding-8b"
    MAX_CONCURRENT_EMBEDDINGS: int = 32
    
    # Database Configuration
    DATABASE_NAME: str = "arxiv_papers"
    DATABASE_DIR: Optional[str] = None  # Will use default if None
    
    # Processing Configuration
    DEFAULT_NUM_DAYS: int = 3000
    DELAY_BETWEEN_DAYS: float = 10.0  # seconds
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.QZ_API_KEY:
            print("❌ QZ_API_KEY is not set. Please set it as environment variable or in config.")
            return False
        
        if not cls.QZ_BASE_URL:
            print("❌ QZ_BASE_URL is not set. Please set it as environment variable or in config.")
            return False
        
        print("✅ Configuration validation passed")
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)."""
        print("=== Preload Configuration ===")
        print(f"QZ_API_KEY: {'***' + cls.QZ_API_KEY[-4:] if cls.QZ_API_KEY else 'Not set'}")
        print(f"QZ_BASE_URL: {cls.QZ_BASE_URL or 'Not set'}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Max Concurrent Embeddings: {cls.MAX_CONCURRENT_EMBEDDINGS}")
        print(f"Database Name: {cls.DATABASE_NAME}")
        print(f"Default Num Days: {cls.DEFAULT_NUM_DAYS}")
        print("=============================")


# Example usage:
if __name__ == "__main__":
    # You can set your credentials here directly (not recommended for production)
    # PreloadConfig.QZ_API_KEY = "your_api_key_here"
    # PreloadConfig.QZ_BASE_URL = "https://your-endpoint.com"
    
    PreloadConfig.print_config()
    PreloadConfig.validate()
