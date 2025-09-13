"""
Configuration file for Arxiv Database Preloader

Set your Qz API credentials here or use environment variables.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class PreloadConfig:
    """Configuration for the Arxiv preloader."""
    
    REMOTE = os.getenv("REMOTE", "false").lower() == "true" 

    # Qz API Configuration
    QZ_API_KEY: str = os.getenv("INF_API_KEY", "")
    QZ_BASE_URL: str = "http://eric-vpn.cpolar.top/r/eric_qwen3_embedding_8b" if REMOTE else "https://jpep8ehg8opgckcqkcc5e5eg9b8ecbcm.openapi-qb.sii.edu.cn"
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "eric-qwen3-embedding-8b"
    MAX_CONCURRENT_EMBEDDINGS: int = 16
    
    # Database Configuration
    DATABASE_NAME: str = "arxiv_papers"
    DATABASE_DIR: Optional[str] = None  # Will use default if None
    
    # Processing Configuration
    DEFAULT_NUM_DAYS: int = 3000
    DELAY_BETWEEN_DAYS: float = 5.0  # seconds
    
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
