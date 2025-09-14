"""
Configuration for Arxiv Sync Service
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class ArxivSyncConfig:
    """Configuration for the Arxiv sync service."""
    
    # VPN Configuration
    USE_ERIC_VPN = os.getenv("USE_ERIC_VPN", "false").lower() == "true"
    ERIC_VPN_URL = os.getenv("ERIC_VPN_URL", "http://eric-vpn.cpolar.top/r/")
    MODEL_PROXY_ROUTE = os.getenv("QWEN3_EMBEDDING_8B_H100_PROXY_ROUTE", "eric_qwen3_embedding_8b_h100")
    MODEL_URL = os.getenv("QWEN3_EMBEDDING_8B_H100_URL", "https://jpep8ehg8opgckcqkcc5e5eg9b8ecbcm.openapi-qb.sii.edu.cn")

    # Qz API Configuration
    QZ_API_KEY: str = os.getenv("QZ_API_KEY", "")
    QZ_BASE_URL: str = ERIC_VPN_URL + MODEL_PROXY_ROUTE if USE_ERIC_VPN else MODEL_URL
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "eric-qwen3-embedding-8b"
    MAX_CONCURRENT_EMBEDDINGS: int = 16
    
    # Database Configuration
    DATABASE_NAME: str = "arxiv_papers"
    DATABASE_DIR: Optional[str] = None  # Will use default if None
    
    # Sync Configuration
    SYNC_INTERVAL_MINUTES: int = 15  # Check every 15 minutes
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_DELAY_SECONDS: float = 5.0
    DELAY_BETWEEN_DAYS: float = 0.1  # seconds
    
    # Service Configuration
    SERVICE_NAME: str = "ArxivSyncService"
    LOG_LEVEL: str = "INFO"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.QZ_API_KEY:
            print("❌ QZ_API_KEY is not set. Please set it as environment variable.")
            return False
        
        if not cls.QZ_BASE_URL:
            print("❌ QZ_BASE_URL is not set. Please set it as environment variable.")
            return False
        
        print("✅ ArxivSync configuration validation passed")
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)."""
        print("=== ArxivSync Configuration ===")
        print(f"QZ_API_KEY: {'***' + cls.QZ_API_KEY[-4:] if cls.QZ_API_KEY else 'Not set'}")
        print(f"QZ_BASE_URL: {cls.QZ_BASE_URL or 'Not set'}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Max Concurrent Embeddings: {cls.MAX_CONCURRENT_EMBEDDINGS}")
        print(f"Database Name: {cls.DATABASE_NAME}")
        print(f"Sync Interval: {cls.SYNC_INTERVAL_MINUTES} minutes")
        print(f"Max Retry Attempts: {cls.MAX_RETRY_ATTEMPTS}")
        print(f"Use Eric VPN: {cls.USE_ERIC_VPN}")
        print("===============================")


if __name__ == "__main__":
    ArxivSyncConfig.print_config()
    ArxivSyncConfig.validate()
