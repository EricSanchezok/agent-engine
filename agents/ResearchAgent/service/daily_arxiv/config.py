"""
Configuration for Daily arXiv Service
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class DailyArxivConfig:
    """Configuration for the daily arXiv service."""
    
    # VPN Configuration
    USE_ERIC_VPN = os.getenv("USE_ERIC_VPN", "false").lower() == "true"
    ERIC_VPN_URL = os.getenv("ERIC_VPN_URL", "http://eric-vpn.cpolar.top/r/")
    EMBEDDING_PROXY_ROUTE = os.getenv("QWEN3_EMBEDDING_8B_H100_PROXY_ROUTE", "eric_qwen3_embedding_8b_h100")
    RERANKER_PROXY_ROUTE = os.getenv("QWEN3_RERANKER_8B_H100_PROXY_ROUTE", "eric_qwen3_reranker_8b_h100")
    EMBEDDING_MODEL_URL = os.getenv("QWEN3_EMBEDDING_8B_H100_URL", "https://jpep8ehg8opgckcqkcc5e5eg9b8ecbcm.openapi-qb.sii.edu.cn")
    RERANKER_MODEL_URL = os.getenv("QWEN3_RERANKER_8B_H100_URL", "https://jpep8ehg8opgckcqkcc5e5eg9b8ecbcm.openapi-qb.sii.edu.cn")

    # Qz API Configuration
    QZ_API_KEY: str = os.getenv("QZ_API_KEY", "")
    EMBEDDING_BASE_URL: str = ERIC_VPN_URL + EMBEDDING_PROXY_ROUTE if USE_ERIC_VPN else EMBEDDING_MODEL_URL
    RERANKER_BASE_URL: str = ERIC_VPN_URL + RERANKER_PROXY_ROUTE if USE_ERIC_VPN else RERANKER_MODEL_URL
    
    # Model Configuration
    EMBEDDING_MODEL: str = "eric-qwen3-embedding-8b"
    RERANKER_MODEL: str = "eric-qwen3-reranker-8b"
    
    # Database Configuration
    ARXIV_DATABASE_NAME: str = "arxiv_papers"
    QIJI_DATABASE_NAME: str = "qiji_memory"
    DATABASE_DIR: Optional[str] = None  # Will use default if None
    
    # PDF Storage Configuration
    PDF_STORAGE_DIR: Optional[str] = None  # Will use default if None
    
    # Service Configuration
    SERVICE_NAME: str = "DailyArxivService"
    LOG_LEVEL: str = "INFO"
    
    # Processing Configuration
    TOP_K_PAPERS: int = int(os.getenv("DAILY_ARXIV_TOP_K", "16"))
    MAX_CONCURRENT_DOWNLOADS: int = int(os.getenv("DAILY_ARXIV_MAX_CONCURRENT", "5"))
    MAX_CONCURRENT_EMBEDDINGS: int = int(os.getenv("DAILY_ARXIV_MAX_CONCURRENT_EMBEDDINGS", "16"))
    
    # Date Configuration
    TARGET_DATE: Optional[str] = os.getenv("DAILY_ARXIV_TARGET_DATE")  # YYYY-MM-DD format
    
    # Filtering Configuration
    MIN_PAPERS_REQUIRED: int = int(os.getenv("DAILY_ARXIV_MIN_PAPERS", "1"))
    REQUIRE_VECTORS: bool = os.getenv("DAILY_ARXIV_REQUIRE_VECTORS", "false").lower() == "true"
    
    # Download Configuration
    MAX_DOWNLOAD_RETRIES: int = int(os.getenv("DAILY_ARXIV_MAX_RETRIES", "3"))
    DOWNLOAD_TIMEOUT_SECONDS: int = int(os.getenv("DAILY_ARXIV_DOWNLOAD_TIMEOUT", "300"))
    DOWNLOAD_DELAY_SECONDS: float = float(os.getenv("DAILY_ARXIV_DOWNLOAD_DELAY", "1.0"))
    
    # Performance Configuration
    BATCH_SIZE: int = int(os.getenv("DAILY_ARXIV_BATCH_SIZE", "100"))
    MEMORY_LIMIT_MB: int = int(os.getenv("DAILY_ARXIV_MEMORY_LIMIT_MB", "1024"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.QZ_API_KEY:
            print("❌ QZ_API_KEY is not set. Please set it as environment variable.")
            return False
        
        if not cls.EMBEDDING_BASE_URL:
            print("❌ EMBEDDING_BASE_URL is not set. Please set it as environment variable.")
            return False
        
        if not cls.RERANKER_BASE_URL:
            print("❌ RERANKER_BASE_URL is not set. Please set it as environment variable.")
            return False
        
        # Validate numeric configurations
        if cls.TOP_K_PAPERS <= 0:
            print("❌ TOP_K_PAPERS must be greater than 0")
            return False
        
        if cls.MAX_CONCURRENT_DOWNLOADS <= 0:
            print("❌ MAX_CONCURRENT_DOWNLOADS must be greater than 0")
            return False
        
        if cls.MAX_CONCURRENT_EMBEDDINGS <= 0:
            print("❌ MAX_CONCURRENT_EMBEDDINGS must be greater than 0")
            return False
        
        print("✅ DailyArxiv configuration validation passed")
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)."""
        print("=== Daily arXiv Service Configuration ===")
        print(f"QZ_API_KEY: {'***' + cls.QZ_API_KEY[-4:] if cls.QZ_API_KEY else 'Not set'}")
        print(f"Embedding Base URL: {cls.EMBEDDING_BASE_URL or 'Not set'}")
        print(f"Reranker Base URL: {cls.RERANKER_BASE_URL or 'Not set'}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Reranker Model: {cls.RERANKER_MODEL}")
        print(f"Arxiv Database Name: {cls.ARXIV_DATABASE_NAME}")
        print(f"Qiji Database Name: {cls.QIJI_DATABASE_NAME}")
        print(f"PDF Storage Dir: {cls.PDF_STORAGE_DIR or 'Default'}")
        print(f"Top K Papers: {cls.TOP_K_PAPERS}")
        print(f"Max Concurrent Downloads: {cls.MAX_CONCURRENT_DOWNLOADS}")
        print(f"Max Concurrent Embeddings: {cls.MAX_CONCURRENT_EMBEDDINGS}")
        print(f"Target Date: {cls.TARGET_DATE or 'Today'}")
        print(f"Min Papers Required: {cls.MIN_PAPERS_REQUIRED}")
        print(f"Require Vectors: {cls.REQUIRE_VECTORS}")
        print(f"Max Download Retries: {cls.MAX_DOWNLOAD_RETRIES}")
        print(f"Download Timeout: {cls.DOWNLOAD_TIMEOUT_SECONDS}s")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Memory Limit: {cls.MEMORY_LIMIT_MB}MB")
        print(f"Use Eric VPN: {cls.USE_ERIC_VPN}")
        print("==========================================")
    
    @classmethod
    def get_pdf_storage_dir(cls) -> str:
        """Get PDF storage directory, using default if not configured."""
        if cls.PDF_STORAGE_DIR:
            return cls.PDF_STORAGE_DIR
        
        # Use default path: agents/ResearchAgent/database/arxiv_pdfs
        from agent_engine.utils import get_current_file_dir
        current_dir = get_current_file_dir()
        return str(current_dir.parent.parent / 'database' / 'arxiv_pdfs')
    
    @classmethod
    def get_database_dir(cls) -> Optional[str]:
        """Get database directory, using default if not configured."""
        if cls.DATABASE_DIR:
            return cls.DATABASE_DIR
        
        # Use default path: agents/ResearchAgent/database
        from agent_engine.utils import get_current_file_dir
        current_dir = get_current_file_dir()
        return str(current_dir.parent.parent / 'database')


if __name__ == "__main__":
    DailyArxivConfig.print_config()
    DailyArxivConfig.validate()
