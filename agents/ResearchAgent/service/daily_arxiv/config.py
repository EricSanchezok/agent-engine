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
    ERIC_VPN_URL = os.getenv("ERIC_VPN_URL", "http://eric-vpn.vip.cpolar.cn/r/")
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
    
    # Processing Configuration
    TOP_K_PAPERS: int = 16
    MAX_CONCURRENT_DOWNLOADS: int = 4
    MAX_CONCURRENT_EMBEDDINGS: int = 16
    
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
        print(f"Top K Papers: {cls.TOP_K_PAPERS}")
        print(f"Max Concurrent Downloads: {cls.MAX_CONCURRENT_DOWNLOADS}")
        print(f"Max Concurrent Embeddings: {cls.MAX_CONCURRENT_EMBEDDINGS}")
        print(f"Use Eric VPN: {cls.USE_ERIC_VPN}")
        print("==========================================")
    
    @classmethod
    def get_pdf_storage_dir(cls) -> str:
        """Get PDF storage directory, using default if not configured."""
        from agent_engine.utils import get_current_file_dir
        current_dir = get_current_file_dir()
        return str(current_dir.parent.parent / 'database' / 'arxiv_pdfs')
    
    @classmethod
    def get_database_dir(cls) -> Optional[str]:
        """Get database directory, using default if not configured."""
        from agent_engine.utils import get_current_file_dir
        current_dir = get_current_file_dir()
        return str(current_dir.parent.parent / 'database')


if __name__ == "__main__":
    DailyArxivConfig.print_config()
    DailyArxivConfig.validate()
