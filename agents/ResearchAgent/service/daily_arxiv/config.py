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
    
    # Swiss Tournament Configuration
    SWISS_TOURNAMENT_TOP_N: int = 8  # Number of top papers to select from TOP_K_PAPERS
    SWISS_TOURNAMENT_MODEL: str = "gpt-4.1"  # Model for pairwise comparison
    SWISS_TOURNAMENT_MAX_TOKENS: int = 640000  # Max tokens for LLM calls
    SWISS_TOURNAMENT_TEMPERATURE: float = 0.1  # Temperature for LLM calls
    SWISS_TOURNAMENT_MAX_CONCURRENT: int = 2  # Max concurrent LLM calls
    
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
        
        # Validate Swiss Tournament configurations
        if cls.SWISS_TOURNAMENT_TOP_N <= 0:
            print("❌ SWISS_TOURNAMENT_TOP_N must be greater than 0")
            return False
        
        if cls.SWISS_TOURNAMENT_TOP_N > cls.TOP_K_PAPERS:
            print("❌ SWISS_TOURNAMENT_TOP_N cannot be greater than TOP_K_PAPERS")
            return False
        
        if cls.SWISS_TOURNAMENT_MAX_CONCURRENT <= 0:
            print("❌ SWISS_TOURNAMENT_MAX_CONCURRENT must be greater than 0")
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
        print("--- Swiss Tournament Configuration ---")
        print(f"Swiss Tournament Top N: {cls.SWISS_TOURNAMENT_TOP_N}")
        print(f"Swiss Tournament Model: {cls.SWISS_TOURNAMENT_MODEL}")
        print(f"Swiss Tournament Max Tokens: {cls.SWISS_TOURNAMENT_MAX_TOKENS}")
        print(f"Swiss Tournament Temperature: {cls.SWISS_TOURNAMENT_TEMPERATURE}")
        print(f"Swiss Tournament Max Concurrent: {cls.SWISS_TOURNAMENT_MAX_CONCURRENT}")
        print("==========================================")
    
    @classmethod
    def get_pdf_storage_dir(cls) -> str:
        """Get PDF storage directory, using default if not configured."""
        from agents.ResearchAgent.config import PDF_STROAGE_DIR
        return str(PDF_STROAGE_DIR)
    
    @classmethod
    def get_database_dir(cls) -> Optional[str]:
        """Get database directory, using default if not configured."""
        from agents.ResearchAgent.config import ARXIV_DATABASE_DIR
        return str(ARXIV_DATABASE_DIR)


if __name__ == "__main__":
    DailyArxivConfig.print_config()
    DailyArxivConfig.validate()
