from dotenv import load_dotenv
import os
from pprint import pprint
import asyncio

# Agent Engine imports
from agent_engine.utils import get_current_file_dir
from agent_engine.llm_client import QzClient
from agent_engine.agent_logger import AgentLogger
from agent_engine.memory.e_memory import EMemory

load_dotenv()

logger = AgentLogger(__name__)

USE_ERIC_VPN = os.getenv('USE_ERIC_VPN', 'false').lower() == 'true'

class QijiLibrary:
    def __init__(self):
        self.embedding_client = QzClient(api_key=os.getenv('QZ_API_KEY'), base_url=os.getenv('QWEN3_EMBEDDING_8B_H100_URL') if not USE_ERIC_VPN else os.getenv('ERIC_VPN_URL') + os.getenv('QWEN3_EMBEDDING_8B_H100_PROXY_ROUTE'))
        self.reranker_client = QzClient(api_key=os.getenv('QZ_API_KEY'), base_url=os.getenv('QWEN3_RERANKER_8B_H100_URL') if not USE_ERIC_VPN else os.getenv('ERIC_VPN_URL') + os.getenv('QWEN3_RERANKER_8B_H100_PROXY_ROUTE'))
        self.embedding_model = "eric-qwen3-embedding-8b"
        self.reranker_model = "eric-qwen3-reranker-8b"
        


async def test():
    qiji_library = QijiLibrary()
    documents = [
        "Document about machine learning",
        "Document about cooking recipes", 
        "Document about artificial intelligence"
    ]

    results = await qiji_library.reranker_client.rerank(
        model_name=qiji_library.reranker_model,
        query="machine learning algorithms",
        documents=documents,
        top_n=2
    )

    pprint(results)


if __name__ == "__main__":
    asyncio.run(test())