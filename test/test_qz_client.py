"""
Test script for QzClient embedding model functionality
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.llm_client import QzClient
from agent_engine.agent_logger import AgentLogger

logger = AgentLogger(__name__)

async def test_qz_client_embedding():
    """Test QzClient embedding functionality"""
    
    # Get API key from environment
    api_key = os.getenv("INF_API_KEY")
    if not api_key:
        logger.error("❌ INF_API_KEY not found in environment variables")
        return
    
    logger.info("🚀 Starting QzClient embedding tests")

    base_url = "https://jpep8ehg8opgckcqkcc5e5eg9b8ecbcm.openapi-qb.sii.edu.cn"
    base_url = "http://eric-vpn.cpolar.top/r/eric_qwen3_embedding_8b"
    
    # Initialize QzClient
    try:
        client = QzClient(api_key=api_key, base_url=base_url)
        logger.info("✅ QzClient initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize QzClient: {e}")
        return
    
    try:
        # Test 1: Single text embedding
        logger.info("\n=== Test 1: Single text embedding ===")
        test_text = "Hello, this is a test sentence for embedding."
        
        embedding = await client.embedding(test_text, "eric-qwen3-embedding-8b")
        if embedding:
            logger.info(f"✅ Single embedding successful, dimension: {len(embedding)}")
            logger.info(f"First 5 values: {embedding[:5]}")
        else:
            logger.error("❌ Single embedding failed")
        
        # Test 2: Multiple texts embedding
        logger.info("\n=== Test 2: Multiple texts embedding ===")
        test_texts = [
            "This is the first test sentence.",
            "This is the second test sentence.",
            "This is the third test sentence."
        ]
        
        embeddings = await client.embedding(test_texts, "eric-qwen3-embedding-8b")
        if embeddings:
            logger.info(f"✅ Multiple embeddings successful, count: {len(embeddings)}")
            for i, emb in enumerate(embeddings):
                logger.info(f"Text {i+1} dimension: {len(emb)}")
        else:
            logger.error("❌ Multiple embeddings failed")
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
    
    finally:
        # Close the client
        await client.close()
        logger.info("✅ QzClient connection closed")

async def main():
    """Main test function"""
    logger.info("🧪 Starting QzClient comprehensive tests")
    
    await test_qz_client_embedding()
    
    logger.info("🏁 All tests completed")

if __name__ == "__main__":
    asyncio.run(main())
