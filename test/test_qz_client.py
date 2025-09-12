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
    
    # Initialize QzClient
    try:
        client = QzClient(api_key=api_key, base_url="https://jpep8ehg8opgckcqkcc5e5eg9b8ecbcm.openapi-qb.sii.edu.cn")
        logger.info("✅ QzClient initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize QzClient: {e}")
        return
    
    try:
        # Test 1: Single text embedding
        logger.info("\n=== Test 1: Single text embedding ===")
        test_text = "Hello, this is a test sentence for embedding."
        
        embedding = await client.embedding(test_text)
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
        
        embeddings = await client.embedding(test_texts)
        if embeddings:
            logger.info(f"✅ Multiple embeddings successful, count: {len(embeddings)}")
            for i, emb in enumerate(embeddings):
                logger.info(f"Text {i+1} dimension: {len(emb)}")
        else:
            logger.error("❌ Multiple embeddings failed")
        
        # Test 3: Different model names
        logger.info("\n=== Test 3: Different model names ===")
        models_to_test = [
            "text-embedding-ada-002",
            "text-embedding-3-small", 
            "text-embedding-3-large"
        ]
        
        for model_name in models_to_test:
            logger.info(f"Testing model: {model_name}")
            embedding = await client.get_embeddings(
                model_name=model_name,
                text="Test text for different model"
            )
            if embedding:
                logger.info(f"✅ Model {model_name} successful, dimension: {len(embedding)}")
            else:
                logger.warning(f"⚠️ Model {model_name} failed or not supported")
        
        # Test 4: Edge cases
        logger.info("\n=== Test 4: Edge cases ===")
        
        # Empty string
        empty_embedding = await client.embedding("")
        if empty_embedding:
            logger.info("✅ Empty string embedding successful")
        else:
            logger.warning("⚠️ Empty string embedding failed")
        
        # Very long text
        long_text = "This is a very long text. " * 100
        long_embedding = await client.embedding(long_text)
        if long_embedding:
            logger.info("✅ Long text embedding successful")
        else:
            logger.warning("⚠️ Long text embedding failed")
        
        # Special characters
        special_text = "Hello! @#$%^&*()_+ 中文测试 🚀"
        special_embedding = await client.embedding(special_text)
        if special_embedding:
            logger.info("✅ Special characters embedding successful")
        else:
            logger.warning("⚠️ Special characters embedding failed")
        
        # Test 5: Error handling
        logger.info("\n=== Test 5: Error handling ===")
        
        # Test with invalid model
        invalid_embedding = await client.get_embeddings(
            model_name="invalid-model-name",
            text="Test text"
        )
        if invalid_embedding:
            logger.warning("⚠️ Invalid model unexpectedly succeeded")
        else:
            logger.info("✅ Invalid model correctly failed")
        
        logger.info("\n🎉 All QzClient embedding tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
    
    finally:
        # Close the client
        await client.close()
        logger.info("✅ QzClient connection closed")

async def test_qz_client_chat():
    """Test QzClient chat functionality (if supported)"""
    
    api_key = os.getenv("INF_API_KEY")
    if not api_key:
        logger.error("❌ INF_API_KEY not found in environment variables")
        return
    
    logger.info("\n🚀 Testing QzClient chat functionality")
    
    try:
        client = QzClient(api_key=api_key)
        
        # Test chat functionality
        response = await client.chat(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello, can you tell me what you are?",
            model_name="gpt-3.5-turbo"
        )
        
        if response:
            logger.info(f"✅ Chat successful: {response}")
        else:
            logger.warning("⚠️ Chat failed or not supported by this endpoint")
        
        await client.close()
        
    except Exception as e:
        logger.error(f"❌ Chat test failed: {e}")

async def main():
    """Main test function"""
    logger.info("🧪 Starting QzClient comprehensive tests")
    
    # Test embedding functionality (primary focus)
    await test_qz_client_embedding()
    
    # Test chat functionality (secondary)
    await test_qz_client_chat()
    
    logger.info("🏁 All tests completed")

if __name__ == "__main__":
    asyncio.run(main())
