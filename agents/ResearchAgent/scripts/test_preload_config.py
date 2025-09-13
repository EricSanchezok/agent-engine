"""
Test script for Arxiv Database Preloader configuration

This script tests the configuration and basic connectivity without
actually running the full preload process.
"""

import asyncio
from preload_config import PreloadConfig
from agent_engine.llm_client.qz_client import QzClient
from agents.ResearchAgent.arxiv_database import ArxivDatabase


async def test_qz_client():
    """Test QzClient connectivity."""
    print("Testing QzClient connectivity...")
    
    try:
        client = QzClient(
            api_key=PreloadConfig.QZ_API_KEY,
            base_url=PreloadConfig.QZ_BASE_URL
        )
        
        # Test embedding generation
        test_text = "This is a test paper summary for embedding generation."
        embedding = await client.get_embeddings(
            model_name=PreloadConfig.EMBEDDING_MODEL,
            text=test_text
        )
        
        if embedding is not None:
            print(f"✅ QzClient test successful - embedding dimension: {len(embedding)}")
            return True
        else:
            print("❌ QzClient test failed - no embedding returned")
            return False
            
    except Exception as e:
        print(f"❌ QzClient test failed: {e}")
        return False
    finally:
        await client.close()


def test_arxiv_database():
    """Test ArxivDatabase initialization."""
    print("Testing ArxivDatabase initialization...")
    
    try:
        db = ArxivDatabase(
            name=PreloadConfig.DATABASE_NAME,
            persist_dir=PreloadConfig.DATABASE_DIR
        )
        
        # Test basic operations
        count = db.count()
        stats = db.get_stats()
        
        print(f"✅ ArxivDatabase test successful - current count: {count}")
        print(f"   Database stats: {stats}")
        return True
        
    except Exception as e:
        print(f"❌ ArxivDatabase test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("=== Arxiv Preloader Configuration Test ===\n")
    
    # Test configuration
    if not PreloadConfig.validate():
        print("❌ Configuration validation failed")
        return
    
    PreloadConfig.print_config()
    print()
    
    # Test components
    tests_passed = 0
    total_tests = 2
    
    # Test QzClient
    if await test_qz_client():
        tests_passed += 1
    print()
    
    # Test ArxivDatabase
    if test_arxiv_database():
        tests_passed += 1
    print()
    
    # Summary
    print("=== Test Summary ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! You're ready to run the preloader.")
    else:
        print("❌ Some tests failed. Please check your configuration.")


if __name__ == "__main__":
    asyncio.run(main())
