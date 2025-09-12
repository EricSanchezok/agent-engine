"""
Test script for QzClient through eric-vpn proxy
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

async def test_qz_client_through_proxy():
    """Test QzClient through eric-vpn proxy"""
    
    # Get API key from environment
    api_key = os.getenv("INF_API_KEY")
    if not api_key:
        logger.error("❌ INF_API_KEY not found in environment variables")
        return
    
    logger.info("🚀 Testing QzClient through eric-vpn proxy")
    
    # Test different proxy configurations
    proxy_configs = [
        {
            "name": "Direct connection (original)",
            "base_url": "https://jpep8ehg8opgckcqkcc5e5eg9b8ecbcm.openapi-qb.sii.edu.cn",
            "description": "Direct connection to the API"
        },
        {
            "name": "Through eric-vpn proxy",
            "base_url": "http://localhost:3000/r/qz_api",  # Assuming proxy runs on localhost:3000
            "description": "Through eric-vpn proxy at localhost:3000"
        },
        {
            "name": "Through eric-vpn proxy (server IP)",
            "base_url": "http://10.244.12.219:3000/r/qz_api",  # Assuming server IP
            "description": "Through eric-vpn proxy at server IP"
        }
    ]
    
    for config in proxy_configs:
        logger.info(f"\n=== Testing: {config['name']} ===")
        logger.info(f"Description: {config['description']}")
        logger.info(f"Base URL: {config['base_url']}")
        
        try:
            # Initialize QzClient with proxy URL
            client = QzClient(api_key=api_key, base_url=config['base_url'])
            logger.info("✅ QzClient initialized successfully")
            
            # Test embedding
            test_text = "Hello, this is a test sentence for embedding through proxy."
            
            embedding = await client.embedding(test_text, "eric-qwen3-embedding-8b")
            if embedding:
                logger.info(f"✅ Embedding successful, dimension: {len(embedding)}")
                logger.info(f"First 5 values: {embedding[:5]}")
            else:
                logger.error("❌ Embedding failed")
            
            await client.close()
            logger.info("✅ Connection closed successfully")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            logger.error(f"   This might be expected if the proxy is not running")

async def test_proxy_health():
    """Test if eric-vpn proxy is running and accessible"""
    
    import httpx
    
    proxy_urls = [
        "http://localhost:3000",
        "http://10.244.12.219:3000"
    ]
    
    for proxy_url in proxy_urls:
        logger.info(f"\n=== Testing proxy health: {proxy_url} ===")
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Test health endpoint
                response = await client.get(f"{proxy_url}/health")
                if response.status_code == 200:
                    logger.info(f"✅ Proxy health check successful: {response.json()}")
                    
                    # Test available routes
                    routes_response = await client.get(f"{proxy_url}/")
                    if routes_response.status_code == 200:
                        routes_data = routes_response.json()
                        logger.info(f"✅ Available routes: {routes_data.get('routes', [])}")
                        
                        if 'qz_api' in routes_data.get('routes', []):
                            logger.info("✅ qz_api route is available")
                        else:
                            logger.warning("⚠️ qz_api route not found in available routes")
                    else:
                        logger.warning(f"⚠️ Could not get routes info: {routes_response.status_code}")
                else:
                    logger.error(f"❌ Proxy health check failed: {response.status_code}")
                    
        except httpx.ConnectError:
            logger.error(f"❌ Cannot connect to proxy at {proxy_url}")
        except Exception as e:
            logger.error(f"❌ Proxy test failed: {e}")

async def test_direct_proxy_call():
    """Test direct API call through proxy using httpx"""
    
    api_key = os.getenv("INF_API_KEY")
    if not api_key:
        logger.error("❌ INF_API_KEY not found in environment variables")
        return
    
    logger.info("\n=== Testing direct proxy call ===")
    
    try:
        import httpx
        import json
        
        # Test through proxy
        proxy_url = "http://localhost:3000/r/qz_api/v1/embeddings"
        
        payload = {
            "model": "eric-qwen3-embedding-8b",
            "input": "Hello, this is a direct test through proxy."
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(proxy_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Direct proxy call successful")
                if 'data' in result and len(result['data']) > 0:
                    embedding = result['data'][0]['embedding']
                    logger.info(f"   Embedding dimension: {len(embedding)}")
                    logger.info(f"   First 5 values: {embedding[:5]}")
            else:
                logger.error(f"❌ Direct proxy call failed: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                
    except Exception as e:
        logger.error(f"❌ Direct proxy test failed: {e}")

async def main():
    """Main test function"""
    logger.info("🧪 Starting QzClient proxy tests")
    
    # Test proxy health first
    await test_proxy_health()
    
    # Test direct proxy call
    await test_direct_proxy_call()
    
    # Test QzClient through proxy
    await test_qz_client_through_proxy()
    
    logger.info("🏁 All proxy tests completed")

if __name__ == "__main__":
    asyncio.run(main())
