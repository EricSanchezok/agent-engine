import asyncio
import sys
import os
from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_engine.llm_client.azure_client import AzureClient

load_dotenv()

async def test_azure_client():
    """Test the new AzureClient functionality"""
    
    api_key = os.getenv("AZURE_API_KEY")
    
    try:
        # Create Azure client instance
        client = AzureClient(api_key=api_key)
        print("✅ AzureClient created successfully")
        
        # Test chat functionality
        system_prompt = "You are a helpful assistant."
        user_prompt = "Hello, how are you?"
        
        print("🚀 Testing chat functionality...")
        response = await client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name='gpt-4o',
            max_tokens=100,
            temperature=0.7
        )
        
        if response:
            print(f"✅ Chat response received: {response[:100]}...")
        else:
            print("❌ Chat response failed")
        
        # Test embedding functionality
        print("🚀 Testing embedding functionality...")
        text = "This is a test text for embeddings."
        embeddings = await client.embedding(
            text=text,
            model_name='text-embedding-ada-002'
        )
        
        if embeddings:
            print(f"✅ Embeddings received, length: {len(embeddings)}")
        else:
            print("❌ Embeddings failed")
        
        # Close the client
        await client.close()
        print("✅ Client closed successfully")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    print("🧪 Testing new AzureClient implementation...")
    asyncio.run(test_azure_client())
