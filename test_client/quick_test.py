import asyncio
import time
import os
from typing import List
from dotenv import load_dotenv
from test_client.azure_client import AzureClient

# Load environment variables
load_dotenv()

def generate_simple_test_strings(num_strings: int = 300, avg_length: int = 500) -> List[str]:
    """Generate simple test strings"""
    import random
    
    # Simple base text
    base_text = "This is a test string for embedding generation. Machine learning and artificial intelligence are fascinating fields."
    
    test_strings = []
    for i in range(num_strings):
        # Create a string by repeating and modifying the base text
        repeat_count = max(1, avg_length // len(base_text))
        text = (base_text + f" Sample {i}. ") * repeat_count
        text = text[:avg_length]  # Trim to exact length
        
        # Add some variation
        if len(text) < avg_length:
            text += " Additional content to reach target length. " * ((avg_length - len(text)) // 20)
            text = text[:avg_length]
        
        test_strings.append(text)
    
    return test_strings

async def quick_concurrent_test(
    client: AzureClient,
    texts: List[str],
    max_concurrent: int = 32
) -> None:
    """Quick concurrent embedding test"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def embed_text(text: str, idx: int):
        async with semaphore:
            start_time = time.time()
            try:
                vector = await client.embedding(text, model_name='text-embedding-3-large')
                end_time = time.time()
                processing_time = end_time - start_time
                
                if vector is not None:
                    print(f"✅ Text {idx}: Success in {processing_time:.2f}s, vector length: {len(vector)}")
                    return True, processing_time
                else:
                    print(f"❌ Text {idx}: Failed - no vector returned")
                    return False, processing_time
                    
            except Exception as e:
                end_time = time.time()
                processing_time = end_time - start_time
                print(f"❌ Text {idx}: Error - {str(e)} in {processing_time:.2f}s")
                return False, processing_time
    
    print(f"🚀 Starting concurrent embedding test with {len(texts)} texts, max {max_concurrent} concurrent...")
    
    # Create tasks
    tasks = [embed_text(text, idx) for idx, text in enumerate(texts)]
    
    # Execute all tasks
    start_total = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_total = time.time()
    
    total_time = end_total - start_total
    
    # Summary
    successful = sum(1 for r in results if isinstance(r, tuple) and r[0])
    failed = len(results) - successful
    
    print(f"\n📊 Test Summary:")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Throughput: {successful / total_time:.2f} embeddings/second")

async def main():
    """Quick test main function"""
    api_key = os.getenv('AZURE_API_KEY')
    if not api_key:
        print("❌ AZURE_API_KEY not found. Please set it in your .env file")
        return
    
    print("🚀 Initializing Azure OpenAI client...")
    client = AzureClient(api_key=api_key)
    
    try:
        # Generate test data
        print("📝 Generating test data...")
        test_strings = generate_simple_test_strings(num_strings=300, avg_length=500)
        print(f"✅ Generated {len(test_strings)} test strings")
        
        # Run quick test
        await quick_concurrent_test(client, test_strings, max_concurrent=32)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await client.close()
        print("✅ Azure client closed")

if __name__ == "__main__":
    asyncio.run(main())
