import asyncio
import time
import os
from typing import List
from dotenv import load_dotenv
from .azure_client import AzureClient

# Load environment variables
load_dotenv()

def generate_test_strings(num_strings: int = 300, avg_length: int = 500) -> List[str]:
    """Generate test strings with specified number and average length"""
    import random
    import string
    
    # Base text for generating realistic strings
    base_text = """
    Machine learning is a subset of artificial intelligence that focuses on developing algorithms and statistical models 
    that enable computers to improve their performance on a specific task through experience. These algorithms build 
    mathematical models based on sample data, known as training data, to make predictions or decisions without being 
    explicitly programmed to perform the task. Machine learning approaches are traditionally divided into three broad 
    categories, depending on the nature of the learning signal or feedback available to a learning system.
    
    Supervised learning algorithms are trained using labeled examples, such as an input where the desired output is known. 
    For example, a piece of equipment could have data points labeled either F (failed) or R (running). The learning 
    algorithm receives a set of inputs along with the corresponding correct outputs, and the algorithm learns by comparing 
    its actual output with correct outputs to find errors. It then modifies the model accordingly.
    
    Unsupervised learning algorithms are used when the information used to train is neither classified nor labeled. 
    Unsupervised learning studies how systems can infer a function to describe a hidden structure from unlabeled data. 
    The system doesn't figure out the right output, but it explores the data and can draw inferences from datasets to 
    describe hidden structures from unlabeled data.
    
    Reinforcement learning is a type of programming that allows a system to automatically determine the ideal behavior 
    within a specific context, to maximize its performance. Simple reward feedback is required for the agent to learn 
    which action is best; this is known as the reward signal.
    """
    
    # Split base text into sentences
    sentences = [s.strip() for s in base_text.split('.') if s.strip()]
    
    test_strings = []
    for i in range(num_strings):
        # Randomly select sentences to create a string of desired length
        current_length = 0
        selected_sentences = []
        
        while current_length < avg_length and sentences:
            # Randomly select a sentence
            sentence = random.choice(sentences)
            selected_sentences.append(sentence)
            current_length += len(sentence) + 1  # +1 for space
            
            # Add some random words to reach target length
            if current_length < avg_length:
                words = ''.join(random.choices(string.ascii_lowercase + ' ', k=random.randint(10, 50)))
                selected_sentences.append(words)
                current_length += len(words)
        
        # Join sentences and add some padding if needed
        result = ' '.join(selected_sentences)
        if len(result) < avg_length:
            padding = ''.join(random.choices(string.ascii_lowercase + ' ', k=avg_length - len(result)))
            result += padding
        
        test_strings.append(result[:avg_length])  # Ensure exact length
    
    return test_strings

async def test_concurrent_embedding(
    client: AzureClient,
    texts: List[str],
    max_concurrent: int = 32,
    model_name: str = 'text-embedding-3-large'
) -> tuple:
    """Test concurrent embedding performance"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def embed_single_text(text: str, idx: int) -> tuple:
        async with semaphore:
            start_time = time.time()
            try:
                vector = await client.embedding(text, model_name=model_name)
                end_time = time.time()
                processing_time = end_time - start_time
                
                if vector is not None:
                    return (idx, True, vector, processing_time, None)
                else:
                    return (idx, False, None, processing_time, "No vector returned")
                    
            except Exception as e:
                end_time = time.time()
                processing_time = end_time - start_time
                return (idx, False, None, processing_time, str(e))
    
    # Create tasks for all texts
    tasks = [embed_single_text(text, idx) for idx, text in enumerate(texts)]
    
    # Execute all tasks concurrently
    start_total = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_total = time.time()
    
    total_time = end_total - start_total
    
    return results, total_time

async def main():
    """Main test function"""
    # Check if API key is available
    api_key = os.getenv('AZURE_API_KEY')
    if not api_key:
        print("❌ AZURE_API_KEY not found in environment variables")
        print("Please set AZURE_API_KEY in your .env file")
        return
    
    # Initialize Azure client
    print("🚀 Initializing Azure OpenAI client...")
    client = AzureClient(api_key=api_key)
    
    try:
        # Generate test data
        print("📝 Generating test data...")
        test_strings = generate_test_strings(num_strings=300, avg_length=500)
        print(f"✅ Generated {len(test_strings)} test strings, average length: {sum(len(s) for s in test_strings) // len(test_strings)}")
        
        # Test different concurrency levels
        concurrency_levels = [1, 4, 8, 16, 32]
        
        for max_concurrent in concurrency_levels:
            print(f"\n🔍 Testing with {max_concurrent} concurrent tasks...")
            
            # Run the test
            results, total_time = await test_concurrent_embedding(
                client=client,
                texts=test_strings,
                max_concurrent=max_concurrent
            )
            
            # Analyze results
            successful = sum(1 for r in results if isinstance(r, tuple) and r[1])
            failed = len(results) - successful
            
            # Calculate average processing time for successful requests
            successful_times = [r[3] for r in results if isinstance(r, tuple) and r[1]]
            avg_time = sum(successful_times) / len(successful_times) if successful_times else 0
            
            # Calculate throughput
            throughput = successful / total_time if total_time > 0 else 0
            
            print(f"📊 Results for {max_concurrent} concurrent tasks:")
            print(f"   Total time: {total_time:.2f} seconds")
            print(f"   Successful: {successful}")
            print(f"   Failed: {failed}")
            print(f"   Average processing time: {avg_time:.3f} seconds")
            print(f"   Throughput: {throughput:.2f} embeddings/second")
            
            # Show some sample results
            if successful > 0:
                sample_result = next(r for r in results if isinstance(r, tuple) and r[1])
                vector_length = len(sample_result[2]) if sample_result[2] else 0
                print(f"   Vector dimension: {vector_length}")
        
        print(f"\n🎯 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        
    finally:
        # Close the client
        await client.close()
        print("✅ Azure client connection closed")

if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
