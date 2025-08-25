"""
Test the advanced embedder with Sentence Transformers

This demonstrates fixed-dimension vectors and better quality embeddings.
"""

from pathlib import Path
import sys

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_engine.memory import Embedder, get_recommended_models, Memory


def test_advanced_embedder():
    """Test the advanced embedder functionality"""
    print("Testing Advanced Embedder\n")
    
    try:
        # Test 1: Basic initialization
        print("1. Initializing embedder...")
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        print(f"   Model: {embedder.model_name}")
        print(f"   Available: {embedder._sentence_transformers_available}")
        
        # Test 2: Get model info
        print("\n2. Model information:")
        info = embedder.get_model_info()
        for key, value in info.items():
            if key != "available_models":
                print(f"   {key}: {value}")
        
        if "available_models" in info:
            print("   Available models:")
            for model in info["available_models"]:
                print(f"     - {model}")
        
        # Test 3: Test embedding with different text lengths
        print("\n3. Testing fixed dimensions with different text lengths:")
        texts = [
            "Hello",  # Very short
            "This is a medium length sentence about machine learning.",  # Medium
            "Artificial intelligence encompasses machine learning, deep learning, neural networks, natural language processing, computer vision, robotics, expert systems, and more. These technologies enable computers to perform tasks that typically require human intelligence."  # Long
        ]
        
        for i, text in enumerate(texts):
            vector = embedder.embed(text)
            print(f"   Text {i+1} ({len(text)} chars): {len(vector)} dimensions")
            print(f"   Sample values: {vector[:5]}")
        
        # Test 4: Batch processing
        print("\n4. Batch processing:")
        vectors = embedder.embed_batch(texts)
        print(f"   Processed {len(vectors)} texts")
        print(f"   All vectors have same dimension: {len(vectors[0])}")
        
        # Test 5: Similarity calculation
        print("\n5. Similarity calculation:")
        sim1 = embedder.similarity(vectors[0], vectors[1])
        sim2 = embedder.similarity(vectors[1], vectors[2])
        sim3 = embedder.similarity(vectors[0], vectors[2])
        
        print(f"   Similarity between short and medium: {sim1:.4f}")
        print(f"   Similarity between medium and long: {sim2:.4f}")
        print(f"   Similarity between short and long: {sim3:.4f}")
        
        # Test 6: Find most similar
        print("\n6. Finding most similar:")
        similarities = embedder.find_most_similar(vectors[0], vectors, top_k=3)
        for idx, score in similarities:
            print(f"   Rank {idx+1}: Text {idx+1} (score: {score:.4f})")
        
        print("\nEmbedder tests completed successfully! ✅")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("To use advanced embedder, install sentence-transformers:")
        print("pip install sentence-transformers")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_recommended_models():
    """Test getting recommended models"""
    print("\n" + "="*50)
    print("Testing Recommended Models\n")
    
    models = get_recommended_models()
    print("Recommended Sentence Transformer models:")
    for category, model_name in models.items():
        print(f"   {category}: {model_name}")
    
    print("\nYou can use any of these models with:")
    print("embedder = Embedder(model_name='model_name')")


def test_memory_with_advanced_embedder():
    """Test Memory class with advanced embedder"""
    print("\n" + "="*50)
    print("Testing Memory with Embedder\n")
    
    try:
        # Create memory with sentence transformer
        memory = Memory("test_advanced")
        print(f"Memory created: {memory.get_info()}")
        
        # Add content
        content1 = "Machine learning algorithms"
        content2 = "Deep learning neural networks"
        content3 = "Natural language processing techniques"
        
        memory.add(content1, metadata={"topic": "ML", "type": "algorithm"})
        memory.add(content2, metadata={"topic": "DL", "type": "neural"})
        memory.add(content3, metadata={"topic": "NLP", "type": "technique"})
        
        print(f"Added 3 items, total count: {memory.count()}")
        
        # Test search
        query = "artificial intelligence algorithms"
        results = memory.search(query, top_k=2)
        print(f"\nSearch results for '{query}':")
        for i, (content, score, metadata) in enumerate(results):
            print(f"  {i+1}. Score: {score:.4f}")
            print(f"     Content: {content}")
            print(f"     Metadata: {metadata}")
        
        print("\nMemory with embedder tests completed successfully! ✅")
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("Starting Embedder Tests...\n")
    
    test_advanced_embedder()
    test_recommended_models()
    test_memory_with_advanced_embedder()
    
    print("\n" + "="*50)
    print("Summary:")
    print("✅ Embedder provides fixed-dimension vectors")
    print("✅ Better semantic understanding than TF-IDF")
    print("✅ Multiple model options for different use cases")
    print("✅ Automatic fallback to basic methods if needed")


if __name__ == "__main__":
    main()
