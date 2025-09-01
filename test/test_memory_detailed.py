"""
Detailed test script to diagnose Memory class issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_engine.memory.memory import Memory
from agent_engine.memory.embedder import Embedder
import numpy as np

def test_embedder_directly():
    """Test embedder directly to understand its behavior"""
    print("=== Testing Embedder Directly ===")
    
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    
    test_texts = [
        "Python is a programming language",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks"
    ]
    
    embedder.fit(test_texts)
    
    print(f"Embedder method: {embedder.method}")
    print(f"Vector dimension: {embedder.get_vector_dimension()}")
    
    test_texts = [
        "Python is a programming language",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks"
    ]
    
    vectors = []
    for text in test_texts:
        vector = embedder.embed(text)
        vectors.append(vector)
        print(f"Text: '{text}'")
        print(f"Vector length: {len(vector)}")
        print(f"Vector norm: {np.linalg.norm(vector):.4f}")
        print(f"First 5 values: {vector[:5]}")
        print()
    
    # Test similarity between vectors
    print("=== Testing Similarities ===")
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            sim = embedder.similarity(vectors[i], vectors[j])
            print(f"Similarity between '{test_texts[i]}' and '{test_texts[j]}': {sim:.4f}")

def test_memory_detailed():
    """Test memory with detailed output"""
    print("\n=== Testing Memory Class ===")
    
    memory = Memory("test_memory_detailed")
    
    # Add test content
    test_contents = [
        "Python is a programming language",
        "Machine learning is a subset of AI", 
        "Deep learning uses neural networks"
    ]
    
    print("Adding content...")
    for content in test_contents:
        memory.add(content)
        vector, _ = memory.get_by_content(content)
        if vector:
            print(f"Added: '{content}'")
            print(f"  Vector length: {len(vector)}")
            print(f"  Vector norm: {np.linalg.norm(vector):.4f}")
    
    print(f"\nTotal items: {memory.count()}")
    
    # Test search with detailed output
    print("\n=== Testing Search ===")
    query = "programming"
    print(f"Searching for: '{query}'")
    
    results = memory.search(query, top_k=3)
    for i, (content, score, metadata) in enumerate(results):
        print(f"  {i+1}. Score: {score:.4f} | Content: '{content}'")
    
    # Test vector-based search
    print("\n=== Testing Vector Search ===")
    query_vector, _ = memory.get_by_content("Machine learning is a subset of AI")
    if query_vector:
        print(f"Searching with vector from 'Machine learning is a subset of AI'")
        print(f"Query vector norm: {np.linalg.norm(query_vector):.4f}")
        
        results = memory.search(query_vector, top_k=3)
        for i, (content, score, metadata) in enumerate(results):
            print(f"  {i+1}. Score: {score:.4f} | Content: '{content}'")
    
    # Test similarity calculation directly
    print("\n=== Testing Direct Similarity Calculation ===")
    vectors = []
    for content in test_contents:
        vector, _ = memory.get_by_content(content)
        if vector:
            vectors.append(vector)
    
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            sim = memory._calculate_similarity(vectors[i], vectors[j])
            print(f"Direct similarity between '{test_contents[i]}' and '{test_contents[j]}': {sim:.4f}")
    
    # Clean up
    memory.clear()

if __name__ == "__main__":
    test_embedder_directly()
    test_memory_detailed()
