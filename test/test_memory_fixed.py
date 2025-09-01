"""
Test script to verify the fixed Memory class functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_engine.memory.memory import Memory

def test_memory_functionality():
    """Test the fixed Memory class functionality"""
    print("Testing Memory class with independent similarity calculation...")
    
    # Create a memory instance
    memory = Memory("test_memory")
    
    # Add some test content
    test_contents = [
        "Python is a programming language",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing deals with text",
        "Computer vision processes images"
    ]
    
    print("Adding test content...")
    for content in test_contents:
        memory.add(content)
        print(f"Added: {content}")
    
    print(f"\nTotal items in memory: {memory.count()}")
    
    # Test search functionality
    print("\nTesting search functionality...")
    search_queries = [
        "programming",
        "artificial intelligence", 
        "neural networks",
        "text processing",
        "image analysis"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = memory.search(query, top_k=3)
        for i, (content, score, metadata) in enumerate(results):
            print(f"  {i+1}. Score: {score:.4f} | Content: {content}")
    
    # Test vector-based search
    print("\nTesting vector-based search...")
    # Get a vector from existing content
    vector, _ = memory.get_by_content("Machine learning is a subset of AI")
    if vector:
        print("Searching using vector from 'Machine learning is a subset of AI'")
        results = memory.search(vector, top_k=3)
        for i, (content, score, metadata) in enumerate(results):
            print(f"  {i+1}. Score: {score:.4f} | Content: {content}")
    
    # Test get_info
    print("\nMemory info:")
    info = memory.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Clean up
    print("\nCleaning up test data...")
    memory.clear()
    print(f"Items after cleanup: {memory.count()}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_memory_functionality()
