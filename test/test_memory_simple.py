"""
Simple test to verify the fixed Memory class basic functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_engine.memory.memory import Memory

def test_memory_basic():
    """Test basic Memory functionality"""
    print("Testing Memory class basic functionality...")
    
    # Create memory instance
    memory = Memory("test_basic")
    
    # Test adding content
    print("1. Testing add functionality...")
    test_content = "This is a test content for memory"
    memory.add(test_content)
    print(f"Added content: {test_content}")
    
    # Test count
    count = memory.count()
    print(f"Memory count: {count}")
    
    # Test get_by_content
    print("2. Testing get_by_content...")
    vector, metadata = memory.get_by_content(test_content)
    if vector:
        print(f"Retrieved vector length: {len(vector)}")
        print(f"Retrieved metadata: {metadata}")
    else:
        print("Failed to retrieve content")
    
    # Test search
    print("3. Testing search...")
    results = memory.search("test content", top_k=1)
    if results:
        content, score, metadata = results[0]
        print(f"Search result: '{content}' with score {score:.4f}")
    else:
        print("No search results")
    
    # Test vector-based search
    print("4. Testing vector-based search...")
    if vector:
        results = memory.search(vector, top_k=1)
        if results:
            content, score, metadata = results[0]
            print(f"Vector search result: '{content}' with score {score:.4f}")
        else:
            print("No vector search results")
    
    # Test get_info
    print("5. Testing get_info...")
    info = memory.get_info()
    print(f"Memory info: {info}")
    
    # Clean up
    print("6. Testing cleanup...")
    memory.clear()
    print(f"Count after cleanup: {memory.count()}")
    
    print("\nBasic test completed!")

if __name__ == "__main__":
    test_memory_basic()
