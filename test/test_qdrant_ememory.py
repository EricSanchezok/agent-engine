#!/usr/bin/env python3
"""
Test script for Qdrant-based EMemory implementation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.memory.e_memory.core import EMemory
from agent_engine.memory.e_memory.models import Record
import numpy as np


def test_basic_operations():
    """Test basic CRUD operations."""
    print("Testing basic operations...")
    
    # Initialize EMemory
    memory = EMemory("test_memory", dimension=128)
    
    # Test adding records
    record1 = Record(
        content="This is a test document about machine learning",
        vector=[0.1] * 128,
        attributes={"type": "document", "topic": "ml"}
    )
    
    record2 = Record(
        content="Another document about artificial intelligence",
        vector=[0.2] * 128,
        attributes={"type": "document", "topic": "ai"}
    )
    
    # Add records
    id1 = memory.add(record1)
    id2 = memory.add(record2)
    print(f"Added records with IDs: {id1}, {id2}")
    
    # Test getting records
    retrieved1 = memory.get(id1)
    retrieved2 = memory.get(id2)
    print(f"Retrieved record 1: {retrieved1.content if retrieved1 else 'None'}")
    print(f"Retrieved record 2: {retrieved2.content if retrieved2 else 'None'}")
    
    # Test counting
    count = memory.count()
    print(f"Total records: {count}")
    
    # Test listing all records
    all_records = memory.list_all()
    print(f"All records count: {len(all_records)}")
    
    # Test vector search
    query_vector = [0.15] * 128  # Similar to record1
    similar = memory.search_similar(query_vector, k=2)
    print(f"Similar records: {similar}")
    
    # Test similar records search
    similar_records = memory.search_similar_records(query_vector, k=2)
    print(f"Similar records with content: {[(r.content, score) for r, score in similar_records]}")
    
    # Test statistics
    stats = memory.get_stats()
    print(f"Memory stats: {stats}")
    
    # Test deletion
    deleted = memory.delete(id1)
    print(f"Deleted record 1: {deleted}")
    
    # Test final count
    final_count = memory.count()
    print(f"Final record count: {final_count}")
    
    # Clean up
    memory.clear()
    print("Cleared memory")


def test_batch_operations():
    """Test batch operations."""
    print("\nTesting batch operations...")
    
    memory = EMemory("test_batch_memory", dimension=64)
    
    # Create multiple records
    records = []
    for i in range(5):
        record = Record(
            content=f"Batch document {i}",
            vector=[float(i) * 0.1] * 64,
            attributes={"batch_id": i}
        )
        records.append(record)
    
    # Add batch
    ids = memory.add_batch(records)
    print(f"Added batch with IDs: {ids}")
    
    # Test count
    count = memory.count()
    print(f"Batch count: {count}")
    
    # Test search
    query_vector = [0.2] * 64
    similar = memory.search_similar(query_vector, k=3)
    print(f"Similar to query: {similar}")
    
    # Clean up
    memory.clear()
    print("Cleared batch memory")


def test_concurrent_operations():
    """Test that operations work without locks (concurrent-safe)."""
    print("\nTesting concurrent operations...")
    
    memory = EMemory("test_concurrent_memory", dimension=32)
    
    # Simulate concurrent operations
    import threading
    import time
    
    def add_records(thread_id, count):
        for i in range(count):
            record = Record(
                content=f"Thread {thread_id} record {i}",
                vector=[float(thread_id + i) * 0.1] * 32,
                attributes={"thread_id": thread_id, "record_id": i}
            )
            memory.add(record)
            time.sleep(0.001)  # Small delay to simulate real work
    
    # Create multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=add_records, args=(i, 10))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check results
    count = memory.count()
    print(f"Concurrent operations result: {count} records")
    
    # Test concurrent search
    def search_records(thread_id):
        query_vector = [0.5] * 32
        results = memory.search_similar(query_vector, k=5)
        print(f"Thread {thread_id} search results: {len(results)}")
    
    # Create search threads
    search_threads = []
    for i in range(3):
        thread = threading.Thread(target=search_records, args=(i,))
        search_threads.append(thread)
        thread.start()
    
    # Wait for search threads
    for thread in search_threads:
        thread.join()
    
    # Clean up
    memory.clear()
    print("Cleared concurrent memory")


if __name__ == "__main__":
    print("Testing Qdrant-based EMemory implementation...")
    
    try:
        test_basic_operations()
        test_batch_operations()
        test_concurrent_operations()
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
