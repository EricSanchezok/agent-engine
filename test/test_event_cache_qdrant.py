#!/usr/bin/env python3
"""
Test script for Qdrant-based EventCache implementation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.ICUAssistantAgent.memory.event_cache import EventCache
from agent_engine.memory.e_memory.models import Record
import numpy as np
import threading
import time


def test_basic_operations():
    """Test basic CRUD operations."""
    print("Testing EventCache basic operations...")
    
    # Initialize EventCache
    cache = EventCache("test_cache", dimension=128, max_elements_per_shard=10)
    
    # Test adding records
    records = []
    for i in range(25):  # This should create multiple shards
        record = Record(
            content=f"Test document {i} about machine learning and AI",
            vector=[float(i) * 0.1] * 128,
            attributes={"type": "document", "index": i}
        )
        records.append(record)
    
    # Add records one by one
    record_ids = []
    for record in records:
        record_id = cache.add(record)
        record_ids.append(record_id)
    
    print(f"Added {len(record_ids)} records")
    print(f"Total records: {cache.count()}")
    print(f"Shard count: {cache._shard_count}")
    
    # Test getting records
    retrieved = cache.get(record_ids[0])
    print(f"Retrieved record: {retrieved.content if retrieved else 'None'}")
    
    # Test listing all records
    all_records = cache.list_all(limit=5)
    print(f"Listed {len(all_records)} records (limited to 5)")
    
    # Test vector search
    query_vector = [0.5] * 128
    similar = cache.search_similar(query_vector, k=3)
    print(f"Similar records: {len(similar)}")
    
    # Test similar records search
    similar_records = cache.search_similar_records(query_vector, k=3)
    print(f"Similar records with content: {len(similar_records)}")
    
    # Test statistics
    stats = cache.get_stats()
    print(f"Cache stats: {stats['name']}, {stats['total_records']} records, {stats['shard_count']} shards")
    
    # Test shard info
    shard_info = cache.get_shard_info()
    print(f"Shard info: {len(shard_info)} shards")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    # Test deletion
    deleted = cache.delete(record_ids[0])
    print(f"Deleted record: {deleted}")
    
    # Test final count
    final_count = cache.count()
    print(f"Final record count: {final_count}")
    
    # Clean up
    cache.clear()
    print("Cleared cache")


def test_batch_operations():
    """Test batch operations."""
    print("\nTesting EventCache batch operations...")
    
    cache = EventCache("test_batch_cache", dimension=64, max_elements_per_shard=5)
    
    # Create multiple records
    records = []
    for i in range(15):  # This should create multiple shards
        record = Record(
            content=f"Batch document {i}",
            vector=[float(i) * 0.1] * 64,
            attributes={"batch_id": i}
        )
        records.append(record)
    
    # Add batch
    ids = cache.add_batch(records)
    print(f"Added batch with {len(ids)} records")
    print(f"Total records: {cache.count()}")
    print(f"Shard count: {cache._shard_count}")
    
    # Test search
    query_vector = [0.2] * 64
    similar = cache.search_similar(query_vector, k=5)
    print(f"Similar to query: {len(similar)}")
    
    # Clean up
    cache.clear()
    print("Cleared batch cache")


def test_concurrent_operations():
    """Test that operations work without locks (concurrent-safe)."""
    print("\nTesting EventCache concurrent operations...")
    
    # Use memory mode to avoid file locking issues
    import time
    cache_name = f"test_concurrent_cache_{int(time.time())}"
    cache = EventCache(cache_name, dimension=32, max_elements_per_shard=10, use_memory=True)
    
    # Simulate concurrent operations
    def add_records(thread_id, count):
        for i in range(count):
            record = Record(
                content=f"Thread {thread_id} record {i}",
                vector=[float(thread_id + i) * 0.1] * 32,
                attributes={"thread_id": thread_id, "record_id": i}
            )
            cache.add(record)
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
    count = cache.count()
    print(f"Concurrent operations result: {count} records")
    print(f"Shard count: {cache._shard_count}")
    
    # Test concurrent search
    def search_records(thread_id):
        query_vector = [0.5] * 32
        results = cache.search_similar(query_vector, k=5)
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
    cache.clear()
    print("Cleared concurrent cache")


def test_shard_distribution():
    """Test that records are properly distributed across shards."""
    print("\nTesting shard distribution...")
    
    cache = EventCache("test_distribution_cache", dimension=16, max_elements_per_shard=3)
    
    # Add records that should trigger shard creation
    records = []
    for i in range(10):
        record = Record(
            content=f"Distribution test {i}",
            vector=[float(i) * 0.1] * 16,
            attributes={"test_id": i}
        )
        records.append(record)
    
    # Add records
    ids = cache.add_batch(records)
    print(f"Added {len(ids)} records")
    
    # Check shard distribution
    shard_info = cache.get_shard_info()
    print(f"Shard distribution:")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    # Test that we can retrieve records from different shards
    for i, record_id in enumerate(ids[:5]):
        retrieved = cache.get(record_id)
        if retrieved:
            print(f"Retrieved record {i}: {retrieved.content}")
        else:
            print(f"Failed to retrieve record {i}")
    
    # Clean up
    cache.clear()
    print("Cleared distribution cache")


if __name__ == "__main__":
    print("Testing Qdrant-based EventCache implementation...")
    
    try:
        test_basic_operations()
        test_batch_operations()
        test_concurrent_operations()
        test_shard_distribution()
        print("\nAll EventCache tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
