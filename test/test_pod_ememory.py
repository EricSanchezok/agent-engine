#!/usr/bin/env python3
"""
Test script for PodEMemory implementation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.memory.e_memory import PodEMemory, Record
import numpy as np
import threading
import time


def test_basic_operations():
    """Test basic CRUD operations."""
    print("Testing PodEMemory basic operations...")
    
    # Initialize PodEMemory with small shard size for testing
    pod = PodEMemory("test_pod", dimension=128, max_elements_per_shard=3)
    
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
    id1 = pod.add(record1)
    id2 = pod.add(record2)
    print(f"Added records with IDs: {id1}, {id2}")
    
    # Test getting records
    retrieved1 = pod.get(id1)
    retrieved2 = pod.get(id2)
    print(f"Retrieved record 1: {retrieved1.content if retrieved1 else 'None'}")
    print(f"Retrieved record 2: {retrieved2.content if retrieved2 else 'None'}")
    
    # Test counting
    count = pod.count()
    print(f"Total records: {count}")
    
    # Test listing all records
    all_records = pod.list_all()
    print(f"All records count: {len(all_records)}")
    
    # Test vector search
    query_vector = [0.15] * 128  # Similar to record1
    similar = pod.search_similar(query_vector, k=2)
    print(f"Similar records: {similar}")
    
    # Test similar records search
    similar_records = pod.search_similar_records(query_vector, k=2)
    print(f"Similar records with content: {[(r.content, score) for r, score in similar_records]}")
    
    # Test statistics
    stats = pod.get_stats()
    print(f"Pod stats: {stats}")
    
    # Test shard info
    shard_info = pod.get_shard_info()
    print(f"Shard info: {shard_info}")
    
    # Test deletion
    deleted = pod.delete(id1)
    print(f"Deleted record 1: {deleted}")
    
    # Test final count
    final_count = pod.count()
    print(f"Final record count: {final_count}")
    
    # Clean up
    pod.clear()
    print("Cleared pod")


def test_shard_creation():
    """Test automatic shard creation when max_elements_per_shard is reached."""
    print("\nTesting PodEMemory shard creation...")
    
    # Initialize PodEMemory with very small shard size
    pod = PodEMemory("test_shard_pod", dimension=64, max_elements_per_shard=2)
    
    # Add records to trigger shard creation
    records = []
    for i in range(5):  # This should create 3 shards (2+2+1)
        record = Record(
            content=f"Shard test document {i}",
            vector=[float(i) * 0.1] * 64,
            attributes={"test_id": i}
        )
        records.append(record)
    
    # Add records one by one to see shard creation
    ids = []
    for i, record in enumerate(records):
        record_id = pod.add(record)
        ids.append(record_id)
        print(f"Added record {i} with ID {record_id}")
        
        # Check shard info after each addition
        shard_info = pod.get_shard_info()
        print(f"  Shards after record {i}: {len(shard_info)}")
        for info in shard_info:
            print(f"    Shard {info['shard_id']}: {info['record_count']} records")
    
    # Test that all records can be retrieved
    print(f"\nTotal records in pod: {pod.count()}")
    
    # Test search across all shards
    query_vector = [0.2] * 64
    similar = pod.search_similar(query_vector, k=3)
    print(f"Similar to query: {similar}")
    
    # Clean up
    pod.clear()
    print("Cleared shard test pod")


def test_batch_operations():
    """Test batch operations with shard distribution."""
    print("\nTesting PodEMemory batch operations...")
    
    pod = PodEMemory("test_batch_pod", dimension=32, max_elements_per_shard=3)
    
    # Create multiple records
    records = []
    for i in range(8):  # This should create 3 shards
        record = Record(
            content=f"Batch document {i}",
            vector=[float(i) * 0.1] * 32,
            attributes={"batch_id": i}
        )
        records.append(record)
    
    # Add batch
    ids = pod.add_batch(records)
    print(f"Added batch with IDs: {len(ids)} records")
    
    # Check shard distribution
    shard_info = pod.get_shard_info()
    print(f"Shard distribution:")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    # Test count
    count = pod.count()
    print(f"Batch count: {count}")
    
    # Test search
    query_vector = [0.2] * 32
    similar = pod.search_similar(query_vector, k=3)
    print(f"Similar to query: {similar}")
    
    # Clean up
    pod.clear()
    print("Cleared batch pod")


def test_concurrent_operations():
    """Test that operations work with concurrent access."""
    print("\nTesting PodEMemory concurrent operations...")
    
    pod = PodEMemory("test_concurrent_pod", dimension=32, max_elements_per_shard=5)
    
    # Simulate concurrent operations
    def add_records(thread_id, count):
        for i in range(count):
            record = Record(
                content=f"Thread {thread_id} record {i}",
                vector=[float(thread_id + i) * 0.1] * 32,
                attributes={"thread_id": thread_id, "record_id": i}
            )
            pod.add(record)
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
    count = pod.count()
    print(f"Concurrent operations result: {count} records")
    
    # Check shard distribution
    shard_info = pod.get_shard_info()
    print(f"Shard distribution after concurrent operations:")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    # Test concurrent search
    def search_records(thread_id):
        query_vector = [0.5] * 32
        results = pod.search_similar(query_vector, k=5)
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
    pod.clear()
    print("Cleared concurrent pod")


def test_file_structure():
    """Test that files are created correctly with proper directory structure."""
    print("\nTesting PodEMemory file structure...")
    
    pod = PodEMemory("test_file_pod", dimension=16, max_elements_per_shard=2)
    
    # Add records to trigger shard creation
    for i in range(3):
        record = Record(
            content=f"File test document {i}",
            vector=[float(i) * 0.1] * 16,
            attributes={"test": True, "id": i}
        )
        pod.add(record)
    
    # Check file structure
    stats = pod.get_stats()
    print(f"Pod directory: {stats['persist_dir']}")
    
    # Verify directory structure
    pod_dir = Path(stats['persist_dir'])
    print(f"Pod directory exists: {pod_dir.exists()}")
    
    # Check shard directories
    shard_info = pod.get_shard_info()
    for info in shard_info:
        shard_dir = Path(info['persist_dir'])
        print(f"Shard {info['shard_id']} directory: {shard_dir}")
        print(f"  Directory exists: {shard_dir.exists()}")
        
        # Check for SQLite and ChromaDB files
        sqlite_file = shard_dir / f"{info['name']}.sqlite"
        chroma_dir = shard_dir / f"{info['name']}_chroma"
        print(f"  SQLite file exists: {sqlite_file.exists()}")
        print(f"  ChromaDB directory exists: {chroma_dir.exists()}")
    
    # Clean up
    pod.clear()
    print("Cleared file structure pod")


if __name__ == "__main__":
    print("Testing PodEMemory implementation...")
    
    try:
        test_basic_operations()
        test_shard_creation()
        test_batch_operations()
        test_concurrent_operations()
        test_file_structure()
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
