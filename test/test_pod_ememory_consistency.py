#!/usr/bin/env python3
"""
Test script for PodEMemory consistency between add and get operations.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.memory.e_memory import PodEMemory, Record
import numpy as np


def test_add_get_consistency():
    """Test that records added can be retrieved from the correct shard."""
    print("Testing PodEMemory add/get consistency...")
    
    # Create PodEMemory with small shard size to force multiple shards
    pod = PodEMemory("test_consistency", dimension=64, max_elements_per_shard=2)
    
    # Add records with known IDs
    test_records = []
    for i in range(5):
        record = Record(
            id=f"test_record_{i}",  # Use known IDs for testing
            content=f"Test document {i}",
            vector=[float(i) * 0.1] * 64,
            attributes={"test_id": i}
        )
        test_records.append(record)
    
    # Add records one by one
    added_ids = []
    for record in test_records:
        result_id = pod.add(record)
        added_ids.append(result_id)
        print(f"Added record {record.id} -> shard {pod._get_shard_for_record(record.id)}")
    
    # Check shard distribution
    shard_info = pod.get_shard_info()
    print(f"\nShard distribution:")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    # Test retrieval consistency
    print(f"\nTesting retrieval consistency:")
    for i, record_id in enumerate(added_ids):
        # Get shard ID using the same method as add
        expected_shard_id = pod._get_shard_for_record(record_id)
        
        # Try to get the record
        retrieved_record = pod.get(record_id)
        
        if retrieved_record:
            print(f"  ✅ Record {record_id}: expected shard {expected_shard_id}, retrieved successfully")
            print(f"     Content: {retrieved_record.content}")
        else:
            print(f"  ❌ Record {record_id}: expected shard {expected_shard_id}, NOT FOUND")
    
    # Test total count
    total_count = pod.count()
    print(f"\nTotal records: {total_count}")
    
    # Clean up
    pod.clear()
    print("Cleared test data")


def test_hash_distribution():
    """Test hash distribution across shards."""
    print("\nTesting hash distribution...")
    
    pod = PodEMemory("test_hash_dist", dimension=32, max_elements_per_shard=10)
    
    # Add many records to see distribution
    records = []
    for i in range(20):
        record = Record(
            id=f"hash_test_{i}",
            content=f"Hash test document {i}",
            vector=[float(i) * 0.1] * 32,
            attributes={"index": i}
        )
        records.append(record)
    
    # Add all records
    pod.add_batch(records)
    
    # Check distribution
    shard_info = pod.get_shard_info()
    print(f"Hash distribution across shards:")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    # Test that all records can be retrieved
    print(f"\nTesting retrieval of all records:")
    success_count = 0
    for i in range(20):
        record_id = f"hash_test_{i}"
        retrieved = pod.get(record_id)
        if retrieved:
            success_count += 1
        else:
            print(f"  ❌ Failed to retrieve {record_id}")
    
    print(f"Successfully retrieved {success_count}/20 records")
    
    # Clean up
    pod.clear()
    print("Cleared hash test data")


def test_existing_data_consistency():
    """Test consistency with existing data (simulate your scenario)."""
    print("\nTesting existing data consistency...")
    
    # Create pod and add some data
    pod1 = PodEMemory("test_existing", dimension=32, max_elements_per_shard=3)
    
    # Add records to create multiple shards
    test_ids = ["existing_001", "existing_002", "existing_003", "existing_004", "existing_005"]
    for i, record_id in enumerate(test_ids):
        record = Record(
            id=record_id,
            content=f"Existing document {i}",
            vector=[float(i) * 0.1] * 32,
            attributes={"index": i}
        )
        pod1.add(record)
        print(f"Added {record_id} -> shard {pod1._get_shard_for_record(record_id)}")
    
    # Check shard distribution
    shard_info = pod1.get_shard_info()
    print(f"\nFirst pod shard distribution:")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    # Close first pod
    del pod1
    
    # Create second pod (should load existing shards)
    pod2 = PodEMemory("test_existing", dimension=32, max_elements_per_shard=3)
    
    # Check shard distribution
    shard_info = pod2.get_shard_info()
    print(f"\nSecond pod shard distribution:")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    # Test retrieval consistency
    print(f"\nTesting retrieval consistency:")
    for record_id in test_ids:
        expected_shard = pod2._get_shard_for_record(record_id)
        retrieved = pod2.get(record_id)
        
        if retrieved:
            print(f"  ✅ {record_id}: expected shard {expected_shard}, retrieved successfully")
        else:
            print(f"  ❌ {record_id}: expected shard {expected_shard}, NOT FOUND")
    
    # Clean up
    pod2.clear()
    print("Cleared existing test data")


if __name__ == "__main__":
    print("Testing PodEMemory consistency...")
    
    try:
        test_add_get_consistency()
        test_hash_distribution()
        test_existing_data_consistency()
        print("\nAll consistency tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
