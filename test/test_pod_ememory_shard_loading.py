#!/usr/bin/env python3
"""
Test script for PodEMemory shard loading functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.memory.e_memory import PodEMemory, Record
import numpy as np


def test_shard_loading():
    """Test that PodEMemory loads existing shards correctly."""
    print("Testing PodEMemory shard loading...")
    
    # Create first PodEMemory instance and add some data
    print("Creating first PodEMemory instance...")
    pod1 = PodEMemory("test_shard_loading", dimension=64, max_elements_per_shard=3)
    
    # Add records to trigger shard creation
    records = []
    for i in range(5):  # This should create 2 shards (3+2)
        record = Record(
            content=f"Test document {i}",
            vector=[float(i) * 0.1] * 64,
            attributes={"test_id": i}
        )
        records.append(record)
    
    # Add records
    ids = pod1.add_batch(records)
    print(f"Added {len(ids)} records to first pod")
    
    # Check shard distribution
    shard_info = pod1.get_shard_info()
    print(f"First pod shard distribution:")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    # Get total count
    total_count = pod1.count()
    print(f"Total records in first pod: {total_count}")
    
    # Close first pod (simulate restart)
    del pod1
    
    # Create second PodEMemory instance (should load existing shards)
    print("\nCreating second PodEMemory instance (should load existing shards)...")
    pod2 = PodEMemory("test_shard_loading", dimension=64, max_elements_per_shard=3)
    
    # Check that existing shards were loaded
    shard_info = pod2.get_shard_info()
    print(f"Second pod shard distribution:")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    # Check total count
    total_count = pod2.count()
    print(f"Total records in second pod: {total_count}")
    
    # Verify we can retrieve records
    print("\nVerifying record retrieval:")
    for i in range(min(3, len(ids))):
        record = pod2.get(ids[i])
        if record:
            print(f"  Retrieved record {i}: {record.content}")
        else:
            print(f"  Failed to retrieve record {i}")
    
    # Test adding new records (should use existing shards or create new ones)
    print("\nAdding new records to second pod...")
    new_record = Record(
        content="New document after restart",
        vector=[0.5] * 64,
        attributes={"new": True}
    )
    new_id = pod2.add(new_record)
    print(f"Added new record with ID: {new_id}")
    
    # Check final shard distribution
    final_shard_info = pod2.get_shard_info()
    print(f"Final shard distribution:")
    for info in final_shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    final_count = pod2.count()
    print(f"Final total records: {final_count}")
    
    # Clean up
    pod2.clear()
    print("Cleared test data")


def test_empty_directory():
    """Test PodEMemory behavior with empty directory."""
    print("\nTesting PodEMemory with empty directory...")
    
    # Create a new pod (should create first shard)
    pod = PodEMemory("test_empty", dimension=32, max_elements_per_shard=2)
    
    shard_info = pod.get_shard_info()
    print(f"Empty directory shard distribution:")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    # Clean up
    pod.clear()
    print("Cleared empty test data")


def test_corrupted_shard():
    """Test PodEMemory behavior with corrupted shard."""
    print("\nTesting PodEMemory with corrupted shard...")
    
    # Create a pod and add some data
    pod1 = PodEMemory("test_corrupted", dimension=32, max_elements_per_shard=2)
    
    # Add some records
    for i in range(3):
        record = Record(
            content=f"Test document {i}",
            vector=[float(i) * 0.1] * 32,
            attributes={"test_id": i}
        )
        pod1.add(record)
    
    print(f"Created pod with {pod1.count()} records")
    
    # Close pod
    del pod1
    
    # Manually create a corrupted shard directory
    corrupted_dir = Path("C:/Eric/projects/agent-engine/.memory/test_corrupted/test_corrupted_shard_999")
    corrupted_dir.mkdir(parents=True, exist_ok=True)
    
    # Create second pod (should skip corrupted shard)
    pod2 = PodEMemory("test_corrupted", dimension=32, max_elements_per_shard=2)
    
    shard_info = pod2.get_shard_info()
    print(f"Pod with corrupted shard distribution:")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    total_count = pod2.count()
    print(f"Total records: {total_count}")
    
    # Clean up
    pod2.clear()
    print("Cleared corrupted test data")


if __name__ == "__main__":
    print("Testing PodEMemory shard loading functionality...")
    
    try:
        test_shard_loading()
        test_empty_directory()
        test_corrupted_shard()
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
