"""
Test script for EventCache automatic sharding functionality.

This script tests whether EventCache properly creates new shards when the current
shard reaches its capacity limit.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.memory.e_memory.models import Record
from agents.ICUAssistantAgent.memory.event_cache import EventCache


def create_test_record(record_id: str, content: str = "test content") -> Record:
    """Create a test record with the given ID."""
    return Record(
        id=record_id,
        content=content,
        attributes={"test": True},
        vector=[0.1] * 3072  # Dummy vector for testing
    )


def test_sharding_with_small_capacity():
    """Test sharding with very small capacity (5 records per shard)."""
    print("=" * 60)
    print("Testing EventCache sharding with 5 records per shard")
    print("=" * 60)
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Create EventCache with very small capacity
        cache = EventCache(
            name="test_sharding",
            persist_dir=temp_dir,
            max_elements_per_shard=5,  # Only 5 records per shard
            dimension=3072
        )
        
        print(f"Initial shard count: {cache._shard_count}")
        print(f"Max elements per shard: {cache.max_elements_per_shard}")
        
        # Test 1: Add records one by one using add() method
        print("\n--- Test 1: Adding records one by one ---")
        for i in range(12):  # Add 12 records, should create 3 shards
            record_id = f"record_{i:03d}"
            record = create_test_record(record_id)
            
            result_id = cache.add(record)
            total_count = cache.count()
            shard_count = cache._shard_count
            
            print(f"Added record {i+1:2d}: ID={record_id}, Total={total_count}, Shards={shard_count}")
            
            # Check shard info
            shard_info = cache.get_shard_info()
            for shard in shard_info:
                print(f"  Shard {shard['shard_id']}: {shard['record_count']} records")
        
        print(f"\nFinal state after individual adds:")
        print(f"Total records: {cache.count()}")
        print(f"Total shards: {cache._shard_count}")
        
        # Test 2: Add records in batch using add_batch() method
        print("\n--- Test 2: Adding records in batch ---")
        
        # Create new cache for batch testing
        batch_cache = EventCache(
            name="test_batch_sharding",
            persist_dir=os.path.join(temp_dir, "batch_test"),
            max_elements_per_shard=5,
            dimension=3072
        )
        
        # Create batch of records
        batch_records = []
        for i in range(12):
            record_id = f"batch_record_{i:03d}"
            record = create_test_record(record_id)
            batch_records.append(record)
        
        print(f"Adding {len(batch_records)} records in batch...")
        record_ids = batch_cache.add_batch(batch_records)
        
        print(f"Batch add completed:")
        print(f"Total records: {batch_cache.count()}")
        print(f"Total shards: {batch_cache._shard_count}")
        
        # Show shard distribution
        shard_info = batch_cache.get_shard_info()
        for shard in shard_info:
            print(f"  Shard {shard['shard_id']}: {shard['record_count']} records")
        
        # Test 3: Verify records can be retrieved
        print("\n--- Test 3: Verifying record retrieval ---")
        for i in range(12):
            record_id = f"record_{i:03d}"
            retrieved = cache.get(record_id)
            if retrieved:
                print(f"✓ Found record: {record_id}")
            else:
                print(f"✗ Missing record: {record_id}")
        
        # Test 4: Check if sharding is actually working
        print("\n--- Test 4: Sharding analysis ---")
        
        # Check if records are distributed across multiple shards
        shard_info = cache.get_shard_info()
        active_shards = [s for s in shard_info if s['record_count'] > 0]
        
        print(f"Active shards: {len(active_shards)}")
        for shard in active_shards:
            print(f"  Shard {shard['shard_id']}: {shard['record_count']} records")
        
        # Check if any shard has more than max_elements_per_shard
        overloaded_shards = [s for s in active_shards if s['record_count'] > cache.max_elements_per_shard]
        if overloaded_shards:
            print(f"⚠️  WARNING: {len(overloaded_shards)} shards exceed capacity limit!")
            for shard in overloaded_shards:
                print(f"  Shard {shard['shard_id']}: {shard['record_count']} records (limit: {cache.max_elements_per_shard})")
        else:
            print("✓ All shards within capacity limits")
        
        return len(active_shards) > 1, len(overloaded_shards) == 0
        
    finally:
        # Clean up - close any open connections first
        try:
            # Force close any open database connections
            import gc
            gc.collect()
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {temp_dir}: {e}")


def test_sharding_logic_analysis():
    """Analyze the sharding logic to understand the issue."""
    print("\n" + "=" * 60)
    print("Analyzing EventCache sharding logic")
    print("=" * 60)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        cache = EventCache(
            name="analysis_test",
            persist_dir=temp_dir,
            max_elements_per_shard=3,
            dimension=3072
        )
        
        print("Current sharding logic analysis:")
        print(f"1. _get_available_shard(): Uses capacity check (current shard count < max_elements_per_shard)")
        print(f"2. _get_shard_for_record(): Uses hash-based distribution (hash % shard_count)")
        print(f"3. add() method: Uses _get_shard_for_record() - NOT _get_available_shard()")
        print(f"4. add_batch() method: Uses _get_shard_for_record() - NOT _get_available_shard()")
        
        print("\nProblem identified:")
        print("- add() and add_batch() use hash-based shard selection")
        print("- This means records are distributed by hash, not by capacity")
        print("- New shards are only created when hash requires a shard that doesn't exist")
        print("- But existing shards can exceed capacity because capacity is never checked")
        
        # Demonstrate the issue
        print("\nDemonstrating the issue:")
        
        # Add records with predictable IDs to see hash distribution
        for i in range(10):
            record_id = f"test_{i}"
            record = create_test_record(record_id)
            cache.add(record)
            
            # Show which shard each record goes to
            shard_id = cache._get_shard_for_record(record_id)
            print(f"Record '{record_id}' -> Shard {shard_id} (hash-based)")
        
        print(f"\nFinal shard distribution:")
        shard_info = cache.get_shard_info()
        for shard in shard_info:
            print(f"  Shard {shard['shard_id']}: {shard['record_count']} records")
        
    finally:
        try:
            # Force close any open database connections
            import gc
            gc.collect()
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up {temp_dir}: {e}")


if __name__ == "__main__":
    print("EventCache Sharding Test")
    print("========================")
    
    # Run sharding logic analysis first
    test_sharding_logic_analysis()
    
    # Run actual sharding test
    sharding_works, capacity_respected = test_sharding_with_small_capacity()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Multiple shards created: {'✓ YES' if sharding_works else '✗ NO'}")
    print(f"Capacity limits respected: {'✓ YES' if capacity_respected else '✗ NO'}")
    
    if not sharding_works:
        print("\n❌ ISSUE: EventCache is not creating multiple shards as expected")
        print("   The problem is that add() and add_batch() use hash-based shard selection")
        print("   instead of capacity-based shard selection.")
    
    if not capacity_respected:
        print("\n⚠️  WARNING: Some shards exceed their capacity limits")
        print("   This confirms that capacity checking is not working properly.")
    
    print("\nRecommendation: Modify add() and add_batch() methods to use")
    print("_get_available_shard() instead of _get_shard_for_record() for new records.")
