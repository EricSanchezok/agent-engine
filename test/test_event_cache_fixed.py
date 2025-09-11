"""
Test EventCache after fixing the sharding issue.

This test verifies that the sharding fix works correctly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.memory.e_memory.models import Record
from agents.ICUAssistantAgent.memory.event_cache import EventCache


def create_test_record(record_id: str) -> Record:
    """Create a simple test record."""
    return Record(
        id=record_id,
        content=f"Fixed test content for {record_id}",
        vector=[0.1] * 3072
    )


def test_individual_adds_fixed():
    """Test adding records one by one after fix."""
    print("=" * 60)
    print("Test 1: Individual adds after sharding fix (5 per shard)")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        cache = EventCache(
            name="fixed_individual_test",
            persist_dir=temp_dir,
            max_elements_per_shard=5,  # 5 records per shard
            dimension=3072
        )
        
        print(f"Initial shards: {cache._shard_count}")
        print(f"Max per shard: {cache.max_elements_per_shard}")
        
        # Add 15 records - should create 3 shards
        print("\nAdding 15 records individually...")
        for i in range(15):
            record_id = f"fixed_individual_{i}"
            record = create_test_record(record_id)
            
            try:
                result_id = cache.add(record)
                # Get the shard this record was added to
                shard_id = cache._get_available_shard()
                print(f"Record {i+1:2d}: {record_id} -> Added successfully")
            except Exception as e:
                print(f"Record {i+1:2d}: {record_id} - ERROR: {e}")
        
        # Check final state
        print(f"\nFinal state:")
        print(f"Total records: {cache.count()}")
        print(f"Total shards: {cache._shard_count}")
        
        shard_info = cache.get_shard_info()
        for shard in shard_info:
            print(f"  Shard {shard['shard_id']}: {shard['record_count']} records")
        
        return cache.count(), cache._shard_count, shard_info
        
    finally:
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def test_batch_adds_fixed():
    """Test adding records in batches after fix."""
    print("\n" + "=" * 60)
    print("Test 2: Batch adds after sharding fix (5 per shard)")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        cache = EventCache(
            name="fixed_batch_test",
            persist_dir=temp_dir,
            max_elements_per_shard=5,  # 5 records per shard
            dimension=3072
        )
        
        print(f"Initial shards: {cache._shard_count}")
        print(f"Max per shard: {cache.max_elements_per_shard}")
        
        # Add 15 records in batch - should create 3 shards
        print("\nAdding 15 records in batch...")
        batch_records = []
        for i in range(15):
            record_id = f"fixed_batch_{i}"
            record = create_test_record(record_id)
            batch_records.append(record)
        
        try:
            record_ids = cache.add_batch(batch_records)
            print(f"✓ Batch add completed, returned {len(record_ids)} IDs")
        except Exception as e:
            print(f"✗ Batch add failed: {e}")
        
        # Check final state
        print(f"\nFinal state:")
        print(f"Total records: {cache.count()}")
        print(f"Total shards: {cache._shard_count}")
        
        shard_info = cache.get_shard_info()
        for shard in shard_info:
            print(f"  Shard {shard['shard_id']}: {shard['record_count']} records")
        
        return cache.count(), cache._shard_count, shard_info
        
    finally:
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def test_retrieval_after_fix():
    """Test that records can still be retrieved after the fix."""
    print("\n" + "=" * 60)
    print("Test 3: Record retrieval after sharding fix")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        cache = EventCache(
            name="retrieval_test",
            persist_dir=temp_dir,
            max_elements_per_shard=3,  # 3 records per shard for testing
            dimension=3072
        )
        
        # Add some records
        test_records = []
        for i in range(10):
            record_id = f"retrieval_test_{i}"
            record = create_test_record(record_id)
            test_records.append(record)
            cache.add(record)
        
        print(f"Added {len(test_records)} records")
        print(f"Total shards: {cache._shard_count}")
        
        # Try to retrieve each record
        print("\nTesting record retrieval:")
        success_count = 0
        for i, record in enumerate(test_records):
            retrieved = cache.get(record.id)
            if retrieved and retrieved.id == record.id:
                print(f"✓ Record {i+1}: {record.id} retrieved successfully")
                success_count += 1
            else:
                print(f"✗ Record {i+1}: {record.id} retrieval failed")
        
        print(f"\nRetrieval success rate: {success_count}/{len(test_records)} ({success_count/len(test_records)*100:.1f}%)")
        
        return success_count == len(test_records)
        
    finally:
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def main():
    """Run comprehensive test after sharding fix."""
    print("EventCache Sharding Fix Test")
    print("============================")
    print("Testing the fixed sharding functionality")
    
    # Test individual adds
    count1, shards1, info1 = test_individual_adds_fixed()
    
    # Test batch adds
    count2, shards2, info2 = test_batch_adds_fixed()
    
    # Test retrieval
    retrieval_success = test_retrieval_after_fix()
    
    # Summary
    print("\n" + "=" * 60)
    print("FIX TEST RESULTS")
    print("=" * 60)
    
    print(f"Individual adds: {count1} records in {shards1} shards")
    print(f"Batch adds: {count2} records in {shards2} shards")
    print(f"Record retrieval: {'✓ SUCCESS' if retrieval_success else '✗ FAILED'}")
    
    # Check if sharding worked
    active_shards1 = [s for s in info1 if s['record_count'] > 0]
    active_shards2 = [s for s in info2 if s['record_count'] > 0]
    
    overloaded1 = [s for s in active_shards1 if s['record_count'] > 5]
    overloaded2 = [s for s in active_shards2 if s['record_count'] > 5]
    
    print(f"\nIndividual adds:")
    print(f"  Active shards: {len(active_shards1)}")
    print(f"  Overloaded shards: {len(overloaded1)}")
    
    print(f"\nBatch adds:")
    print(f"  Active shards: {len(active_shards2)}")
    print(f"  Overloaded shards: {len(overloaded2)}")
    
    # Determine if fix worked
    fix_worked = (
        len(active_shards1) > 1 and len(overloaded1) == 0 and
        len(active_shards2) > 1 and len(overloaded2) == 0 and
        retrieval_success
    )
    
    if fix_worked:
        print("\n🎉 SUCCESS: Sharding fix is working correctly!")
        print("   ✓ Multiple shards created")
        print("   ✓ No overloaded shards")
        print("   ✓ Records can be retrieved")
    else:
        print("\n❌ ISSUE: Sharding fix may not be working properly")
        if len(active_shards1) == 1:
            print("   - Individual adds: Still only creating one shard")
        if len(overloaded1) > 0:
            print("   - Individual adds: Some shards are overloaded")
        if len(active_shards2) == 1:
            print("   - Batch adds: Still only creating one shard")
        if len(overloaded2) > 0:
            print("   - Batch adds: Some shards are overloaded")
        if not retrieval_success:
            print("   - Record retrieval is failing")


if __name__ == "__main__":
    main()
