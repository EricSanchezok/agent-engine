"""
Comprehensive test for EventCache sharding with 5 records per shard.

This test demonstrates the sharding issue with a specific configuration
of 5 records per shard.
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
        content=f"Comprehensive test content for {record_id}",
        vector=[0.1] * 3072
    )


def test_individual_adds():
    """Test adding records one by one."""
    print("=" * 60)
    print("Test 1: Adding records individually (5 per shard)")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        cache = EventCache(
            name="individual_test",
            persist_dir=temp_dir,
            max_elements_per_shard=5,  # 5 records per shard
            dimension=3072
        )
        
        print(f"Initial shards: {cache._shard_count}")
        print(f"Max per shard: {cache.max_elements_per_shard}")
        
        # Add 15 records - should create 3 shards
        print("\nAdding 15 records individually...")
        for i in range(15):
            record_id = f"individual_{i}"
            record = create_test_record(record_id)
            
            try:
                result_id = cache.add(record)
                print(f"Record {i+1:2d}: {record_id} -> Shard {cache._get_shard_for_record(record_id)}")
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


def test_batch_adds():
    """Test adding records in batches."""
    print("\n" + "=" * 60)
    print("Test 2: Adding records in batches (5 per shard)")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        cache = EventCache(
            name="batch_test",
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
            record_id = f"batch_{i}"
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


def analyze_sharding_logic():
    """Analyze why sharding is not working."""
    print("\n" + "=" * 60)
    print("Analysis: Why sharding is not working")
    print("=" * 60)
    
    print("Current EventCache logic:")
    print("1. add() method uses _get_shard_for_record(record_id)")
    print("2. _get_shard_for_record() uses: hash(record_id) % shard_count")
    print("3. This means records are distributed by hash, not by capacity")
    print("4. New shards are only created when hash % shard_count requires a non-existent shard")
    print("5. But existing shards can exceed capacity because capacity is never checked")
    
    print("\nThe problem:")
    print("- _get_available_shard() exists and checks capacity properly")
    print("- But add() and add_batch() don't use _get_available_shard()")
    print("- Instead they use _get_shard_for_record() which ignores capacity")
    
    print("\nExpected behavior with max_elements_per_shard=5:")
    print("- Records 1-5 should go to Shard 0")
    print("- Records 6-10 should go to Shard 1") 
    print("- Records 11-15 should go to Shard 2")
    
    print("\nActual behavior:")
    print("- All records go to Shard 0 (based on hash distribution)")
    print("- Shard 0 exceeds capacity limit")
    print("- HNSW index reports 'exceeds the specified limit' errors")


def main():
    """Run comprehensive sharding test."""
    print("EventCache Comprehensive Sharding Test")
    print("======================================")
    print("Testing with max_elements_per_shard=5")
    print("Expected: 15 records should create 3 shards (5 records each)")
    
    # Test individual adds
    count1, shards1, info1 = test_individual_adds()
    
    # Test batch adds
    count2, shards2, info2 = test_batch_adds()
    
    # Analyze the problem
    analyze_sharding_logic()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Individual adds: {count1} records in {shards1} shards")
    print(f"Batch adds: {count2} records in {shards2} shards")
    
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
    
    if len(active_shards1) == 1 and len(overloaded1) > 0:
        print("\n❌ CONFIRMED: Individual add method has sharding issue")
    if len(active_shards2) == 1 and len(overloaded2) > 0:
        print("❌ CONFIRMED: Batch add method has sharding issue")
    
    print("\n🔧 RECOMMENDATION:")
    print("Modify EventCache.add() and EventCache.add_batch() methods")
    print("to use _get_available_shard() instead of _get_shard_for_record()")
    print("for capacity-based shard selection.")


if __name__ == "__main__":
    main()
