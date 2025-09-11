"""
Test EventCache add_batch method sharding.

This test verifies that add_batch method also has the same sharding issue.
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
        content=f"Batch test content for {record_id}",
        vector=[0.1] * 3072
    )


def main():
    """Test add_batch sharding."""
    print("EventCache Batch Sharding Test")
    print("==============================")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Temp directory: {temp_dir}")
    
    try:
        # Create EventCache with small capacity
        cache = EventCache(
            name="batch_test",
            persist_dir=temp_dir,
            max_elements_per_shard=5,  # 5 records per shard
            dimension=3072
        )
        
        print(f"Initial shards: {cache._shard_count}")
        print(f"Max per shard: {cache.max_elements_per_shard}")
        
        # Create batch of 12 records - should create multiple shards
        print("\nCreating batch of 12 records...")
        batch_records = []
        for i in range(12):
            record_id = f"batch_record_{i}"
            record = create_test_record(record_id)
            batch_records.append(record)
        
        print(f"Adding batch of {len(batch_records)} records...")
        try:
            record_ids = cache.add_batch(batch_records)
            print(f"✓ Batch add completed, returned {len(record_ids)} IDs")
        except Exception as e:
            print(f"✗ Batch add failed: {e}")
        
        # Check final state
        print(f"\nFinal state:")
        print(f"Total records: {cache.count()}")
        print(f"Total shards: {cache._shard_count}")
        
        # Show shard distribution
        shard_info = cache.get_shard_info()
        for shard in shard_info:
            print(f"  Shard {shard['shard_id']}: {shard['record_count']} records")
        
        # Check if sharding worked
        active_shards = [s for s in shard_info if s['record_count'] > 0]
        overloaded_shards = [s for s in active_shards if s['record_count'] > cache.max_elements_per_shard]
        
        print(f"\nResults:")
        print(f"Active shards: {len(active_shards)}")
        print(f"Overloaded shards: {len(overloaded_shards)}")
        
        if len(active_shards) == 1 and len(overloaded_shards) > 0:
            print("\n❌ PROBLEM CONFIRMED: add_batch also has sharding issue!")
            print("   Only one shard created and it's overloaded.")
        elif len(active_shards) > 1:
            print("\n✓ Multiple shards created - batch sharding might be working")
        else:
            print("\n? Unexpected result")
            
    finally:
        # Clean up
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up: {temp_dir}")
        except Exception as e:
            print(f"Cleanup warning: {e}")


if __name__ == "__main__":
    main()
