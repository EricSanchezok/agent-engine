"""
Simple test to verify basic sharding logic works.

This test bypasses EMemory complexity and focuses on the core sharding logic.
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
        content=f"Simple test content for {record_id}",
        vector=[0.1] * 3072
    )


def main():
    """Simple sharding test."""
    print("Simple EventCache Sharding Test")
    print("===============================")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Temp directory: {temp_dir}")
    
    try:
        # Create EventCache with very small capacity
        cache = EventCache(
            name="simple_test",
            persist_dir=temp_dir,
            max_elements_per_shard=2,  # Only 2 records per shard
            dimension=3072
        )
        
        print(f"Initial state:")
        print(f"  Shards: {cache._shard_count}")
        print(f"  Current shard: {cache._current_shard}")
        print(f"  Max per shard: {cache.max_elements_per_shard}")
        
        # Add records one by one
        print("\nAdding records...")
        for i in range(6):  # Should create 3 shards (2+2+2)
            record_id = f"record_{i}"
            record = create_test_record(record_id)
            
            print(f"Adding record {i+1}: {record_id}")
            
            try:
                result_id = cache.add(record)
                print(f"  ✓ Added successfully")
                
                # Show current state
                print(f"  Current shard: {cache._current_shard}")
                print(f"  Total shards: {cache._shard_count}")
                
            except Exception as e:
                print(f"  ✗ Failed to add: {e}")
                break
            
            print()
        
        # Check final state
        print("Final state:")
        print(f"  Total records: {cache.count()}")
        print(f"  Total shards: {cache._shard_count}")
        print(f"  Current shard: {cache._current_shard}")
        
        # Show shard distribution
        shard_info = cache.get_shard_info()
        for shard in shard_info:
            print(f"    Shard {shard['shard_id']}: {shard['record_count']} records")
        
        # Check if sharding worked
        active_shards = [s for s in shard_info if s['record_count'] > 0]
        overloaded_shards = [s for s in active_shards if s['record_count'] > cache.max_elements_per_shard]
        
        print(f"\nResults:")
        print(f"  Active shards: {len(active_shards)}")
        print(f"  Overloaded shards: {len(overloaded_shards)}")
        
        if len(active_shards) > 1 and len(overloaded_shards) == 0:
            print("\n🎉 SUCCESS: Sharding is working!")
        else:
            print("\n❌ ISSUE: Sharding not working properly")
        
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
