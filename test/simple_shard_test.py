"""
Simple test to understand the shard behavior.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.memory.e_memory.pod_ememory import PodEMemory
from agent_engine.memory.e_memory.models import Record


def simple_test():
    """Simple test with just one record per shard."""
    temp_dir = tempfile.mkdtemp(prefix="simple_shard_test_")
    print(f"Test directory: {temp_dir}")
    
    # Initialize PodEMemory with small shard size
    pod_memory = PodEMemory(
        name="test_pod",
        persist_dir=temp_dir,
        max_elements_per_shard=1,  # Only 1 record per shard
        distance_metric="cosine"
    )
    
    # Add just 2 records to create 2 shards
    records = []
    for i in range(2):
        record = Record(
            id=f"record_{i}",
            content=f"Content {i}",
            vector=[0.1 * i, 0.2 * i, 0.3 * i],
            attributes={"index": i}
        )
        records.append(record)
        success = pod_memory.add(record)
        print(f"Added record_{i}: success={success}")
    
    # Print shard info
    shard_info = pod_memory.get_shard_info()
    print(f"\nShard info:")
    for info in shard_info:
        print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    # Test individual operations
    print(f"\n=== Individual Operations ===")
    for record in records:
        exists = pod_memory.exists(record.id)
        has_vector = pod_memory.has_vector(record.id)
        print(f"{record.id}: exists={exists}, has_vector={has_vector}")
    
    # Test batch operations
    print(f"\n=== Batch Operations ===")
    record_ids = [r.id for r in records]
    
    exists_batch = pod_memory.exists_batch(record_ids)
    print(f"exists_batch: {exists_batch}")
    
    has_vector_batch = pod_memory.has_vector_batch(record_ids)
    print(f"has_vector_batch: {has_vector_batch}")
    
    # Test each shard individually
    print(f"\n=== Shard Individual Tests ===")
    for shard_id, shard in pod_memory._shards.items():
        print(f"\nShard {shard_id}:")
        
        # Test all records on this shard
        for record in records:
            exists = shard.exists(record.id)
            has_vector = shard.has_vector(record.id)
            print(f"  {record.id}: exists={exists}, has_vector={has_vector}")
        
        # Test batch operations on this shard
        shard_exists_batch = shard.exists_batch(record_ids)
        shard_has_vector_batch = shard.has_vector_batch(record_ids)
        print(f"  exists_batch({record_ids}): {shard_exists_batch}")
        print(f"  has_vector_batch({record_ids}): {shard_has_vector_batch}")


if __name__ == "__main__":
    simple_test()
