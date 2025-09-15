"""
Test PodEMemory batch operations (exists_batch and has_vector_batch).

This test verifies that the batch methods work correctly by:
1. Creating a PodEMemory instance with multiple shards
2. Adding records with and without vectors
3. Testing exists_batch and has_vector_batch operations
4. Verifying results match individual operations
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import List

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.memory.e_memory.pod_ememory import PodEMemory
from agent_engine.memory.e_memory.models import Record


class PodEMemoryBatchTester:
    """Test class for PodEMemory batch operations."""
    
    def __init__(self):
        """Initialize the tester."""
        self.temp_dir = None
        self.pod_memory = None
        self.test_records = []
        
    def setup(self):
        """Set up test environment."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp(prefix="pod_ememory_test_")
        print(f"Test directory: {self.temp_dir}")
        
        # Initialize PodEMemory with small shard size for testing
        self.pod_memory = PodEMemory(
            name="test_pod",
            persist_dir=self.temp_dir,
            max_elements_per_shard=3,  # Small shard size to test multiple shards
            distance_metric="cosine"
        )
        
        print(f"PodEMemory initialized with max_elements_per_shard=3")
        
    def create_test_records(self) -> List[Record]:
        """Create test records with and without vectors."""
        records = []
        
        # Records with vectors
        for i in range(10):
            record = Record(
                id=f"record_with_vector_{i}",
                content=f"Content for record {i} with vector",
                vector=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],  # Simple test vector
                attributes={"type": "with_vector", "index": i}
            )
            records.append(record)
        
        # Records without vectors
        for i in range(5):
            record = Record(
                id=f"record_without_vector_{i}",
                content=f"Content for record {i} without vector",
                vector=None,  # No vector
                attributes={"type": "without_vector", "index": i}
            )
            records.append(record)
        
        # Non-existent record IDs for testing
        self.non_existent_ids = [f"non_existent_{i}" for i in range(3)]
        
        self.test_records = records
        return records
    
    def add_records_to_memory(self):
        """Add test records to memory."""
        print(f"Adding {len(self.test_records)} test records to PodEMemory...")
        
        # Add records one by one to test individual operations
        for record in self.test_records:
            success = self.pod_memory.add(record)
            if not success:
                print(f"Failed to add record {record.id}")
        
        print(f"Successfully added {len(self.test_records)} records")
        
        # Print shard info
        shard_info = self.pod_memory.get_shard_info()
        print(f"Created {len(shard_info)} shards:")
        for info in shard_info:
            print(f"  Shard {info['shard_id']}: {info['record_count']} records")
    
    def test_individual_operations(self):
        """Test individual exists and has_vector operations."""
        print("\n=== Testing Individual Operations ===")
        
        individual_exists = {}
        individual_has_vector = {}
        
        all_ids = [r.id for r in self.test_records] + self.non_existent_ids
        
        for record_id in all_ids:
            exists = self.pod_memory.exists(record_id)
            has_vector = self.pod_memory.has_vector(record_id)
            
            individual_exists[record_id] = exists
            individual_has_vector[record_id] = has_vector
            
            print(f"  {record_id}: exists={exists}, has_vector={has_vector}")
        
        return individual_exists, individual_has_vector
    
    def test_batch_operations(self):
        """Test batch exists and has_vector operations."""
        print("\n=== Testing Batch Operations ===")
        
        all_ids = [r.id for r in self.test_records] + self.non_existent_ids
        
        # Test exists_batch
        print("Testing exists_batch...")
        batch_exists = self.pod_memory.exists_batch(all_ids)
        print(f"exists_batch results: {batch_exists}")
        
        # Test has_vector_batch
        print("Testing has_vector_batch...")
        batch_has_vector = self.pod_memory.has_vector_batch(all_ids)
        print(f"has_vector_batch results: {batch_has_vector}")
        
        return batch_exists, batch_has_vector
    
    def compare_results(self, individual_exists, individual_has_vector, batch_exists, batch_has_vector):
        """Compare individual vs batch operation results."""
        print("\n=== Comparing Results ===")
        
        all_ids = list(individual_exists.keys())
        
        # Compare exists results
        exists_match = True
        print("Comparing exists results:")
        for record_id in all_ids:
            individual = individual_exists[record_id]
            batch = batch_exists[record_id]
            match = individual == batch
            exists_match = exists_match and match
            
            status = "✓" if match else "✗"
            print(f"  {status} {record_id}: individual={individual}, batch={batch}")
        
        # Compare has_vector results
        has_vector_match = True
        print("\nComparing has_vector results:")
        for record_id in all_ids:
            individual = individual_has_vector[record_id]
            batch = batch_has_vector[record_id]
            match = individual == batch
            has_vector_match = has_vector_match and match
            
            status = "✓" if match else "✗"
            print(f"  {status} {record_id}: individual={individual}, batch={batch}")
        
        print(f"\n=== Final Results ===")
        print(f"exists_batch correctness: {'PASS' if exists_match else 'FAIL'}")
        print(f"has_vector_batch correctness: {'PASS' if has_vector_match else 'FAIL'}")
        
        return exists_match and has_vector_match
    
    def test_performance(self):
        """Test performance comparison between individual and batch operations."""
        print("\n=== Performance Test ===")
        
        import time
        
        all_ids = [r.id for r in self.test_records] + self.non_existent_ids
        
        # Test individual operations timing
        start_time = time.time()
        for _ in range(3):  # Run multiple times for average
            for record_id in all_ids:
                self.pod_memory.exists(record_id)
                self.pod_memory.has_vector(record_id)
        individual_time = time.time() - start_time
        
        # Test batch operations timing
        start_time = time.time()
        for _ in range(3):  # Run multiple times for average
            self.pod_memory.exists_batch(all_ids)
            self.pod_memory.has_vector_batch(all_ids)
        batch_time = time.time() - start_time
        
        print(f"Individual operations time: {individual_time:.4f} seconds")
        print(f"Batch operations time: {batch_time:.4f} seconds")
        print(f"Speedup: {individual_time / batch_time:.2f}x")
        
        return individual_time, batch_time
    
    def cleanup(self):
        """Clean up test environment with safe file handling."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            import time
            import gc
            
            # Force garbage collection to release any remaining file handles
            gc.collect()
            
            # Try to clean up with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # First, try to close any open connections explicitly
                    if hasattr(self, 'pod_memory') and self.pod_memory:
                        # Close all shard connections
                        for shard in self.pod_memory._shards.values():
                            if hasattr(shard, 'chroma_client'):
                                try:
                                    shard.chroma_client = None
                                except:
                                    pass
                    
                    # Force garbage collection again
                    gc.collect()
                    
                    # Wait a bit for file handles to be released
                    time.sleep(0.1)
                    
                    # Try to remove the directory
                    shutil.rmtree(self.temp_dir)
                    print(f"Successfully cleaned up test directory: {self.temp_dir}")
                    break
                    
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        print(f"Cleanup attempt {attempt + 1} failed, retrying in 1 second...")
                        time.sleep(1)
                        continue
                    else:
                        print(f"Warning: Could not clean up test directory {self.temp_dir}")
                        print(f"Error: {e}")
                        print("You may need to manually delete this directory later.")
                except Exception as e:
                    print(f"Unexpected error during cleanup: {e}")
                    break
    
    def run_all_tests(self):
        """Run all tests."""
        try:
            print("Starting PodEMemory Batch Operations Test")
            print("=" * 50)
            
            # Setup
            self.setup()
            
            # Create and add test records
            self.create_test_records()
            self.add_records_to_memory()
            
            # Run tests
            individual_exists, individual_has_vector = self.test_individual_operations()
            batch_exists, batch_has_vector = self.test_batch_operations()
            
            # Compare results
            all_correct = self.compare_results(
                individual_exists, individual_has_vector,
                batch_exists, batch_has_vector
            )
            
            # Performance test
            self.test_performance()
            
            # Final result
            print(f"\n{'=' * 50}")
            print(f"Overall Test Result: {'PASS' if all_correct else 'FAIL'}")
            
            return all_correct
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            self.cleanup()


def main():
    """Main test function."""
    tester = PodEMemoryBatchTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
