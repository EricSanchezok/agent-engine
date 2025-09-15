"""
Test PodEMemory with proper context manager usage.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.memory.e_memory.pod_ememory import PodEMemory
from agent_engine.memory.e_memory.models import Record


def test_context_manager():
    """Test PodEMemory with context manager for proper cleanup."""
    temp_dir = tempfile.mkdtemp(prefix="context_manager_test_")
    print(f"Test directory: {temp_dir}")
    
    try:
        # Use context manager for automatic cleanup
        with PodEMemory(
            name="test_pod",
            persist_dir=temp_dir,
            max_elements_per_shard=2,
            distance_metric="cosine"
        ) as pod_memory:
            
            print("PodEMemory created with context manager")
            
            # Add some test records
            records = []
            for i in range(4):
                record = Record(
                    id=f"record_{i}",
                    content=f"Content {i}",
                    vector=[0.1 * i, 0.2 * i, 0.3 * i],
                    attributes={"index": i}
                )
                records.append(record)
                success = pod_memory.add(record)
                print(f"Added record_{i}: success={success}")
            
            # Test operations
            print("\n=== Testing Operations ===")
            for record in records:
                exists = pod_memory.exists(record.id)
                has_vector = pod_memory.has_vector(record.id)
                print(f"{record.id}: exists={exists}, has_vector={has_vector}")
            
            # Test batch operations
            record_ids = [r.id for r in records]
            exists_batch = pod_memory.exists_batch(record_ids)
            has_vector_batch = pod_memory.has_vector_batch(record_ids)
            
            print(f"\nBatch operations:")
            print(f"exists_batch: {exists_batch}")
            print(f"has_vector_batch: {has_vector_batch}")
            
            print("\nContext manager will automatically close connections...")
        
        # Test cleanup
        print("\n=== Testing Cleanup ===")
        import shutil
        import time
        import gc
        
        # Force garbage collection
        gc.collect()
        time.sleep(0.2)  # Wait for file handles to be released
        
        try:
            shutil.rmtree(temp_dir)
            print(f"✅ Successfully cleaned up test directory: {temp_dir}")
            return True
        except PermissionError as e:
            print(f"❌ Still cannot clean up directory: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during cleanup: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_manual_close():
    """Test manual close method."""
    temp_dir = tempfile.mkdtemp(prefix="manual_close_test_")
    print(f"\nManual close test directory: {temp_dir}")
    
    try:
        # Create PodEMemory without context manager
        pod_memory = PodEMemory(
            name="test_pod",
            persist_dir=temp_dir,
            max_elements_per_shard=2,
            distance_metric="cosine"
        )
        
        print("PodEMemory created, testing manual close...")
        
        # Add a test record
        record = Record(
            id="test_record",
            content="Test content",
            vector=[0.1, 0.2, 0.3],
            attributes={"test": True}
        )
        success = pod_memory.add(record)
        print(f"Added test record: success={success}")
        
        # Manually close
        pod_memory.close()
        print("Manually closed PodEMemory")
        
        # Test cleanup
        import shutil
        import time
        import gc
        
        gc.collect()
        time.sleep(0.2)
        
        try:
            shutil.rmtree(temp_dir)
            print(f"✅ Successfully cleaned up manual test directory: {temp_dir}")
            return True
        except PermissionError as e:
            print(f"❌ Still cannot clean up directory: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during cleanup: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Manual close test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Testing PodEMemory with proper resource management")
    print("=" * 60)
    
    # Test context manager
    context_success = test_context_manager()
    
    # Test manual close
    manual_success = test_manual_close()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Context Manager Test: {'✅ PASS' if context_success else '❌ FAIL'}")
    print(f"Manual Close Test: {'✅ PASS' if manual_success else '❌ FAIL'}")
    
    overall_success = context_success and manual_success
    print(f"Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
