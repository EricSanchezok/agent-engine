"""
Safe cleanup utilities for Windows file locking issues.
"""

import os
import shutil
import time
import gc
import atexit
from pathlib import Path
from typing import List


class SafeCleanup:
    """Safe cleanup utility for handling Windows file locking issues."""
    
    def __init__(self):
        self.temp_dirs: List[str] = []
        # Register cleanup function to run on exit
        atexit.register(self.cleanup_all)
    
    def register_temp_dir(self, temp_dir: str):
        """Register a temporary directory for cleanup."""
        self.temp_dirs.append(temp_dir)
    
    def cleanup_dir(self, temp_dir: str, max_retries: int = 5) -> bool:
        """
        Safely clean up a directory with retry mechanism.
        
        Args:
            temp_dir: Directory to clean up
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if cleanup successful, False otherwise
        """
        if not os.path.exists(temp_dir):
            return True
        
        # Force garbage collection
        gc.collect()
        
        for attempt in range(max_retries):
            try:
                # Wait progressively longer between attempts
                if attempt > 0:
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                    time.sleep(wait_time)
                
                # Try to remove the directory
                shutil.rmtree(temp_dir)
                print(f"✅ Successfully cleaned up: {temp_dir}")
                return True
                
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Cleanup attempt {attempt + 1}/{max_retries} failed, retrying...")
                    continue
                else:
                    print(f"❌ Could not clean up {temp_dir} after {max_retries} attempts")
                    print(f"   Error: {e}")
                    print(f"   You may need to manually delete this directory later.")
                    return False
                    
            except Exception as e:
                print(f"❌ Unexpected error cleaning up {temp_dir}: {e}")
                return False
        
        return False
    
    def cleanup_all(self):
        """Clean up all registered temporary directories."""
        if not self.temp_dirs:
            return
        
        print(f"\n🧹 Cleaning up {len(self.temp_dirs)} temporary directories...")
        
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                self.cleanup_dir(temp_dir)


# Global cleanup manager
_cleanup_manager = SafeCleanup()


def register_temp_dir(temp_dir: str):
    """Register a temporary directory for automatic cleanup."""
    _cleanup_manager.register_temp_dir(temp_dir)


def safe_cleanup(temp_dir: str, max_retries: int = 5) -> bool:
    """Safely clean up a directory."""
    return _cleanup_manager.cleanup_dir(temp_dir, max_retries)


def cleanup_all():
    """Clean up all registered directories."""
    _cleanup_manager.cleanup_all()
