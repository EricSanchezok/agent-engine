import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))


import os
import threading
from pathlib import Path
from typing import Optional


class LogDirectoryManager:
    """
    Manages log directory for agent processes.
    Uses thread-local storage to ensure each thread/process has its own log directory.
    """
    
    def __init__(self):
        self._thread_local = threading.local()
    
    def set_agent_log_dir(self, log_dir: str) -> None:
        """
        Set the log directory for the current thread/process.
        
        Args:
            log_dir: Path to the log directory
        """
        self._thread_local.agent_log_dir = str(log_dir)
        # Set environment variable for the current process
        os.environ['AGENT_LOG_DIR'] = str(log_dir)
    
    def get_agent_log_dir(self) -> Optional[str]:
        """
        Get the log directory for the current thread/process.
        
        Returns:
            Log directory path or None if not set
        """
        return getattr(self._thread_local, 'agent_log_dir', None)
    
    def clear_agent_log_dir(self) -> None:
        """Clear the log directory for the current thread/process."""
        if hasattr(self._thread_local, 'agent_log_dir'):
            delattr(self._thread_local, 'agent_log_dir')
        # Remove environment variable
        if 'AGENT_LOG_DIR' in os.environ:
            del os.environ['AGENT_LOG_DIR']


# Global instance
log_directory_manager = LogDirectoryManager()


def set_agent_log_directory(log_dir: str) -> None:
    """
    Convenience function to set the agent log directory.
    
    Args:
        log_dir: Path to the log directory
    """
    log_directory_manager.set_agent_log_dir(log_dir)


def get_agent_log_directory() -> Optional[str]:
    """
    Convenience function to get the current agent log directory.
    
    Returns:
        Log directory path or None if not set
    """
    return log_directory_manager.get_agent_log_dir()


def clear_agent_log_directory() -> None:
    """Convenience function to clear the current agent log directory."""
    log_directory_manager.clear_agent_log_dir()
