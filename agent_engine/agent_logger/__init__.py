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


from .agent_logger import AgentLogger
from .log_directory_manager import (
    LogDirectoryManager, 
    set_agent_log_directory, 
    get_agent_log_directory, 
    clear_agent_log_directory
)

__all__ = [
    'AgentLogger', 
    'LogDirectoryManager',
    'set_agent_log_directory',
    'get_agent_log_directory', 
    'clear_agent_log_directory'
]
