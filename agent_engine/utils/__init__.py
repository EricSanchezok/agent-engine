"""
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


Tools package for agent engine.
"""
from .project_root import find_project_root
from .network_utils import get_local_ip, get_all_local_ips, get_public_ip
from .file_utils import get_current_file_dir, get_relative_path_from_current_file, get_package_relative_path
from .id_utils import generate_unique_id, generate_simple_unique_id, generate_uuid_based_id

__all__ = [
    'find_project_root', 
    'get_local_ip', 
    'get_all_local_ips', 
    'get_public_ip',
    'get_current_file_dir',
    'get_relative_path_from_current_file', 
    'get_package_relative_path',
    'generate_unique_id',
    'generate_simple_unique_id',
    'generate_uuid_based_id'
]