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


File utility functions for agent engine.
"""
import os
import inspect
from pathlib import Path
from typing import Union


def get_current_file_dir() -> Path:
    """
    Get the directory of the current Python file that calls this function.
    
    Returns:
        Path: The directory path of the calling file
        
    Example:
        # If called from agent/skill.py, this will return the agent/ directory
        current_dir = get_current_file_dir()
        yaml_path = current_dir / "prompts.yaml"
    """
    # Get the frame of the calling function
    frame = inspect.currentframe()
    
    # Go up the call stack to find the actual calling file
    while frame:
        frame_info = inspect.getframeinfo(frame)
        filename = frame_info.filename
        
        # Skip if it's this utility function itself
        if 'file_utils.py' not in filename:
            # Return the directory of the calling file
            return Path(filename).parent
        
        frame = frame.f_back
    
    # Fallback: return current working directory
    return Path.cwd()


def get_relative_path_from_current_file(relative_path: Union[str, Path]) -> Path:
    """
    Get an absolute path by combining the current file's directory with a relative path.
    
    Args:
        relative_path (Union[str, Path]): The relative path from the current file's directory
        
    Returns:
        Path: The absolute path
        
    Example:
        # If called from agent/skill.py and you want agent/prompts.yaml
        yaml_path = get_relative_path_from_current_file("prompts.yaml")
        # This will resolve to the full path of agent/prompts.yaml
    """
    current_dir = get_current_file_dir()
    return current_dir / relative_path


def get_package_relative_path(relative_path: Union[str, Path]) -> Path:
    """
    Get a path relative to the current package directory.
    This is useful when you need to reference files within the same package.
    
    Args:
        relative_path (Union[str, Path]): The relative path from the package root
        
    Returns:
        Path: The absolute path
        
    Example:
        # If called from agent/skill.py and you want agent/prompts.yaml
        yaml_path = get_package_relative_path("agent/prompts.yaml")
        # This will resolve to the full path of agent/prompts.yaml
    """
    current_dir = get_current_file_dir()
    
    # Find the package root by looking for __init__.py files
    package_root = current_dir
    while package_root.parent.exists():
        if (package_root / "__init__.py").exists():
            break
        package_root = package_root.parent
    
    return package_root / relative_path
