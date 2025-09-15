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


Project root directory finder utility.

This module provides functions to find the root directory of a project
using various methods including pyproject.toml, .gitignore, and fallback strategies.
"""

import os
import pathlib
from typing import Optional, Union


def find_project_root(start_path: Optional[Union[str, pathlib.Path]] = None) -> pathlib.Path:
    """
    Find the project root directory using multiple strategies.
    
    This function searches for the project root by looking for common project files
    and directories. It uses the following priority order:
    1. pyproject.toml file
    2. .gitignore file
    3. Fallback strategies (common project structure patterns)
    
    Args:
        start_path: Starting path for the search. If None, uses current working directory.
        
    Returns:
        Path to the project root directory.
        
    Raises:
        FileNotFoundError: If project root cannot be determined.
    """
    if start_path is None:
        start_path = pathlib.Path.cwd()
    elif isinstance(start_path, str):
        start_path = pathlib.Path(start_path)
    
    # Ensure the start path is absolute
    start_path = start_path.resolve()
    
    # Strategy 1: Look for pyproject.toml
    root = _find_by_pyproject_toml(start_path)
    if root:
        return root
    
    # Strategy 2: Look for .gitignore
    root = _find_by_gitignore(start_path)
    if root:
        return root
    
    # Strategy 3: Fallback strategies
    root = _find_by_fallback_strategies(start_path)
    if root:
        return root
    
    # If all strategies fail, raise an error
    raise FileNotFoundError(
        f"Could not determine project root from {start_path}. "
        "No pyproject.toml, .gitignore, or recognizable project structure found."
    )


def _find_by_pyproject_toml(start_path: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Find project root by looking for pyproject.toml file.
    
    Args:
        start_path: Starting path for the search.
        
    Returns:
        Path to directory containing pyproject.toml, or None if not found.
    """
    current = start_path
    
    while current != current.parent:
        pyproject_file = current / "pyproject.toml"
        if pyproject_file.exists() and pyproject_file.is_file():
            return current
        current = current.parent
    
    return None


def _find_by_gitignore(start_path: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Find project root by looking for .gitignore file.
    
    Args:
        start_path: Starting path for the search.
        
    Returns:
        Path to directory containing .gitignore, or None if not found.
    """
    current = start_path
    
    while current != current.parent:
        gitignore_file = current / ".gitignore"
        if gitignore_file.exists() and gitignore_file.is_file():
            return current
        current = current.parent
    
    return None


def _find_by_fallback_strategies(start_path: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Find project root using fallback strategies.
    
    This function looks for common project structure patterns:
    - src/ directory
    - tests/ or test/ directory
    - requirements.txt or setup.py
    - README.md
    - .git/ directory
    
    Args:
        start_path: Starting path for the search.
        
    Returns:
        Path to directory matching project structure patterns, or None if not found.
    """
    current = start_path
    
    while current != current.parent:
        # Check for common project structure indicators
        has_src = (current / "src").exists() and (current / "src").is_dir()
        has_tests = ((current / "tests").exists() and (current / "tests").is_dir()) or \
                   ((current / "test").exists() and (current / "test").is_dir())
        has_requirements = (current / "requirements.txt").exists() or \
                         (current / "setup.py").exists() or \
                         (current / "pyproject.toml").exists()
        has_readme = (current / "README.md").exists() or \
                    (current / "README.rst").exists() or \
                    (current / "README.txt").exists()
        has_git = (current / ".git").exists() and (current / ".git").is_dir()
        
        # If we find multiple indicators, this is likely the project root
        indicators = [has_src, has_tests, has_requirements, has_readme, has_git]
        if sum(indicators) >= 2:  # At least 2 indicators suggest this is a project root
            return current
        
        current = current.parent
    
    return None


def get_project_root() -> pathlib.Path:
    """
    Convenience function to get project root from current working directory.
    
    Returns:
        Path to the project root directory.
        
    Raises:
        FileNotFoundError: If project root cannot be determined.
    """
    return find_project_root()


def is_project_root(path: Union[str, pathlib.Path]) -> bool:
    """
    Check if the given path is a project root directory.
    
    Args:
        path: Path to check.
        
    Returns:
        True if the path appears to be a project root, False otherwise.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    
    path = path.resolve()
    
    # Check for pyproject.toml first (most reliable)
    if (path / "pyproject.toml").exists():
        return True
    
    # Check for .gitignore
    if (path / ".gitignore").exists():
        return True
    
    # Check for multiple project indicators
    indicators = [
        (path / "src").exists() and (path / "src").is_dir(),
        ((path / "tests").exists() and (path / "tests").is_dir()) or \
        ((path / "test").exists() and (path / "test").is_dir()),
        (path / "requirements.txt").exists() or \
        (path / "setup.py").exists(),
        (path / "README.md").exists() or \
        (path / "README.rst").exists() or \
        (path / "README.txt").exists(),
        (path / ".git").exists() and (path / ".git").is_dir()
    ]
    
    return sum(indicators) >= 2
