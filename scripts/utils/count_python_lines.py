#!/usr/bin/env python3
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


Script to count total lines in all Python files in the project,
excluding files and directories specified in .gitignore.
"""

import os
import fnmatch
from pathlib import Path
from typing import List, Set


def load_gitignore_patterns(gitignore_path: str) -> List[str]:
    """Load patterns from .gitignore file."""
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    patterns.append(line)
    return patterns


def should_ignore_path(path: str, gitignore_patterns: List[str]) -> bool:
    """Check if a path should be ignored based on gitignore patterns."""
    path_obj = Path(path)
    
    # Check each pattern
    for pattern in gitignore_patterns:
        # Handle directory patterns (ending with /)
        if pattern.endswith('/'):
            dir_pattern = pattern[:-1]
            if fnmatch.fnmatch(path_obj.name, dir_pattern) or fnmatch.fnmatch(str(path_obj), f"*/{dir_pattern}/*"):
                return True
        # Handle file patterns
        elif fnmatch.fnmatch(path_obj.name, pattern) or fnmatch.fnmatch(str(path_obj), f"*/{pattern}"):
            return True
        # Handle patterns that match any part of the path
        elif fnmatch.fnmatch(str(path_obj), pattern):
            return True
    
    return False


def count_lines_in_file(file_path: str) -> int:
    """Count lines in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except (UnicodeDecodeError, PermissionError, OSError):
        # Skip files that can't be read
        return 0


def find_python_files(root_dir: str, gitignore_patterns: List[str]) -> List[str]:
    """Find all Python files in the project, excluding gitignore patterns."""
    python_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Remove ignored directories from dirs list to prevent walking into them
        dirs[:] = [d for d in dirs if not should_ignore_path(os.path.join(root, d), gitignore_patterns)]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if not should_ignore_path(file_path, gitignore_patterns):
                    python_files.append(file_path)
    
    return python_files


def main():
    """Main function to count Python file lines."""
    # Get project root directory (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    print(f"Project root: {project_root}")
    
    # Load gitignore patterns
    gitignore_path = project_root / '.gitignore'
    gitignore_patterns = load_gitignore_patterns(str(gitignore_path))
    print(f"Loaded {len(gitignore_patterns)} gitignore patterns")
    
    # Find all Python files
    python_files = find_python_files(str(project_root), gitignore_patterns)
    print(f"Found {len(python_files)} Python files")
    
    # Count lines in each file
    total_lines = 0
    file_stats = []
    
    for file_path in python_files:
        lines = count_lines_in_file(file_path)
        total_lines += lines
        file_stats.append((file_path, lines))
    
    # Sort files by line count (descending)
    file_stats.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\nTotal lines in Python files: {total_lines:,}")
    print(f"Total Python files: {len(python_files)}")
    
    # Print top 10 files by line count
    print(f"\nTop 10 files by line count:")
    print("-" * 80)
    for i, (file_path, lines) in enumerate(file_stats[:10]):
        relative_path = os.path.relpath(file_path, project_root)
        print(f"{i+1:2d}. {relative_path:<50} {lines:>6,} lines")
    
    # Print summary statistics
    if file_stats:
        avg_lines = total_lines / len(file_stats)
        max_lines = max(file_stats, key=lambda x: x[1])[1]
        min_lines = min(file_stats, key=lambda x: x[1])[1]
        
        print(f"\nSummary statistics:")
        print(f"Average lines per file: {avg_lines:.1f}")
        print(f"Largest file: {max_lines:,} lines")
        print(f"Smallest file: {min_lines:,} lines")


if __name__ == "__main__":
    main()
