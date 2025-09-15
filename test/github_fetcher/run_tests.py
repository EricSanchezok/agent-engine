"""
Test runner for GitHub Fetcher tests

This script runs the GitHub fetcher tests with proper configuration.
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

from test_config import load_test_env, is_github_token_available


def main():
    """Run GitHub fetcher tests."""
    print("GitHub Fetcher Test Runner")
    print("=" * 40)
    
    # Load environment variables
    load_test_env()
    
    # Check if GitHub token is available
    if is_github_token_available():
        print("✓ GitHub API token found - running all tests")
        test_args = ["-v", "--tb=short"]
    else:
        print("⚠ GitHub API token not found - running unit tests only")
        test_args = ["-v", "--tb=short", "-k", "not Integration"]
    
    # Import and run pytest
    try:
        import pytest
        test_file = Path(__file__).parent / "test_github_fetcher.py"
        pytest.main([str(test_file)] + test_args)
    except ImportError:
        print("Error: pytest not installed")
        print("Please install pytest: pip install pytest")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
