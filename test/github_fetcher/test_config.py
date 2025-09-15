"""
Configuration helper for GitHub Fetcher tests

This module provides utilities for loading environment variables
and configuring test settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def load_test_env():
    """Load environment variables from .env file for testing."""
    # Get project root
    current_file = Path(__file__).resolve()
    project_root = current_file
    while project_root.parent != project_root:
        if (project_root / "pyproject.toml").exists():
            break
        project_root = project_root.parent
    
    # Load .env file if it exists
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment variables from: {env_file}")
    else:
        print(f"No .env file found at: {env_file}")
        print("Please create a .env file with GITHUB_API_KEY=your_token_here")


def get_github_token():
    """Get GitHub API token from environment variables."""
    token = os.getenv('GITHUB_API_KEY')
    if not token:
        print("Warning: GITHUB_API_KEY not found in environment variables")
        print("Some integration tests will be skipped")
    return token


def is_github_token_available():
    """Check if GitHub API token is available."""
    return get_github_token() is not None


if __name__ == "__main__":
    load_test_env()
    token = get_github_token()
    if token:
        print(f"GitHub token found: {token[:10]}...")
    else:
        print("GitHub token not found")
        print("\nTo set up GitHub API access:")
        print("1. Create a .env file in the project root")
        print("2. Add: GITHUB_API_KEY=your_github_token_here")
        print("3. Get your token from: https://github.com/settings/tokens")
