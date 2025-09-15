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


Arxiv Sync Service

A robust service for synchronizing arXiv papers to the database.
Provides daily sync functionality with rolling 7-day window and embedding generation.
"""
