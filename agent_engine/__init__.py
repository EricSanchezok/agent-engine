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


Agent Engine - A Python library for building intelligent agents

This package provides tools and utilities for building AI agents,
including memory management, LLM clients, and automatic service generation.
"""

from .service import AutoService, expose

__version__ = "0.1.0"
__all__ = ['AutoService', 'expose']
