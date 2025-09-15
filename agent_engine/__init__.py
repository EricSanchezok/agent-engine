"""
Agent Engine - A Python library for building intelligent agents

This package provides tools and utilities for building AI agents,
including memory management, LLM clients, and automatic service generation.
"""

from .service import AutoService, expose

__version__ = "0.1.0"
__all__ = ['AutoService', 'expose']
