"""
Service module for agent_engine

This module provides automatic HTTP service generation capabilities.
"""

from .auto_service import AutoService, expose

__all__ = ['AutoService', 'expose']
