"""
Deep Research Engine

This module provides the main entry point for deep research functionality,
managing user sessions, research plans, and data storage.
"""

import os
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from agent_engine.agent_logger import AgentLogger
from agent_engine.memory.e_memory import EMemory
from agent_engine.utils import get_relative_path_from_current_file

from planner import DeepResearchPlanner
from model import ResearchPlan, ResearchTask, ContextSummary


class DeepResearchEngine:
    """
    Main engine for deep research functionality.
    
    This class manages:
    - User sessions and research instances
    - Data storage with hierarchical directory structure
    - Research plan execution coordination
    - Context management between tasks
    """
    
    def __init__(self, base_storage_path: Optional[str] = None):
        """
        Initialize the deep research engine.
        
        Args:
            base_storage_path: Base path for data storage (defaults to agents/ResearchAgent/runtime)
        """
        self.logger = AgentLogger('DeepResearchEngine')
        
        # Set up storage paths
        if base_storage_path is None:
            # Default to agents/ResearchAgent/runtime
            current_dir = Path(__file__).parent
            self.base_storage_path = current_dir.parent / "runtime"
        else:
            self.base_storage_path = Path(base_storage_path)
        
        # Ensure base storage directory exists
        self.base_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize planner
        self.planner = DeepResearchPlanner()
        
        self.logger.info(f"DeepResearchEngine initialized with storage path: {self.base_storage_path}")
    
    async def start_deep_research(
        self, 
        user_id: str, 
        session_id: str, 
        user_input: str,
        use_deep_plan: bool = True
    ) -> Dict[str, Any]:
        """
        Start a new deep research session.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session
            user_input: The research query from the user
            use_deep_plan: Whether to use deep planning (hypothesize -> explore -> plan)
            
        Returns:
            Dictionary containing research session information
        """
        try:
            # Generate unique deep research ID
            deep_research_id = str(uuid.uuid4())
            
            # Create storage directory structure
            storage_path = self._get_storage_path(user_id, session_id, deep_research_id)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Starting deep research session: {deep_research_id}")
            self.logger.info(f"User: {user_id}, Session: {session_id}")
            self.logger.info(f"Storage path: {storage_path}")
            
            # Generate research plan
            if use_deep_plan:
                plan_data = await self.planner.deep_plan(user_input)
            else:
                plan_data = await self.planner.plan(user_input)
            
            if not plan_data or "tasks" not in plan_data:
                raise ValueError("Failed to generate research plan")
            
            # Create research plan object
            research_plan = ResearchPlan(plan_data["tasks"], deep_research_id)
            
            # Initialize research notes (EMemory)
            research_notes = self._initialize_research_notes(storage_path, deep_research_id)
            
            # Save initial session data
            session_data = {
                "deep_research_id": deep_research_id,
                "user_id": user_id,
                "session_id": session_id,
                "user_input": user_input,
                "created_at": datetime.utcnow().isoformat(),
                "status": "initialized",
                "plan": research_plan.to_dict(),
                "storage_path": str(storage_path)
            }
            
            self._save_session_data(storage_path, session_data)
            
            # Return session information
            return {
                "deep_research_id": deep_research_id,
                "status": "initialized",
                "plan_summary": research_plan.get_progress(),
                "storage_path": str(storage_path),
                "message": "Deep research session initialized successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start deep research: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to initialize deep research session"
            }
    
    def get_research_session(self, user_id: str, session_id: str, deep_research_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an existing research session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            deep_research_id: Deep research identifier
            
        Returns:
            Session data if found, None otherwise
        """
        try:
            storage_path = self._get_storage_path(user_id, session_id, deep_research_id)
            session_file = storage_path / "session.json"
            
            if not session_file.exists():
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Failed to get research session: {e}")
            return None
    
    def list_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session information
        """
        try:
            user_path = self.base_storage_path / user_id
            if not user_path.exists():
                return []
            
            sessions = []
            for session_dir in user_path.iterdir():
                if session_dir.is_dir():
                    # Look for deep research sessions in this session directory
                    for deep_research_dir in session_dir.iterdir():
                        if deep_research_dir.is_dir():
                            session_file = deep_research_dir / "session.json"
                            if session_file.exists():
                                try:
                                    with open(session_file, 'r', encoding='utf-8') as f:
                                        session_data = json.load(f)
                                        sessions.append({
                                            "session_id": session_dir.name,
                                            "deep_research_id": deep_research_dir.name,
                                            "created_at": session_data.get("created_at"),
                                            "status": session_data.get("status"),
                                            "user_input": session_data.get("user_input", "")[:100] + "..." if len(session_data.get("user_input", "")) > 100 else session_data.get("user_input", "")
                                        })
                                except Exception as e:
                                    self.logger.warning(f"Failed to read session file {session_file}: {e}")
            
            # Sort by creation time (newest first)
            sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return sessions
            
        except Exception as e:
            self.logger.error(f"Failed to list user sessions: {e}")
            return []
    
    def get_research_plan(self, user_id: str, session_id: str, deep_research_id: str) -> Optional[ResearchPlan]:
        """
        Get the research plan for a session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            deep_research_id: Deep research identifier
            
        Returns:
            ResearchPlan object if found, None otherwise
        """
        try:
            session_data = self.get_research_session(user_id, session_id, deep_research_id)
            if not session_data or "plan" not in session_data:
                return None
            
            return ResearchPlan.from_dict(session_data["plan"])
            
        except Exception as e:
            self.logger.error(f"Failed to get research plan: {e}")
            return None
    
    def get_research_notes(self, user_id: str, session_id: str, deep_research_id: str) -> Optional[EMemory]:
        """
        Get the research notes (EMemory) for a session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            deep_research_id: Deep research identifier
            
        Returns:
            EMemory object if found, None otherwise
        """
        try:
            storage_path = self._get_storage_path(user_id, session_id, deep_research_id)
            memory_path = storage_path / "research_notes"
            
            if not memory_path.exists():
                return None
            
            return EMemory(
                name=f"research_notes_{deep_research_id}",
                persist_dir=str(memory_path)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get research notes: {e}")
            return None
    
    def update_session_status(self, user_id: str, session_id: str, deep_research_id: str, status: str) -> bool:
        """
        Update the status of a research session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            deep_research_id: Deep research identifier
            status: New status
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            storage_path = self._get_storage_path(user_id, session_id, deep_research_id)
            session_file = storage_path / "session.json"
            
            if not session_file.exists():
                return False
            
            # Load existing session data
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Update status
            session_data["status"] = status
            session_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Save updated data
            self._save_session_data(storage_path, session_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update session status: {e}")
            return False
    
    def _get_storage_path(self, user_id: str, session_id: str, deep_research_id: str) -> Path:
        """
        Get the storage path for a specific research session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            deep_research_id: Deep research identifier
            
        Returns:
            Path object for the storage directory
        """
        return self.base_storage_path / user_id / session_id / deep_research_id
    
    def _initialize_research_notes(self, storage_path: Path, deep_research_id: str) -> EMemory:
        """
        Initialize EMemory for research notes.
        
        Args:
            storage_path: Storage directory path
            deep_research_id: Deep research identifier
            
        Returns:
            Initialized EMemory object
        """
        memory_path = storage_path / "research_notes"
        return EMemory(
            name=f"research_notes_{deep_research_id}",
            persist_dir=str(memory_path)
        )
    
    def _save_session_data(self, storage_path: Path, session_data: Dict[str, Any]) -> None:
        """
        Save session data to file.
        
        Args:
            storage_path: Storage directory path
            session_data: Session data to save
        """
        session_file = storage_path / "session.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    def cleanup_old_sessions(self, user_id: str, days_old: int = 30) -> int:
        """
        Clean up old research sessions.
        
        Args:
            user_id: User identifier
            days_old: Number of days after which to clean up sessions
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            user_path = self.base_storage_path / user_id
            if not user_path.exists():
                return 0
            
            cutoff_date = datetime.utcnow().timestamp() - (days_old * 24 * 60 * 60)
            cleaned_count = 0
            
            for session_dir in user_path.iterdir():
                if session_dir.is_dir():
                    for deep_research_dir in session_dir.iterdir():
                        if deep_research_dir.is_dir():
                            session_file = deep_research_dir / "session.json"
                            if session_file.exists():
                                try:
                                    file_time = session_file.stat().st_mtime
                                    if file_time < cutoff_date:
                                        # Remove the entire deep research directory
                                        import shutil
                                        shutil.rmtree(deep_research_dir)
                                        cleaned_count += 1
                                        self.logger.info(f"Cleaned up old session: {deep_research_dir}")
                                except Exception as e:
                                    self.logger.warning(f"Failed to clean up session {deep_research_dir}: {e}")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {e}")
            return 0


# Convenience function for easy access
def create_deep_research_engine(base_storage_path: Optional[str] = None) -> DeepResearchEngine:
    """
    Create a new DeepResearchEngine instance.
    
    Args:
        base_storage_path: Optional base storage path
        
    Returns:
        DeepResearchEngine instance
    """
    return DeepResearchEngine(base_storage_path)
