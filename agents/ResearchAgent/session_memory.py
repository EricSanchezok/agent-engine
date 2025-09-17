"""
SessionMemory module for managing conversation history and context.

This module provides session-based memory management with:
- Short-term conversation history (recent Q&A pairs)
- Long-term summarized history
- Automatic summarization when short history exceeds token limits
- Persistent storage using EMemory database
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

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from agent_engine.memory.e_memory import EMemory, Record
from agent_engine.agent_logger.agent_logger import AgentLogger

logger = AgentLogger(__name__)


@dataclass
class QA:
    """Question and Answer pair"""
    question: str
    answer: str
    timestamp: str
    qa_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ContextHistory:
    """Context history containing short and long term memory"""
    short_history: List[QA] = field(default_factory=list)
    long_history: str = field(default_factory=str)


class SessionMemory:
    """
    Session-based memory management for conversation history.
    
    Features:
    - Maintains short-term conversation history (recent Q&A pairs)
    - Maintains long-term summarized history
    - Automatic summarization when short history exceeds token limits
    - Persistent storage using EMemory database
    - Context management for different user sessions
    """
    
    def __init__(
        self, 
        user_id: str, 
        session_id: str,
        max_short_history_tokens: int = 2000,
        summarization_threshold: int = 1500
    ):
        """
        Initialize SessionMemory for a specific user and session.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session
            max_short_history_tokens: Maximum tokens in short history before summarization
            summarization_threshold: Token threshold to trigger summarization
        """
        self.user_id = user_id
        self.session_id = session_id
        self.max_short_history_tokens = max_short_history_tokens
        self.summarization_threshold = summarization_threshold
        
        # Create unique memory name for this user session
        self.memory_name = f"session_{user_id}_{session_id}"
        
        # Initialize EMemory database
        self.memory = EMemory(
            name=self.memory_name,
            persist_dir=None  # Use default .memory directory
        )
        
        # Initialize context history
        self.context_history = ContextHistory()
        
        # Load existing context from database
        self._load_context_history()
        
        logger.info(f"SessionMemory initialized for user={user_id}, session={session_id}")
        logger.info(f"Memory name: {self.memory_name}")
        logger.info(f"Short history: {len(self.context_history.short_history)} Q&A pairs")
        logger.info(f"Long history length: {len(self.context_history.long_history)} chars")
    
    def _load_context_history(self) -> None:
        """Load context history from EMemory database"""
        try:
            # Get all records from the memory
            records = self.memory.list_all()
            
            # Separate short history and long history records
            short_history_records = []
            long_history_record = None
            
            for record in records:
                attributes = record.attributes or {}
                record_type = attributes.get('type', '')
                
                if record_type == 'short_history':
                    # Parse QA from content
                    try:
                        qa_data = json.loads(record.content)
                        qa = QA(
                            question=qa_data['question'],
                            answer=qa_data['answer'],
                            timestamp=qa_data['timestamp'],
                            qa_id=qa_data.get('qa_id', str(uuid.uuid4()))
                        )
                        short_history_records.append(qa)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse short history record: {e}")
                elif record_type == 'long_history':
                    long_history_record = record
            
            # Sort short history by timestamp
            short_history_records.sort(key=lambda x: x.timestamp)
            self.context_history.short_history = short_history_records
            
            # Set long history
            if long_history_record:
                self.context_history.long_history = long_history_record.content or ""
            
            logger.info(f"Loaded context history: {len(short_history_records)} short Q&A pairs")
            
        except Exception as e:
            logger.error(f"Failed to load context history: {e}")
            # Initialize empty context history
            self.context_history = ContextHistory()
    
    def _save_context_history(self) -> None:
        """Save context history to EMemory database"""
        try:
            # Clear existing context records
            self._clear_context_records()
            
            # Save short history
            for qa in self.context_history.short_history:
                qa_data = {
                    'question': qa.question,
                    'answer': qa.answer,
                    'timestamp': qa.timestamp,
                    'qa_id': qa.qa_id
                }
                
                record = Record(
                    id=f"short_{qa.qa_id}",
                    content=json.dumps(qa_data),
                    attributes={'type': 'short_history', 'qa_id': qa.qa_id},
                    timestamp=qa.timestamp
                )
                self.memory.add(record)
            
            # Save long history
            if self.context_history.long_history:
                record = Record(
                    id="long_history",
                    content=self.context_history.long_history,
                    attributes={'type': 'long_history'},
                    timestamp=datetime.utcnow().isoformat() + "Z"
                )
                self.memory.add(record)
            
            logger.debug("Context history saved to EMemory")
            
        except Exception as e:
            logger.error(f"Failed to save context history: {e}")
    
    def _clear_context_records(self) -> None:
        """Clear existing context records from memory"""
        try:
            records = self.memory.list_all()
            for record in records:
                attributes = record.attributes or {}
                record_type = attributes.get('type', '')
                if record_type in ['short_history', 'long_history']:
                    self.memory.delete(record.id)
        except Exception as e:
            logger.warning(f"Failed to clear context records: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Simple estimation: ~4 characters per token for English text.
        """
        return len(text) // 4
    
    def _should_summarize(self) -> bool:
        """Check if short history should be summarized"""
        if not self.context_history.short_history:
            return False
        
        # Calculate total tokens in short history
        total_tokens = 0
        for qa in self.context_history.short_history:
            total_tokens += self._estimate_tokens(qa.question)
            total_tokens += self._estimate_tokens(qa.answer)
        
        return total_tokens >= self.summarization_threshold
    
    def add_qa(self, question: str, answer: str) -> None:
        """
        Add a new Q&A pair to the conversation history.
        
        Args:
            question: User's question
            answer: Agent's answer
        """
        try:
            # Create new QA pair
            qa = QA(
                question=question,
                answer=answer,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            
            # Add to short history
            self.context_history.short_history.append(qa)
            
            # Check if summarization is needed
            if self._should_summarize():
                logger.info("Short history exceeds threshold, triggering summarization")
                self._summarize_history()
            
            # Save to database
            self._save_context_history()
            
            logger.info(f"Added Q&A pair to session memory. Short history: {len(self.context_history.short_history)} pairs")
            
        except Exception as e:
            logger.error(f"Failed to add Q&A pair: {e}")
    
    def _summarize_history(self) -> None:
        """
        Summarize short history and merge with long history.
        This method creates a summary of recent conversations and updates long history.
        """
        try:
            if not self.context_history.short_history:
                return
            
            # Create summary of short history
            short_summary_parts = []
            for qa in self.context_history.short_history:
                short_summary_parts.append(f"Q: {qa.question}\nA: {qa.answer}")
            
            short_summary = "\n\n".join(short_summary_parts)
            
            # Combine with existing long history
            if self.context_history.long_history:
                combined_history = f"{self.context_history.long_history}\n\n--- Recent Conversation Summary ---\n{short_summary}"
            else:
                combined_history = f"--- Conversation Summary ---\n{short_summary}"
            
            # Update long history
            self.context_history.long_history = combined_history
            
            # Clear short history (keep only the most recent Q&A pair for context)
            if len(self.context_history.short_history) > 1:
                # Keep only the last Q&A pair
                self.context_history.short_history = self.context_history.short_history[-1:]
            
            logger.info("History summarization completed")
            logger.info(f"Long history length: {len(self.context_history.long_history)} chars")
            logger.info(f"Short history reduced to: {len(self.context_history.short_history)} pairs")
            
        except Exception as e:
            logger.error(f"Failed to summarize history: {e}")
    
    def get_context_for_llm(self, max_context_tokens: int = 4000) -> str:
        """
        Get formatted context for LLM input.
        
        Args:
            max_context_tokens: Maximum tokens for context
            
        Returns:
            Formatted context string
        """
        try:
            context_parts = []
            
            # Add long history if available
            if self.context_history.long_history:
                context_parts.append("=== Previous Conversation Summary ===")
                context_parts.append(self.context_history.long_history)
            
            # Add recent short history
            if self.context_history.short_history:
                context_parts.append("\n=== Recent Conversation ===")
                for qa in self.context_history.short_history:
                    context_parts.append(f"User: {qa.question}")
                    context_parts.append(f"Assistant: {qa.answer}")
            
            context = "\n".join(context_parts)
            
            # Truncate if too long
            if self._estimate_tokens(context) > max_context_tokens:
                # Keep long history and truncate short history
                if self.context_history.long_history:
                    remaining_tokens = max_context_tokens - self._estimate_tokens(self.context_history.long_history) - 200  # Buffer
                    if remaining_tokens > 0:
                        recent_context = []
                        current_tokens = 0
                        for qa in reversed(self.context_history.short_history):
                            qa_text = f"User: {qa.question}\nAssistant: {qa.answer}"
                            qa_tokens = self._estimate_tokens(qa_text)
                            if current_tokens + qa_tokens <= remaining_tokens:
                                recent_context.insert(0, qa_text)
                                current_tokens += qa_tokens
                            else:
                                break
                        
                        context = f"=== Previous Conversation Summary ===\n{self.context_history.long_history}\n\n=== Recent Conversation ===\n" + "\n".join(recent_context)
                    else:
                        context = f"=== Previous Conversation Summary ===\n{self.context_history.long_history}"
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get context for LLM: {e}")
            return ""
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        try:
            short_history_tokens = 0
            for qa in self.context_history.short_history:
                short_history_tokens += self._estimate_tokens(qa.question)
                short_history_tokens += self._estimate_tokens(qa.answer)
            
            return {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "memory_name": self.memory_name,
                "short_history_count": len(self.context_history.short_history),
                "short_history_tokens": short_history_tokens,
                "long_history_length": len(self.context_history.long_history),
                "long_history_tokens": self._estimate_tokens(self.context_history.long_history),
                "total_records_in_memory": self.memory.count(),
                "should_summarize": self._should_summarize()
            }
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {}
    
    def clear_session(self) -> None:
        """Clear all session data"""
        try:
            # Clear context history
            self.context_history = ContextHistory()
            
            # Clear memory database
            self.memory.clear(confirm=False)
            
            logger.info(f"Cleared session memory for user={self.user_id}, session={self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
    
    def close(self) -> None:
        """Close the session memory and cleanup resources"""
        try:
            if hasattr(self, 'memory'):
                self.memory.close()
            logger.info(f"SessionMemory closed for user={self.user_id}, session={self.session_id}")
        except Exception as e:
            logger.error(f"Failed to close session memory: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.close()
