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
from dotenv import load_dotenv
import tiktoken

from agent_engine.memory.e_memory import EMemory, Record
from agent_engine.agent_logger.agent_logger import AgentLogger
from agent_engine.utils import get_current_file_dir, get_relative_path_from_current_file
from agent_engine.prompt import PromptLoader
from agent_engine.llm_client import AzureClient

logger = AgentLogger(__name__)

load_dotenv()


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
    total_tokens: int = field(default=0)


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
        
        self.llm_client = AzureClient(api_key=os.getenv("AZURE_API_KEY"))
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file("prompts.yaml"))
        self.model_name = "gpt-4.1"

        # Create database directory structure
        self.db_dir = Path("agents/ResearchAgent/database/runtime") / user_id / session_id
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique memory name for this user session
        self.memory_name = f"session_{user_id}_{session_id}"
        
        # Initialize EMemory database with custom persist_dir
        self.memory = EMemory(
            name=self.memory_name,
            persist_dir=str(self.db_dir)
        )
        
        # Context history file path
        self.context_history_file = self.db_dir / "context_history.json"
        
        # Initialize context history
        self.context_history = ContextHistory()
        
        # Load existing context from file
        self._load_context_history()
        
        logger.info(f"SessionMemory initialized for user={user_id}, session={session_id}")
        logger.info(f"Database directory: {self.db_dir}")
        logger.info(f"Short history: {len(self.context_history.short_history)} Q&A pairs")
        logger.info(f"Long history length: {len(self.context_history.long_history)} chars")
        logger.info(f"Total tokens: {self.context_history.total_tokens}")
    
    def _load_context_history(self) -> None:
        """Load context history from JSON file"""
        try:
            if self.context_history_file.exists():
                with open(self.context_history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load short history
                short_history_data = data.get('short_history', [])
                short_history_records = []
                for qa_data in short_history_data:
                    qa = QA(
                        question=qa_data['question'],
                        answer=qa_data['answer'],
                        timestamp=qa_data['timestamp'],
                        qa_id=qa_data.get('qa_id', str(uuid.uuid4()))
                    )
                    short_history_records.append(qa)
                
                # Load long history and total tokens
                self.context_history.short_history = short_history_records
                self.context_history.long_history = data.get('long_history', '')
                self.context_history.total_tokens = data.get('total_tokens', 0)
                
                logger.info(f"Loaded context history: {len(short_history_records)} short Q&A pairs")
                logger.info(f"Total tokens: {self.context_history.total_tokens}")
            else:
                logger.info("No existing context history file found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load context history: {e}")
            # Initialize empty context history
            self.context_history = ContextHistory()
    
    def _save_context_history(self) -> None:
        """Save context history to JSON file"""
        try:
            # Prepare data for JSON serialization
            data = {
                'short_history': [
                    {
                        'question': qa.question,
                        'answer': qa.answer,
                        'timestamp': qa.timestamp,
                        'qa_id': qa.qa_id
                    }
                    for qa in self.context_history.short_history
                ],
                'long_history': self.context_history.long_history,
                'total_tokens': self.context_history.total_tokens
            }
            
            # Save to JSON file
            with open(self.context_history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Context history saved to JSON file")
            
        except Exception as e:
            logger.error(f"Failed to save context history: {e}")
    
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text using tiktoken.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        try:
            # Use tiktoken to get accurate token count for the model
            encoding = tiktoken.encoding_for_model(self.model_name)
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Failed to estimate tokens with tiktoken: {e}")
            # Fallback to simple estimation: ~4 characters per token for English text
            return len(text) // 4
    
    def _should_summarize(self) -> bool:
        """Check if short history should be summarized"""
        return self.context_history.total_tokens >= self.summarization_threshold
    
    async def add_qa(self, question: str, answer: str) -> None:
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
            
            # Calculate tokens for this Q&A pair
            qa_tokens = self._estimate_tokens(question) + self._estimate_tokens(answer)
            
            # Add to short history
            self.context_history.short_history.append(qa)
            
            # Update total tokens
            self.context_history.total_tokens += qa_tokens
            
            # Add to EMemory database
            qa_data = {
                'question': question,
                'answer': answer,
                'timestamp': qa.timestamp,
                'qa_id': qa.qa_id
            }
            
            record = Record(
                id=f"qa_{qa.qa_id}",
                content=json.dumps(qa_data),
                attributes={'type': 'qa_pair', 'qa_id': qa.qa_id, 'tokens': qa_tokens},
                timestamp=qa.timestamp
            )
            self.memory.add(record)
            
            # Check if summarization is needed
            if self._should_summarize():
                logger.info("Short history exceeds threshold, triggering summarization")
                await self._summarize_history()
            
            # Save context history to JSON file
            self._save_context_history()
            
            logger.info(f"Added Q&A pair to session memory. Short history: {len(self.context_history.short_history)} pairs, Total tokens: {self.context_history.total_tokens}")
            
        except Exception as e:
            logger.error(f"Failed to add Q&A pair: {e}")
    
    async def _summarize_history(self) -> None:
        """
        Summarize short history and merge with long history.
        This method creates a summary of recent conversations and updates long history.
        """
        try:
            if not self.context_history.short_history:
                return
            
            system_prompt = self.prompt_loader.get_prompt(
                section="inno_researcher_summarize",
                prompt_type="system",
                long_history=self.context_history.long_history,
                short_history=self.context_history.short_history
            )
            user_prompt = self.prompt_loader.get_prompt(
                section="inno_researcher_summarize",
                prompt_type="user"
            )

            try:
                result = await self.llm_client.chat(system_prompt, user_prompt, model_name=self.model_name, max_tokens=32000)
                self.context_history.long_history = result
                self.context_history.short_history = []
                self.context_history.total_tokens = self._estimate_tokens(result)
            except Exception as e:
                logger.error(f"Failed to summarize history: {e}")
                return
            
            logger.info("History summarization completed")
            logger.info(f"Long history length: {len(self.context_history.long_history)} chars")
            logger.info(f"Short history reduced to: {len(self.context_history.short_history)} pairs")
            logger.info(f"Total tokens reset to: {self.context_history.total_tokens}")
            
        except Exception as e:
            logger.error(f"Failed to summarize history: {e}")
    
    def get_context_for_llm(self) -> Tuple[str, str]:
        """
        Get formatted context for LLM input.
            
        Returns:
            Formatted context string
        """
        try:
            long_history = ""
            if self.context_history.long_history:
                long_history = self.context_history.long_history

            short_history_parts = []
            if self.context_history.short_history:
                for qa in self.context_history.short_history:
                    short_history_parts.append(f"User: {qa.question}")
                    short_history_parts.append(f"Assistant: {qa.answer}")
            
            short_history = "\n".join(short_history_parts)
            return long_history, short_history
            
        except Exception as e:
            logger.error(f"Failed to get context for LLM: {e}")
            return "", ""
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        try:
            return {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "memory_name": self.memory_name,
                "db_dir": str(self.db_dir),
                "short_history_count": len(self.context_history.short_history),
                "total_tokens": self.context_history.total_tokens,
                "long_history_length": len(self.context_history.long_history),
                "long_history_tokens": self._estimate_tokens(self.context_history.long_history),
                "total_records_in_memory": self.memory.count(),
                "should_summarize": self._should_summarize(),
                "summarization_threshold": self.summarization_threshold
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
            
            # Remove context history file if it exists
            if self.context_history_file.exists():
                self.context_history_file.unlink()
            
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
