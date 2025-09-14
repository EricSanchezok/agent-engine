import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, AsyncIterator
from openai import RateLimitError, APITimeoutError, APIConnectionError

from ..agent_logger.agent_logger import AgentLogger
from .llm_monitor import LLMChatMonitor

logger = AgentLogger('LLMClient')

class LLMClient(ABC):
    """Base class for LLM clients with common functionality and retry logic"""
    
    NO_TEMP_MODELS = {
        'o1',
        'o3-mini',
        'o3',
        'o4-mini',
        'o3-deep-research'
    }
    
    def __init__(self):
        self.logger = logger
        # Initialize monitor (non-vector storage under .llm_monitoring)
        try:
            # Allow frontend notify url from env
            notify_url = os.getenv("LLM_MONITOR_NOTIFY_URL")
            self.monitor = LLMChatMonitor(name="llm_chats", enable_vectors=False, notify_url=notify_url)
        except Exception as e:
            self.monitor = None  # type: ignore
            self.logger.warning(f"LLM monitor initialization failed: {e}")
    
    # Old file-based monitoring is removed in favor of LLMChatMonitor
    
    async def async_retry_on_exception(
        self, 
        func, 
        max_retries: int = 3, 
        delay: int = 2, 
        backoff: int = 2, 
        exceptions: tuple = (RateLimitError, APITimeoutError, APIConnectionError),
        *args, 
        **kwargs
    ):
        """Retry function execution with exponential backoff"""
        retries = 0
        current_delay = delay
        
        while True:
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                retries += 1
                self.logger.error(f"⚠️ Attempt {retries} failed: {type(e).__name__}: {e}")
                self.logger.error(f"⚠️ Error details: {str(e)}")
                
                if retries > max_retries:
                    self.logger.error(f"❌ Failed after {max_retries} retries, final error: {type(e).__name__}: {e}")
                    raise e
                    
                self.logger.error(f"⏳ Retrying in {current_delay} seconds (attempt {retries + 1})...")
                await asyncio.sleep(current_delay)
                current_delay += backoff
    
    @abstractmethod
    async def close(self):
        """Close the client connection"""
        pass
    
    @abstractmethod
    async def call_llm(
        self, 
        model_name: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 8000,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        """Call the LLM with the given parameters"""
        pass
    
    @abstractmethod
    async def get_embeddings(
        self,
        model_name: str,
        text: Union[str, List[str]],
        **kwargs
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """Get embeddings for the given text"""
        pass

    @abstractmethod
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> Optional[str]:
        """Chat with the LLM"""
        pass
    
    @abstractmethod
    async def embedding(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """Get embeddings for the given text"""
        pass

    @abstractmethod
    async def rerank(
        self,
        model_name: str,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        **kwargs
    ) -> Optional[List[Dict[str, Any]]]:
        """Rerank documents based on query relevance"""
        pass

    @abstractmethod
    async def chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str = 'o3-mini',
        max_tokens: int = 8000,
        temperature: Optional[float] = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream chat completion yielding text chunks"""
        pass
    
    def _should_use_temperature(self, model_name: str) -> bool:
        """Check if temperature parameter should be used for the given model"""
        return model_name not in self.NO_TEMP_MODELS
    
    def _prepare_chat_params(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 8000,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare parameters for chat completion"""
        params = {
            "model": model_name,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            **kwargs
        }
        
        # Only add temperature if the model supports it
        if temperature is not None and self._should_use_temperature(model_name):
            params["temperature"] = temperature
        
        return params
