import asyncio
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Any
from openai import RateLimitError, APITimeoutError, APIConnectionError

from ..agent_logger.agent_logger import AgentLogger

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
