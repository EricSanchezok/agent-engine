import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from openai import RateLimitError, APITimeoutError, APIConnectionError


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
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup monitoring directory for LLM calls"""
        # Get project root directory (similar to agent_logger.py)
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            if (parent / "pyproject.toml").exists():
                project_root = parent
                break
        else:
            project_root = Path.cwd()
        
        # Create logs/llm directory
        self.monitor_dir = project_root / "logs" / "llm"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"LLM monitoring directory initialized: {self.monitor_dir}")
    
    def _save_chat_monitoring(
        self,
        system_prompt: str,
        user_prompt: str,
        response: str,
        model_name: str,
        max_tokens: int,
        temperature: Optional[float],
        **kwargs
    ):
        """Save chat input-output pair to monitoring file"""
        try:
            # Create monitoring record
            monitor_record = {
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response": response,
                "additional_params": kwargs
            }
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"llm_chat_{timestamp}.json"
            filepath = self.monitor_dir / filename
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(monitor_record, f, ensure_ascii=False, indent=2)
            
            print(f"Chat monitoring saved: {filepath}")
            
        except Exception as e:
            print(f"Failed to save chat monitoring: {e}")
    
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
                print(f"⚠️ Attempt {retries} failed: {type(e).__name__}: {e}")
                print(f"⚠️ Error details: {str(e)}")
                
                if retries > max_retries:
                    print(f"❌ Failed after {max_retries} retries, final error: {type(e).__name__}: {e}")
                    raise e
                    
                print(f"⏳ Retrying in {current_delay} seconds (attempt {retries + 1})...")
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
