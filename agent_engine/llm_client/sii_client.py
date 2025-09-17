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


import asyncio
import json
from typing import Optional, List, Union, Dict, Any, AsyncIterator
import httpx

from .llm_client import LLMClient

class SiiClient(LLMClient):
    """Sii API client implementation using HTTP requests"""
    
    # Available models configuration
    AVAILABLE_MODELS = {
        'qwen25': {
            'name': 'Qwen2.5-72B-128K',
            'url': 'http://qwen2-5-72b-128k.api.sii.edu.cn',
            'context_length': 128000
        },
        'deepseek-r1-0528': {
            'name': 'Deepseek-R1-0528-671B-16K-int8',
            'url': 'http://ds-r1-0528-671b-16k-int8.api.sii.edu.cn',
            'context_length': 16000
        },
        'deepseek-v3-0324': {
            'name': 'Deepseek-V3-0324-671B-16K-int8',
            'url': 'http://ds-v3-0324-671b-16k-int8.api.sii.edu.cn',
            'context_length': 16000
        },
        'qwen3-235b': {
            'name': 'Qwen3-235B-A22B-128K',
            'url': 'http://qwen3-235b-128k.api.sii.edu.cn',
            'context_length': 128000
        },
        'qwen25-vl': {
            'name': 'Qwen2.5-VL-72B',
            'url': 'http://qwen2-5-vl-72b-64k.api.sii.edu.cn',
            'context_length': 64000
        }
    }
    
    def __init__(self, api_key: str = "dummy_key"):
        super().__init__()
        self.api_key = api_key  # Sii API doesn't require authentication, but keeping for compatibility
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            timeout=httpx.Timeout(300.0),  # 5 minutes timeout
            verify=False  # Disable SSL verification as shown in examples
        )
        
        self.logger.info("✅ SiiClient initialized")
    
    async def close(self):
        """Close the HTTP client connection"""
        if self.client:
            await self.client.aclose()
            self.logger.info("✅ SiiClient connection closed")
    
    def _get_model_url(self, model_name: str) -> str:
        """Get the base URL for the given model"""
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} is not available. Available models: {list(self.AVAILABLE_MODELS.keys())}")
        
        return self.AVAILABLE_MODELS[model_name]['url']
    
    async def call_llm(
        self, 
        model_name: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 8000,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        """Call Sii API chat completion"""
        trace_id = None
        if hasattr(self, "monitor") and self.monitor is not None:
            try:
                trace_id = self.monitor.new_trace_id()
                # Persist start record
                params_for_record: Dict[str, Any] = {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                params_for_record.update(kwargs or {})
                await self.monitor.start_chat(
                    trace_id=trace_id,
                    provider="sii",
                    model_name=model_name,
                    messages=messages,
                    params=params_for_record,
                )
            except Exception as e:
                self.logger.warning(f"LLM monitor start failed: {e}")

        self.logger.info(f"🚀 Requesting Sii API: model={model_name}, max_tokens={max_tokens}, trace_id={trace_id}")
        
        async def api_call():
            params = self._prepare_chat_params(
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            base_url = self._get_model_url(model_name)
            url = f"{base_url}/v1/chat/completions"
            
            response = await self.client.post(url, json=params)
            response.raise_for_status()
            return response.json()
        
        try:
            completion = await self.async_retry_on_exception(
                api_call,
                max_retries=3,
                delay=5,
                backoff=2
            )
            
            # Extract content from response
            if 'choices' in completion and len(completion['choices']) > 0:
                content = completion['choices'][0]['message']['content']
            else:
                self.logger.error("❌ Invalid response format from Sii API")
                return None
            
            self.logger.info(f"✅ Sii API response received successfully, trace_id={trace_id}")

            # Extract usage if available
            usage: Optional[Dict[str, Any]] = None
            try:
                if 'usage' in completion:
                    usage_obj = completion['usage']
                    usage = {
                        "input_tokens": usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens"),
                        "output_tokens": usage_obj.get("completion_tokens") or usage_obj.get("output_tokens"),
                        "total_tokens": usage_obj.get("total_tokens"),
                    }
            except Exception:
                usage = None

            if trace_id and hasattr(self, "monitor") and self.monitor is not None:
                try:
                    await self.monitor.complete_chat(
                        trace_id=trace_id,
                        response_text=content,
                        usage=usage,
                        raw=completion,
                    )
                except Exception as e:
                    self.logger.warning(f"LLM monitor complete failed: {e}")
            return content
            
        except Exception as e:
            self.logger.error(f"❌ Failed to call Sii API after all retries: {e}, trace_id={trace_id}")
            if trace_id and hasattr(self, "monitor") and self.monitor is not None:
                try:
                    await self.monitor.fail_chat(
                        trace_id=trace_id,
                        error_message=str(e),
                        raw=None,
                    )
                except Exception as e2:
                    self.logger.warning(f"LLM monitor fail record failed: {e2}")
            return None
    
    async def get_embeddings(
        self,
        model_name: str,
        text: Union[str, List[str]],
        **kwargs
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """Get embeddings from Sii API - Not supported"""
        self.logger.warning("⚠️ Embeddings are not supported by Sii API")
        return None
    
    async def chat(
        self, 
        system_prompt: str, 
        user_prompt: str,
        model_name: str = 'qwen25',
        max_tokens: int = 8000,
        temperature: Optional[float] = 0.7,
        **kwargs
    ) -> Optional[str]:
        """Convenience method for chat completion with system and user prompts"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.call_llm(
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        return response
    
    async def embedding(
        self, 
        text: Union[str, List[str]], 
        model_name: str = 'qwen25',
        **kwargs
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """Convenience method for getting embeddings - Not supported"""
        self.logger.warning("⚠️ Embeddings are not supported by Sii API")
        return None

    async def rerank(
        self,
        model_name: str,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        **kwargs
    ) -> Optional[List[Dict[str, Any]]]:
        """Rerank documents based on query relevance - Not supported"""
        self.logger.warning("⚠️ Rerank is not supported by Sii API")
        return None

    async def chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str = 'qwen25',
        max_tokens: int = 8000,
        temperature: Optional[float] = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream chat completion yielding text chunks"""
        trace_id = None
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Start monitor
        if hasattr(self, "monitor") and self.monitor is not None:
            try:
                trace_id = self.monitor.new_trace_id()
                params_for_record: Dict[str, Any] = {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                }
                params_for_record.update(kwargs or {})
                await self.monitor.start_chat(
                    trace_id=trace_id,
                    provider="sii",
                    model_name=model_name,
                    messages=messages,
                    params=params_for_record,
                )
            except Exception as e:
                self.logger.warning(f"LLM monitor start failed: {e}")

        self.logger.info(f"🚀 Streaming from Sii API: model={model_name}, max_tokens={max_tokens}, trace_id={trace_id}")

        # Prepare params
        params = self._prepare_chat_params(
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        params["stream"] = True

        accumulated_parts: List[str] = []

        try:
            base_url = self._get_model_url(model_name)
            url = f"{base_url}/v1/chat/completions"
            
            async with self.client.stream("POST", url, json=params) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                choice = data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    piece = choice['delta']['content']
                                    if piece:
                                        accumulated_parts.append(piece)
                                        yield piece
                        except json.JSONDecodeError:
                            self.logger.warning(f"Failed to parse streaming data: {data_str}")
                            continue

            # complete monitor after stream ends
            final_text = "".join(accumulated_parts)
            if trace_id and hasattr(self, "monitor") and self.monitor is not None:
                try:
                    await self.monitor.complete_chat(
                        trace_id=trace_id,
                        response_text=final_text,
                        usage=None,
                        raw=None,
                    )
                except Exception as me:
                    self.logger.warning(f"LLM monitor complete failed: {me}")
            self.logger.info(f"✅ Sii API streaming finished, trace_id={trace_id}")

        except Exception as e:
            self.logger.error(f"❌ Sii API streaming failed: {e}, trace_id={trace_id}")
            if trace_id and hasattr(self, "monitor") and self.monitor is not None:
                try:
                    await self.monitor.fail_chat(
                        trace_id=trace_id,
                        error_message=str(e),
                        raw=None,
                    )
                except Exception as e2:
                    self.logger.warning(f"LLM monitor fail record failed: {e2}")
            return
