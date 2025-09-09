from typing import Optional, List, Union, Dict, Any, AsyncIterator
from openai import AsyncAzureOpenAI

from .llm_client import LLMClient

class AzureClient(LLMClient):
    """Azure OpenAI client implementation"""
    
    def __init__(self, api_key: str, base_url: str = 'https://gpt.yunstorm.com/', api_version: str = '2025-04-01-preview'):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        
        # Validate API key
        if not self.api_key:
            self.logger.error("❌ API key not provided")
            raise ValueError("API key is required")
        
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.base_url,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        self.logger.info(f"✅ AzureClient initialized with endpoint: {self.base_url}")
    
    async def close(self):
        """Close the Azure OpenAI client connection"""
        if self.client:
            await self.client.close()
            self.logger.info("✅ AzureClient connection closed")
    
    async def call_llm(
        self, 
        model_name: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 8000,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        """Call Azure OpenAI chat completion API"""
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
                    provider="azure",
                    model_name=model_name,
                    messages=messages,
                    params=params_for_record,
                )
            except Exception as e:
                self.logger.warning(f"LLM monitor start failed: {e}")

        self.logger.info(f"🚀 Requesting Azure OpenAI: model={model_name}, max_tokens={max_tokens}, trace_id={trace_id}")
        
        async def api_call():
            params = self._prepare_chat_params(
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            response = await self.client.chat.completions.create(**params)
            return response
        
        try:
            completion = await self.async_retry_on_exception(
                api_call,
                max_retries=3,
                delay=5,
                backoff=2
            )
            
            content = completion.choices[0].message.content
            self.logger.info(f"✅ Azure OpenAI response received successfully, trace_id={trace_id}")

            # Extract usage if available
            usage: Optional[Dict[str, Any]] = None
            try:
                usage_obj = getattr(completion, "usage", None)
                if usage_obj is not None:
                    # OpenAI SDK style: input_tokens, output_tokens, total_tokens
                    usage = {
                        "input_tokens": getattr(usage_obj, "prompt_tokens", None) or getattr(usage_obj, "input_tokens", None),
                        "output_tokens": getattr(usage_obj, "completion_tokens", None) or getattr(usage_obj, "output_tokens", None),
                        "total_tokens": getattr(usage_obj, "total_tokens", None),
                    }
            except Exception:
                usage = None

            if trace_id and hasattr(self, "monitor") and self.monitor is not None:
                try:
                    await self.monitor.complete_chat(
                        trace_id=trace_id,
                        response_text=content,
                        usage=usage,
                        raw=None,
                    )
                except Exception as e:
                    self.logger.warning(f"LLM monitor complete failed: {e}")
            return content
            
        except Exception as e:
            self.logger.error(f"❌ Failed to call Azure OpenAI after all retries: {e}, trace_id={trace_id}")
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
        """Get embeddings from Azure OpenAI"""
        self.logger.info(f"🚀 Requesting Azure OpenAI embeddings: model={model_name}")
        
        async def api_call():
            response = await self.client.embeddings.create(
                model=model_name,
                input=text,
                **kwargs
            )
            return response
        
        try:
            response = await self.async_retry_on_exception(
                api_call,
                max_retries=3,
                delay=5,
                backoff=2
            )
            
            # Handle both single and batch embeddings
            if isinstance(text, str):
                # Single text input
                embeddings = response.data[0].embedding
            else:
                # Multiple texts input - return list of embeddings
                embeddings = [item.embedding for item in response.data]
            
            self.logger.info(f"✅ Azure OpenAI embeddings received successfully")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get embeddings after all retries: {e}")
            return None
    
    async def chat(
        self, 
        system_prompt: str, 
        user_prompt: str,
        model_name: str = 'o3-mini',
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
        model_name: str = 'text-embedding-ada-002',
        **kwargs
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """Convenience method for getting embeddings"""
        return await self.get_embeddings(
            model_name=model_name,
            text=text,
            **kwargs
        )

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
                    provider="azure",
                    model_name=model_name,
                    messages=messages,
                    params=params_for_record,
                )
            except Exception as e:
                self.logger.warning(f"LLM monitor start failed: {e}")

        self.logger.info(f"🚀 Streaming from Azure OpenAI: model={model_name}, max_tokens={max_tokens}, trace_id={trace_id}")

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
            stream = await self.client.chat.completions.create(**params)

            async for event in stream:
                try:
                    piece: Optional[str] = None
                    choices = getattr(event, "choices", None)
                    if choices and len(choices) > 0:
                        delta = getattr(choices[0], "delta", None)
                        if delta is not None:
                            piece = getattr(delta, "content", None)
                        else:
                            # fallback for SDKs that place content on message
                            msg = getattr(choices[0], "message", None)
                            if msg is not None:
                                piece = getattr(msg, "content", None)
                    if piece:
                        accumulated_parts.append(piece)
                        yield piece
                except Exception as ie:
                    self.logger.warning(f"Unexpected stream chunk structure: {ie}")

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
            self.logger.info(f"✅ Azure OpenAI streaming finished, trace_id={trace_id}")

        except Exception as e:
            self.logger.error(f"❌ Azure OpenAI streaming failed: {e}, trace_id={trace_id}")
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
