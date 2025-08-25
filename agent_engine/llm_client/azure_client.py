from typing import Optional, List, Union, Dict, Any
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
        self.logger.info(f"🚀 Requesting Azure OpenAI: model={model_name}, max_tokens={max_tokens}")
        
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
            self.logger.info(f"✅ Azure OpenAI response received successfully")
            return content
            
        except Exception as e:
            self.logger.error(f"❌ Failed to call Azure OpenAI after all retries: {e}")
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
            
            embeddings = response.data[0].embedding
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
        
        # Save monitoring data if response is successful
        if response is not None:
            self._save_chat_monitoring(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response,
                model_name=model_name,
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
