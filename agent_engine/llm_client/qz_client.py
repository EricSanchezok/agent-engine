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

class QzClient(LLMClient):
    """Qz API client implementation using HTTP requests"""
    
    def __init__(self, api_key: str, base_url: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        
        # Validate API key
        if not self.api_key:
            self.logger.error("❌ API key not provided")
            raise ValueError("API key is required")
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            },
            timeout=httpx.Timeout(60.0)
        )
        
        self.logger.info(f"✅ QzClient initialized with endpoint: {self.base_url}")
    
    async def close(self):
        """Close the HTTP client connection"""
        if self.client:
            await self.client.aclose()
            self.logger.info("✅ QzClient connection closed")
    
    async def call_llm(
        self, 
        model_name: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 8000,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        """Call Qz API chat completion"""
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
                    provider="qz",
                    model_name=model_name,
                    messages=messages,
                    params=params_for_record,
                )
            except Exception as e:
                self.logger.warning(f"LLM monitor start failed: {e}")

        self.logger.info(f"🚀 Requesting Qz API: model={model_name}, max_tokens={max_tokens}, trace_id={trace_id}")
        
        async def api_call():
            params = self._prepare_chat_params(
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            response = await self.client.post("/v1/chat/completions", json=params)
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
                self.logger.error("❌ Invalid response format from Qz API")
                return None
            
            self.logger.info(f"✅ Qz API response received successfully, trace_id={trace_id}")

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
            self.logger.error(f"❌ Failed to call Qz API after all retries: {e}, trace_id={trace_id}")
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
        """Get embeddings from Qz API"""
        self.logger.info(f"🚀 Requesting Qz API embeddings: model={model_name}")
        
        async def api_call():
            payload = {
                "model": model_name,
                "input": text,
                **kwargs
            }
            
            response = await self.client.post("/v1/embeddings", json=payload)
            response.raise_for_status()
            return response.json()
        
        try:
            response = await self.async_retry_on_exception(
                api_call,
                max_retries=3,
                delay=5,
                backoff=2
            )
            
            # Handle both single and batch embeddings
            if 'data' in response and len(response['data']) > 0:
                if isinstance(text, str):
                    # Single text input
                    embeddings = response['data'][0]['embedding']
                else:
                    # Multiple texts input - return list of embeddings
                    embeddings = [item['embedding'] for item in response['data']]
            else:
                self.logger.error("❌ Invalid embeddings response format from Qz API")
                return None
            
            self.logger.info(f"✅ Qz API embeddings received successfully")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get embeddings after all retries: {e}")
            return None
    
    async def chat(
        self, 
        system_prompt: str, 
        user_prompt: str,
        model_name: str,
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
        model_name: str,
        **kwargs
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """Convenience method for getting embeddings"""
        return await self.get_embeddings(
            model_name=model_name,
            text=text,
            **kwargs
        )

    async def rerank(
        self,
        model_name: str,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        **kwargs
    ) -> Optional[List[Dict[str, Any]]]:
        """Rerank documents based on query relevance using Qz API"""
        trace_id = None
        if hasattr(self, "monitor") and self.monitor is not None:
            try:
                trace_id = self.monitor.new_trace_id()
                # Note: monitor doesn't have rerank tracking yet, but we can use chat tracking
                # as a fallback for monitoring purposes
                self.logger.info(f"Rerank request started, trace_id={trace_id}")
            except Exception as e:
                self.logger.warning(f"LLM monitor start failed: {e}")

        self.logger.info(f"🚀 Requesting Qz API rerank: model={model_name}, docs_count={len(documents)}, top_n={top_n}, trace_id={trace_id}")
        
        async def api_call():
            payload = {
                "model": model_name,
                "query": query,
                "documents": documents,
                **kwargs
            }
            
            # Add top_n if specified
            if top_n is not None:
                payload["top_n"] = top_n
            
            self.logger.debug(f"Rerank payload: {payload}")
            response = await self.client.post("/v1/rerank", json=payload)
            response.raise_for_status()
            result = response.json()
            self.logger.debug(f"Rerank raw response: {result}")
            return result
        
        try:
            response = await self.async_retry_on_exception(
                api_call,
                max_retries=3,
                delay=5,
                backoff=2
            )
            
            # Parse rerank response
            if 'results' in response:
                results = response['results']
                # Convert to standard format with index and score
                reranked_results = []
                for item in results:
                    if isinstance(item, dict):
                        # Handle different response formats
                        if 'index' in item and 'score' in item:
                            # Standard format: {"index": 0, "score": 0.95}
                            reranked_results.append({
                                "index": item["index"],
                                "document": documents[item["index"]] if item["index"] < len(documents) else "",
                                "score": item["score"]
                            })
                        elif 'index' in item and 'relevance_score' in item:
                            # Qz API format: {"index": 0, "document": {...}, "relevance_score": 0.95}
                            reranked_results.append({
                                "index": item["index"],
                                "document": documents[item["index"]] if item["index"] < len(documents) else "",
                                "score": item["relevance_score"]
                            })
                        elif 'document' in item and 'score' in item:
                            # Alternative format with document content
                            reranked_results.append({
                                "index": documents.index(item["document"]) if item["document"] in documents else -1,
                                "document": item["document"],
                                "score": item["score"]
                            })
                        elif 'document' in item and 'relevance_score' in item:
                            # Alternative format with document content and relevance_score
                            doc_text = item["document"].get("text", "") if isinstance(item["document"], dict) else str(item["document"])
                            try:
                                doc_index = documents.index(doc_text)
                            except ValueError:
                                doc_index = -1
                            reranked_results.append({
                                "index": doc_index,
                                "document": doc_text,
                                "score": item["relevance_score"]
                            })
                        else:
                            self.logger.warning(f"Unexpected rerank result format: {item}")
                            # Try to extract basic info if possible
                            if 'index' in item:
                                try:
                                    doc_index = int(item['index'])
                                    if 0 <= doc_index < len(documents):
                                        # Try to find any score-like field
                                        score = None
                                        for key in ['score', 'relevance_score', 'relevance', 'rank_score']:
                                            if key in item:
                                                score = float(item[key])
                                                break
                                        
                                        if score is not None:
                                            reranked_results.append({
                                                "index": doc_index,
                                                "document": documents[doc_index],
                                                "score": score
                                            })
                                        else:
                                            self.logger.warning(f"No score field found in item: {item}")
                                    else:
                                        self.logger.warning(f"Invalid index {doc_index} for documents list of length {len(documents)}")
                                except (ValueError, TypeError) as e:
                                    self.logger.warning(f"Error parsing index from item: {e}, item: {item}")
                    elif isinstance(item, (int, float)):
                        # Simple score array format
                        idx = len(reranked_results)
                        if idx < len(documents):
                            reranked_results.append({
                                "index": idx,
                                "document": documents[idx],
                                "score": float(item)
                            })
                
                # Sort results by score in descending order if not already sorted
                reranked_results.sort(key=lambda x: x["score"], reverse=True)
                
                # Apply top_n filter if specified
                if top_n is not None and top_n > 0:
                    reranked_results = reranked_results[:top_n]
                
                self.logger.info(f"✅ Qz API rerank completed successfully, returned {len(reranked_results)} results, trace_id={trace_id}")
                return reranked_results
            else:
                self.logger.error(f"❌ Invalid rerank response format from Qz API - missing 'results' field: {response}")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to rerank documents after all retries: {e}, trace_id={trace_id}")
            return None

    async def chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
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
                    provider="qz",
                    model_name=model_name,
                    messages=messages,
                    params=params_for_record,
                )
            except Exception as e:
                self.logger.warning(f"LLM monitor start failed: {e}")

        self.logger.info(f"🚀 Streaming from Qz API: model={model_name}, max_tokens={max_tokens}, trace_id={trace_id}")

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
            async with self.client.stream("POST", "/v1/chat/completions", json=params) as response:
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
            self.logger.info(f"✅ Qz API streaming finished, trace_id={trace_id}")

        except Exception as e:
            self.logger.error(f"❌ Qz API streaming failed: {e}, trace_id={trace_id}")
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
