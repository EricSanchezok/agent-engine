from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..agent_logger.agent_logger import AgentLogger
from ..utils.project_root import get_project_root
from ..memory.scalable_memory import ScalableMemory


class LLMChatMonitor:
    """Monitor for persisting LLM chat sessions using ScalableMemory without vectors."""

    def __init__(
        self,
        *,
        name: str = "llm_chats",
        enable_vectors: bool = False,
        notify_url: Optional[str] = None,
    ) -> None:
        self.logger = AgentLogger("LLMMonitor")
        self.name = name
        self.notify_url = notify_url

        # Ensure storage under project_root/.llm_monitoring (no extra subfolder)
        project_root = get_project_root()
        persist_dir = project_root / ".llm_monitoring"

        # Initialize ScalableMemory in non-vector mode
        self.memory = ScalableMemory(
            name=name,
            enable_vectors=enable_vectors,
            persist_dir=str(persist_dir),
            db_backend="sqlite",
        )

    def new_trace_id(self, prefix: str = "llm") -> str:
        now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        rand = uuid.uuid4().hex[:8]
        return f"{prefix}_{now}_{rand}"

    async def start_chat(
        self,
        *,
        trace_id: str,
        provider: str,
        model_name: str,
        messages: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> None:
        try:
            started_at = datetime.now().isoformat()

            content: Dict[str, Any] = {
                "trace_id": trace_id,
                "provider": provider,
                "model_name": model_name,
                "request": {
                    "messages": messages,
                    "params": params,
                },
                "response": {
                    "content": None,
                    "usage": None,
                    "raw": None,
                },
                "timing": {
                    "started_at": started_at,
                    "ended_at": None,
                    "duration_ms": None,
                },
                "status": "pending",
            }

            metadata: Dict[str, Any] = {
                "trace_id": trace_id,
                "provider": provider,
                "model_name": model_name,
                "status": "pending",
                "started_at": started_at,
                "ended_at": None,
            }

            await self.memory.add(
                content=json.dumps(content, ensure_ascii=False),
                metadata=metadata,
                item_id=trace_id,
            )
            # notify frontend if configured
            try:
                if self.notify_url:
                    try:
                        import aiohttp  # type: ignore
                        async with aiohttp.ClientSession() as sess:
                            await sess.post(self.notify_url, json={"id": trace_id})
                    except Exception:
                        # fallback to stdlib if aiohttp not available
                        import urllib.request
                        import urllib.error
                        import json as _json
                        req = urllib.request.Request(self.notify_url, data=_json.dumps({"id": trace_id}).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
                        try:
                            with urllib.request.urlopen(req, timeout=2) as _:
                                pass
                        except Exception as _e:
                            # best-effort
                            self.logger.warning(f"LLM monitor notify fallback failed: {_e}")
            except Exception as ne:
                self.logger.warning(f"LLM monitor notify failed: {ne}")
        except Exception as e:
            self.logger.warning(f"LLM monitor start_chat failed: {e}")

    async def complete_chat(
        self,
        *,
        trace_id: str,
        response_text: str,
        usage: Optional[Dict[str, Any]] = None,
        raw: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            ended_at = datetime.now().isoformat()

            old_content_str, _, old_meta = self.memory.get_by_id(trace_id)
            content = json.loads(old_content_str) if old_content_str else {}

            started_at_str = None
            try:
                started_at_str = content.get("timing", {}).get("started_at")
            except Exception:
                started_at_str = None

            duration_ms: Optional[int] = None
            if started_at_str:
                try:
                    started = datetime.fromisoformat(started_at_str)
                    ended = datetime.fromisoformat(ended_at)
                    duration_ms = int((ended - started).total_seconds() * 1000)
                except Exception:
                    duration_ms = None

            # Update content
            content.setdefault("response", {})
            content["response"]["content"] = response_text
            content["response"]["usage"] = usage
            content["response"]["raw"] = raw

            content.setdefault("timing", {})
            content["timing"]["ended_at"] = ended_at
            content["timing"]["duration_ms"] = duration_ms
            content["status"] = "success"

            # Update metadata
            old_meta = old_meta or {}
            old_meta["status"] = "success"
            old_meta["ended_at"] = ended_at

            await self.memory.upsert(
                item_id=trace_id,
                content=json.dumps(content, ensure_ascii=False),
                metadata=old_meta,
            )
            # notify frontend if configured
            try:
                if self.notify_url:
                    try:
                        import aiohttp  # type: ignore
                        async with aiohttp.ClientSession() as sess:
                            await sess.post(self.notify_url, json={"id": trace_id})
                    except Exception:
                        import urllib.request
                        import urllib.error
                        import json as _json
                        req = urllib.request.Request(self.notify_url, data=_json.dumps({"id": trace_id}).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
                        try:
                            with urllib.request.urlopen(req, timeout=2) as _:
                                pass
                        except Exception as _e:
                            self.logger.warning(f"LLM monitor notify fallback failed: {_e}")
            except Exception as ne:
                self.logger.warning(f"LLM monitor notify failed: {ne}")
        except Exception as e:
            self.logger.warning(f"LLM monitor complete_chat failed: {e}")

    async def fail_chat(
        self,
        *,
        trace_id: str,
        error_message: str,
        raw: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            ended_at = datetime.now().isoformat()
            old_content_str, _, old_meta = self.memory.get_by_id(trace_id)
            content = json.loads(old_content_str) if old_content_str else {}

            started_at_str = None
            try:
                started_at_str = content.get("timing", {}).get("started_at")
            except Exception:
                started_at_str = None

            duration_ms: Optional[int] = None
            if started_at_str:
                try:
                    started = datetime.fromisoformat(started_at_str)
                    ended = datetime.fromisoformat(ended_at)
                    duration_ms = int((ended - started).total_seconds() * 1000)
                except Exception:
                    duration_ms = None

            # Update content
            content.setdefault("response", {})
            content["response"]["content"] = None
            content["response"]["usage"] = None
            content["response"]["raw"] = raw
            content["response"]["error"] = error_message

            content.setdefault("timing", {})
            content["timing"]["ended_at"] = ended_at
            content["timing"]["duration_ms"] = duration_ms
            content["status"] = "failed"

            # Update metadata
            old_meta = old_meta or {}
            old_meta["status"] = "failed"
            old_meta["ended_at"] = ended_at

            await self.memory.upsert(
                item_id=trace_id,
                content=json.dumps(content, ensure_ascii=False),
                metadata=old_meta,
            )
            # notify frontend if configured
            try:
                if self.notify_url:
                    try:
                        import aiohttp  # type: ignore
                        async with aiohttp.ClientSession() as sess:
                            await sess.post(self.notify_url, json={"id": trace_id})
                    except Exception:
                        import urllib.request
                        import urllib.error
                        import json as _json
                        req = urllib.request.Request(self.notify_url, data=_json.dumps({"id": trace_id}).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
                        try:
                            with urllib.request.urlopen(req, timeout=2) as _:
                                pass
                        except Exception as _e:
                            self.logger.warning(f"LLM monitor notify fallback failed: {_e}")
            except Exception as ne:
                self.logger.warning(f"LLM monitor notify failed: {ne}")
        except Exception as e:
            self.logger.warning(f"LLM monitor fail_chat failed: {e}")

    def get_chat(self, trace_id: str) -> Dict[str, Any]:
        try:
            content_str, _, meta = self.memory.get_by_id(trace_id)
            content = json.loads(content_str) if content_str else {}
            return {"content": content, "metadata": meta or {}}
        except Exception as e:
            self.logger.warning(f"LLM monitor get_chat failed: {e}")
            return {"content": {}, "metadata": {}}


