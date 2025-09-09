from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

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
        self.persist_dir: Path = persist_dir

        # Initialize ScalableMemory in non-vector mode
        self.memory = ScalableMemory(
            name=name,
            enable_vectors=enable_vectors,
            persist_dir=str(persist_dir),
            db_backend="sqlite",
        )

        # Lazy-loaded pricing table cache
        self._pricing_table: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None

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



    # ---------------------- Pricing & Usage Aggregation ----------------------
    def _pricing_file_path(self) -> Path:
        return self.persist_dir / "pricing.json"

    def _default_pricing(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """USD per 1M tokens. Users can override via pricing.json under .llm_monitoring.

        Note: Values are placeholders for convenience and may not reflect current vendor pricing.
        Update pricing.json for accurate accounting.
        """
        return {
            "azure": {
                "gpt-4o": {"input": 2.50, "output": 10.00},
                "gpt-4o-mini": {"input": 0.15, "output": 0.60},
                "o3": {"input": 2.00, "output": 8.00},
                "o3-mini": {"input": 1.10, "output": 4.40},
                "o4-mini": {"input": 1.10, "output": 4.40},
            },

        }

    def _load_pricing_table(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        if self._pricing_table is not None:
            return self._pricing_table
        try:
            path = self._pricing_file_path()
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._pricing_table = data  # type: ignore[assignment]
                        return self._pricing_table
        except Exception as e:
            self.logger.warning(f"Failed to load pricing.json, using defaults: {e}")
        self._pricing_table = self._default_pricing()
        return self._pricing_table

    def save_pricing_table(self, table: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """Persist a new pricing table (USD per 1M tokens) to pricing.json under .llm_monitoring."""
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            with self._pricing_file_path().open("w", encoding="utf-8") as f:
                json.dump(table, f, ensure_ascii=False, indent=2)
            self._pricing_table = table
            self.logger.info("Pricing table saved to pricing.json")
        except Exception as e:
            self.logger.warning(f"Failed to save pricing table: {e}")

    def _get_unit_prices(self, provider: str, model_name: str) -> Optional[Tuple[float, float]]:
        """Return (input_per_1m_usd, output_per_1m_usd) or None if unknown."""
        try:
            table = self._load_pricing_table()
            provider_key = str(provider or "").lower()
            model_key = str(model_name or "")
            # Try exact match first
            provider_map = table.get(provider_key) or {}
            prices = provider_map.get(model_key)
            if prices and "input" in prices and "output" in prices:
                return float(prices["input"]), float(prices["output"])
            # Try case-insensitive model name matching
            for mk, pr in provider_map.items():
                if mk.lower() == model_key.lower() and "input" in pr and "output" in pr:
                    return float(pr["input"]), float(pr["output"])
        except Exception:
            pass
        return None

    def _extract_usage_tokens(self, usage: Optional[Dict[str, Any]]) -> Tuple[int, int, int, bool]:
        """Extract (input_tokens, output_tokens, total_tokens, has_breakdown)."""
        if not usage or not isinstance(usage, dict):
            return 0, 0, 0, False
        try:
            input_tokens = int(
                usage.get("input_tokens")
                or usage.get("prompt_tokens")
                or 0
            )
        except Exception:
            input_tokens = 0
        try:
            output_tokens = int(
                usage.get("output_tokens")
                or usage.get("completion_tokens")
                or 0
            )
        except Exception:
            output_tokens = 0
        try:
            total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
        except Exception:
            total_tokens = input_tokens + output_tokens
        has_breakdown = (input_tokens > 0 or output_tokens > 0)
        return input_tokens, output_tokens, total_tokens, has_breakdown

    def _estimate_tokens(self, provider: str, model_name: str, messages: List[Dict[str, Any]], response_text: Optional[str]) -> Tuple[int, int, int]:
        """Best-effort token estimation using tiktoken if available. Returns (input, output, total)."""
        try:
            import tiktoken  # type: ignore
        except Exception:
            return 0, 0, 0

        try:
            # Map some model aliases to encodings; fallback to cl100k_base
            model_hint = str(model_name or "").lower()
            encoding_name = None
            if "gpt-4o" in model_hint or "o3" in model_hint:
                # OpenAI omni/o3 map to o200k_base in tiktoken
                encoding_name = "o200k_base"
            enc = tiktoken.get_encoding(encoding_name) if encoding_name else tiktoken.get_encoding("cl100k_base")
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                return 0, 0, 0

        def count_text(text: str) -> int:
            try:
                return len(enc.encode(text))
            except Exception:
                return 0

        in_tokens = 0
        for msg in messages or []:
            try:
                content = msg.get("content")
                if isinstance(content, str):
                    in_tokens += count_text(content)
                elif isinstance(content, list):
                    # content might be list of parts
                    for part in content:
                        if isinstance(part, dict):
                            val = part.get("text") or part.get("content")
                            if isinstance(val, str):
                                in_tokens += count_text(val)
            except Exception:
                continue

        out_tokens = count_text(response_text) if isinstance(response_text, str) else 0
        total = in_tokens + out_tokens
        return in_tokens, out_tokens, total

    def summarize_usage_cost(self) -> Dict[str, Any]:
        """Aggregate total tokens and cost across all stored chats.

        Returns a dict with total tokens, total cost, and per-model breakdown.
        """
        items = self.memory.get_all()
        total_input = 0
        total_output = 0
        total_tokens = 0
        total_cost_usd = 0.0

        per_model: Dict[str, Dict[str, Any]] = {}
        unknown_pricing: Dict[str, int] = {}
        estimated_count = 0

        for content_str, _vec, meta in items:
            try:
                content = json.loads(content_str) if content_str else {}
            except Exception:
                content = {}

            provider = (meta or {}).get("provider") or content.get("provider") or ""
            model_name = (meta or {}).get("model_name") or content.get("model_name") or ""
            status = (meta or {}).get("status") or content.get("status")

            # Only include items that have finished (success or failed); pending has no usage
            if status not in ("success", "failed"):
                continue

            resp = content.get("response") or {}
            usage = resp.get("usage") if isinstance(resp, dict) else None

            inp, out, tot, has_breakdown = self._extract_usage_tokens(usage)

            # If missing usage, try estimation from messages/response
            if (inp == 0 and out == 0 and tot == 0):
                messages = (content.get("request") or {}).get("messages") or []
                response_text = resp.get("content") if isinstance(resp, dict) else None
                est_in, est_out, est_tot = self._estimate_tokens(provider, model_name, messages, response_text)
                if est_tot > 0:
                    inp, out, tot = est_in, est_out, est_tot
                    has_breakdown = True
                    estimated_count += 1

            total_input += inp
            total_output += out
            total_tokens += tot

            key = f"{str(provider).lower()}::{str(model_name)}"
            if key not in per_model:
                per_model[key] = {
                    "provider": provider,
                    "model_name": model_name,
                    "chats": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            per_model[key]["chats"] += 1
            per_model[key]["input_tokens"] += inp
            per_model[key]["output_tokens"] += out
            per_model[key]["total_tokens"] += tot

            # Cost
            prices = self._get_unit_prices(str(provider), str(model_name))
            if prices is None:
                unknown_pricing[key] = unknown_pricing.get(key, 0) + 1
                continue
            input_per_1m, output_per_1m = prices
            # If no breakdown but have total, conservatively skip cost to avoid distortion
            if not has_breakdown and tot > 0:
                unknown_pricing[key] = unknown_pricing.get(key, 0) + 1
                continue
            cost = (inp / 1_000_000.0) * input_per_1m + (out / 1_000_000.0) * output_per_1m
            per_model[key]["cost_usd"] += cost
            total_cost_usd += cost

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost_usd, 6),
            "estimated_records_used": estimated_count,
            "by_model": per_model,
            "unknown_pricing": unknown_pricing,
            "pricing_table_path": str(self._pricing_file_path()),
        }