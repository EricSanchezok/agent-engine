import json
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from agent_engine.memory.scalable_memory import ScalableMemory
from agent_engine.utils.project_root import get_project_root


_EVENT_LOCK = threading.Lock()
_EVENT_QUEUE = []  # list of {"type": "new", "id": trace_id}

class LLMMonitorAPIHandler(SimpleHTTPRequestHandler):
    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/llm/"):
            return self.handle_api_get(parsed)
        else:
            # Serve static files from this directory
            return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/llm/"):
            # only API accepts POST
            self._set_headers(404)
            self.wfile.write(b"{}")
            return
        try:
            if parsed.path == "/api/llm/notify":
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except Exception:
                    payload = {}
                trace_id = payload.get("id") or payload.get("trace_id")
                if not trace_id:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": "missing id"}).encode("utf-8"))
                    return
                # enqueue event for SSE consumers
                with _EVENT_LOCK:
                    _EVENT_QUEUE.append({"type": "new", "id": str(trace_id)})
                self._set_headers(200)
                self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
                return
            self._set_headers(404)
            self.wfile.write(b"{}")
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def handle_api_get(self, parsed):
        try:
            if parsed.path == "/api/llm/sessions":
                return self._handle_sessions(parsed)
            if parsed.path.startswith("/api/llm/sessions/"):
                trace_id = parsed.path.rsplit("/", 1)[-1]
                return self._handle_session_detail(trace_id)
            if parsed.path == "/api/llm/stats":
                return self._handle_stats()
            if parsed.path == "/api/llm/stream":
                return self._handle_stream()
            self._set_headers(404)
            self.wfile.write(b"{}")
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def _get_memory(self) -> ScalableMemory:
        # Use the same persist_dir as the monitor: project_root/.llm_monitoring
        project_root = get_project_root()
        persist_dir = project_root / ".llm_monitoring"
        return ScalableMemory(name="llm_chats", enable_vectors=False, persist_dir=str(persist_dir), db_backend="sqlite")

    def _handle_sessions(self, parsed):
        qs = parse_qs(parsed.query or "")
        status = (qs.get("status", [""])[0] or "").strip()
        model = (qs.get("model", [""])[0] or "").strip()
        provider = (qs.get("provider", [""])[0] or "").strip()
        q = (qs.get("q", [""])[0] or "").strip().lower()
        trace_id_filter = (qs.get("trace_id", [""])[0] or "").strip()

        mem = self._get_memory()
        metas = mem.get_all_metadata()

        items = []
        for md in metas:
            if trace_id_filter and (md.get("id") != trace_id_filter):
                continue
            if status and (md.get("status") != status):
                continue
            if model and (md.get("model_name") != model):
                continue
            if provider and (md.get("provider") != provider):
                continue
            trace_id = md.get("id")
            if q:
                content_str, _, _ = mem.get_by_id(trace_id)
                hay = (content_str or "").lower()
                if q not in hay:
                    continue
            # Build list item
            items.append({
                "trace_id": trace_id,
                "model_name": md.get("model_name"),
                "provider": md.get("provider"),
                "status": md.get("status"),
                "started_at": md.get("started_at"),
                "ended_at": md.get("ended_at"),
            })

        # Sort by started_at desc and limit 50
        items.sort(key=lambda x: x.get("started_at") or "", reverse=True)
        items = items[:50]

        # Enrich with total_tokens from stored content (if available)
        for it in items:
            try:
                trace_id = it.get("trace_id")
                if not trace_id:
                    continue
                content_str, _, _ = mem.get_by_id(trace_id)
                if not content_str:
                    continue
                obj = json.loads(content_str)
                usage = (obj.get("response", {}) or {}).get("usage") if isinstance(obj, dict) else None
                if isinstance(usage, dict):
                    total = usage.get("total_tokens")
                    # Fallback: compute if only input/output present
                    if total is None:
                        try:
                            inp = usage.get("input_tokens") or 0
                            out = usage.get("output_tokens") or 0
                            total = (int(inp) if isinstance(inp, int) else 0) + (int(out) if isinstance(out, int) else 0)
                        except Exception:
                            total = None
                    if total is not None:
                        it["total_tokens"] = int(total)
            except Exception:
                # best-effort enrichment
                pass

        self._set_headers(200)
        self.wfile.write(json.dumps({"items": items, "total": len(items)}).encode("utf-8"))

    def _handle_session_detail(self, trace_id: str):
        mem = self._get_memory()
        content_str, _, meta = mem.get_by_id(trace_id)
        content = json.loads(content_str) if content_str else {}
        self._set_headers(200)
        self.wfile.write(json.dumps({"content": content, "metadata": meta or {}}).encode("utf-8"))

    def _handle_stats(self):
        mem = self._get_memory()
        metas = mem.get_all_metadata()
        total = len(metas)
        success = sum(1 for m in metas if m.get("status") == "success")
        failed = sum(1 for m in metas if m.get("status") == "failed")
        pending = sum(1 for m in metas if m.get("status") == "pending")
        # compute total tokens best-effort from stored content
        total_tokens = 0
        try:
            for md in metas:
                tid = md.get("id")
                if not tid:
                    continue
                content_str, _, _ = mem.get_by_id(tid)
                if not content_str:
                    continue
                obj = json.loads(content_str)
                usage = (obj.get("response", {}) or {}).get("usage") if isinstance(obj, dict) else None
                if isinstance(usage, dict):
                    tok = usage.get("total_tokens")
                    if tok is None:
                        try:
                            inp = usage.get("input_tokens") or 0
                            out = usage.get("output_tokens") or 0
                            tok = (int(inp) if isinstance(inp, int) else 0) + (int(out) if isinstance(out, int) else 0)
                        except Exception:
                            tok = None
                    if isinstance(tok, int):
                        total_tokens += tok
        except Exception:
            pass
        self._set_headers(200)
        self.wfile.write(json.dumps({
            "total": total,
            "success": success,
            "failed": failed,
            "pending": pending,
            "total_tokens": total_tokens,
        }).encode("utf-8"))

    def _handle_stream(self):
        # Server-Sent Events: push a notification when total count changes; send keep-alives otherwise
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            mem = self._get_memory()
            last_count = None
            last_success = None
            last_failed = None
            last_pending = None
            while True:
                try:
                    # flush any pending new-id events first
                    pending = []
                    with _EVENT_LOCK:
                        if _EVENT_QUEUE:
                            pending = _EVENT_QUEUE[:]
                            _EVENT_QUEUE.clear()
                    for ev in pending:
                        try:
                            payload = json.dumps({"type": "new", "id": ev.get("id")})
                            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                            self.wfile.flush()
                        except Exception:
                            pass

                    # compute status counts to detect updates without total change
                    metas = mem.get_all_metadata()
                    cur = len(metas)
                    cur_success = sum(1 for m in metas if m.get("status") == "success")
                    cur_failed = sum(1 for m in metas if m.get("status") == "failed")
                    cur_pending = sum(1 for m in metas if m.get("status") == "pending")

                    if last_count is None:
                        last_count = cur
                        last_success = cur_success
                        last_failed = cur_failed
                        last_pending = cur_pending
                        # initial keep-alive
                        self.wfile.write(b": init\n\n")
                        self.wfile.flush()
                    elif (cur != last_count) or (cur_success != last_success) or (cur_failed != last_failed) or (cur_pending != last_pending):
                        payload = json.dumps({
                            "type": "update",
                            "count": cur,
                            "success": cur_success,
                            "failed": cur_failed,
                            "pending": cur_pending,
                        })
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        last_count = cur
                        last_success = cur_success
                        last_failed = cur_failed
                        last_pending = cur_pending
                    else:
                        # keep connection alive
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                    time.sleep(1.5)
                except BrokenPipeError:
                    break
                except ConnectionResetError:
                    break
                except Exception:
                    # On unexpected errors, break the loop to avoid tight spins
                    break
        except Exception:
            # If headers sending failed or other IO error, just return
            return


def run_server(host: str = "127.0.0.1", port: int = 8765):
    web_dir = Path(__file__).parent
    os.chdir(web_dir)
    httpd = ThreadingHTTPServer((host, port), LLMMonitorAPIHandler)
    print(f"LLM Monitor web running at http://{host}:{port}")
    print("Open index.html in your browser.")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM Monitor web server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    args = parser.parse_args()
    run_server(args.host, args.port)


