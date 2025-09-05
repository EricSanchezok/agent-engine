import json
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from agent_engine.memory.scalable_memory import ScalableMemory
from agent_engine.utils.project_root import get_project_root


class LLMMonitorAPIHandler(SimpleHTTPRequestHandler):
    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
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

        mem = self._get_memory()
        metas = mem.get_all_metadata()

        items = []
        for md in metas:
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

        # Sort by started_at desc
        items.sort(key=lambda x: x.get("started_at") or "", reverse=True)

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
        self._set_headers(200)
        self.wfile.write(json.dumps({
            "total": total,
            "success": success,
            "failed": failed,
            "pending": pending,
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
            while True:
                try:
                    cur = mem.count()
                    if last_count is None:
                        last_count = cur
                        # initial keep-alive
                        self.wfile.write(b": init\n\n")
                        self.wfile.flush()
                    elif cur != last_count:
                        payload = json.dumps({"type": "update", "count": cur})
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        last_count = cur
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
    run_server()


