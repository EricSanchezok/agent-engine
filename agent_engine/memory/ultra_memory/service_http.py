from __future__ import annotations
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



import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from agent_engine.agent_logger.agent_logger import AgentLogger

from .core import UltraMemory, UltraMemoryConfig
from .models import CollectionSpec, Record, Filter


class UltraMemoryHTTPHandler(BaseHTTPRequestHandler):
    server_version = "UltraMemoryHTTP/0.1"

    def _send(self, code: int, obj: Any) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path == "/healthz":
            return self._send(200, self.server.um.health())
        if self.path == "/stats":
            return self._send(200, self.server.um.stats())
        return self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(body or "{}")
        except Exception:
            return self._send(400, {"error": "invalid json"})

        if self.path == "/v1/collections":
            spec = CollectionSpec(**payload)
            self.server.um.create_collection(spec)
            return self._send(200, {"ok": True})
        if self.path.endswith(":upsert"):
            collection = self.path.split("/")[-1].split(":")[0]
            records = [Record(**r) for r in payload.get("records", [])]
            ids = self.server.um.upsert(collection, records)
            return self._send(200, {"ids": ids})
        if self.path.endswith(":query"):
            collection = self.path.split("/")[-1].split(":")[0]
            flt = Filter(**payload)
            rows = self.server.um.query(collection, flt)
            return self._send(200, {"data": rows})
        if self.path.endswith(":search"):
            collection = self.path.split("/")[-1].split(":")[0]
            vector = payload.get("vector")
            top_k = int(payload.get("top_k", 5))
            threshold = float(payload.get("threshold", 0.0))
            flt = Filter(**(payload.get("filter") or {}))
            res = self.server.um.search_vectors(collection, vector, top_k=top_k, threshold=threshold, flt=flt)
            return self._send(200, {"data": res})
        return self._send(404, {"error": "not found"})


class UltraMemoryHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, um: UltraMemory) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.um = um


def serve(host: str, port: int, config: Dict[str, Any]) -> None:
    logger = AgentLogger("UltraMemoryHTTPServer")
    um = UltraMemory(UltraMemoryConfig(**config))
    server = UltraMemoryHTTPServer((host, port), UltraMemoryHTTPHandler, um)
    logger.info(f"UltraMemory HTTP serving on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        um.close()
        server.server_close()


