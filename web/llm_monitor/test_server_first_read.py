import subprocess
import sys
import time
from pathlib import Path
from urllib import request as urlrequest, error as urlerror

from agent_engine.agent_logger import AgentLogger


logger = AgentLogger("LLMMonitorFirstReadTest")


def _http_get(url: str, timeout: float = 3.0) -> None:
    try:
        logger.info(f"GET {url}")
        with urlrequest.urlopen(url, timeout=timeout) as resp:
            status = resp.status
            chunk = resp.read(512)
            logger.info(f"GET {url} -> status={status}, head={chunk[:128]!r}")
    except urlerror.HTTPError as e:
        logger.error(f"GET {url} -> HTTP {e.code}: {e.reason}")
        try:
            body = e.read(256)
            logger.error(f"Error body head: {body!r}")
        except Exception:
            pass
    except Exception as e:
        logger.error(f"GET {url} -> failed: {e}")


def _sse_probe(url: str, timeout: float = 3.0) -> None:
    try:
        logger.info(f"SSE probe {url}")
        req = urlrequest.Request(url)
        # Open and read a tiny chunk; server should keep-alive, we'll close immediately
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            head = resp.read(64)
            logger.info(f"SSE {url} -> status={status}, head={head!r}")
    except Exception as e:
        logger.error(f"SSE {url} -> failed: {e}")


def main():
    web_dir = Path(__file__).parent
    server = subprocess.Popen([sys.executable, str(web_dir / "run.py")])
    try:
        time.sleep(1.2)  # wait server up
        base = "http://127.0.0.1:8765"
        # 1) 直接触发会话列表（首次读取 DB）
        _http_get(f"{base}/api/llm/sessions")
        # 2) 触发 SSE 首次连接（也会读取 DB 统计）
        _sse_probe(f"{base}/api/llm/stream")
        logger.info("Done. Press Ctrl+C to stop server…")
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.terminate()


if __name__ == "__main__":
    main()


