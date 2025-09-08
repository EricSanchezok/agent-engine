import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def _find_free_port(host: str = "127.0.0.1", preferred: int = 8765) -> int:
    # Try preferred first
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, preferred))
            return preferred
        except OSError:
            pass
    # Fallback to ephemeral
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def main(host: str = "127.0.0.1", port: int | None = None):
    web_dir = Path(__file__).parent
    port = port or _find_free_port(host, 8765)
    server = subprocess.Popen([sys.executable, str(web_dir / "server.py"), "--host", host, "--port", str(port)])
    try:
        time.sleep(1.5)
        webbrowser.open(f"http://{host}:{port}/index.html")
        print("Server started. Press Ctrl+C to stop.")
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.terminate()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Open LLM Monitor web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to connect (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port to use; auto-find if not set")
    args = parser.parse_args()
    main(args.host, args.port)


