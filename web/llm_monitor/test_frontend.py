import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def main():
    web_dir = Path(__file__).parent
    server = subprocess.Popen([sys.executable, str(web_dir / "run.py")])
    try:
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8765/index.html")
        print("Server started. Press Ctrl+C to stop.")
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.terminate()


if __name__ == "__main__":
    main()


