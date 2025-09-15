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

from agent_engine.llm_client import LLMChatMonitor
from pprint import pprint

if __name__ == "__main__":
    monitor = LLMChatMonitor()
    summary = monitor.summarize_usage_cost()
    pprint(summary)