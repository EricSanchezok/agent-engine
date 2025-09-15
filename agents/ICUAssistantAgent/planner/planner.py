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


from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.utils import get_relative_path_from_current_file

import os

class planner:
    def __init__(self):
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))

    

    