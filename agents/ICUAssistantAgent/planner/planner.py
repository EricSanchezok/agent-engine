from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.utils import get_relative_path_from_current_file

import os

class planner:
    def __init__(self):
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))

    

    