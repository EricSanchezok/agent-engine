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


from dotenv import load_dotenv
import os
import asyncio
from typing import List, Dict, Any
import json
from pprint import pprint

# Agent Engine imports
from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_relative_path_from_current_file

logger = AgentLogger('FastJudger')

load_dotenv()

class FastJudger:
    def __init__(self):
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))

    async def invoke(self, task_content: str, task_result: str) -> bool:
        if isinstance(task_result, dict):
            task_result = json.dumps(task_result, ensure_ascii=False)

        logger.info(f"Judging task: {task_content} with result: \n{task_result[:4000]}")
        system_prompt = self.prompt_loader.get_prompt(
            section='task_analyzer',
            prompt_type='system'
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='task_analyzer',
            prompt_type='user',
            task_content=task_content,
            task_result=task_result[:4000]
        )
        
        try:
            response = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
            response = json.loads(response)
        except Exception as e:
            logger.error(f"❌ Failed to judge task: {e}")
            return False
        
        logger.info(f"Judged result: {response}")
        if response.get('success'):
            logger.info(f"Task {task_content} was judged as successful")
            return True
        else:
            logger.info(f"Task {task_content} was judged as failed")
            return False

        