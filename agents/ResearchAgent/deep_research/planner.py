from dotenv import load_dotenv
import asyncio
import os



# Agent Engine imports
from agent_engine.prompt import PromptLoader
from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.utils import get_relative_path_from_current_file

load_dotenv()

class DeepResearchPlanner:
    def __init__(self):
        self.logger = AgentLogger('DeepResearchPlanner')
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        self.model = 'gpt-4.1'

    async def plan(self, user_query: str):
        """Stream planning response chunk by chunk"""
        prompt = self.prompt_loader.get_prompt(
            section='hypothesize',
            prompt_type='system',
            model=self.model
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='hypothesize',
            prompt_type='user',
            user_query=user_query
        )
        
        try:
            async for chunk in self.llm_client.chat_stream(
                system_prompt=prompt,
                user_prompt=user_prompt
            ):
                yield chunk
        except Exception as e:
            self.logger.error(f"Error planning deep research: {e}")


async def test():
    planner = DeepResearchPlanner()
    async for chunk in planner.plan(user_query="What is the latest research on AI agents?"):
        print(chunk, end='', flush=True)
    print()  # Add newline at the end

if __name__ == "__main__":
    asyncio.run(test())