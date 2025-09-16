from unittest import result
from dotenv import load_dotenv
import asyncio
import os
import json
from typing import List, Dict, Any
from pprint import pprint
from tavily import TavilyClient

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
        self.model_name = 'gpt-4.1'
        self.tavily_client = TavilyClient(os.getenv('TAVILY_API_KEY'))
        # response = client.search(
        #     query="What is the latest research on AI agents?",
        #     include_answer="basic",
        #     max_results=20
        # )
        # print(response)

    async def plan(self, user_query: str) -> Dict[str, Any]:
        system_prompt = self.prompt_loader.get_prompt(
            section='plan',
            prompt_type='system',
            model_name=self.model_name
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='plan',
            prompt_type='user',
            user_query=user_query
        )
        try:
            response = await self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=self.model_name,
                max_tokens=12000
            )
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error planning: {e}")
            return {}

    async def deep_plan(self, user_query: str) -> Dict[str, Any]:
        hypothesis_list = await self.hypothesize(user_query)
        research_sketch = await self.explore(user_query, hypothesis_list)

        system_prompt = self.prompt_loader.get_prompt(
            section='deep_plan',
            prompt_type='system',
            model_name=self.model_name,
            research_sketch=research_sketch
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='deep_plan',
            prompt_type='user',
            user_query=user_query
        )
        try:
            response = await self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=self.model_name,
                max_tokens=12000
            )
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error deep planning: {e}")
            return {}


    async def hypothesize(self, user_query: str) -> List[str]:
        """Stream planning response chunk by chunk"""
        system_prompt = self.prompt_loader.get_prompt(
            section='hypothesize',
            prompt_type='system',
            model_name=self.model_name
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='hypothesize',
            prompt_type='user',
            user_query=user_query
        )
        
        try:
            response = await self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=self.model_name
            )
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error hypothesizing: {e}")
            return []

    async def explore(self, user_query: str, hypothesis_list: List[str]) -> Dict[str, Any]:
        search_result = await self.search_by_tavily(hypothesis_list)
        system_prompt = self.prompt_loader.get_prompt(
            section='explore',
            prompt_type='system',
            model_name=self.model_name,
            search_results=search_result
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='explore',
            prompt_type='user',
            user_query=user_query
        )

        try:
            response = await self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=self.model_name,
                max_tokens=12000
            )
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error exploring: {e}")
            return {}
        


    async def search_by_tavily(self, query_list: List[str]) -> str:
        semaphore = asyncio.Semaphore(8)
        
        async def search_query(query: str):
            async with semaphore:
                response = self.tavily_client.search(
                    query=query,
                    include_answer="basic",
                    max_results=10
                )
                return [result.get("content") for result in response.get("results")]
        
        tasks = [search_query(query) for query in query_list]
        contents = await asyncio.gather(*tasks)
        return "\n".join([item for sublist in contents for item in sublist])

    

    # async def hypothesize(self, user_query: str):
    #     """Stream planning response chunk by chunk"""
    #     prompt = self.prompt_loader.get_prompt(
    #         section='hypothesize',
    #         prompt_type='system',
    #         model_name=self.model_name
    #     )
    #     user_prompt = self.prompt_loader.get_prompt(
    #         section='hypothesize',
    #         prompt_type='user',
    #         user_query=user_query
    #     )
        
    #     try:
    #         async for chunk in self.llm_client.chat_stream(
    #             system_prompt=prompt,
    #             user_prompt=user_prompt,
    #             model_name=self.model_name
    #         ):
    #             yield chunk
    #     except Exception as e:
    #         self.logger.error(f"Error hypothesizing: {e}")


async def test():
    planner = DeepResearchPlanner()
    response = await planner.plan(user_query="What is the latest research on AI agents?")
    pprint(response)
    # async for chunk in planner.hypothesize(user_query="What is the latest research on AI agents?"):
    #     print(chunk, end='', flush=True)
    # print()  # Add newline at the end

if __name__ == "__main__":
    asyncio.run(test())