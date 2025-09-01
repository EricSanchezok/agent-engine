from dotenv import load_dotenv
import os
import asyncio
from typing import List, Dict, Any
import json
from pprint import pprint

# AgentEngine imports
from agent_engine.memory import Memory
from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.utils import get_relative_path_from_current_file, get_current_file_dir

# Core imports
from core.holos import get_all_agent_cards

logger = AgentLogger(__name__)

load_dotenv()

class CapabilityBuilder:
    def __init__(self):
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))
        self.capability_memory = Memory(name='capability_memory', db_path=get_current_file_dir() / 'database' / 'capability_memory.db')
        self.semaphore = asyncio.Semaphore(32)

    async def invoke(self):
        capabilities = await self.run_capability_extractor(load=False)
        self.capability_memory.clear()

        for capability in capabilities:
            capability_content = {
                'name': capability.get('name'),
                'definition': capability.get('definition'),
                'alias': [capability.get('name')],
                'agents': [capability.get('agent')]
            }
            text = {
                'name': capability.get('name'),
                'definition': capability.get('definition'),
            }
            text = json.dumps(text, ensure_ascii=False, indent=4)
            vector = await self.llm_client.embedding(text, model_name='text-embedding-3-small')

            if self.capability_memory.count() == 0:
                self.capability_memory.add(json.dumps(capability_content, ensure_ascii=False, indent=4), vector)
                continue
            
            result = self.capability_memory.search(vector, top_k=5)
            similar_capabilities = [json.loads(_result[0]) for _result in result if _result[1] > 0.50]

            if similar_capabilities:
                system_prompt = self.prompt_loader.get_prompt(
                    section='capability_merger',
                    prompt_type='system'
                )
                user_prompt = self.prompt_loader.get_prompt(
                    section='capability_merger',
                    prompt_type='user',
                    new_capability=capability,
                    existing_capabilities=similar_capabilities
                )
                response = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
                response = json.loads(response)

                target_name = response.get('target_name')
                if target_name:
                    target_capability = next((_capability for _capability in similar_capabilities if _capability.get('name') == target_name), None)
                    if target_capability:
                        self.capability_memory.delete_by_content(json.dumps(target_capability, ensure_ascii=False, indent=4))
                        target_capability['name'] = response.get('new_name') if response.get('new_name') else target_capability['name']
                        target_capability['definition'] = response.get('new_definition') if response.get('new_definition') else target_capability['definition']
                        if capability.get('name') not in target_capability['alias']:
                            target_capability['alias'].append(capability.get('name'))
                        if capability.get('agent') not in target_capability['agents']:
                            target_capability['agents'].append(capability.get('agent'))
                        self.capability_memory.add(json.dumps(target_capability, ensure_ascii=False, indent=4), vector)
                    else:
                        self.capability_memory.add(json.dumps(capability_content, ensure_ascii=False, indent=4), vector)
                else:
                    self.capability_memory.add(json.dumps(capability_content, ensure_ascii=False, indent=4), vector)
                continue
        
            self.capability_memory.add(json.dumps(capability_content, ensure_ascii=False, indent=4), vector)

        await self.save()


    async def run_capability_extractor(self, load: bool = False) -> List[Dict[str, Any]]:
        # Try to load from local JSON file if load is True
        if load:
            try:
                json_file_path = get_current_file_dir() / 'capabilities.json'
                if json_file_path.exists():
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        capabilities = json.load(f)
                        logger.info(f"Loaded {len(capabilities)} capabilities from local JSON file")
                        return capabilities
            except Exception as e:
                logger.error(f"Failed to load from local JSON file: {e}")
                logger.info("Proceeding with normal extraction process...")
        
        # Normal extraction process
        cards = get_all_agent_cards()
        tasks = [self._capability_extract(card) for card in cards]
        results = await asyncio.gather(*tasks)
        capabilities = []
        for result in results:
            capabilities.extend(result)
        
        # Save to local JSON file for future use
        try:
            json_file_path = get_current_file_dir() / 'capabilities.json'
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(capabilities, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved {len(capabilities)} capabilities to local JSON file")
        except Exception as e:
            logger.error(f"Failed to save to local JSON file: {e}")
        
        return capabilities

    async def _capability_extract(self, card: Dict[str, Any]) -> List[Dict[str, Any]]:
        async with self.semaphore:
            system_prompt = self.prompt_loader.get_prompt(
                section='capability_extractor',
                prompt_type='system'
            )

            agent_info = {
                'name': card.get('name'),
                'description': card.get('description'),
                'skills': card.get('skills')
            }

            user_prompt = self.prompt_loader.get_prompt(
                section='capability_extractor',
                prompt_type='user',
                agent_info=agent_info
            )   

            response = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
            response = json.loads(response)
            result = []
            for _capability in response:
                _capability['agent'] = {
                    'name': card.get('name'),
                    'url': card.get('url')
                }
                result.append(_capability)
            return result

    async def save(self):
        capabilities = []
        for capability in self.capability_memory.get_all_contents():
            capabilities.append(json.loads(capability))

        with open(get_current_file_dir() / 'view_capabilities.json', 'w', encoding='utf-8') as f:
            json.dump(capabilities, f, ensure_ascii=False, indent=4)

    async def test(self):
        # text = {
        #     "name": "Browse web",
        #     "definition": "This capability enables an agent to browse the web to answer questions. It processes user queries by searching online for up-to-date information and returning relevant results, such as weather updates or topic-specific data.",
        # }

        # text = json.dumps(text, ensure_ascii=False, indent=4)
        # vector = await self.llm_client.embedding(text, model_name='text-embedding-3-small')

        # result = self.capability_memory.search(vector, top_k=5)
        # pprint(result)

        capability_A = {
            "name": "Automate browser tasks",
            "definition": "This capability allows the agent to control a headless browser for web automation. It can navigate web pages, perform searches, click elements, extract web content, automate form filling, and take screenshots.",
        }
        
        capability_B = {
            "name": "Browse web",
            "definition": "This capability enables an agent to browse the web to answer questions. It processes user queries by searching online for up-to-date information and returning relevant results, such as weather updates or topic-specific data.",
        }
        
        system_prompt = self.prompt_loader.get_prompt(
            section='capability_merger',
            prompt_type='system'
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='capability_merger',
            prompt_type='user',
            new_capability=capability_A,
            existing_capabilities=[capability_B]
        )
        response = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
        response = json.loads(response)
        pprint(response)


if __name__ == '__main__':
    builder = CapabilityBuilder()
    # asyncio.run(builder.invoke())
    asyncio.run(builder.test())