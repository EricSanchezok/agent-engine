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

# AgentEngine imports
from agent_engine.memory import Memory
from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.utils import get_relative_path_from_current_file, get_current_file_dir

# Core imports
from core.holos import get_all_agent_cards

# Local imports
from service.record_server.record_memory import RecordMemory

logger = AgentLogger(__name__)

load_dotenv()

class CapabilityBuilder:
    def __init__(self):
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))
        self.capability_memory = RecordMemory(name='record_memory', db_path='database/record_memory.sqlite')
        self.semaphore = asyncio.Semaphore(32)

    async def invoke(self):
        capabilities = await self.run_capability_extractor(load=False)
        self.capability_memory.clear()

        async def add_capability(capability: Dict[str, Any]):
            await self.capability_memory.add_capability(
                name=capability.get('name'),
                definition=capability.get('definition'),
                alias=[capability.get('name')],
                agents=[capability.get('agent')]
            )

        for capability in capabilities:
            if self.capability_memory.count() == 0:
                await add_capability(capability)
                continue
            
            result = await self.capability_memory.search_similar_capabilities(capability.get('name'), capability.get('definition'), top_k=5, threshold=0.7)
            similar_capabilities = []
            for similar_cap in result:
                similar_cap_content = {
                    'name': similar_cap['name'],
                    'definition': similar_cap['definition'],
                    'alias': similar_cap.get('metadata', {}).get('alias', []),
                    'agents': similar_cap.get('metadata', {}).get('agents', [])
                }
                similar_capabilities.append(similar_cap_content)

            if similar_capabilities:
                # print("*"*100)
                # pprint(capability)
                # print("-"*100)
                # pprint(similar_capabilities)
                # print("-"*100)
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
                    # print("x"*100)
                    # pprint(response)
                    # print("x"*100)
                    target_capability = next((_capability for _capability in similar_capabilities if _capability.get('name') == target_name), None)
                    if target_capability:
                        # Find the actual capability to delete using similarity search
                        actual_capability_to_delete = None
                        for similar_cap in result:
                            if similar_cap['name'] == target_name:
                                actual_capability_to_delete = similar_cap
                                break
                        
                        if actual_capability_to_delete:
                            # Delete the actual capability found in memory
                            await self.capability_memory.delete_capability(
                                actual_capability_to_delete['name'], 
                                actual_capability_to_delete['definition']
                            )
                        else:
                            logger.error(f"Target capability not found in memory: {target_name}")
                            await add_capability(capability)
                            continue
                        
                        # Update target capability
                        new_name = response.get('new_name') if response.get('new_name') else target_capability['name']
                        new_definition = response.get('new_definition') if response.get('new_definition') else target_capability['definition']
                        
                        # Update alias and agents
                        if capability.get('name') not in target_capability['alias']:
                            target_capability['alias'].append(capability.get('name'))
                        if capability.get('agent') not in target_capability['agents']:
                            target_capability['agents'].append(capability.get('agent'))
                        
                        # Add updated capability
                        await self.capability_memory.add_capability(
                            name=new_name,
                            definition=new_definition,
                            alias=target_capability['alias'],
                            agents=target_capability['agents']
                        )
                        # pprint({
                        #     'name': new_name,
                        #     'definition': new_definition,
                        #     'alias': target_capability['alias'],
                        #     'agents': target_capability['agents']
                        # })
                        # print("-"*100)
                    else:
                        logger.error(f"Target capability not found: {target_name}")
                        await add_capability(capability)
                else:
                    logger.warning(f"Target name not found: {target_name}")
                    await add_capability(capability)
                continue
        
            await add_capability(capability)

        await self.save()

    async def run_capability_extractor(self, load: bool = False) -> List[Dict[str, Any]]:
        # Try to load from local JSON file if load is True
        if load:
            try:
                json_file_path = get_current_file_dir() / 'raw_capabilities.json'
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
        
        # Apply capability validation to filter out invalid capabilities (concurrent execution)
        validation_tasks = [self._validate_capability(capability) for capability in capabilities]
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        validated_capabilities = []
        for i, result in enumerate(validation_results):
            if isinstance(result, Exception):
                logger.error(f"Validation failed for capability {capabilities[i].get('name')}: {result}")
                continue
            if result:
                validated_capabilities.append(capabilities[i])
            else:
                logger.info(f"Rejected capability: {capabilities[i].get('name')} - {capabilities[i].get('definition')}")
        
        logger.info(f"Extracted {len(capabilities)} capabilities, validated {len(validated_capabilities)} capabilities")
        
        # Save to local JSON file for future use
        try:
            json_file_path = get_current_file_dir() / 'raw_capabilities.json'
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(validated_capabilities, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved {len(validated_capabilities)} validated capabilities to local JSON file")
        except Exception as e:
            logger.error(f"Failed to save to local JSON file: {e}")
        
        return validated_capabilities

    async def _validate_capability(self, capability: Dict[str, Any]) -> bool:
        """Validate if a capability is user-facing and valuable"""
        async with self.semaphore:
            system_prompt = self.prompt_loader.get_prompt(
                section='capability_validator',
                prompt_type='system'
            )
            user_prompt = self.prompt_loader.get_prompt(
                section='capability_validator',
                prompt_type='user',
                capability_name=capability.get('name'),
                capability_definition=capability.get('definition')
            )
            
            try:
                response = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
                validation_result = json.loads(response)
                
                # Log detailed validation result
                is_valid = validation_result.get('is_valid', False)
                reason = validation_result.get('reason', 'No reason provided')
                category = validation_result.get('category', 'Unknown')
                score = validation_result.get('business_value_score', 0)
                
                if not is_valid:
                    logger.info(f"Rejected capability: {capability.get('name')} - Category: {category}, Score: {score}, Reason: {reason}")
                else:
                    logger.info(f"Accepted capability: {capability.get('name')} - Category: {category}, Score: {score}")
                
                return is_valid
            except Exception as e:
                logger.error(f"Error validating capability {capability.get('name')}: {e}")
                return False

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

    def all_capabilities(self) -> List[Dict[str, Any]]:
        capabilities = []
        all_items = self.capability_memory.get_all()
        
        for content_str, vector, metadata in all_items:
            content = json.loads(content_str)
            
            # Combine content with metadata
            capability = {
                'name': content['name'],
                'definition': content['definition'],
                'alias': metadata.get('alias', []),
                'agents': metadata.get('agents', [])
            }
            capabilities.append(capability)
        return capabilities

    async def save(self):
        capabilities = self.all_capabilities()

        with open(get_current_file_dir() / 'capabilities.json', 'w', encoding='utf-8') as f:
            json.dump(capabilities, f, ensure_ascii=False, indent=4)

    async def run_task_generator(self) -> List[Dict[str, Any]]:
        capabilities = self.all_capabilities()
        tasks = [self._task_generator(capability) for capability in capabilities]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and empty results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task generation failed: {result}")
                continue
            if result:
                valid_results.append(result)

        with open(get_current_file_dir() / 'capabilities_with_tasks.json', 'w', encoding='utf-8') as f:
            json.dump(valid_results, f, ensure_ascii=False, indent=4)

        return valid_results

    async def _task_generator(self, capability: Dict[str, Any]) -> Dict[str, Any]:
        async with self.semaphore:
            system_prompt = self.prompt_loader.get_prompt(
                section='capability_filter',
                prompt_type='system'
            )
            user_prompt = self.prompt_loader.get_prompt(
                section='capability_filter',
                prompt_type='user',
                capability_name=capability.get('name'),
                capability_definition=capability.get('definition')
            )
            response = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')

            if 'yes' in response or len(capability.get('agents')) <= 1:
                return {}

            system_prompt = self.prompt_loader.get_prompt(
                section='task_generator',
                prompt_type='system'
            )
            user_prompt = self.prompt_loader.get_prompt(
                section='task_generator',
                prompt_type='user',
                capability_name=capability.get('name'),
                capability_definition=capability.get('definition'),
                num_tasks=10
            )
            response = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
            response = json.loads(response)

            tasks = response.get('tasks')
            if not tasks:
                return {}

            capability['tasks'] = tasks
            return capability


if __name__ == '__main__':
    builder = CapabilityBuilder()
    asyncio.run(builder.invoke())
    asyncio.run(builder.run_task_generator())