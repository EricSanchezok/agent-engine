from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import asyncio


# Agent Engine imports
from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.memory import ScalableMemory
from agent_engine.utils import get_relative_path_from_current_file

# Local imports
from agents.ICUMemoryAgent.utils import save_test_events

logger = AgentLogger(__name__)

load_dotenv()

class ICUMemoryCollector:
    def __init__(self):
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))

    async def invoke(self, user_query: str, total_events: int, event_time_range: Tuple[str, str]) -> List[Dict[str, Any]]:
        sub_type_descriptions = json.load(open(get_relative_path_from_current_file('sub_type_descriptions.json'), 'r', encoding='utf-8'))
        
        system_prompt = self.prompt_loader.get_prompt(
            section='collector',
            prompt_type='system',
            sub_type_descriptions=sub_type_descriptions
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='collector',
            prompt_type='user',
            user_query=user_query,
            current_time=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            total_events=total_events,
            event_time_range=event_time_range
        )

        try:
            response = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
            response = json.loads(response)
            return response
        except Exception as e:
            logger.error(f"ICUMemoryCollector error: {e}")
            return []

async def main():
    from agents.ICUDataIngestionAgent.agent import ICUDataIngestionAgent
    from agents.ICUMemoryAgent.agent import ICUMemoryAgent
    from pprint import pprint

    UPDATES = 100
    PATIENT_ID = "1125112810"
    user_query = "What was the most recent surgery the patient had?"

    ingestion = ICUDataIngestionAgent()
    patient_json_path = f"database/icu_raw/{PATIENT_ID}.json"
    ingestion.load_patient(patient_json_path)
    patient_id = ingestion.patient_id or PATIENT_ID
    memory = ICUMemoryAgent()
    memory.delete_patient_memory(ingestion.patient_id)

    total_written = 0
    for i in range(1, UPDATES + 1):
        batch = await ingestion.update()
        if not batch:
            logger.info("No more events from ingestion; stopping early at update %s", i)
            break
        ids = await memory.add_events(patient_id, batch)
        total_written += len(ids)
        logger.info(f"Update {i}: wrote {len(ids)} events (total={total_written})")

    events = await memory.query_search(patient_id, user_query, top_k=20)
    save_test_events(events)

    print(memory.get_event_count(patient_id))
    print(memory.get_event_time_range(patient_id))

    collector = ICUMemoryCollector()
    result = await collector.invoke(user_query, memory.get_event_count(patient_id), memory.get_event_time_range(patient_id))
    print(result)

if __name__ == "__main__":
    asyncio.run(main())