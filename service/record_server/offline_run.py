import asyncio
import json
import sys
import os
import random
from pathlib import Path
from typing import List, Dict, Any
from pprint import pprint

# Agent Engine imports
from agent_engine.a2a_client import send_message_to_a2a_agent_streaming, send_message_to_a2a_agent
from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_relative_path_from_current_file

# Core imports
from core.holos.config import PROXY_URL

# Local imports
from service.record_server.config import ROUTING_AGENT_BASE_URL
from service.record_server.record_memory import RecordMemory
from service.record_server.fast_judger import FastJudger

logger = AgentLogger(__name__)

capabilities_path = get_relative_path_from_current_file("capabilities_with_tasks.json")
semaphore = asyncio.Semaphore(32)

def random_choose():
    with open(capabilities_path, "r", encoding="utf-8") as f:
        capabilities = json.load(f)
    random_capability = random.choice(capabilities)
    random_task = random.choice(random_capability["tasks"])
    random_agent = random.choice(random_capability["agents"])
    return random_task, random_agent


async def run(memory: RecordMemory, agent: Dict[str, Any], capability: Dict[str, Any], task_content: str):
    async with semaphore:
        fast_judger = FastJudger()
        response = None
        try:
            response = await send_message_to_a2a_agent(
                base_url = agent.get("url"),
                message = task_content,
                proxy_url = None
            )
            logger.info(f"Response: {response}")
        except Exception as e:
            logger.error(f"❌ Failed to send message to agent: {e}")

        if response:
            success = await fast_judger.invoke(task_content, response)
            logger.info(f"Task {task_content} was judged as {success}")

            await memory.add_task_result(
                agent_name=agent.get("name"),
                agent_url=agent.get("url"),
                capability_name=capability.get("name"),
                capability_definition=capability.get("definition"),
                success=success,
                task_content=task_content,
                task_result=response
            )

def save_history():
    memory = RecordMemory()

    with open(capabilities_path, "r", encoding="utf-8") as f:
        capabilities = json.load(f)

    history = []

    for capability in capabilities:
        print("="*100)
        history.append(asyncio.run(memory.get_capability_history(capability["name"], capability["definition"])))
    
    with open("database/tasks_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    memory = RecordMemory()
    memory.delete_all_task_history()

    with open(capabilities_path, "r", encoding="utf-8") as f:
        capabilities = json.load(f)

    pending_tasks = []

    for capability in capabilities:
        for agent in capability["agents"]:
            for task in capability["tasks"]:
                pending_tasks.append((agent, capability, task))

    print(len(pending_tasks))
    tasks = [run(memory, agent, capability, task) for agent, capability, task in pending_tasks]
    
    async def main():
        await asyncio.gather(*tasks)
    
    asyncio.run(main())

    save_history()
    