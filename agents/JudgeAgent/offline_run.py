import asyncio
import json
import sys
import os
import random
from pathlib import Path
from typing import List, Dict, Any
from pprint import pprint

# Agent Engine imports
from agent_engine.a2a_client import send_message_to_a2a_agent
from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_relative_path_from_current_file

# Core imports
from core.holos.config import PROXY_URL

# Local imports
from agents.JudgeAgent.config import ROUTING_AGENT_BASE_URL
from agents.JudgeAgent.judge_memory import JudgeMemory

logger = AgentLogger(__name__)

capabilities_path = get_relative_path_from_current_file("capabilities_with_tasks.json")

def random_choose():
    with open(capabilities_path, "r", encoding="utf-8") as f:
        capabilities = json.load(f)
    random_capability = random.choice(capabilities)
    random_task = random.choice(random_capability["tasks"])
    random_agent = random.choice(random_capability["agents"])
    return random_task, random_agent


async def run(task, agent):
    message = {
        "test_task": {
            "task_description": task,
            "depends_on": [],
            "agent": agent.get("name"),
            "agent_url": agent.get("url")
        }
    }
    response = await send_message_to_a2a_agent(
        base_url = ROUTING_AGENT_BASE_URL,
        message = json.dumps(message, ensure_ascii=False, indent=4),
        proxy_url = PROXY_URL
    )
    return response


if __name__ == "__main__":
    task, agent = random_choose()
    response = asyncio.run(run(task, agent))
    pprint(response)