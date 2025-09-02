import asyncio
import json
from pprint import pprint


# Agent Engine imports
from agent_engine.a2a_client import send_message_to_a2a_agent


# Local imports
from agents.JudgeAgent.config import ROUTING_AGENT_BASE_URL

if __name__ == "__main__":
    message = {"search_papers": {"task_description": "搜索7月13号AI 领域的论文", "depends_on": [],"agent": "ArxivSearchAgent", "agent_url": ""}}

    response = asyncio.run(send_message_to_a2a_agent(
        base_url=ROUTING_AGENT_BASE_URL,
        message=json.dumps(message, ensure_ascii=False, indent=4),
        message_id=None,
        role="user",
        timeout=30.0
    ))
    pprint(response)