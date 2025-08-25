from dotenv import load_dotenv
import os
import asyncio
from typing import List
import json
import datetime
import pytz
from uuid import uuid4

# A2A framework imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication
from a2a.types import AgentCard
from a2a.utils import (
    new_agent_text_message, new_agent_parts_message, new_artifact,
    new_data_artifact, new_task, new_text_artifact
)
from a2a.types import (
    Artifact, Message, Role, Task, TaskStatus, TaskState, 
    FilePart, Part, FileWithBytes, TextPart, MessageSendParams
)

# AgentEngine imports
from agent_engine.agent import BaseA2AAgent
from agent_engine.llm_client import AzureClient
from agent_engine.agent import SkillIdentifier
from agent_engine.prompt import PromptLoader
from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_relative_path_from_current_file

# Core imports
from core.arxiv import ArXivFetcher, Paper, CATEGORIES_QUERY_STRING
from core.utils import DateFormatter

# Local imports
from agents.ArxivSearchAgent.config import AGENT_CARD
from agents.ArxivSearchAgent.category_navigator import ArXivCategoryNavigator
from agents.ArxivSearchAgent.query_parser import ArXivQueryParser

logger = AgentLogger(__name__)

class ArxivSearchAgent(BaseA2AAgent):
    def __init__(self):
        super().__init__(agent_card=AGENT_CARD)
        load_dotenv()
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        self.skill_identifier = SkillIdentifier(llm_client=self.llm_client, model_name='o3-mini')
        self.arxiv_fetcher = ArXivFetcher()
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))
        self.category_navigator = ArXivCategoryNavigator()
        self.query_parser = ArXivQueryParser()
        self.date_formatter = DateFormatter()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id if context.task_id else str(uuid4())
        context_id = context.context_id if context.context_id else str(uuid4())
        user_input = context.get_user_input()

        skill_id, reason = await self.skill_identifier.invoke(user_input, AGENT_CARD.skills)
        logger.info(f"Skill ID: {skill_id}, Reason: {reason}")

        if skill_id == 'search_papers_with_text':
            logger.info("Processing query parsing and category navigation")

            query_info_task = self.query_parser.invoke(user_input)
            categories_task = self.category_navigator.invoke(user_input)
            
            query_info, categories = await asyncio.gather(query_info_task, categories_task)
            query_info['cat'] = categories
            logger.debug(f"Query info: {json.dumps(query_info, ensure_ascii=False, indent=4)}")

            logger.info("Building ArXiv query")
            system_prompt = self.prompt_loader.get_prompt(
                section='query_builder',
                prompt_type='system'
            )
            user_prompt = self.prompt_loader.get_prompt(
                section='query_builder',
                prompt_type='user',
                user_input=user_input,
                query_info=query_info
            )

            try:
                result = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
                result = json.loads(result)
                query_string = result.get("result", "")
                query_string = self.date_formatter.reformat_dates_in_query(query_string)
                logger.info(f"Generated query: {query_string}")
            except Exception as e:
                logger.error(f"Query building failed: {str(e)}")
                await self._task_failed(context, event_queue, f"Query builder error: {e}")
                return

            if not categories:
                query_string = f"{query_string} AND {CATEGORIES_QUERY_STRING}"

        elif skill_id == 'search_papers_with_query_string':
            query_string = user_input
            
        else:
            await self._task_failed(context, event_queue, f"Can't find the skill: {skill_id}")
            return

        logger.info(f"Using query string: {query_string}")
        try:
            papers: List[Paper] = await self.arxiv_fetcher.search(query_string)
            logger.info(f"Found {len(papers)} papers")
        except Exception as e:
            logger.error(f"Paper fetching failed: {str(e)}")
            await self._task_failed(context, event_queue, f"ArXiv fetcher error: {e}")
            return

        json_content = json.dumps([paper.info for paper in papers], ensure_ascii=False, indent=4)
        parts = [Part(root=TextPart(text=json_content))]

        artifact = Artifact(
            artifact_id=str(uuid4()),
            parts=parts,
            name="arxiv_collect_papers",
            description=f"ArXiv收集的论文，共{len(papers)}篇",
        )
        message = Message(
            role=Role.agent,
            task_id=task_id,
            message_id=str(uuid4()),
            content_id=context_id,
            parts=parts
        )
        task = Task(
            id=task_id,
            contextId=context_id,
            artifacts=[artifact],
            history=[message],
            status=TaskStatus(
                state=TaskState.completed,
                timestamp=datetime.datetime.now(pytz.timezone('Asia/Shanghai')).replace(microsecond=0).isoformat(),
            ),
        )
        await self._put_event(event_queue, task)
        logger.info(f"Collection results sent - {len(papers)} papers metadata returned")

async def main():
    agent = ArxivSearchAgent()
    await agent.run_user_input("submittedDate:[20250821 TO 20250822] AND (cat:cs.CL OR cat:cs.NE OR cat:physics.comp-ph OR cat:q-bio.BM OR cat:eess.AS OR cat:cs.MM OR cat:math.IT OR cat:q-bio.QM OR cat:I.2.10; I.4.8; I.2.6; I.2.7; I.5.4; I.5.1 OR cat:physics.chem-ph OR cat:cs.SD OR cat:cs.CV OR cat:cs.AR OR cat:cond-mat.soft OR cat:cond-mat.mtrl-sci OR cat:cs.RO OR cat:cs.MA OR cat:I.2.1 OR cat:cs.IT OR cat:cs.HC OR cat:eess.IV OR cat:cs.IR OR cat:cs.AI OR cat:cs.CY OR cat:I.4.9 OR cat:cs.LG OR cat:cs.NI OR cat:cond-mat.stat-mech OR cat:cs.DC)")


if __name__ == "__main__":
    asyncio.run(main())