import os
import re
import docx
from pathlib import Path
import json
from typing import List, Tuple
import asyncio
from dotenv import load_dotenv
from uuid import uuid4
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy
import numpy as np
import datetime
import pytz
import random
import ssl, certifi
import socket
import aiohttp
from tqdm.asyncio import tqdm

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
from agent_engine.memory import Memory
from agent_engine.utils import get_relative_path_from_current_file, get_current_file_dir


# Core imports
from core.arxiv import ArXivFetcher, Paper, CATEGORIES_QUERY_STRING, ArxivIdParser
from core.utils import DateFormatter

# Local imports
from agents.PaperFetchAgent.config import AGENT_CARD, LOG_DIR

logger = AgentLogger(__name__)

load_dotenv()

class PaperFetchAgent(BaseA2AAgent):
    def __init__(self):
        super().__init__(agent_card=AGENT_CARD)
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        self.skill_identifier = SkillIdentifier(llm_client=self.llm_client, model_name='o3-mini')
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))
        self.arxiv_fetcher = ArXivFetcher()
        self.date_formatter = DateFormatter()
        self.semaphore = asyncio.Semaphore(32)
        self.arxiv_id_parser = ArxivIdParser(self.llm_client)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id if context.task_id else str(uuid4())
        context_id = context.context_id if context.context_id else str(uuid4())
        user_input = context.get_user_input()

        skill_id, reason = await self.skill_identifier.invoke(user_input, AGENT_CARD.skills)
        logger.info(f"Skill ID: {skill_id}, Reason: {reason}")

        async def download_with_semaphore(paper, session):
            async with self.semaphore:
                await asyncio.sleep(random.uniform(0.1, 0.5))
                _paper = await self.arxiv_fetcher.download(paper, session)
                return _paper

        if skill_id == 'fetch_papers_as_pdf':
            # Try to parse arxiv_ids from user input
            arxiv_ids = await self.arxiv_id_parser.extract_arxiv_ids(user_input)
            
            if not arxiv_ids:
                await self._task_failed(context, event_queue, "No arxiv ids found in the input")
                return

            try:
                papers: List[Paper] = await self.arxiv_fetcher.search(id_list=arxiv_ids)
            except Exception as e:
                await self._task_failed(context, event_queue, f"Can't fetch the papers: {e}")
                return

            if not papers:
                await self._task_failed(context, event_queue, f"No papers found")
                return

            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(
                ssl=ssl_ctx,
                limit=32,
                limit_per_host=16, # This is key for connection pooling to arXiv
                family=socket.AF_INET,
                ttl_dns_cache=300,
                enable_cleanup_closed=True
            )

            async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
                download_tasks = [
                    download_with_semaphore(paper, session)
                    for paper in papers
                ]
                
                downloaded_papers = []
                
                logger.info(f"Starting parallel download of {len(papers)} papers using a shared connection pool")
                
                for coro in tqdm.as_completed(download_tasks, total=len(papers), desc="Downloading Papers"):
                    paper = await coro
                    downloaded_papers.append(paper)

            success_papers = []
            failed_papers = []

            for paper in downloaded_papers:
                if isinstance(paper, Exception):
                    failed_papers.append(paper)
                elif isinstance(paper, Paper) and paper.pdf_bytes:
                    success_papers.append(paper)
                else:
                    failed_papers.append(paper)
            
            logger.info(f"Download statistics - Successful: {len(success_papers)}, Failed: {len(failed_papers)}, Total: {len(papers)}")
            
            if not success_papers:
                logger.error("No papers downloaded successfully")
                error_info = f"Download operation failed: None of the {len(papers)} papers were downloaded successfully. This could be due to network connectivity issues, file access permissions, or ArXiv service problems."
                await self._task_failed(context, event_queue, error_info)
                return

            parts = []
            for paper in success_papers:
                file_part = Part(root=FilePart(
                    file=FileWithBytes(
                        name=f"{paper.id}.pdf",
                        mime_type="application/pdf",
                        bytes=paper.pdf_bytes
                    )
                ))
                parts.append(file_part)

            message = Message(
                role=Role.agent,
                task_id=task_id,
                message_id=str(uuid4()),
                content_id=context_id,
                parts=parts
            )
            artifact = Artifact(
                artifact_id=str(uuid4()),
                parts=parts,
                name="arxiv_download_papers",
                description=f"ArXiv下载的论文，共{len(success_papers)}篇",
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
            logger.info(f"Download artifacts sent - {len(success_papers)} papers successfully processed")

        else:
            await self._task_failed(context, event_queue, f"Can't find the skill: {skill_id}")
            return

async def main():
    agent = PaperFetchAgent()
    user_input = '帮我下载2508.15697,2508.15692,2508_15678'
    await agent.run_user_input(user_input)


if __name__ == "__main__":
    asyncio.run(main())