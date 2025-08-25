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
import base64
import io
from PyPDF2 import PdfReader
from a2a.types import FileWithBytes

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
from agents.PaperAnalysisAgent.config import AGENT_CARD
from agents.PaperFetchAgent.agent import PaperFetchAgent

logger = AgentLogger(__name__)

load_dotenv()

class PaperAnalysisAgent(BaseA2AAgent):
    def __init__(self):
        super().__init__(agent_card=AGENT_CARD)
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        self.skill_identifier = SkillIdentifier(llm_client=self.llm_client, model_name='o3-mini')
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))
        self.semaphore = asyncio.Semaphore(32)
        self.paper_fetch_agent = PaperFetchAgent()
        self.arxiv_id_parser = ArxivIdParser(self.llm_client)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id if context.task_id else str(uuid4())
        context_id = context.context_id if context.context_id else str(uuid4())
        user_input = context.get_user_input()

        skill_id, reason = await self.skill_identifier.invoke(user_input, AGENT_CARD.skills)
        logger.info(f"Skill ID: {skill_id}, Reason: {reason}")

        papers_base64 = []
        if skill_id == 'analyze_by_arxiv_ids':
            event = await self.paper_fetch_agent.run_user_input(user_input)
            if event:
                artifact: Artifact = event.artifacts[0]
                parts = artifact.parts
                for part in parts:
                    pdf_base64 = part.root.file
                    papers_base64.append(pdf_base64)
            else:
                await self._task_failed(context, event_queue, f"No event returned")
                return

        elif skill_id == 'analyze_from_message_files':
            message = context.message
            if message:
                parts = message.parts
                for part in parts:
                    pdf_base64 = part.root.file
                    papers_base64.append(pdf_base64)
            else:
                await self._task_failed(context, event_queue, f"No message provided")
                return
        else:
            await self._task_failed(context, event_queue, f"Unknown skill ID: {skill_id}")
            return

        if not papers_base64:
            await self._task_failed(context, event_queue, f"No papers base64 provided")
            return

        analyze_tasks = [self.analyze_paper(pdf_base64) for pdf_base64 in papers_base64]
        analyze_results = await asyncio.gather(*analyze_tasks, return_exceptions=True)

        success_results = []
        failed_results = []

        for result in analyze_results:
            if isinstance(result, Exception) or not result:
                failed_results.append(result)
            else:
                success_results.append(result)

        if not success_results:
            await self._task_failed(context, event_queue, f"No papers analyzed successfully")
            return
        
        parts = []
        for result in success_results:
            part = Part(root=TextPart(text=result))
            parts.append(part)

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
            name="arxiv_analysis_papers",
            description=f"ArXiv解读的论文，共{len(success_results)}篇",
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
        logger.info(f"Analysis artifacts sent - {len(success_results)} papers successfully processed")
    
    async def draft_report(self, paper_full_text: str) -> str:
        system_prompt = self.prompt_loader.get_prompt(
            section='paper_parser',
            prompt_type='system'
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='paper_parser',
            prompt_type='user',
            paper_full_text=paper_full_text
        )
        try:
            result = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3', max_tokens=100000)
            return result
        except Exception as e:
            logger.error(f"Paper parser failed: {str(e)}")
            return ''

    async def review_paper(self,draft_report: str, paper_full_text: str) -> str:
        system_prompt = self.prompt_loader.get_prompt(
            section='paper_reviewer',
            prompt_type='system'
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='paper_reviewer',
            prompt_type='user',
            draft_report=draft_report,
            paper_full_text=paper_full_text
        )

        try:
            result = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3', max_tokens=100000)
            return result
        except Exception as e:
            logger.error(f"Paper review failed: {str(e)}")
            return ''

    async def analyze_paper(self, pdf_b64_or_obj) -> str:
        async with self.semaphore:
            paper_text = self._pdf_base64_to_text(pdf_b64_or_obj)
            if not paper_text:
                logger.error("Empty text extracted from PDF. Skipping analysis.")
                return ''

            draft_report = await self.draft_report(paper_text)
            if not draft_report:
                logger.error("Empty draft report generated. Skipping analysis.")
                return ''

            review_report = await self.review_paper(draft_report, paper_text)
            if not review_report:
                logger.error("Empty review report generated. Skipping analysis.")
                return ''

            return review_report

    def _pdf_base64_to_text(self, pdf_base64: str | FileWithBytes) -> str:
        """Decode a PDF provided as base64 (string) or FileWithBytes and return extracted plain text."""
        # Determine actual base64 string
        if isinstance(pdf_base64, FileWithBytes):
            b64_str = pdf_base64.bytes
        else:
            b64_str = pdf_base64

        try:
            pdf_bytes = base64.b64decode(b64_str)
        except Exception as e:
            logger.error(f"Base64 decode failed: {e}")
            return ""

        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return ""

async def main():
    ids = [
        '2508_15697',
        '2508.15692',
        '2508.15678'
    ]
    user_input = json.dumps({
        "arxiv_ids": ids
    }, indent=4, ensure_ascii=False)
    agent = PaperAnalysisAgent()
    await agent.run_user_input(user_input)


if __name__ == "__main__":
    asyncio.run(main())