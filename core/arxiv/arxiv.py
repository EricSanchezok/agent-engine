import math
from typing import List, Tuple
import re
import arxiv
import os
import requests
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import aiohttp
import asyncio
import ssl, certifi
import socket
import datetime
import random

# AgentEngine imports
from agent_engine.agent_logger import AgentLogger

# Internal imports
from .paper_db import Paper, ArxivPaperDB

logger = AgentLogger(__name__)

class ArXivFetcher:
    def __init__(self):
        self.arxiv_paper_db = ArxivPaperDB('database/arxiv_paper_db.sqlite')

    async def search(self, query_string: str = "", id_list: List[str] = [], max_results: int = 10000) -> List[Paper]:
        if query_string == "":
            logger.info(f"Fetching papers with id_list: {id_list}")
        else:
            logger.info(f"Fetching papers with query: {query_string}")

        papers = []
        arxiv_client = arxiv.Client(page_size=500, delay_seconds=3)

        if id_list:
            _id_list = [paper_id.replace('_', '.') for paper_id in id_list]
            id_list = _id_list

        try:
            search = arxiv.Search(
                query=query_string,
                id_list=id_list,
                sort_by=arxiv.SortCriterion.Relevance
            )

            for result in arxiv_client.results(search):
                _paper = Paper.from_arxiv_result(result)
                papers.append(_paper)
                if len(papers) >= max_results:
                    break
                
            logger.info(f"Found {len(papers)} papers")
            return papers
            
        except arxiv.UnexpectedEmptyPageError as e:
            logger.warning(f"arXiv returned empty page, likely reached end of results: {e}")
            logger.info(f"Successfully retrieved {len(papers)} papers before empty page")
            return papers
        except Exception as e:
            logger.error(f"arXiv fetch error: {e}", exc_info=True)
            return []
        
    async def download(self, paper: Paper, session: aiohttp.ClientSession) -> Paper:
        # 若数据库中已经存在并且 pdf 已下载，则直接返回
        cached = self.arxiv_paper_db.get(paper.id)
        if cached and cached.pdf_bytes:
            logger.info(f"Paper {paper.id} already downloaded in DB, skip downloading")
            return cached

        max_retries = 3
        base_retry_delay = 5  # seconds

        pdf_buffer: bytearray = bytearray()
        
        try:
            for attempt in range(max_retries):
                try:
                    async with session.head(paper.info['pdf_url'], timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        resp.raise_for_status()
                        total_size = int(resp.headers.get('content-length', 0))

                    async with session.get(paper.info['pdf_url'], timeout=aiohttp.ClientTimeout(total=300)) as resp:
                        resp.raise_for_status()
                        total_bytes = 0
                        async for chunk in resp.content.iter_chunked(8192):
                            if chunk:
                                pdf_buffer.extend(chunk)
                                total_bytes += len(chunk)

                        if total_size > 0 and total_bytes < total_size:
                            raise aiohttp.ClientError(
                                f"Download incomplete. Expected {total_size} bytes, got {total_bytes}")

                        paper_size_mb = float(total_bytes) / (1024 * 1024)

                        import base64
                        paper.pdf_bytes = base64.b64encode(bytes(pdf_buffer)).decode("ascii")
                        # 保存至数据库
                        self.arxiv_paper_db.add(paper, overwrite=True)
                        logger.info(
                            f"✅ Successfully downloaded and stored paper {paper.id}. Size: {paper_size_mb:.2f} MB"
                        )
                        return paper

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for paper {paper.id}: {e}")
                    if attempt + 1 == max_retries:
                        logger.error(f"❌ All {max_retries} attempts failed for paper {paper.id}. Giving up.")
                        break
                    
                    wait_time = base_retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Waiting for {wait_time:.2f} seconds before retrying...")
                    await asyncio.sleep(wait_time)

        finally:
            # 清空缓冲区以释放内存
            pdf_buffer.clear()

        return paper

    async def get_random_papers(self, query_string: str = "", max_results: int = 10000) -> List[Paper]:
        start_date = datetime.datetime.strptime("20230810", "%Y%m%d").date()
        end_date = datetime.datetime.strptime("20250810", "%Y%m%d").date()

        total_days = (end_date - start_date).days

        while True:
            random_day_offset = random.randint(0, total_days)
            random_date = start_date + datetime.timedelta(days=random_day_offset)
            if random_date.weekday() < 5:
                break
        next_day = random_date + datetime.timedelta(days=1)

        first_day_str = random_date.strftime("%Y%m%d")
        next_day_str = next_day.strftime("%Y%m%d")
        
        _query_string = f"(ti:artificial intelligence OR cat:cs.AI) AND submittedDate:[{first_day_str} TO {next_day_str}]" if query_string == "" else query_string
        logger.info(f"query_string: {_query_string}")

        papers = await self.invoke(_query_string, max_results=max_results)
        return papers