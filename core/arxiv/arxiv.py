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
from .paper import Paper

logger = AgentLogger(__name__)

class ArXivFetcher:
    async def invoke(self, query_string: str = "", id_list: List[str] = [], max_results: int = 10000) -> List[Paper]:
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
        pdf_filename = paper.get_pdf_filename()
        # Only try to load if the paper has been downloaded before
        if paper.load():
            logger.info(f"Paper {paper.id} already downloaded, skip downloading")
            return paper

        max_retries = 3
        base_retry_delay = 5  # seconds
        
        target_dir = os.path.dirname(pdf_filename)
        os.makedirs(target_dir, exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".pdf", dir=target_dir
        )
        temp_path = temp_file.name
        
        try:
            for attempt in range(max_retries):
                try:
                    temp_file.seek(0)
                    temp_file.truncate()

                    async with session.head(paper.pdf_url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        resp.raise_for_status()
                        total_size = int(resp.headers.get('content-length', 0))

                    async with session.get(paper.pdf_url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                        resp.raise_for_status()

                        total_bytes = 0
                        async for chunk in resp.content.iter_chunked(8192):
                            if chunk:
                                temp_file.write(chunk)
                                total_bytes += len(chunk)

                        if total_size > 0 and total_bytes < total_size:
                            raise aiohttp.ClientError(f"Download incomplete. Expected {total_size} bytes, got {total_bytes}")

                        paper_size_mb = float(total_bytes) / (1024 * 1024)
                        temp_file.flush()
                        os.fsync(temp_file.fileno())
                        temp_file.close()
                        shutil.move(temp_path, pdf_filename)
                        # logger.info(f"✅ Successfully downloaded paper {paper.id} on attempt {attempt + 1}/{max_retries}. Size: {paper_size_mb:.2f} MB")
                        paper.pdf_filename = str(pdf_filename)
                        paper.downloaded = True
                        paper.save()
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
            if not temp_file.closed:
                temp_file.close()
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info(f"Cleaned up temporary file {temp_path}")

        return paper
        
    # async def download(self, paper: Paper) -> Paper:
    #     pdf_filename = paper.get_pdf_filename()
    #     # Only try to load if the paper has been downloaded before
    #     if paper.load():
    #         logger.info(f"Paper {paper.id} already downloaded, skip downloading")
    #         return paper

    #     max_retries = 3
    #     base_retry_delay = 5  # seconds
        
    #     target_dir = os.path.dirname(pdf_filename)
    #     os.makedirs(target_dir, exist_ok=True)

    #     temp_file = tempfile.NamedTemporaryFile(
    #         mode="wb", delete=False, suffix=".pdf", dir=target_dir
    #     )
    #     temp_path = temp_file.name
        
    #     try:
    #         for attempt in range(max_retries):
    #             try:
    #                 temp_file.seek(0)
    #                 temp_file.truncate()

    #                 ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    #                 connector = aiohttp.TCPConnector(
    #                     ssl=ssl_ctx,
    #                     limit=32,
    #                     limit_per_host=16,
    #                     family=socket.AF_INET,
    #                     ttl_dns_cache=300,
    #                     enable_cleanup_closed=True
    #                 )
                    
    #                 async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
    #                     async with session.head(paper.pdf_url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
    #                         resp.raise_for_status()
    #                         total_size = int(resp.headers.get('content-length', 0))

    #                     timeout_settings = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)
    #                     async with session.get(paper.pdf_url, timeout=timeout_settings) as resp:
    #                         resp.raise_for_status()

    #                         total_bytes = 0
    #                         async for chunk in resp.content.iter_chunked(8192):
    #                             if chunk:
    #                                 temp_file.write(chunk)
    #                                 total_bytes += len(chunk)

    #                         if total_size > 0 and total_bytes < total_size:
    #                             raise aiohttp.ClientError(f"Download incomplete. Expected {total_size} bytes, got {total_bytes}")

    #                         paper_size_mb = float(total_bytes) / (1024 * 1024)
                            
    #                         temp_file.flush()
    #                         os.fsync(temp_file.fileno())
    #                         temp_file.close()

    #                         shutil.move(temp_path, pdf_filename)
                            
    #                         logger.info(f"✅ Successfully downloaded paper {paper.id} on attempt {attempt + 1}/{max_retries}. Size: {paper_size_mb:.2f} MB")
                            
    #                         paper.pdf_filename = str(pdf_filename)
    #                         paper.downloaded = True
    #                         paper.save()
                            
    #                         return paper

    #             except (aiohttp.ClientError, asyncio.TimeoutError) as e:
    #                 logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for paper {paper.id}: {e}")
    #                 if attempt + 1 == max_retries:
    #                     logger.error(f"❌ All {max_retries} attempts failed for paper {paper.id}. Giving up.")
    #                     break
                    
    #                 wait_time = base_retry_delay * (2 ** attempt) + random.uniform(0, 1)
    #                 logger.info(f"Waiting for {wait_time:.2f} seconds before retrying...")
    #                 await asyncio.sleep(wait_time)

    #     finally:
    #         if not temp_file.closed:
    #             temp_file.close()
    #         if os.path.exists(temp_path):
    #             os.unlink(temp_path)
    #             logger.info(f"Cleaned up temporary file {temp_path}")

    #     return paper
    

    # async def download(self, paper: Paper) -> Paper:
    #     pdf_filename = paper.get_pdf_filename()
    #     # Only try to load if the paper has been downloaded before
    #     if paper.load():
    #         logger.info(f"Paper {paper.id} already downloaded, skip downloading")
    #         return paper

    #     temp_file = None
    #     try:
    #         pdf_url = paper.pdf_url
    #         os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)

    #         temp_file = tempfile.NamedTemporaryFile(
    #             mode="wb",
    #             delete=False,
    #             suffix=".pdf",
    #             dir=os.path.dirname(pdf_filename)
    #         )
    #         temp_path = temp_file.name

    #         ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    #         connector = aiohttp.TCPConnector(
    #             ssl=ssl_ctx,
    #             limit=32,
    #             limit_per_host=16,
    #             family=socket.AF_INET,
    #             ttl_dns_cache=300,
    #             enable_cleanup_closed=True
    #         )
            
    #         async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
    #             async with session.head(pdf_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
    #                 resp.raise_for_status()
    #                 total_size = int(resp.headers.get('content-length', 0))

    #             async with session.get(pdf_url, timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=30)) as resp:
    #                 resp.raise_for_status()

    #                 total_bytes = 0
    #                 download_success = False
    #                 try:
    #                     async for chunk in resp.content.iter_chunked(8192):
    #                         if chunk:
    #                             temp_file.write(chunk)
    #                             total_bytes += len(chunk)

    #                     temp_file.flush()
    #                     os.fsync(temp_file.fileno())
    #                     download_success = True

    #                     temp_file.close()
    #                     if download_success and total_bytes > 0:
    #                         paper_size = float(total_bytes) / (1024 * 1024)
    #                         paper_size_string = f"{paper_size:.2f} MB"

    #                         if total_size > 0 and total_bytes < total_size:
    #                             logger.warning(f"Download incomplete for {paper.id}: size: {paper_size_string}")
    #                             download_success = False
                            
    #                         if download_success:
    #                             shutil.move(temp_path, pdf_filename)
    #                             logger.info(f"Downloaded paper {paper.id} to {pdf_filename}, size: {paper_size_string}")
    #                             paper.pdf_filename = str(pdf_filename)
    #                             paper.downloaded = True
    #                             paper.save()
    #                             return paper
    #                         else:
    #                             if os.path.exists(temp_path):
    #                                 os.unlink(temp_path)
    #                                 logger.info(f"Cleaned up incomplete download for {paper.id}")
    #                             return paper
    #                     else:
    #                         if os.path.exists(temp_path):
    #                             os.unlink(temp_path)
    #                             logger.info(f"Cleaned up failed download for {paper.id}")
    #                         return paper
    #                 except Exception as e:
    #                     logger.error(f"Download paper {paper.id} failed: {e}", exc_info=True)
    #                     return paper

    #     except Exception as e:
    #         if temp_file is not None:
    #             temp_file.close()
    #             temp_path = temp_file.name
    #             if os.path.exists(temp_path):
    #                 os.unlink(temp_path)
    #                 logger.info(f"Cleaned up temp file after exception for {paper.id}")
            
    #         logger.error(f"Download paper {paper.id} failed: {e}", exc_info=True)
    #         return paper

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