from core.arxiv import ArXivFetcher
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

async def main():
    paper_id = "2508.19750"
    arxiv_fetcher = ArXivFetcher()
    papers = await arxiv_fetcher.search(id_list=[paper_id])
    paper = papers[0]
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
        paper = await arxiv_fetcher.download(paper, session)
        
        
if __name__ == "__main__":
    asyncio.run(main())
    