import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))


import asyncio
from core.arxiv_fetcher import ArxivFetcher
from agents.ResearchAgent.arxiv_database import ArxivDatabase

async def test():
    arxiv_fetcher = ArxivFetcher()
    arxiv_database = ArxivDatabase(name="test_arxiv_database", persist_dir=".memory")
    print(arxiv_database.count())
    papers = await arxiv_fetcher.search_papers(query="submittedDate:[20240911 TO 20240912]", max_results=11)
    
    arxiv_database.add_papers(papers)


if __name__ == "__main__":
    asyncio.run(test())