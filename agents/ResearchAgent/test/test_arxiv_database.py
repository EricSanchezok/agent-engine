
import asyncio
from agents.ResearchAgent.arxiv_database import ArxivDatabase
from core.arxiv_fetcher import ArxivFetcher, ArxivPaper

async def test():
    db = ArxivDatabase()
    fetcher = ArxivFetcher()
    papers = await fetcher.get_random_papers(query="submittedDate:[20240911 TO 20240912]")
    print(len(papers))

if __name__ == "__main__":
    asyncio.run(test())