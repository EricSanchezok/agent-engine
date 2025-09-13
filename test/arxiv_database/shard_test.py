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