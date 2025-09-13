

import asyncio
from core.arxiv_fetcher import ArxivFetcher


async def test():
    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()
    arxiv_fetcher = ArxivFetcher()
    papers = await arxiv_fetcher.search_papers(query="submittedDate:[20240911 TO 20240912]", max_results=10000)
    print(len(papers))
    profiler.stop()
    profiler.print()


if __name__ == "__main__":
    asyncio.run(test())