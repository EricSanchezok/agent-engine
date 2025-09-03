import asyncio

# Core imports
from core.arxiv.arxiv import ArXivFetcher


if __name__ == "__main__":
    query = "submittedDate:[20250802 TO 20250803] AND (cat:cs.CL OR cat:cs.NE OR cat:physics.comp-ph OR cat:q-bio.BM OR cat:eess.AS OR cat:cs.MM OR cat:math.IT OR cat:q-bio.QM OR cat:I.2.10; I.4.8; I.2.6; I.2.7; I.5.4; I.5.1 OR cat:physics.chem-ph OR cat:cs.SD OR cat:cs.CV OR cat:cs.AR OR cat:cond-mat.soft OR cat:cond-mat.mtrl-sci OR cat:cs.RO OR cat:cs.MA OR cat:I.2.1 OR cat:cs.IT OR cat:cs.HC OR cat:eess.IV OR cat:cs.IR OR cat:cs.AI OR cat:cs.CY OR cat:I.4.9 OR cat:cs.LG OR cat:cs.NI OR cat:cond-mat.stat-mech OR cat:cs.DC)"
    fetcher = ArXivFetcher()
    papers = asyncio.run(fetcher.search(query))
    print(len(papers))