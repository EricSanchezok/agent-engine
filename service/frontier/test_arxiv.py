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
import json

# Core imports
from core.arxiv.arxiv import ArXivFetcher


if __name__ == "__main__":
    query = "submittedDate:[20250802 TO 20250803] AND (cat:cs.CL OR cat:cs.NE OR cat:physics.comp-ph OR cat:q-bio.BM OR cat:eess.AS OR cat:cs.MM OR cat:math.IT OR cat:q-bio.QM OR cat:I.2.10; I.4.8; I.2.6; I.2.7; I.5.4; I.5.1 OR cat:physics.chem-ph OR cat:cs.SD OR cat:cs.CV OR cat:cs.AR OR cat:cond-mat.soft OR cat:cond-mat.mtrl-sci OR cat:cs.RO OR cat:cs.MA OR cat:I.2.1 OR cat:cs.IT OR cat:cs.HC OR cat:eess.IV OR cat:cs.IR OR cat:cs.AI OR cat:cs.CY OR cat:I.4.9 OR cat:cs.LG OR cat:cs.NI OR cat:cond-mat.stat-mech OR cat:cs.DC)"
    fetcher = ArXivFetcher()
    papers = asyncio.run(fetcher.search(query))
    papers = [paper.model_dump() for paper in papers]
    print(len(papers))
    path = "service/frontier/test_papers.json"
    # with open(path, "w") as f:
    #     json.dump(papers[:10], f, ensure_ascii=False, indent=4)