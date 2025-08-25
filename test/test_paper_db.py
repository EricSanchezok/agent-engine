# -*- coding: utf-8 -*-
"""Basic test for ArxivPaperDB and Paper.
Run with: run.bat test\test_paper_db.py
"""
from core.arxiv.paper_db import Paper, ArxivPaperDB


def main():
    db = ArxivPaperDB()  # 默认路径 core/arxiv/paper_db.sqlite

    # 构造简单 Paper
    info = {
        "id": "2507_18009v2",
        "title": "Example Paper",
        "authors": ["Alice", "Bob"],
        "categories": ["cs.AI"],
        "timestamp": "20250715T1010",
        "summary": "Just a test.",
        "comment": "",
        "journal_ref": "",
        "doi": "",
        "links": [],
        "pdf_url": "https://arxiv.org/pdf/2507.18009.pdf",
    }

    paper = Paper(info)

    # 写入
    db.add(paper, overwrite=True)

    # 查询
    fetched = db.get("2507_18009")
    print("Fetched:", fetched.info)

    # 按日期查询
    same_day = db.get_by_date("20250715")
    print("Same day count:", len(same_day))

    # 删除
    db.delete("2507.18009")
    print("Count after delete:", db.count())


if __name__ == "__main__":
    main()
