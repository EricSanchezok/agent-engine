from core.arxiv.paper_db import ArxivPaperDB

if __name__ == "__main__":
    db = ArxivPaperDB("database/arxiv_paper_db.sqlite")
    print(db.count())
    print(db.all_ids())
