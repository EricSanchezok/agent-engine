"""
Check if papers in database have vectors
"""

from agents.ResearchAgent.arxiv_database import ArxivDatabase
from datetime import datetime, date

def check_vectors():
    db = ArxivDatabase()
    papers = db.get_papers_by_date(datetime.combine(date(2025, 9, 11), datetime.min.time()), limit=5)
    print(f'Found {len(papers)} papers')

    if papers:
        paper = papers[0]
        print(f'First paper: {paper.full_id}')
        print(f'Title: {paper.title[:50]}...')
        
        # Check if paper has vector
        has_vec = db.has_vector(paper)
        print(f'Has vector: {has_vec}')
        
        # Get vector
        vector = db.get_vector(paper)
        print(f'Vector: {vector is not None}')
        if vector:
            print(f'Vector length: {len(vector)}')
        
        # Check a few more papers
        print("\nChecking more papers:")
        for i, paper in enumerate(papers[:3]):
            has_vec = db.has_vector(paper)
            print(f"Paper {i+1}: {paper.full_id} - Has vector: {has_vec}")

if __name__ == "__main__":
    check_vectors()
