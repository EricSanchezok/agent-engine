"""
Test script for new EMemory and PodEMemory methods.
"""

import asyncio
from datetime import datetime

from agents.ResearchAgent.arxiv_database import ArxivDatabase
from core.arxiv_fetcher import ArxivFetcher, ArxivPaper


async def test_new_methods():
    """Test the new exists and has_vector methods."""
    print("=== Testing New EMemory/PodEMemory Methods ===\n")
    
    # Initialize database
    db = ArxivDatabase(name="test_arxiv_db", max_elements_per_shard=1000)
    print(f"ArxivDatabase initialized")
    
    # Initialize fetcher
    fetcher = ArxivFetcher()
    print(f"ArxivFetcher initialized")
    
    # Search for some papers
    print("\n--- Step 1: Search for papers ---")
    papers = await fetcher.search_papers(query="cat:cs.AI", max_results=5)
    print(f"Found {len(papers)} papers")
    
    if not papers:
        print("No papers found, skipping test")
        return
    
    # Show first few papers
    for i, paper in enumerate(papers[:3]):
        print(f"{i+1}. {paper.full_id}: {paper.title[:60]}...")
    
    # Test exists methods before adding
    print("\n--- Step 2: Test exists methods before adding ---")
    paper_ids = [paper.full_id for paper in papers[:3]]
    
    # Test single exists
    for paper_id in paper_ids:
        exists = db.exists(paper_id)
        print(f"Paper {paper_id} exists: {exists}")
    
    # Test batch exists
    exists_results = db.exists_batch(paper_ids)
    print(f"Batch exists results: {exists_results}")
    
    # Test has_vector methods before adding
    print("\n--- Step 3: Test has_vector methods before adding ---")
    for paper_id in paper_ids:
        has_vec = db.has_vector(paper_id)
        print(f"Paper {paper_id} has vector: {has_vec}")
    
    has_vector_results = db.has_vector_batch(paper_ids)
    print(f"Batch has_vector results: {has_vector_results}")
    
    # Add papers without embeddings
    print("\n--- Step 4: Add papers without embeddings ---")
    papers_to_add = papers[:3]
    record_ids = db.add_papers(papers_to_add)
    print(f"Added {len(record_ids)} papers")
    
    # Test exists methods after adding
    print("\n--- Step 5: Test exists methods after adding ---")
    for paper_id in paper_ids:
        exists = db.exists(paper_id)
        print(f"Paper {paper_id} exists: {exists}")
    
    exists_results = db.exists_batch(paper_ids)
    print(f"Batch exists results: {exists_results}")
    
    # Test has_vector methods after adding (should be False since no embeddings)
    print("\n--- Step 6: Test has_vector methods after adding ---")
    for paper_id in paper_ids:
        has_vec = db.has_vector(paper_id)
        print(f"Paper {paper_id} has vector: {has_vec}")
    
    has_vector_results = db.has_vector_batch(paper_ids)
    print(f"Batch has_vector results: {has_vector_results}")
    
    # Add papers with fake embeddings
    print("\n--- Step 7: Add papers with fake embeddings ---")
    fake_embeddings = [[0.1] * 100 for _ in papers_to_add]  # 100-dim fake embeddings
    
    # Update papers with embeddings
    for i, paper in enumerate(papers_to_add):
        success = db.update_paper(paper, fake_embeddings[i])
        print(f"Updated paper {paper.full_id} with embedding: {success}")
    
    # Test has_vector methods after adding embeddings
    print("\n--- Step 8: Test has_vector methods after adding embeddings ---")
    for paper_id in paper_ids:
        has_vec = db.has_vector(paper_id)
        print(f"Paper {paper_id} has vector: {has_vec}")
    
    has_vector_results = db.has_vector_batch(paper_ids)
    print(f"Batch has_vector results: {has_vector_results}")
    
    # Test database stats
    print("\n--- Step 9: Database statistics ---")
    stats = db.get_stats()
    print(f"Total papers: {stats['total_records']}")
    print(f"Shard count: {stats['shard_count']}")
    
    # Clean up
    print("\n--- Step 10: Clean up ---")
    db.clear()
    print("Database cleared")
    
    print("\n=== Test completed successfully ===")


if __name__ == "__main__":
    asyncio.run(test_new_methods())
