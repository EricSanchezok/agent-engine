"""
Swiss Tournament Paper Ranking System

This module provides a Swiss tournament-based ranking system for academic papers
using pairwise LLM comparisons. It can load papers from PDF files and rank them
using the arena_referee prompt for detailed comparison.
"""

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
import random
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.utils import get_relative_path_from_current_file

from agents.ResearchAgent.service.daily_arxiv.config import DailyArxivConfig


@dataclass
class PaperContent:
    """Represents a paper with its content and metadata."""
    paper_id: str
    title: str
    content: str
    pdf_path: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of a pairwise comparison."""
    paper_one_id: str
    paper_two_id: str
    winner: str  # "Paper 1" or "Paper 2"
    justification: str
    comparative_analysis: Dict[str, str]


class SwissTournamentRanker:
    """
    Swiss tournament-based paper ranking system.
    
    This class implements a Swiss tournament algorithm where papers are compared
    pairwise using LLM-based evaluation. The system maintains scores and rankings
    based on comparison results.
    """
    
    def __init__(self, pdf_storage_dir: Optional[str] = None):
        """
        Initialize the Swiss tournament ranker.
        
        Args:
            pdf_storage_dir: Directory to look for PDF files (optional, uses config if None)
        """
        self.logger = AgentLogger(self.__class__.__name__)
        
        # Initialize LLM client
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        
        # Initialize prompt loader
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))
        
        # Set up PDF storage directory
        if pdf_storage_dir is None:
            self.pdf_storage_dir = Path(DailyArxivConfig.get_pdf_storage_dir())
        else:
            self.pdf_storage_dir = Path(pdf_storage_dir)
        
        # Tournament state
        self.papers: List[PaperContent] = []
        self.scores: Dict[str, float] = defaultdict(float)
        self.comparisons: List[ComparisonResult] = []
        self.rankings: List[Tuple[str, float]] = []
        
        self.logger.info(f"SwissTournamentRanker initialized with PDF storage: {self.pdf_storage_dir}")
    
    async def rank_papers_from_pdf_paths(
        self, 
        pdf_paths: List[str], 
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Rank papers from a list of PDF file paths.
        
        Args:
            pdf_paths: List of PDF file paths
            top_n: Number of top papers to return (uses config if None)
            
        Returns:
            Dictionary with ranking results and statistics
        """
        if top_n is None:
            top_n = DailyArxivConfig.SWISS_TOURNAMENT_TOP_N
        
        self.logger.info(f"Starting Swiss tournament ranking for {len(pdf_paths)} papers")
        
        try:
            # Step 1: Load and parse PDFs
            papers = await self._load_papers_from_pdfs(pdf_paths)
            if len(papers) < 2:
                return {
                    "success": False,
                    "reason": f"Need at least 2 papers for ranking, got {len(papers)}",
                    "papers_processed": len(papers),
                    "top_papers": []
                }
            
            self.logger.info(f"Successfully loaded {len(papers)} papers")
            
            # Step 2: Run Swiss tournament
            rankings = await self._run_swiss_tournament(papers)
            
            # Step 3: Select top N papers
            top_papers = rankings[:top_n]
            
            result = {
                "success": True,
                "papers_processed": len(papers),
                "total_comparisons": len(self.comparisons),
                "top_papers": [
                    {
                        "paper_id": paper_id,
                        "title": next(p.title for p in papers if p.paper_id == paper_id),
                        "score": score,
                        "rank": i + 1
                    }
                    for i, (paper_id, score) in enumerate(top_papers)
                ],
                "all_rankings": [
                    {
                        "paper_id": paper_id,
                        "title": next(p.title for p in papers if p.paper_id == paper_id),
                        "score": score,
                        "rank": i + 1
                    }
                    for i, (paper_id, score) in enumerate(rankings)
                ]
            }
            
            self.logger.info(f"Swiss tournament completed: {len(top_papers)} top papers selected")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Swiss tournament ranking: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "papers_processed": 0,
                "top_papers": []
            }
    
    async def rank_papers_from_date(
        self, 
        target_date: date, 
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Rank papers from a specific date by loading PDFs from the storage directory.
        
        Args:
            target_date: Date to load papers for
            top_n: Number of top papers to return (uses config if None)
            
        Returns:
            Dictionary with ranking results and statistics
        """
        self.logger.info(f"Loading papers for date: {target_date}")
        
        # Find PDF files for the target date
        pdf_paths = self._find_pdfs_for_date(target_date)
        
        if not pdf_paths:
            return {
                "success": False,
                "reason": f"No PDF files found for date {target_date}",
                "target_date": target_date.isoformat(),
                "papers_processed": 0,
                "top_papers": []
            }
        
        self.logger.info(f"Found {len(pdf_paths)} PDF files for {target_date}")
        
        # Add target_date to the result
        result = await self.rank_papers_from_pdf_paths(pdf_paths, top_n)
        result["target_date"] = target_date.isoformat()
        
        return result
    
    async def _load_papers_from_pdfs(self, pdf_paths: List[str]) -> List[PaperContent]:
        """Load and parse PDF files to extract paper content."""
        papers = []
        
        for pdf_path in pdf_paths:
            try:
                paper_content = await self._parse_pdf_content(pdf_path)
                if paper_content:
                    papers.append(paper_content)
                    self.logger.info(f"Successfully parsed PDF: {Path(pdf_path).name}")
                else:
                    self.logger.warning(f"Failed to parse PDF: {Path(pdf_path).name}")
            except Exception as e:
                self.logger.error(f"Error parsing PDF {pdf_path}: {e}")
                continue
        
        return papers
    
    async def _parse_pdf_content(self, pdf_path: str) -> Optional[PaperContent]:
        """Parse PDF content using PyPDF2."""
        try:
            from PyPDF2 import PdfReader
            
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                self.logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            # Extract paper ID from filename
            paper_id = pdf_file.stem
            
            # Read PDF content
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                
                # Extract text from all pages
                content_parts = []
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            content_parts.append(text)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_num} of {pdf_path}: {e}")
                        continue
                
                if not content_parts:
                    self.logger.warning(f"No text content found in PDF: {pdf_path}")
                    return None
                
                # Combine all text content
                full_content = "\n\n".join(content_parts)
                
                # Extract title from first few lines (simple heuristic)
                lines = full_content.split('\n')
                title = ""
                for line in lines[:10]:  # Check first 10 lines
                    line = line.strip()
                    if len(line) > 10 and len(line) < 200:  # Reasonable title length
                        title = line
                        break
                
                if not title:
                    title = f"Paper {paper_id}"
                
                return PaperContent(
                    paper_id=paper_id,
                    title=title,
                    content=full_content[:50000],  # Limit content length for LLM processing
                    pdf_path=pdf_path
                )
                
        except ImportError:
            self.logger.error("PyPDF2 not available. Please install it: pip install PyPDF2")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {e}")
            return None
    
    def _find_pdfs_for_date(self, target_date: date) -> List[str]:
        """Find PDF files for a specific date in the storage directory."""
        pdf_paths = []
        
        # Look for PDFs in the date-based directory structure: YYYY/MM/DD/
        year_dir = str(target_date.year)
        month_dir = f"{target_date.month:02d}"
        day_dir = f"{target_date.day:02d}"
        
        date_dir = self.pdf_storage_dir / year_dir / month_dir / day_dir
        
        if date_dir.exists():
            for pdf_file in date_dir.glob("*.pdf"):
                pdf_paths.append(str(pdf_file))
        
        return pdf_paths
    
    async def _run_swiss_tournament(self, papers: List[PaperContent]) -> List[Tuple[str, float]]:
        """Run Swiss tournament algorithm to rank papers."""
        self.logger.info(f"Starting Swiss tournament with {len(papers)} papers")
        
        # Initialize scores
        self.scores = defaultdict(float)
        self.comparisons = []
        
        # Calculate number of rounds needed
        num_rounds = self._calculate_rounds(len(papers))
        self.logger.info(f"Swiss tournament will run {num_rounds} rounds")
        
        # Run tournament rounds
        for round_num in range(num_rounds):
            self.logger.info(f"Starting round {round_num + 1}/{num_rounds}")
            
            # Pair papers for this round
            pairs = self._pair_papers_for_round(papers, round_num)
            
            if not pairs:
                self.logger.warning(f"No pairs found for round {round_num + 1}")
                break
            
            self.logger.info(f"Round {round_num + 1}: {len(pairs)} pairs to compare")
            
            # Compare pairs concurrently
            round_comparisons = await self._compare_pairs_concurrently(pairs)
            
            # Update scores based on comparison results
            for comparison in round_comparisons:
                self.comparisons.append(comparison)
                self._update_scores_from_comparison(comparison)
            
            self.logger.info(f"Round {round_num + 1} completed: {len(round_comparisons)} comparisons")
        
        # Generate final rankings
        rankings = self._generate_rankings(papers)
        
        self.logger.info(f"Swiss tournament completed: {len(rankings)} papers ranked")
        return rankings
    
    def _calculate_rounds(self, num_papers: int) -> int:
        """Calculate number of rounds needed for Swiss tournament."""
        # Swiss tournament typically needs log2(num_papers) rounds
        # But we'll use a more conservative approach for academic papers
        if num_papers <= 4:
            return 4
        elif num_papers <= 8:
            return 5
        elif num_papers <= 16:
            return 6
        else:
            return 7
    
    def _pair_papers_for_round(self, papers: List[PaperContent], round_num: int) -> List[Tuple[PaperContent, PaperContent]]:
        """Pair papers for a tournament round."""
        pairs = []
        
        # For the first round, pair randomly
        if round_num == 0:
            shuffled_papers = papers.copy()
            random.shuffle(shuffled_papers)
            
            for i in range(0, len(shuffled_papers) - 1, 2):
                pairs.append((shuffled_papers[i], shuffled_papers[i + 1]))
        
        else:
            # For subsequent rounds, pair based on current scores
            # Sort papers by score (descending)
            sorted_papers = sorted(papers, key=lambda p: self.scores[p.paper_id], reverse=True)
            
            # Pair papers with similar scores
            used_papers = set()
            for i, paper1 in enumerate(sorted_papers):
                if paper1.paper_id in used_papers:
                    continue
                
                # Find the best available opponent
                best_opponent = None
                best_score_diff = float('inf')
                
                for j, paper2 in enumerate(sorted_papers[i + 1:], i + 1):
                    if paper2.paper_id in used_papers:
                        continue
                    
                    # Check if these papers have already been compared
                    if self._have_been_compared(paper1.paper_id, paper2.paper_id):
                        continue
                    
                    score_diff = abs(self.scores[paper1.paper_id] - self.scores[paper2.paper_id])
                    if score_diff < best_score_diff:
                        best_opponent = paper2
                        best_score_diff = score_diff
                
                if best_opponent:
                    pairs.append((paper1, best_opponent))
                    used_papers.add(paper1.paper_id)
                    used_papers.add(best_opponent.paper_id)
        
        return pairs
    
    def _have_been_compared(self, paper1_id: str, paper2_id: str) -> bool:
        """Check if two papers have already been compared."""
        for comparison in self.comparisons:
            if ((comparison.paper_one_id == paper1_id and comparison.paper_two_id == paper2_id) or
                (comparison.paper_one_id == paper2_id and comparison.paper_two_id == paper1_id)):
                return True
        return False
    
    async def _compare_pairs_concurrently(self, pairs: List[Tuple[PaperContent, PaperContent]]) -> List[ComparisonResult]:
        """Compare pairs concurrently using LLM."""
        semaphore = asyncio.Semaphore(DailyArxivConfig.SWISS_TOURNAMENT_MAX_CONCURRENT)
        
        async def compare_with_semaphore(pair: Tuple[PaperContent, PaperContent]) -> Optional[ComparisonResult]:
            async with semaphore:
                return await self._compare_pair(pair[0], pair[1])
        
        # Run comparisons concurrently
        tasks = [compare_with_semaphore(pair) for pair in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        comparisons = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Comparison failed for pair {i}: {result}")
            elif result:
                comparisons.append(result)
        
        return comparisons
    
    async def _compare_pair(self, paper1: PaperContent, paper2: PaperContent) -> Optional[ComparisonResult]:
        """Compare two papers using LLM."""
        try:
            # Get prompts
            system_prompt = self.prompt_loader.get_prompt(
                section='arena_referee',
                prompt_type='system'
            )
            user_prompt = self.prompt_loader.get_prompt(
                section='arena_referee',
                prompt_type='user',
                paper_one_content=f"Title: {paper1.title}\n\nContent: {paper1.content[:20000]}",
                paper_two_content=f"Title: {paper2.title}\n\nContent: {paper2.content[:20000]}"
            )
            
            # Call LLM
            response = await self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=DailyArxivConfig.SWISS_TOURNAMENT_MODEL,
                max_tokens=DailyArxivConfig.SWISS_TOURNAMENT_MAX_TOKENS,
                temperature=DailyArxivConfig.SWISS_TOURNAMENT_TEMPERATURE
            )
            
            if not response:
                self.logger.error(f"Empty response from LLM for comparison: {paper1.paper_id} vs {paper2.paper_id}")
                return None
            
            # Parse JSON response
            try:
                result_data = json.loads(response.strip())
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response as JSON: {e}")
                self.logger.error(f"Response: {response[:500]}")
                return None
            
            # Create comparison result
            comparison = ComparisonResult(
                paper_one_id=paper1.paper_id,
                paper_two_id=paper2.paper_id,
                winner=result_data.get("recommendation", ""),
                justification=result_data.get("justification", ""),
                comparative_analysis=result_data.get("comparative_analysis", {})
            )
            
            self.logger.info(f"Comparison completed: {paper1.paper_id} vs {paper2.paper_id} -> {comparison.winner}")
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing papers {paper1.paper_id} vs {paper2.paper_id}: {e}")
            return None
    
    def _update_scores_from_comparison(self, comparison: ComparisonResult):
        """Update scores based on comparison result."""
        if comparison.winner == "Paper 1":
            self.scores[comparison.paper_one_id] += 1.0
        elif comparison.winner == "Paper 2":
            self.scores[comparison.paper_two_id] += 1.0
        else:
            # Tie or invalid result - give both papers 0.5 points
            self.scores[comparison.paper_one_id] += 0.5
            self.scores[comparison.paper_two_id] += 0.5
    
    def _generate_rankings(self, papers: List[PaperContent]) -> List[Tuple[str, float]]:
        """Generate final rankings based on scores."""
        # Create list of (paper_id, score) tuples
        paper_scores = [(paper.paper_id, self.scores[paper.paper_id]) for paper in papers]
        
        # Sort by score (descending), then by paper_id for consistency
        rankings = sorted(paper_scores, key=lambda x: (-x[1], x[0]))
        
        return rankings


async def main():
    """Test the Swiss tournament ranker."""
    ranker = SwissTournamentRanker()
    
    # Test with today's date
    today = date(2025, 9, 16)
    result = await ranker.rank_papers_from_date(today)
    
    print("Swiss Tournament Results:")
    print(f"Success: {result['success']}")
    print(f"Target Date: {result.get('target_date', 'N/A')}")
    print(f"Papers Processed: {result['papers_processed']}")
    print(f"Total Comparisons: {result.get('total_comparisons', 0)}")
    
    if result['success']:
        print(f"Top Papers: {len(result['top_papers'])}")
        for paper_info in result['top_papers']:
            print(f"  {paper_info['rank']}. {paper_info['paper_id']}: {paper_info['title'][:50]}... (Score: {paper_info['score']:.1f})")
    else:
        print(f"Error: {result.get('error', result.get('reason', 'Unknown error'))}")


if __name__ == "__main__":
    asyncio.run(main())
