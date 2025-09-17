"""
Paper Condenser for Daily arXiv Service

This module provides functionality to condense academic papers into structured
markdown reports using LLM-based analysis. It processes the top papers selected
by the Swiss tournament ranking system.
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
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.utils import get_relative_path_from_current_file

from agents.ResearchAgent.service.daily_arxiv.config import DailyArxivConfig


@dataclass
class CondensedReport:
    """Represents a condensed paper report."""
    paper_id: str
    title: str
    markdown_content: str
    pdf_path: str
    generated_at: datetime
    file_path: Optional[str] = None


class PaperCondenser:
    """
    Paper condenser that converts academic papers into structured markdown reports.
    
    This class takes the top papers from Swiss tournament ranking and generates
    detailed markdown reports using the paper_condenser prompt.
    """
    
    def __init__(self, pdf_storage_dir: Optional[str] = None):
        """
        Initialize the paper condenser.
        
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
        
        self.logger.info(f"PaperCondenser initialized with PDF storage: {self.pdf_storage_dir}")
    
    async def condense_papers(
        self, 
        top_papers: List[Dict[str, Any]], 
        pdf_paths: List[str],
        max_concurrent: Optional[int] = None
    ) -> List[CondensedReport]:
        """
        Condense multiple papers into markdown reports.
        
        Args:
            top_papers: List of top papers from Swiss tournament (with paper_id, title, etc.)
            pdf_paths: List of PDF file paths corresponding to the papers
            max_concurrent: Maximum concurrent LLM calls (uses config if None)
            
        Returns:
            List of CondensedReport objects
        """
        if max_concurrent is None:
            max_concurrent = DailyArxivConfig.PAPER_CONDENSER_MAX_CONCURRENT
        
        self.logger.info(f"Starting paper condensation for {len(top_papers)} papers")
        
        # Create mapping from paper_id to PDF path
        pdf_mapping = self._create_pdf_mapping(top_papers, pdf_paths)
        
        # Create semaphore for concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def condense_with_semaphore(paper_info: Dict[str, Any]) -> Optional[CondensedReport]:
            async with semaphore:
                return await self._condense_single_paper(paper_info, pdf_mapping)
        
        # Process papers concurrently
        tasks = [condense_with_semaphore(paper_info) for paper_info in top_papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        condensed_reports = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Condensation failed for paper {top_papers[i]['paper_id']}: {result}")
            elif result:
                condensed_reports.append(result)
        
        self.logger.info(f"Paper condensation completed: {len(condensed_reports)} reports generated")
        return condensed_reports
    
    def _create_pdf_mapping(self, top_papers: List[Dict[str, Any]], pdf_paths: List[str]) -> Dict[str, str]:
        """Create mapping from paper_id to PDF path."""
        pdf_mapping = {}
        
        for pdf_path in pdf_paths:
            pdf_file = Path(pdf_path)
            paper_id = pdf_file.stem  # Extract paper_id from filename
            
            # Check if this paper is in our top papers list
            for paper_info in top_papers:
                if paper_info['paper_id'] == paper_id:
                    pdf_mapping[paper_id] = pdf_path
                    break
        
        self.logger.info(f"Created PDF mapping for {len(pdf_mapping)} papers")
        return pdf_mapping
    
    async def _condense_single_paper(
        self, 
        paper_info: Dict[str, Any], 
        pdf_mapping: Dict[str, str]
    ) -> Optional[CondensedReport]:
        """Condense a single paper into a markdown report."""
        paper_id = paper_info['paper_id']
        title = paper_info['title']
        
        # Get PDF path
        pdf_path = pdf_mapping.get(paper_id)
        if not pdf_path:
            self.logger.warning(f"No PDF path found for paper {paper_id}")
            return None
        
        try:
            # Load PDF content
            paper_content = await self._load_pdf_content(pdf_path)
            if not paper_content:
                self.logger.error(f"Failed to load PDF content for {paper_id}")
                return None
            
            # Generate markdown report using LLM
            markdown_content = await self._generate_markdown_report(paper_content)
            if not markdown_content:
                self.logger.error(f"Failed to generate markdown report for {paper_id}")
                return None
            
            # Create condensed report
            report = CondensedReport(
                paper_id=paper_id,
                title=title,
                markdown_content=markdown_content,
                pdf_path=pdf_path,
                generated_at=datetime.now()
            )
            
            self.logger.info(f"Successfully condensed paper: {paper_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error condensing paper {paper_id}: {e}")
            return None
    
    async def _load_pdf_content(self, pdf_path: str) -> Optional[str]:
        """Load PDF content using PyPDF2."""
        try:
            from PyPDF2 import PdfReader
            
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                self.logger.error(f"PDF file not found: {pdf_path}")
                return None
            
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
                return full_content
                
        except ImportError:
            self.logger.error("PyPDF2 not available. Please install it: pip install PyPDF2")
            return None
        except Exception as e:
            self.logger.error(f"Error loading PDF content from {pdf_path}: {e}")
            return None
    
    async def _generate_markdown_report(self, paper_content: str) -> Optional[str]:
        """Generate markdown report using LLM."""
        try:
            # Get prompts
            system_prompt = self.prompt_loader.get_prompt(
                section='paper_condenser',
                prompt_type='system'
            )
            user_prompt = self.prompt_loader.get_prompt(
                section='paper_condenser',
                prompt_type='user',
                paper_full_content=paper_content[:100000]  # Limit content length for LLM processing
            )
            
            # Call LLM
            response = await self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=DailyArxivConfig.PAPER_CONDENSER_MODEL,
                max_tokens=DailyArxivConfig.PAPER_CONDENSER_MAX_TOKENS,
                temperature=DailyArxivConfig.PAPER_CONDENSER_TEMPERATURE
            )
            
            if not response:
                self.logger.error("Empty response from LLM for paper condensation")
                return None
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating markdown report: {e}")
            return None
    
    def save_reports_to_files(
        self, 
        reports: List[CondensedReport], 
        output_dir: str
    ) -> List[Dict[str, Any]]:
        """
        Save condensed reports to markdown files.
        
        Args:
            reports: List of CondensedReport objects
            output_dir: Directory to save markdown files
            
        Returns:
            List of file information dictionaries
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for report in reports:
            try:
                # Create filename based on paper_id
                filename = f"{report.paper_id}.md"
                file_path = output_path / filename
                
                # Write markdown content to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report.markdown_content)
                
                # Update report with file path
                report.file_path = str(file_path)
                
                saved_files.append({
                    "paper_id": report.paper_id,
                    "title": report.title,
                    "markdown_file": filename,
                    "file_path": str(file_path),
                    "generated_at": report.generated_at.isoformat()
                })
                
                self.logger.info(f"Saved condensed report: {filename}")
                
            except Exception as e:
                self.logger.error(f"Error saving report for {report.paper_id}: {e}")
                continue
        
        self.logger.info(f"Saved {len(saved_files)} condensed reports to {output_dir}")
        return saved_files

