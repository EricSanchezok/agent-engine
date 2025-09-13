"""
ArxivFetcher - A focused arXiv data fetching utility.

This module provides clean interfaces for searching and downloading arXiv papers
without coupling to storage or database concerns.
"""

from __future__ import annotations

import asyncio
import base64
import arxiv
import io
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import pyinstrument
import aiohttp
import arxiv

from agent_engine.agent_logger import AgentLogger

from .arxiv_paper import ArxivPaper


def validate_pdf_integrity(pdf_data: bytes) -> bool:
    """
    Validate PDF integrity.
    
    Args:
        pdf_data: PDF data as bytes
        
    Returns:
        True if PDF is valid, False otherwise
    """
    try:
        # Check minimum size
        if len(pdf_data) < 8:
            return False
        
        # Try to read with PyPDF2 if available
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(pdf_data), strict=False)
            num_pages = len(reader.pages)
            if num_pages == 0:
                return False
            
            return True
            
        except ImportError:
            # PyPDF2 not available, use basic validation
            return True
            
        except Exception:
            return False
            
    except Exception:
        return False


def get_pdf_storage_path(paper: ArxivPaper, pdf_storage_dir: str) -> Path:
    """
    Generate the storage path for a PDF file based on paper and storage directory.
    
    Directory structure: YYYY/MM/DD/full_id.pdf
    
    Args:
        paper: ArxivPaper object
        pdf_storage_dir: Base directory for PDF storage
        
    Returns:
        Path to where the PDF should be stored
    """
    pdf_storage_path = Path(pdf_storage_dir)
    
    # Determine date for directory structure
    paper_date = paper.published_date or paper.submitted_date
    if paper_date is None:
        # Fallback to current date if no date available
        paper_date = datetime.now()
    
    # Create date-based directory structure: YYYY/MM/DD
    year_dir = str(paper_date.year)
    month_dir = f"{paper_date.month:02d}"
    day_dir = f"{paper_date.day:02d}"
    
    # Create the full directory path
    date_dir = pdf_storage_path / year_dir / month_dir / day_dir
    
    # Create filename using full_id only
    filename = f"{paper.full_id}.pdf"
    
    return date_dir / filename


def check_pdf_exists_and_valid(paper: ArxivPaper, pdf_storage_dir: str) -> bool:
    """
    Check if PDF file exists and is valid.
    
    Args:
        paper: ArxivPaper object
        pdf_storage_dir: Base directory for PDF storage
        
    Returns:
        True if PDF exists and is valid, False otherwise
    """
    try:
        pdf_path = get_pdf_storage_path(paper, pdf_storage_dir)
        
        # Check if file exists
        if not pdf_path.exists():
            return False
        
        # Check if file is valid PDF
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        
        return validate_pdf_integrity(pdf_data)
        
    except Exception:
        return False


def normalize_arxiv_id(paper_id: str) -> str:
    """
    Normalize arXiv ID for arXiv API compatibility.
    
    arXiv.Search id_list only accepts xxx.xxx format, but some IDs might have
    underscores, dashes, or other variations that need to be normalized.
    
    Args:
        paper_id: Raw arXiv ID
        
    Returns:
        Normalized arXiv ID compatible with arXiv API
        
    Examples:
        >>> normalize_arxiv_id("2507_18009")
        "2507.18009"
        >>> normalize_arxiv_id("2507-18009")
        "2507.18009"
        >>> normalize_arxiv_id("2507.18009")
        "2507.18009"
    """
    if not paper_id:
        raise ValueError("Paper ID cannot be empty")
    
    # Replace common variations with dots
    normalized = paper_id.replace('_', '.').replace('-', '.')
    
    # Ensure it follows the pattern: YYMM.NNNNN or YYYY.NNNNN
    # Remove any extra dots or invalid characters
    parts = normalized.split('.')
    if len(parts) >= 2:
        # Keep only the first two parts (year/month and paper number)
        normalized = f"{parts[0]}.{parts[1]}"
    
    return normalized


class ArxivFetcher:
    """
    A focused arXiv data fetching utility.
    
    This class handles searching and downloading arXiv papers without
    coupling to storage or database concerns. It returns ArxivPaper
    objects that can be processed by other components.
    """
    
    def __init__(self, pdf_storage_dir: Optional[str] = None):
        """
        Initialize the arXiv fetcher.
        
        Args:
            pdf_storage_dir: Directory to store downloaded PDFs (optional)
        """
        self.logger = AgentLogger(self.__class__.__name__)
        
        # PDF storage directory
        if pdf_storage_dir:
            self.pdf_storage_dir = Path(pdf_storage_dir)
            self.pdf_storage_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.pdf_storage_dir = None
        
        # HTTP client configuration
        self.max_retries = 3
        self.base_retry_delay = 5  # seconds
        self.timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
        
        self.logger.info(f"ArxivFetcher initialized with PDF storage: {self.pdf_storage_dir}")
    
    async def search_papers(
        self,
        query: str = "",
        paper_ids: List[str] = None,
        max_results: int = 10000,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> List[ArxivPaper]:
        """
        Search for papers on arXiv.
        
        Args:
            query: Search query string
            paper_ids: List of specific arXiv IDs to fetch
            max_results: Maximum number of results to return
            sort_by: Sort criterion for results
            
        Returns:
            List of ArxivPaper objects
        """
        if not query and not paper_ids:
            self.logger.warning("No search query or paper IDs provided")
            return []
        
        if paper_ids:
            self.logger.info(f"Fetching {len(paper_ids)} specific papers")
        else:
            self.logger.info(f"Searching arXiv with query: {query}")
        
        papers = []
        arxiv_client = arxiv.Client(page_size=500, delay_seconds=3)
        
        try:
            # Prepare search parameters
            if paper_ids:
                # Normalize paper IDs for arXiv API
                clean_ids = [normalize_arxiv_id(pid) for pid in paper_ids]
                search = arxiv.Search(
                    query="",
                    id_list=clean_ids,
                    sort_by=sort_by
                )
            else:
                search = arxiv.Search(
                    query=query,
                    sort_by=sort_by
                )
            
            # Fetch results
            for result in arxiv_client.results(search):
                try:
                    paper = ArxivPaper.from_arxiv_result(result)
                    papers.append(paper)
                    
                    if len(papers) >= max_results:
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process arXiv result: {e}")
                    continue
            
            self.logger.info(f"Successfully fetched {len(papers)} papers")
            return papers
            
        except arxiv.UnexpectedEmptyPageError as e:
            self.logger.warning(f"arXiv returned empty page: {e}")
            self.logger.info(f"Successfully retrieved {len(papers)} papers before empty page")
            return papers
            
        except Exception as e:
            self.logger.error(f"arXiv search error: {e}", exc_info=True)
            return []
    
    async def download_papers(
        self,
        papers: List[ArxivPaper],
        session: Optional[aiohttp.ClientSession] = None,
        max_concurrent: int = 5
    ) -> List[Tuple[bool, ArxivPaper]]:
        """
        Download PDFs for multiple papers.
        
        Args:
            papers: List of ArxivPaper objects to download
            session: Optional aiohttp session (will create one if not provided)
            max_concurrent: Maximum concurrent downloads
            
        Returns:
            List of (success, ArxivPaper) tuples
        """
        if not papers:
            return []
        
        if not self.pdf_storage_dir:
            self.logger.error("PDF storage directory not configured")
            return [(False, paper) for paper in papers]
        
        self.logger.info(f"Starting download of {len(papers)} papers")
        
        # Create session if not provided
        if session is None:
            # Configure SSL context to handle certificate issues
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(
                limit=10, 
                limit_per_host=5,
                ssl=ssl_context
            )
            session = aiohttp.ClientSession(connector=connector, timeout=self.timeout)
            should_close_session = True
        else:
            should_close_session = False
        
        try:
            # Create semaphore to limit concurrent downloads
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def download_with_semaphore(paper: ArxivPaper) -> Tuple[bool, ArxivPaper]:
                async with semaphore:
                    success = await self._download_single_paper(paper, session)
                    return success, paper
            
            # Download all papers concurrently
            tasks = [download_with_semaphore(paper) for paper in papers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to download paper {papers[i].full_id}: {result}")
                    final_results.append((False, papers[i]))
                else:
                    final_results.append(result)
            
            successful = sum(1 for success, _ in final_results if success)
            self.logger.info(f"Download completed: {successful}/{len(papers)} successful")
            return final_results
            
        finally:
            if should_close_session:
                await session.close()
    
    async def _download_single_paper(
        self,
        paper: ArxivPaper,
        session: aiohttp.ClientSession
    ) -> bool:
        """Download PDF for a single paper."""
        if not paper.pdf_url:
            self.logger.warning(f"No PDF URL available for paper {paper.full_id}")
            return False
        
        try:
            # Download PDF
            pdf_data = await self._download_pdf_with_retry(paper.pdf_url, session)
            
            if pdf_data is None:
                self.logger.error(f"Failed to download PDF for paper {paper.full_id}")
                return False
            
            # Validate PDF integrity
            if not validate_pdf_integrity(pdf_data):
                self.logger.error(f"Downloaded PDF failed integrity validation for paper {paper.full_id}")
                return False
            
            # Save to file with date-based directory structure
            save_success = self._save_pdf_to_file(paper, pdf_data)
            if not save_success:
                return False
            
            self.logger.info(f"Successfully downloaded PDF for paper {paper.full_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading paper {paper.full_id}: {e}")
            return False
    
    
    async def _download_pdf_with_retry(
        self,
        pdf_url: str,
        session: aiohttp.ClientSession
    ) -> Optional[bytes]:
        """Download PDF with retry logic."""
        pdf_buffer = bytearray()
        
        for attempt in range(self.max_retries):
            try:
                # Get content length first
                async with session.head(pdf_url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    resp.raise_for_status()
                    total_size = int(resp.headers.get('content-length', 0))
                
                # Download PDF
                async with session.get(pdf_url, timeout=self.timeout) as resp:
                    resp.raise_for_status()
                    total_bytes = 0
                    
                    async for chunk in resp.content.iter_chunked(8192):
                        if chunk:
                            pdf_buffer.extend(chunk)
                            total_bytes += len(chunk)
                    
                    # Check if download is complete
                    if total_size > 0 and total_bytes < total_size:
                        raise aiohttp.ClientError(
                            f"Download incomplete. Expected {total_size} bytes, got {total_bytes}"
                        )
                    
                    return bytes(pdf_buffer)
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if attempt + 1 == self.max_retries:
                    self.logger.error(f"All {self.max_retries} attempts failed for {pdf_url}")
                    break
                
                # Wait before retry
                wait_time = self.base_retry_delay * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
                
            finally:
                # Clear buffer for next attempt
                pdf_buffer.clear()
        
        return None
    
    
    def _save_pdf_to_file(self, paper: ArxivPaper, pdf_data: bytes) -> bool:
        """
        Save PDF data to file system with date-based directory structure.
        
        Args:
            paper: ArxivPaper object
            pdf_data: PDF data as bytes
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Get the storage path using the external function
            pdf_path = get_pdf_storage_path(paper, str(self.pdf_storage_dir))
            
            # Create parent directories
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write PDF data
            with open(pdf_path, 'wb') as f:
                f.write(pdf_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save PDF for paper {paper.full_id}: {e}")
            return False
    
    async def get_random_papers(
        self,
        query: str = "",
        max_results: int = 10,
        date_range: tuple = ("20240101", "20241231")
    ) -> List[ArxivPaper]:
        """
        Get random papers from a date range.
        
        Args:
            query: Base query string
            max_results: Maximum number of results
            date_range: Tuple of (start_date, end_date) in YYYYMMDD format
            
        Returns:
            List of ArxivPaper objects
        """
        import datetime
        
        start_date = datetime.datetime.strptime(date_range[0], "%Y%m%d").date()
        end_date = datetime.datetime.strptime(date_range[1], "%Y%m%d").date()
        
        total_days = (end_date - start_date).days
        
        # Find a random weekday
        while True:
            random_day_offset = random.randint(0, total_days)
            random_date = start_date + datetime.timedelta(days=random_day_offset)
            if random_date.weekday() < 5:  # Monday = 0, Friday = 4
                break
        
        next_day = random_date + datetime.timedelta(days=1)
        
        first_day_str = random_date.strftime("%Y%m%d")
        next_day_str = next_day.strftime("%Y%m%d")
        
        # Build query with date range
        if not query:
            query = f"(ti:artificial intelligence OR cat:cs.AI) AND submittedDate:[{first_day_str} TO {next_day_str}]"
        else:
            query = f"{query} AND submittedDate:[{first_day_str} TO {next_day_str}]"
        
        self.logger.info(f"Random papers query: {query}")
        
        return await self.search_papers(query, max_results=max_results)