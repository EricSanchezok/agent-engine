from typing import List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import json
import os
from pathlib import Path, WindowsPath, PosixPath
import re
import arxiv
import time
from PyPDF2 import PdfReader

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption

# AgentEngine imports
from agent_engine.agent_logger import AgentLogger

# Internal imports
from .figure import Figure
from .table import Table
from .page import Page

logger = AgentLogger(__name__)

PAPER_DOWNLOAD_DIR = "data/arxiv_papers"
IMAGE_RESOLUTION_SCALE = 2.0

class Paper(BaseModel):
    paper_dir: str = Field(default=PAPER_DOWNLOAD_DIR, description="Paper directory")
    
    # Core identification information
    id: str = Field("test", description="Paper ID")
    title: str = Field("test", description="Paper title")
    
    # Author and category information
    authors: List[str] = Field(default_factory=list, description="Author names")
    categories: List[str] = Field(default_factory=list, description="arXiv categories")
    
    # Time information
    timestamp: str = Field("20250805T0505", description="Update timestamp, format: YYYYMMDDTHHMM")
    
    # Summary information
    summary: str = Field(default="", description="Paper summary")

    # Additional information
    comment: Optional[str] = Field(default="", description="Paper comment")
    journal_ref: Optional[str] = Field(default="", description="Paper journal reference")
    doi: Optional[str] = Field(default="", description="Paper DOI")
    links: List[str] = Field(default_factory=list, description="Paper links")
    
    # Pdf information
    pdf_url: str = Field(default="", description="PDF download link")
    downloaded: bool = Field(default=False, description="Is Paper downloaded")

    # File information
    pdf_filename: str = Field(default="", description="PDF filename")
    md_filename: str = Field(default="", description="Markdown file path")
    html_filename: str = Field(default="", description="HTML file path")
    figures: List[Figure] = Field(default_factory=list, description="Figures")
    tables: List[Table] = Field(default_factory=list, description="Tables")
    pages: List[Page] = Field(default_factory=list, description="Pages")

    # Arena information
    recommended_points: int = Field(default=0, description="Recommended points")
    opponents_played: set = Field(default_factory=set, description="Opponents played")
    
    class Config:
        from_attributes = True
        extra = "ignore"
        populate_by_name = True

    def get_pdf_filename(self) -> str:
        timestamp_date = datetime.strptime(self.timestamp, "%Y%m%dT%H%M").strftime("%Y%m%d")
        return os.path.join(self.paper_dir, timestamp_date, self.id, f"{self.title}.pdf")

    def add_figure(self, figure: Figure):
        self.figures.append(figure)

    def add_table(self, table: Table):
        self.tables.append(table)

    def add_page(self, page: Page):
        self.pages.append(page)

    def get_text(self, use_md: bool = False) -> str:
        use_md = False  # Default to False to avoid unnecessary conversion
        if use_md and not self.md_filename:
            self.convert()

        if use_md:
            with open(self.md_filename, "r", encoding="utf-8") as f:
                return f.read()
        else:
            reader = PdfReader(self.pdf_filename)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text

    def get_report(self) -> str:
        print(self.pdf_filename)
        report_filename = os.path.join(Path(self.pdf_filename).parent, "report.md")
        print(report_filename)
        with open(report_filename, "r", encoding="utf-8") as f:
            return f.read()

    def convert(self):
        if not self.pdf_filename:
            logger.warning(f"Paper {self.id} is not downloaded, skip converting")
            return
        
        if not self.md_filename or not self.html_filename:
            converter = Converter()
            converter.convert(self)
            self.save()
        else:
            logger.info(f"Paper {self.id} already converted, skip converting")

    def package(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "categories": self.categories,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "doi": self.doi,
            "links": self.links,
            "pdf_url": self.pdf_url
        }

    def save(self):
        if not self.downloaded and not self.pdf_filename:
            logger.warning(f"Paper {self.id} is not downloaded, skip saving")
            return
        save_dir = os.path.dirname(self.pdf_filename)
        os.makedirs(save_dir, exist_ok=True)

        metadata_filename = os.path.join(save_dir, "metadata.json")
        metadata = {
            # Core identification information
            "id": self.id,
            "title": self.title,

            # Author and category information
            "authors": self.authors,
            "categories": self.categories,

            # Time information
            "timestamp": self.timestamp,

            # Summary information
            "summary": self.summary,

            # Additional information
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "doi": self.doi,
            "links": self.links,

            # Pdf information
            "pdf_url": self.pdf_url,
            "downloaded": self.downloaded,

            # File information
            "pdf_filename": str(self.pdf_filename),
            "md_filename": str(self.md_filename),
            "html_filename": str(self.html_filename),
            "figures": [str(figure.filename) for figure in self.figures],
            "tables": [str(table.filename) for table in self.tables],
            "pages": [str(page.filename) for page in self.pages],
        }
        with open(metadata_filename, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        for figure in self.figures:
            figure.save()
        for table in self.tables:
            table.save()
        for page in self.pages:
            page.save()

    def load(self, paper_path: str = None) -> bool:        
        if not paper_path:
            pdf_filename = self.get_pdf_filename()
            metadata_filename = os.path.join(os.path.dirname(pdf_filename), "metadata.json")
        else:
            metadata_filename = os.path.join(paper_path, "metadata.json")
        try:
            with open(metadata_filename, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except:
            return False
            
            
        # Core identification information
        self.id = metadata["id"]
        self.title = metadata["title"]

        # Author and category information
        self.authors = metadata["authors"]
        self.categories = metadata["categories"]

        # Time information
        self.timestamp = metadata["timestamp"]

        # Summary information
        self.summary = metadata["summary"]

        # Additional information
        self.comment = metadata.get("comment", "")
        self.journal_ref = metadata.get("journal_ref", "")
        self.doi = metadata.get("doi", "")
        self.links = metadata.get("links", [])

        # Pdf information
        self.pdf_url = metadata["pdf_url"]
        self.downloaded = metadata.get("downloaded", False)

        # File information
        self.pdf_filename = metadata.get("pdf_filename", "")
        self.md_filename = metadata.get("md_filename", "")
        self.html_filename = metadata.get("html_filename", "")
        self.figures = [Figure.from_filename(figure_filename) for figure_filename in metadata.get("figures", [])]
        self.tables = [Table.from_filename(table_filename) for table_filename in metadata.get("tables", [])]
        self.pages = [Page.from_filename(page_filename) for page_filename in metadata.get("pages", [])]

        return True

    @classmethod
    def from_arxiv_result(cls, paper: arxiv.Result) -> "Paper":
        # Get arXiv ID safely
        try:
            arxiv_id = paper.entry_id.split("/")[-1].replace('.', '_')
            if 'v' in arxiv_id:
                arxiv_id = arxiv_id.split('v')[0]
        except Exception as e:
            logger.warning(f"Get arXiv ID failed: {e}")
            arxiv_id = "unknown_id"

        # Process title: clean, standardize, truncate
        try:
            title = paper.title.strip() if paper.title else "unknown_title"
            clean_title = re.sub(r"[^\w\s]", "", title.lower())
            clean_title = re.sub(r"\s+", "_", clean_title)
            clean_title = clean_title.strip("_")
            clean_title = clean_title[0].upper() + clean_title[1:]
        except Exception as e:
            logger.warning(f"Process title failed: {e}")
            clean_title = "unknown_title"

        # Get authors safely
        try:
            authors = [author.name for author in paper.authors]
        except Exception as e:
            logger.warning(f"Get author info failed: {e}")
            authors = []
        
        # Get categories safely
        try:
            categories = paper.categories if paper.categories else []
        except Exception as e:
            logger.warning(f"Get categories failed: {e}")
            categories = []
                
        # Format timestamp
        try:
            timestamp = paper.published.strftime("%Y%m%dT%H%M") if paper.published else ""
        except Exception as e:
            logger.warning(f"Format timestamp failed: {e}")
            timestamp = ""
        
        # Get summary safely
        try:
            summary = paper.summary if paper.summary else ""
        except Exception as e:
            logger.warning(f"Get summary failed: {e}")
            summary = ""
        
        # Get PDF URL safely
        try:
            pdf_url = paper.pdf_url if hasattr(paper, "pdf_url") else ""
            if not pdf_url:
                pdf_url = paper.entry_id.replace("/abs/", "/pdf/")
        except Exception as e:
            logger.warning(f"Get PDF URL failed: {e}")
            pdf_url = ""

        # Additional information
        try:
            comment = paper.comment if paper.comment else ""
        except Exception as e:
            logger.warning(f"Get comment failed: {e}")
            comment = ""

        try:
            journal_ref = paper.journal_ref if paper.journal_ref else ""
        except Exception as e:
            logger.warning(f"Get journal_ref failed: {e}")
            journal_ref = ""

        try:
            doi = paper.doi if paper.doi else ""
        except Exception as e:
            logger.warning(f"Get doi failed: {e}")
            doi = ""

        try:
            # Convert arxiv.Result.Link objects to strings
            if paper.links:
                links = [str(link.href) for link in paper.links if hasattr(link, 'href')]
            else:
                links = []
        except Exception as e:
            logger.warning(f"Get links failed: {e}")
            links = []
        
        return cls(
            paper_dir=PAPER_DOWNLOAD_DIR,
            # Core identification information
            id=arxiv_id,
            title=clean_title,

            # Author and category information
            authors=authors,
            categories=categories,

            # Time information
            timestamp=timestamp,

            # Summary information
            summary=summary,

            # Additional information
            comment=comment,
            journal_ref=journal_ref,
            doi=doi,
            links=links,

            # Pdf information
            pdf_url=pdf_url,
            downloaded=False,

            # File information
            pdf_filename="",
            md_filename="",
            html_filename="",
            figures=[],
            tables=[],
            pages=[],
        )

class Converter:
    def __init__(self, gpu_id: int = 0): 
        self.gpu_id = gpu_id
        self._detect_cuda()    
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.accelerator_options = self.accelerator_options

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        settings.debug.profile_pipeline_timings = True

    def _detect_cuda(self):
        try:
            import torch
            if torch.cuda.is_available():
                device_str = f"cuda:{self.gpu_id}"
                
                if self.gpu_id >= torch.cuda.device_count():
                    logger.error(f"指定的GPU ID {self.gpu_id} 不可用，将使用CPU。")
                    raise ValueError("Invalid GPU ID")

                cuda_name = torch.cuda.get_device_name(self.gpu_id)
                # logger.info(f"✅ Converter instance will use GPU {self.gpu_id}: {cuda_name}")

                self.accelerator_options = AcceleratorOptions(
                    num_threads=8,
                    device=device_str, 
                    cuda_use_flash_attention2=True
                )
                
            else:
                logger.info("❌ CUDA not available, using CPU")
                self.accelerator_options = AcceleratorOptions(
                    num_threads=32, 
                    device=AcceleratorDevice.CPU
                )
                
        except ImportError:
            logger.warning("⚠️ PyTorch not installed, CUDA detection skipped")
            self.accelerator_options = AcceleratorOptions(
                num_threads=32, 
                device=AcceleratorDevice.CPU
            )
        except Exception as e:
            logger.error(f"❌ Error during CUDA detection: {e}")
            self.accelerator_options = AcceleratorOptions(
                num_threads=32, 
                device=AcceleratorDevice.CPU
            )

    def convert(self, paper: Paper) -> Paper:
        if not paper.pdf_filename:
            logger.warning(f"PDF file path is not set for paper {paper.id}")
            return paper
        
        _start_time = time.time()

        paper_dir = Path(os.path.dirname(paper.pdf_filename))
        paper_dir.mkdir(parents=True, exist_ok=True)

        conv_res = self.converter.convert(paper.pdf_filename)
        doc_filename = conv_res.input.file.stem

        for page_no, page in conv_res.document.pages.items():
            page_no = page.page_no
            page_image_filename = paper_dir / f"pages" / f"page_{page_no}.png"
            page = Page(
                filename=page_image_filename,
                image=page.image.pil_image,
                number=page_no
            )
            page.save()
            paper.add_page(page)

        table_counter = 0
        picture_counter = 0
        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                element_image_filename = (
                    paper_dir / f"tables" / f"table_{table_counter}.png"
                )
                table = Table(
                    filename=element_image_filename,
                    image=element.get_image(conv_res.document),
                    number=table_counter
                )
                table.save()
                paper.add_table(table)

            elif isinstance(element, PictureItem):
                picture_counter += 1
                element_image_filename = (
                    paper_dir / f"figures" / f"figure_{picture_counter}.png"
                )
                figure = Figure(
                    filename=element_image_filename,
                    image=element.get_image(conv_res.document),
                    number=picture_counter,
                    caption=element.caption_text(conv_res.document)
                )
                figure.save()
                paper.add_figure(figure)

        md_filename = paper_dir / f"{doc_filename}.md"
        conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)
        paper.md_filename = md_filename

        html_filename = paper_dir / f"{doc_filename}.html"
        conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)
        paper.html_filename = html_filename

        end_time = time.time()
        logger.info(f"Converted paper: {paper.id} in {end_time - _start_time:.2f} seconds")
        return paper

    def _get_paper_dir(self, paper: Paper) -> Path:
        return Path(paper.pdf_filename).parent
    
    
import multiprocessing as mp
from itertools import islice

def _worker_convert(paper: Paper, gpu_id: int) -> Paper:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    converter = Converter(gpu_id=0)
    logger.info(f"Worker on physical GPU {gpu_id} starting conversion for paper {paper.id}")
    
    updated_paper = converter.convert(paper)
    updated_paper.save()
    return updated_paper


class PaperProcessor:
    def __init__(self):
        self.num_gpus = 0
        try:
            import torch
            if torch.cuda.is_available():
                self.num_gpus = torch.cuda.device_count()
        except ImportError:
            self.num_gpus = 0

    def convert_papers_in_batches(self, papers: List[Paper]) -> List[Paper]:
        if self.num_gpus > 0:
            num_workers = self.num_gpus
            logger.info(f"🚀 Detected {num_workers} GPUs, starting parallel conversion for {len(papers)} papers.")
        else:
            num_workers = min(8, mp.cpu_count() if mp.cpu_count() else 4)
            logger.info(f"ℹ️ No GPUs detected. Using {num_workers} CPU processes for parallel conversion of {len(papers)} papers.")


        papers_to_convert = [p for p in papers if not p.md_filename or not os.path.exists(p.md_filename)]
        if not papers_to_convert:
            logger.info("✅ All papers are already converted.")
            return papers
            
        logger.info(f"🚀 Starting parallel conversion for {len(papers_to_convert)} papers on {self.num_gpus} GPUs.")
        
        paper_iterator = iter(papers_to_convert)
        batch_size = num_workers * 4
        
        all_processed_papers = []

        with mp.get_context("spawn").Pool(self.num_gpus) as pool:
            while True:
                current_batch = list(islice(paper_iterator, batch_size))
                if not current_batch:
                    break

                logger.info(f"--- Processing a new batch of {len(current_batch)} papers ---")

                tasks_for_batch = []
                for i, paper in enumerate(current_batch):
                    gpu_id = i % self.num_gpus
                    tasks_for_batch.append((paper, gpu_id))

                processed_in_batch = pool.starmap(_worker_convert, tasks_for_batch)
                
                all_processed_papers.extend(processed_in_batch)

        logger.info("✅ All batches processed.")

        processed_dict = {p.id: p for p in all_processed_papers}
        final_papers = [processed_dict.get(p.id, p) for p in papers]
        
        return final_papers