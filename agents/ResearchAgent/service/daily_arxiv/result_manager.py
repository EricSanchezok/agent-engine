"""
Result Manager for Daily arXiv Service

This module manages the storage of daily arXiv processing results and tracks
the processing status to prevent duplicate runs on the same day.
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

import json
from datetime import date, datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from agent_engine.agent_logger import AgentLogger

from agents.ResearchAgent.service.daily_arxiv.config import DailyArxivConfig


@dataclass
class ProcessingStatus:
    """Represents the processing status for a specific date."""
    date: str  # YYYY-MM-DD format
    status: str  # "completed", "in_progress", "failed"
    started_at: Optional[str] = None  # ISO format timestamp
    completed_at: Optional[str] = None  # ISO format timestamp
    error_message: Optional[str] = None
    result_file_path: Optional[str] = None


@dataclass
class DailyResult:
    """Represents the complete result of daily arXiv processing."""
    metadata: Dict[str, Any]
    filter_result: Dict[str, Any]
    swiss_tournament_result: Dict[str, Any]
    condensed_reports: List[Dict[str, Any]]


class ResultManager:
    """
    Manages daily arXiv processing results and status tracking.
    
    This class handles:
    1. Saving complete processing results to JSON files
    2. Tracking daily processing status to prevent duplicate runs
    3. Organizing results by date in directory structure
    """
    
    def __init__(self):
        """Initialize the result manager."""
        self.logger = AgentLogger(self.__class__.__name__)
        
        # Set up directories
        self.result_storage_dir = Path(DailyArxivConfig.get_result_storage_dir())
        self.status_file_path = Path(DailyArxivConfig.get_status_file_path())
        
        # Ensure directories exist
        self.result_storage_dir.mkdir(parents=True, exist_ok=True)
        self.status_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ResultManager initialized with storage: {self.result_storage_dir}")
    
    def is_date_processed(self, target_date: date) -> bool:
        """
        Check if a specific date has already been processed.
        
        Args:
            target_date: Date to check
            
        Returns:
            True if date has been processed, False otherwise
        """
        status = self._load_processing_status()
        date_str = target_date.isoformat()
        
        if date_str in status:
            processing_status = status[date_str]
            return processing_status.get("status") == "completed"
        
        return False
    
    def mark_date_in_progress(self, target_date: date) -> bool:
        """
        Mark a date as being processed.
        
        Args:
            target_date: Date to mark as in progress
            
        Returns:
            True if successfully marked, False if already processed
        """
        if self.is_date_processed(target_date):
            self.logger.warning(f"Date {target_date} has already been processed")
            return False
        
        status = self._load_processing_status()
        date_str = target_date.isoformat()
        
        status[date_str] = {
            "date": date_str,
            "status": "in_progress",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error_message": None,
            "result_file_path": None
        }
        
        self._save_processing_status(status)
        self.logger.info(f"Marked date {target_date} as in progress")
        return True
    
    def mark_date_completed(
        self, 
        target_date: date, 
        result_file_path: str
    ) -> bool:
        """
        Mark a date as completed processing.
        
        Args:
            target_date: Date to mark as completed
            result_file_path: Path to the result JSON file
            
        Returns:
            True if successfully marked
        """
        status = self._load_processing_status()
        date_str = target_date.isoformat()
        
        if date_str not in status:
            self.logger.error(f"Date {target_date} not found in processing status")
            return False
        
        status[date_str].update({
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result_file_path": result_file_path,
            "error_message": None
        })
        
        self._save_processing_status(status)
        self.logger.info(f"Marked date {target_date} as completed")
        return True
    
    def mark_date_failed(self, target_date: date, error_message: str) -> bool:
        """
        Mark a date as failed processing.
        
        Args:
            target_date: Date to mark as failed
            error_message: Error message describing the failure
            
        Returns:
            True if successfully marked
        """
        status = self._load_processing_status()
        date_str = target_date.isoformat()
        
        if date_str not in status:
            self.logger.error(f"Date {target_date} not found in processing status")
            return False
        
        status[date_str].update({
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error_message": error_message
        })
        
        self._save_processing_status(status)
        self.logger.info(f"Marked date {target_date} as failed: {error_message}")
        return True
    
    def save_daily_result(
        self, 
        target_date: date,
        filter_result: Dict[str, Any],
        swiss_tournament_result: Dict[str, Any],
        condensed_reports: List[Dict[str, Any]]
    ) -> str:
        """
        Save complete daily processing result to JSON file.
        
        Args:
            target_date: Date of processing
            filter_result: Result from paper filtering
            swiss_tournament_result: Result from Swiss tournament ranking
            condensed_reports: List of condensed report information
            
        Returns:
            Path to the saved result file
        """
        # Create date-based directory structure: YYYY/MM/DD/
        year_dir = str(target_date.year)
        month_dir = f"{target_date.month:02d}"
        day_dir = f"{target_date.day:02d}"
        
        date_dir = self.result_storage_dir / year_dir / month_dir / day_dir
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # Create result data structure
        result_data = {
            "metadata": {
                "date": target_date.isoformat(),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
                "total_papers_processed": filter_result.get("papers_processed", 0),
                "total_comparisons": swiss_tournament_result.get("total_comparisons", 0)
            },
            "filter_result": filter_result,
            "swiss_tournament_result": swiss_tournament_result,
            "condensed_reports": condensed_reports
        }
        
        # Save to JSON file
        result_filename = f"daily_result_{target_date.isoformat()}.json"
        result_file_path = date_dir / result_filename
        
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved daily result to: {result_file_path}")
        return str(result_file_path)
    
    def load_daily_result(self, target_date: date) -> Optional[Dict[str, Any]]:
        """
        Load daily processing result from JSON file.
        
        Args:
            target_date: Date to load result for
            
        Returns:
            Result data dictionary or None if not found
        """
        # Create expected file path
        year_dir = str(target_date.year)
        month_dir = f"{target_date.month:02d}"
        day_dir = f"{target_date.day:02d}"
        
        date_dir = self.result_storage_dir / year_dir / month_dir / day_dir
        result_filename = f"daily_result_{target_date.isoformat()}.json"
        result_file_path = date_dir / result_filename
        
        if not result_file_path.exists():
            self.logger.warning(f"No result file found for date {target_date}")
            return None
        
        try:
            with open(result_file_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            self.logger.info(f"Loaded daily result from: {result_file_path}")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Error loading result for date {target_date}: {e}")
            return None
    
    def get_processing_status_summary(self) -> Dict[str, Any]:
        """
        Get summary of processing status for all dates.
        
        Returns:
            Dictionary with processing statistics
        """
        status = self._load_processing_status()
        
        total_dates = len(status)
        completed_dates = sum(1 for s in status.values() if s.get("status") == "completed")
        failed_dates = sum(1 for s in status.values() if s.get("status") == "failed")
        in_progress_dates = sum(1 for s in status.values() if s.get("status") == "in_progress")
        
        return {
            "total_dates": total_dates,
            "completed_dates": completed_dates,
            "failed_dates": failed_dates,
            "in_progress_dates": in_progress_dates,
            "success_rate": completed_dates / total_dates if total_dates > 0 else 0
        }
    
    def _load_processing_status(self) -> Dict[str, Any]:
        """Load processing status from file."""
        if not self.status_file_path.exists():
            return {}
        
        try:
            with open(self.status_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading processing status: {e}")
            return {}
    
    def _save_processing_status(self, status: Dict[str, Any]) -> None:
        """Save processing status to file."""
        try:
            with open(self.status_file_path, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving processing status: {e}")


def main():
    """Test the result manager."""
    manager = ResultManager()
    
    # Test with today's date
    today = date.today()
    
    print("Result Manager Test:")
    print(f"Testing with date: {today}")
    print(f"Is processed: {manager.is_date_processed(today)}")
    
    # Mark as in progress
    if manager.mark_date_in_progress(today):
        print("Marked as in progress")
        
        # Simulate saving a result
        sample_filter_result = {
            "papers_processed": 50,
            "papers_downloaded": 15,
            "successful_pdf_paths": ["path1", "path2"]
        }
        
        sample_swiss_result = {
            "papers_processed": 15,
            "total_comparisons": 120,
            "top_papers": [
                {"paper_id": "2509.12345", "title": "Sample Paper", "score": 4.5, "rank": 1}
            ]
        }
        
        sample_reports = [
            {"paper_id": "2509.12345", "markdown_file": "2509.12345.md"}
        ]
        
        result_path = manager.save_daily_result(
            today, sample_filter_result, sample_swiss_result, sample_reports
        )
        
        # Mark as completed
        manager.mark_date_completed(today, result_path)
        print("Marked as completed")
    
    # Check status summary
    summary = manager.get_processing_status_summary()
    print(f"Status Summary: {summary}")


if __name__ == "__main__":
    main()
