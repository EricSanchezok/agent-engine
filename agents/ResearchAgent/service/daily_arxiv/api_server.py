import os
import json
import asyncio
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS

from agent_engine.agent_logger import AgentLogger
from agent_engine.utils.network_utils import get_local_ip
from agents.ResearchAgent.service.daily_arxiv.config import DailyArxivConfig

app = Flask(__name__)
CORS(app)

class PaperReportAPI:
    def __init__(self):
        self.logger = AgentLogger(self.__class__.__name__)
        self.base_dir = Path(DailyArxivConfig.get_result_storage_dir())
        self.logger.info(f"PaperReportAPI initialized with base directory: {self.base_dir}")
    
    def get_reports_by_date_range(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get paper reports within date range"""
        try:
            start_dt = datetime.strptime(start_date, "%Y%m%d").date()
            end_dt = datetime.strptime(end_date, "%Y%m%d").date()
            
            if start_dt > end_dt:
                return {
                    "code": 0,
                    "message": "Invalid date range: start_date cannot be later than end_date",
                    "data": []
                }
            
            reports = []
            current_date = start_dt
            
            while current_date <= end_dt:
                date_reports = self._get_reports_for_date(current_date)
                reports.extend(date_reports)
                current_date = self._get_next_date(current_date)
            
            if not reports:
                return {
                    "code": 0,
                    "message": f"No paper reports found for date range {start_date} to {end_date}. "
                              f"This could be because: 1) No papers were processed for these dates, "
                              f"2) The daily arXiv processing has not run yet, or "
                              f"3) The reports are still being generated.",
                    "data": []
                }
            
            return {
                "code": 0,
                "message": "success",
                "data": reports
            }
            
        except ValueError as e:
            return {
                "code": 1,
                "message": f"Invalid date format: {str(e)}. Expected format: YYYYMMDD",
                "data": []
            }
        except Exception as e:
            self.logger.error(f"Error getting reports by date range: {e}", exc_info=True)
            return {
                "code": 1,
                "message": f"Internal server error: {str(e)}",
                "data": []
            }
    
    def _get_reports_for_date(self, target_date: date) -> List[Dict[str, Any]]:
        """Get reports for a specific date"""
        date_dir = self.base_dir / str(target_date.year) / f"{target_date.month:02d}" / f"{target_date.day:02d}"
        
        if not date_dir.exists():
            self.logger.info(f"No reports directory found for date: {target_date}")
            return []
        
        reports = []
        
        # First, try to load metadata from JSON result file
        json_file = date_dir / f"daily_result_{target_date.strftime('%Y-%m-%d')}.json"
        metadata = {}
        
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    
                    # Extract metadata from paper_metadata section (preferred)
                    if 'paper_metadata' in json_data:
                        for paper_id, paper_data in json_data['paper_metadata'].items():
                            metadata[paper_id] = {
                                'title': paper_data.get('title', ''),
                                'authors': paper_data.get('authors', []),
                                'categories': paper_data.get('categories', []),
                                'summary': paper_data.get('summary', ''),
                                'published_date': paper_data.get('published_date'),
                                'pdf_url': paper_data.get('pdf_url', ''),
                                'doi': paper_data.get('doi'),
                                'journal_ref': paper_data.get('journal_ref'),
                                'comment': paper_data.get('comment')
                            }
                    
                    # Fallback: Extract metadata from Swiss tournament results
                    elif 'swiss_tournament_result' in json_data and 'top_papers' in json_data['swiss_tournament_result']:
                        for paper in json_data['swiss_tournament_result']['top_papers']:
                            paper_id = paper.get('paper_id', '')
                            metadata[paper_id] = {
                                'title': paper.get('title', ''),
                                'pdf_path': paper.get('pdf_path', ''),
                                'rank': paper.get('rank', 0)
                            }
            except Exception as e:
                self.logger.error(f"Error loading JSON metadata from {json_file}: {e}")
        
        # Look for markdown files
        for md_file in date_dir.glob("*.md"):
            if md_file.name == "README.md":
                continue
                
            try:
                report_data = self._parse_markdown_report(md_file, target_date, metadata)
                if report_data:
                    reports.append(report_data)
            except Exception as e:
                self.logger.error(f"Error parsing markdown file {md_file}: {e}")
                continue
        
        return reports
    
    def _parse_markdown_report(self, md_file: Path, target_date: date, metadata: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Parse markdown report file"""
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract paper ID from filename (e.g., "2509.13311v1.md")
            paper_id = md_file.stem
            
            # Parse markdown content to extract title and other info
            lines = content.split('\n')
            title = ""
            authors = []
            categories = []
            report_content = content
            
            # Extract title from the first ### line
            for line in lines:
                if line.startswith('### '):
                    title = line[4:].strip()
                    break
            
            # Use metadata if available
            if metadata and paper_id in metadata:
                meta = metadata[paper_id]
                title = meta.get('title', title) or f"Paper {paper_id}"
                authors = meta.get('authors', ["Unknown Author"])
                categories = meta.get('categories', ["Unknown Category"])
                pdf_url = meta.get('pdf_url', f"https://arxiv.org/pdf/{paper_id}.pdf")
            else:
                title = title or f"Paper {paper_id}"
                authors = ["Unknown Author"]
                categories = ["Unknown Category"]
                pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            
            return {
                "title": title,
                "authors": authors,
                "categories": categories,
                "timestamp": target_date.strftime("%Y%m%dT0000"),
                "pdf_url": pdf_url,
                "report": report_content
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing markdown file {md_file}: {e}")
            return None
    
    def _get_next_date(self, current_date: date) -> date:
        """Get next date"""
        from datetime import timedelta
        return current_date + timedelta(days=1)

# Initialize API instance
api = PaperReportAPI()

@app.route('/api/paper-reports', methods=['POST'])
def get_paper_reports():
    """Get paper reports by date range"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "code": 1,
                "message": "Request body is required",
                "data": []
            })
        
        date_range = data.get('date_range')
        if not date_range:
            return jsonify({
                "code": 1,
                "message": "date_range is required in request body",
                "data": []
            })
        
        start_date = date_range.get('start_date')
        end_date = date_range.get('end_date')
        
        if not start_date or not end_date:
            return jsonify({
                "code": 1,
                "message": "start_date and end_date are required in date_range",
                "data": []
            })
        
        result = api.get_reports_by_date_range(start_date, end_date)
        return jsonify(result)
        
    except Exception as e:
        api.logger.error(f"Error in API endpoint: {e}", exc_info=True)
        return jsonify({
            "code": 1,
            "message": f"Internal server error: {str(e)}",
            "data": []
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "code": 0,
        "message": "API server is running",
        "data": {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get processing status"""
    try:
        status_file = Path(DailyArxivConfig.get_status_file_path())
        if status_file.exists():
            with open(status_file, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
        else:
            status_data = {"message": "No status file found"}
        
        return jsonify({
            "code": 0,
            "message": "success",
            "data": status_data
        })
    except Exception as e:
        return jsonify({
            "code": 1,
            "message": f"Error reading status: {str(e)}",
            "data": []
        })

def start_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Start the Flask server"""
    api.logger.info(f"Starting API server on {host}:{port}")
    api.logger.info(f"Base directory: {api.base_dir}")
    
    # Get local IP for convenience
    local_ip = get_local_ip()
    api.logger.info(f"Local IP: {local_ip}")
    api.logger.info(f"Access URL: http://{local_ip}:{port}")
    
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    start_server(debug=True)
