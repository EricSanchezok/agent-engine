#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SignalFrontier 服务器
提供论文数据查询服务的Flask应用
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify

# AgentEngine imports
from agent_engine.agent_logger import AgentLogger

# Core imports
from core.arxiv.paper_db import Paper, ArxivPaperDB

import hashlib
import json

# 配置常量
PAPER_PATH = "database/arxiv_paper_db.sqlite"
DEFAULT_PORT = 6000
DEFAULT_HOST = "0.0.0.0"

# 初始化logger和Flask应用
logger = AgentLogger(__name__)
app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

# 配置日志
logger.info("SignalFrontier server starting...")
logger.info(f"Paper data path: {PAPER_PATH}")
logger.info(f"Server address: {DEFAULT_HOST}:{DEFAULT_PORT}")


def validate_date_format(date_str: str) -> Optional[datetime]:
    """
    验证日期格式并转换为datetime对象
    
    Args:
        date_str: 日期字符串，格式应为YYYYMMDD
        
    Returns:
        datetime对象或None（如果格式无效）
    """
    try:
        return datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        logger.warning(f"Invalid date format: {date_str}")
        return None


def get_papers(start_date: str, end_date: str) -> List[Paper]:
    """
    Fetch papers within a date range from the SQLite database.

    Only papers that contain a non-empty ``report`` field inside ``paper.metadata``
    will be returned.
    """
    logger.info(f"Fetching papers from database, date range: {start_date} to {end_date}")

    papers: List[Paper] = []

    # Validate date formats
    start_dt = validate_date_format(start_date)
    end_dt = validate_date_format(end_date)

    if not start_dt or not end_dt:
        logger.error("Date format validation failed")
        return papers

    db = ArxivPaperDB(PAPER_PATH)
    current_dt = start_dt

    while current_dt < end_dt:
        date_str = current_dt.strftime("%Y%m%d")
        logger.debug(f"Querying papers for date: {date_str}")
        try:
            day_papers = db.get_by_date(date_str)
            for paper in day_papers:
                if get_report(paper):  # ensure report exists and is non-empty
                    papers.append(paper)
        except Exception as e:
            logger.error(f"Error fetching papers for date {date_str}: {e}")

        current_dt += timedelta(days=1)

    logger.info(f"Paper data fetching completed, total {len(papers)} papers")
    return papers

def filter_papers_by_categories(papers: List[Paper], categories: List[str]) -> List[Paper]:
    """
    根据分类过滤论文
    
    Args:
        papers: 论文列表
        categories: 分类列表
        
    Returns:
        过滤后的论文列表
    """
    if not categories:
        logger.info("No category filter specified, returning all papers")
        return papers
    
    logger.info(f"Starting to filter papers by categories: {categories}")
    
    filtered_papers = []
    for paper in papers:
        paper_categories = paper.info.get("categories", [])
        if any(category in paper_categories for category in categories):
            filtered_papers.append(paper)
    
    logger.info(f"Category filtering completed, filtered {len(filtered_papers)} papers from {len(papers)} total papers")
    return filtered_papers


# -----------------------------------------------------------------------------
# Helper to extract report content from Paper.metadata
# -----------------------------------------------------------------------------

def get_report(paper: Paper) -> str:
    """Return daily pipeline report text if ``metadata['report_source']`` is ``'daily'``."""
    if getattr(paper, "metadata", None):
        if paper.metadata.get("report_source") == "daily":
            return paper.metadata.get("report", "") or ""
    return ""

def generate_paper_id(paper: Paper) -> str:
    paper_hash = hashlib.sha256(paper.info.get("title", "").encode("utf-8")).hexdigest()
    return paper_hash

def generate_content_id(paper: Paper) -> str:
    report_content = get_report(paper)
    content_hash = hashlib.sha256(report_content.encode('utf-8')).hexdigest()
    return content_hash

def build_paper_response(paper: Paper) -> Dict[str, Any]:
    paper_id = generate_paper_id(paper)
    content_id = generate_content_id(paper)

    title = paper.info.get("title", "").replace("_", " ")
    authors = paper.info.get("authors", [])
    categories = paper.info.get("categories", [])
    timestamp = paper.info.get("timestamp", "")
    pdf_url = paper.info.get("pdf_url", "")
    report_text = get_report(paper)

    return {
        "paper_id": paper_id,
        "content_id": content_id,
        "title": title,
        "authors": authors,
        "categories": categories,
        "timestamp": timestamp,
        "pdf_url": pdf_url,
        "report": report_text,
    }


@app.route('/papers/fetch', methods=['POST'])
def handle_fetch_papers():
    """
    处理论文获取请求
    
    Returns:
        JSON响应，包含论文数据或错误信息
    """
    logger.info("Received paper fetch request")
    
    try:
        # 解析请求数据
        request_data = request.get_json()
        if not request_data or "date_range" not in request_data:
            raise ValueError("Request body must be a JSON object with a 'date_range' key.")
        
        start_date_str = request_data["date_range"].get("start_date")
        end_date_str = request_data["date_range"].get("end_date")
        categories = request_data.get("categories", [])

        logger.info(f"Request parameters - Start date: {start_date_str}, End date: {end_date_str}, Categories: {categories}")

        # 验证必需参数
        if not start_date_str or not end_date_str:
            raise ValueError("Both 'start_date' and 'end_date' are required.")
            
        # 获取论文数据
        papers = get_papers(start_date_str, end_date_str)
        
        # 按分类过滤
        filtered_papers = filter_papers_by_categories(papers, categories)
        
        # 构建响应数据
        response_data = [build_paper_response(paper) for paper in filtered_papers]
        
        response_payload = {
            "code": 0,
            "message": "success",
            "data": response_data
        }
        
        logger.info(f"Request processed successfully, returning {len(response_data)} papers")
        return jsonify(response_payload), 200

    except (ValueError, TypeError) as e:
        error_msg = f"Invalid request format: {e}"
        logger.error(f"Request format error: {error_msg}")
        return jsonify({"code": -200, "message": error_msg}), 400
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(f"Unexpected error occurred while processing request: {error_msg}")
        return jsonify({"code": -500, "message": error_msg}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查端点
    
    Returns:
        服务器状态信息
    """
    logger.debug("Received health check request")
    return jsonify({
        "status": "healthy",
        "service": "SignalFrontier",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/', methods=['GET'])
def root():
    """
    根路径端点
    
    Returns:
        服务信息
    """
    logger.debug("Received root path request")
    return jsonify({
        "service": "SignalFrontier",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "服务信息",
            "GET /health": "健康检查",
            "POST /papers/fetch": "获取论文数据"
        }
    }), 200


if __name__ == '__main__':
    logger.info("Starting Flask application...")
    try:
        app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=True)
        logger.info("Flask application started successfully")
    except Exception as e:
        logger.error(f"Flask application failed to start: {e}")
        raise

