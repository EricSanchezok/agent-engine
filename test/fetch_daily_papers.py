#!/usr/bin/env python3
"""Utility script to query SignalFrontier server for papers of a specific day.

Usage
-----
run.bat test/fetch_daily_papers.py 20250821

The script sends a POST request to the running `frontier_server.py` (default
`http://localhost:6000`) and prints a concise summary of the returned papers.
It automatically converts the single input day into the inclusive date range
expected by the server (start_date <= date < end_date).
"""

import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

import requests

SERVER_URL = "http://localhost:6000/papers/fetch"
DEFAULT_DAY = "20250820"  # 修改为你想查询的日期
OUTPUT_PATH = "database/daily_papers.json"
OUTPUT_DIR = "database/daily_papers"

def build_payload(day: str) -> Dict[str, Any]:
    """Build the JSON payload accepted by `/papers/fetch` endpoint.

    The backend expects a half-open interval [start_date, end_date) so we add one
    day to the given date for *end_date*.
    """
    try:
        date_obj = datetime.strptime(day, "%Y%m%d")
    except ValueError as exc:
        raise SystemExit(f"Invalid date format: {day}. Expected YYYYMMDD") from exc

    next_day = date_obj + timedelta(days=1)

    return {
        "date_range": {
            "start_date": day,
            "end_date": next_day.strftime("%Y%m%d"),
        },
        # Add category filters here if needed, leave empty to fetch all
        "categories": [],
    }

def fetch_papers(day: str) -> List[Dict[str, Any]]:
    payload = build_payload(day)
    response = requests.post(SERVER_URL, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"Request failed with status {response.status_code}: {response.text}"
        )

    resp_json = response.json()
    if resp_json.get("code") != 0:
        raise RuntimeError(
            f"Server returned error code {resp_json.get('code')}: {resp_json.get('message')}"
        )

    return resp_json.get("data", [])

def main():
    # 如果命令行提供日期则使用，否则使用 DEFAULT_DAY
    day = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_DAY
    papers = fetch_papers(day)

    print(f"Fetched {len(papers)} papers for {day}")
    for idx, paper in enumerate(papers, 1):
        title = paper.get("title", "<unknown title>")
        categories = ", ".join(paper.get("categories", []))
        print(f"{idx:03d}. {title} | {categories}")

    # 保存 report 为 markdown 文件
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for idx, paper in enumerate(papers, 1):
        report_md = paper.get("report", "")
        md_path = os.path.join(OUTPUT_DIR, f"{idx}.md")
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(report_md)
        except Exception as e:
            print(f"Failed to write {md_path}: {e}")

    # 将结果保存为 JSON 文件
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"date": day, "papers": papers}, f, ensure_ascii=False, indent=2)
        print(f"Saved data to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Failed to save JSON: {e}")

if __name__ == "__main__":
    main()
