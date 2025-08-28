# -*- coding: utf-8 -*-
"""
A lightweight database-backed implementation of an arXiv `Paper` object.

Design goals
------------
1. **与旧实现解耦**：不依赖旧的 Figure / Table / Page 等类，完全使用内置 `dict` 结构保存所有信息。
2. **单表结构**：SQLite 表 `papers`，包含 5 列：
   - `id` TEXT 主键，形如 "2507.18009"。
   - `info_json`   TEXT，核心元数据，见 `Paper.info`。
   - `pdf_base64`  BLOB，可为空；有内容表示已下载。
   - `converted_json` TEXT，可为空；存储转换后的 figures/tables/pages。
   - `metadata_json` TEXT，可为空；额外元数据（推荐分等）。
3. **API**：
   - `ArxivPaperDB.add(paper)` / `delete(paper_id)`
   - `get(paper_id)` -> `Paper`
   - `get_by_date('YYYYMMDD')` -> list[`Paper`]

使用示例见 `test/test_paper_db.py`。
"""
from __future__ import annotations

import sqlite3
import json
import os
import re
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 为保持日志风格统一，沿用 agent_engine 的日志器
from agent_engine.agent_logger import AgentLogger

logger = AgentLogger(__name__)

###############################################################################
# Helper functions
###############################################################################

def _clean_arxiv_id(raw_id: str) -> str:
    """清洗 arXiv ID：
    1. 将 `_` 替换为 `.`
    2. 去掉版本号 `v\d+$`

    Examples
    --------
    >>> _clean_arxiv_id("2507_18009v3")
    '2507.18009'
    """
    if not raw_id:
        raise ValueError("arXiv id 不能为空")
    # `_` -> `.`
    fixed = raw_id.replace("_", ".")
    # 删除版本号
    fixed = re.sub(r"v\d+$", "", fixed)
    return fixed

###############################################################################
# Paper dataclass (dict-based)
###############################################################################

class Paper:
    """以字典存储信息的轻量 Paper 对象。"""

    # ----- 核心信息字段 -----
    CORE_FIELDS = [
        "id",
        "title",
        "authors",
        "categories",
        "timestamp",
        "summary",
        "comment",
        "journal_ref",
        "doi",
        "links",
        "pdf_url",
    ]

    def __init__(
        self,
        info: Dict[str, Any],
        pdf_base64: Optional[str] = None,
        converted: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Parameters
        ----------
        info : dict
            包含 CORE_FIELDS 的字典（id 必填）
        pdf_base64 : str | None
            原始 PDF 文件的字节内容，base64 编码；如果 None 代表尚未下载。
        converted : dict | None
            转换后的 figures/tables/pages 字典。
        metadata : dict | None
            额外元数据。
        """
        # 深拷贝保证外部修改不影响内部
        self.info: Dict[str, Any] = {k: info.get(k) for k in self.CORE_FIELDS}
        if not self.info.get("id"):
            raise ValueError("Paper.info 必须包含 'id'")
        self.info["id"] = _clean_arxiv_id(self.info["id"])

        # base64 字符串；None 代表未下载
        self.pdf_bytes: Optional[str] = pdf_base64
        self.converted: Dict[str, Any] = converted or {}
        self.metadata: Dict[str, Any] = metadata or {}

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------
    @property
    def id(self) -> str:  # noqa: D401  (one-liner ok in chinese)
        """返回清洗后的 arXiv id"""
        return self.info["id"]

    @classmethod
    def from_arxiv_result(cls, result: "arxiv.Result") -> "Paper":  # type: ignore
        """根据 `arxiv.Result` 构建 Paper.

        由于本文件不直接依赖外部 arxiv 包，这里进行延迟导入。
        """
        import re as _re
        import arxiv  # type: ignore

        if not isinstance(result, arxiv.Result):
            raise TypeError("result 必须是 arxiv.Result 类型")

        # 安全提取字段
        raw_id = result.entry_id.split("/")[-1]
        authors = [a.name for a in result.authors] if result.authors else []
        categories = result.categories if result.categories else []
        pdf_url = getattr(result, "pdf_url", "") or result.entry_id.replace("/abs/", "/pdf/")

        info = {
            "id": raw_id,
            "title": (result.title or "").strip(),
            "authors": authors,
            "categories": categories,
            "timestamp": result.published.strftime("%Y%m%dT%H%M") if result.published else "",
            "summary": result.summary or "",
            "comment": result.comment or "",
            "journal_ref": result.journal_ref or "",
            "doi": result.doi or "",
            "links": [str(link.href) for link in (result.links or []) if hasattr(link, "href")],
            "pdf_url": pdf_url,
        }
        # 清洗 title（与旧逻辑保持一致，可自定义）
        clean_title = _re.sub(r"[^\w\s]", "", info["title"].lower())
        clean_title = _re.sub(r"\s+", "_", clean_title).strip("_")
        info["title"] = clean_title.capitalize() if clean_title else "unknown_title"
        return cls(info)

    # ------------------------------------------------------------------
    # 序列化 / 反序列化（与 DB 行互转）
    # ------------------------------------------------------------------
    def to_db_row(self) -> Tuple[str, str, Optional[bytes], str, str]:
        """转换为可插入 SQLite 的行。"""
        info_json = json.dumps(self.info, ensure_ascii=False)
        pdf_blob = self.pdf_bytes.encode("ascii") if self.pdf_bytes else None
        converted_json = json.dumps(self.converted, ensure_ascii=False) if self.converted else None
        metadata_json = json.dumps(self.metadata, ensure_ascii=False) if self.metadata else None
        return (self.id, info_json, pdf_blob, converted_json, metadata_json)

    @classmethod
    def from_db_row(
        cls, row: Tuple[str, str, Optional[bytes], Optional[str], Optional[str]]
    ) -> "Paper":
        """根据 SQLite 行创建 Paper 对象。"""
        paper_id, info_json, pdf_blob, converted_json, metadata_json = row
        info = json.loads(info_json)
        pdf_base64 = pdf_blob.decode("ascii") if pdf_blob else None
        converted = json.loads(converted_json) if converted_json else {}
        metadata = json.loads(metadata_json) if metadata_json else {}
        # 确保 id 一致
        info["id"] = paper_id
        return cls(info, pdf_base64=pdf_base64, converted=converted, metadata=metadata)

###############################################################################
# ArxivPaperDB
###############################################################################

class ArxivPaperDB:
    """SQLite 后端的 Paper 数据库。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path(__file__).with_suffix(".sqlite"))
        self._init_db()

    # ------------------------------------------------------------------
    def _init_db(self):
        parent = Path(self.db_path).parent
        parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    info_json TEXT NOT NULL,
                    pdf_base64 BLOB,
                    converted_json TEXT,
                    metadata_json TEXT
                )
                """
            )
            conn.commit()
        logger.info(f"ArxivPaperDB 初始化完成，数据库路径: {self.db_path}")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(self, paper: Paper, overwrite: bool = False):
        """插入或更新 Paper。"""
        if not isinstance(paper, Paper):
            raise TypeError("paper 必须是 Paper 类型")
        row = paper.to_db_row()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO papers (id, info_json, pdf_base64, converted_json, metadata_json) VALUES (?, ?, ?, ?, ?)",
                    row,
                )
            except sqlite3.IntegrityError:
                if overwrite:
                    cursor.execute(
                        """
                        UPDATE papers SET info_json=?, pdf_base64=?, converted_json=?, metadata_json=? WHERE id=?
                        """,
                        (row[1], row[2], row[3], row[4], row[0]),
                    )
                else:
                    logger.warning(f"Paper {paper.id} 已存在，跳过插入。设置 overwrite=True 可覆盖。")
            conn.commit()
        logger.info(f"Paper {paper.id} 已写入数据库")

    def delete(self, paper_id: str):
        cid = _clean_arxiv_id(paper_id)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM papers WHERE id=?", (cid,))
            conn.commit()
            if cursor.rowcount:
                logger.info(f"Paper {cid} 已删除")
            else:
                logger.warning(f"Paper {cid} 不存在")

    def get(self, paper_id: str) -> Optional[Paper]:
        cid = _clean_arxiv_id(paper_id)
        logger.info(f"Getting paper {cid} from database")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, info_json, pdf_base64, converted_json, metadata_json FROM papers WHERE id=?",
                (cid,),
            )
            row = cursor.fetchone()
            if row:
                return Paper.from_db_row(row)  # type: ignore[arg-type]
            return None

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_by_date(self, date_str: str) -> List[Paper]:
        """按日期(YYYYMMDD)检索论文。"""
        if not re.match(r"^\d{8}$", date_str):
            raise ValueError("date_str 应为 8 位 YYYYMMDD")
        like_pattern = f'"timestamp": "{date_str}'  # info_json LIKE '%"timestamp": "20240715%'
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, info_json, pdf_base64, converted_json, metadata_json
                FROM papers
                WHERE info_json LIKE ?
                """,
                (f"%{like_pattern}%",),
            )
            rows = cursor.fetchall()
        return [Paper.from_db_row(row) for row in rows]  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM papers")
            return cursor.fetchone()[0]

    def all_ids(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM papers")
            return [row[0] for row in cursor.fetchall()]
