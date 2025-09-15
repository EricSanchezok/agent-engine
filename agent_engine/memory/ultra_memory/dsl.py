from __future__ import annotations
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



from typing import Any, Dict, List, Tuple

from .models import Filter


def _json_path(path: str) -> str:
    # Map dotted path to PostgreSQL JSONB accessors: attributes #>> '{a,b}'
    if path in ("id", "content", "ts", "timestamp"):
        # Prefer ts as the timestamp column name; allow timestamp alias
        col = "ts" if path in ("ts", "timestamp") else path
        return f'"{col}"'
    parts = [p.strip() for p in path.split(".") if p.strip()]
    if not parts:
        return 'attributes'
    inner = ",".join(parts)
    return f"attributes #>> '{{{inner}}}'"


def _build_expr(expr: Dict[str, Any], params: List[Any]) -> str:
    if not expr:
        return "TRUE"
    if "and" in expr:
        subs = expr["and"] or []
        return "(" + " AND ".join([_build_expr(s, params) for s in subs]) + ")"
    if "or" in expr:
        subs = expr["or"] or []
        return "(" + " OR ".join([_build_expr(s, params) for s in subs]) + ")"
    if "not" in expr:
        return "(NOT " + _build_expr(expr["not"], params) + ")"
    if "eq" in expr:
        field, value = expr["eq"][0], expr["eq"][1]
        lhs = _json_path(str(field))
        params.append(value)
        return f"{lhs} = %s"
    if "in" in expr:
        field, arr = expr["in"][0], expr["in"][1]
        lhs = _json_path(str(field))
        params.append(list(arr))
        return f"{lhs} = ANY(%s::text[])"
    if "range" in expr:
        obj = expr["range"]
        # {"range": {"field": [lo, hi]}}
        if isinstance(obj, dict):
            ((field, bounds),) = obj.items()
            lhs = _json_path(str(field))
            lo, hi = (bounds[0], bounds[1])
            params.extend([lo, hi])
            return f"{lhs} BETWEEN %s AND %s"
    if "like" in expr:
        field, pattern = expr["like"][0], expr["like"][1]
        lhs = _json_path(str(field))
        params.append(pattern)
        return f"{lhs} LIKE %s"
    if "contains" in expr:
        # JSON containment on attributes
        val = expr["contains"]
        params.append(val)
        return "attributes @> %s::jsonb"
    # Fallback true
    return "TRUE"


def build_select(table: str, columns: List[str], flt: Filter) -> Tuple[str, List[Any]]:
    params: List[Any] = []
    where_sql = _build_expr(flt.expr or {}, params)
    cols = ", ".join([f'"{c}"' for c in columns])
    sql = f"SELECT {cols} FROM \"{table}\" WHERE {where_sql}"
    if flt.order_by:
        order = []
        for col, direc in flt.order_by:
            order.append(f'{_json_path(col)} {"DESC" if str(direc).lower()=="desc" else "ASC"}')
        sql += " ORDER BY " + ", ".join(order)
    if flt.limit is not None:
        sql += " LIMIT %s"
        params.append(int(flt.limit))
    if flt.offset is not None:
        sql += " OFFSET %s"
        params.append(int(flt.offset))
    return sql, params


def build_where(flt: Filter) -> Tuple[str, List[Any]]:
    """Return only WHERE clause SQL (without the 'WHERE' keyword) and params."""
    params: List[Any] = []
    where_sql = _build_expr(flt.expr or {}, params)
    return where_sql, params


def build_count(table: str, flt: Filter) -> Tuple[str, List[Any]]:
    params: List[Any] = []
    where_sql = _build_expr(flt.expr or {}, params)
    sql = f'SELECT COUNT(*) FROM "{table}" WHERE {where_sql}'
    return sql, params


def build_exists(table: str, flt: Filter) -> Tuple[str, List[Any]]:
    params: List[Any] = []
    where_sql = _build_expr(flt.expr or {}, params)
    sql = f'SELECT 1 FROM "{table}" WHERE {where_sql} LIMIT 1'
    return sql, params


