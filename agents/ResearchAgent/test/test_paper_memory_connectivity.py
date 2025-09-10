from __future__ import annotations

import os
from typing import Tuple, List

from agent_engine.agent_logger.agent_logger import AgentLogger

from agents.ResearchAgent.config import DATABASE_URL
from agents.ResearchAgent.paper_memory import PaperMemory, PaperMemoryConfig
from agent_engine.memory.ultra_memory.adapters.postgres_pgvector import PostgresPgvectorAdapter


logger = AgentLogger("PaperMemoryConnectivityTest")


def _parse_host_port(url: str) -> Tuple[str, int]:
    s = str(url).strip()
    if not s:
        raise ValueError("DATABASE_URL is empty")
    if "://" in s:
        try:
            from urllib.parse import urlparse

            u = urlparse(s)
            host = u.hostname or ""
            port = int(u.port or 5432)
            if not host:
                raise ValueError("Hostname missing in DATABASE_URL")
            return host, port
        except Exception as e:
            raise ValueError(f"Failed to parse DATABASE_URL as URI: {e}")
    # Fallback: host:port
    if ":" in s:
        host, port_s = s.rsplit(":", 1)
        host = host.strip()
        try:
            port = int(port_s)
        except Exception as e:
            raise ValueError(f"Invalid port in DATABASE_URL: {e}")
        if not host:
            raise ValueError("Hostname missing in DATABASE_URL")
        return host, port
    # Host only
    return s, 5432


def main() -> None:
    # Credentials: set here if needed. You can also override via env PG_USER/PG_PASSWORD.
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "postgres777")

    host, port = _parse_host_port(DATABASE_URL)
    dsn_template = f"postgresql://{user}:{password}@{host}:{port}/{{db}}"

    logger.info(f"Prepared DSN template (masked): postgresql://{user}:***@{host}:{port}/{{db}}")

    # Segments to test: adjust as needed
    segments: List[str] = [
        "2022H1", "2022H2",
        "2023H1", "2023H2",
        "2024H1", "2024H2",
        "2025H1",
    ]

    pm = PaperMemory(PaperMemoryConfig(
        dsn_template=dsn_template,
        collection_name="papers",
        vector_field="text_vec",
        vector_dim=3072,
        metric="cosine",
        index_params={"lists": 50},
    ))

    ok = 0
    for seg in segments:
        try:
            # Resolve DSN actually used
            dsn_used = pm._dsn_for_segment(seg)  # type: ignore[attr-defined]
            masked = dsn_used
            try:
                at = dsn_used.find('@')
                if at > 0 and '://' in dsn_used:
                    scheme = dsn_used.split('://', 1)[0]
                    user_part = dsn_used.split('://', 1)[1].split('@', 1)[0].split(':', 1)[0]
                    hostpart = dsn_used.split('@', 1)[1]
                    masked = f"{scheme}://{user_part}:***@{hostpart}"
            except Exception:
                masked = dsn_used
            logger.info(f"Connecting segment {seg} using DSN: {masked}")

            um = pm._get_segment_um(seg)  # type: ignore[attr-defined]
            adapter = getattr(um, "adapter", None)
            if not isinstance(adapter, PostgresPgvectorAdapter):
                logger.error(f"Segment {seg}: adapter is not PostgresPgvectorAdapter")
                continue
            conn = getattr(adapter, "_conn", None)
            if conn is None:
                logger.error(f"Segment {seg}: connection not established (adapter fell back to in-memory)")
                continue

            # Issue a simple query
            cur = conn.cursor()
            try:
                cur.execute("SELECT 1")
                row = cur.fetchone()
                logger.info(f"Segment {seg}: SELECT 1 -> {row}")
                ok += 1
            finally:
                cur.close()
        except Exception as e:
            logger.error(f"Segment {seg}: connectivity test failed: {e}")

    logger.info(f"Connectivity OK segments: {ok}/{len(segments)}")


if __name__ == "__main__":
    main()


