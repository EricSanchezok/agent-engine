from __future__ import annotations

from agent_engine.memory.ultra_memory import UltraMemory, UltraMemoryConfig, CollectionSpec, Record, Filter
from agent_engine.memory.ultra_memory.docker_pg import ensure_docker_postgres, ensure_docker_timescale


def main() -> None:
    pg_dsn = ensure_docker_postgres() or ""
    ts_dsn = ensure_docker_timescale() or ""
    if not (pg_dsn and ts_dsn):
        print("Failed to prepare containers")
        return

    # 简化配置后，推荐分别实例化两个 UltraMemory：
    um_pg = UltraMemory(UltraMemoryConfig(backend="postgres_pgvector", dsn=pg_dsn))
    um_ts = UltraMemory(UltraMemoryConfig(backend="timescaledb", dsn=ts_dsn))

    # vector collection on pg
    um_pg.create_collection(CollectionSpec(name="docs", mode="vector", vector_dimensions={"text_vec": 3}))
    um_pg.upsert("docs", [Record(id="x1", attributes={"k": "v"}, content="hello", vector=[1, 0, 0])])
    print("vector search:", um_pg.search_vectors("docs", [1, 0, 0], top_k=1))

    # timeseries collection on ts
    um_ts.create_collection(CollectionSpec(name="cpu", mode="timeseries"))
    from agent_engine.memory.ultra_memory.models import Point

    um_ts.append_points("cpu", [Point(metric="cpu", timestamp="2025-01-01T00:00:00Z", tags={}, fields={"u": 0.5})])
    print("ts query:", um_ts.query_timeseries("cpu", Filter(expr={"range": {"timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"]}})))


if __name__ == "__main__":
    main()


