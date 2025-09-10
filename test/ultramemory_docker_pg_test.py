from __future__ import annotations

from agent_engine.memory.ultra_memory import UltraMemory, UltraMemoryConfig, CollectionSpec, Record, Filter
from agent_engine.memory.ultra_memory.docker_pg import ensure_docker_postgres, DockerPGConfig


def main() -> None:
    dsn = ensure_docker_postgres(DockerPGConfig())
    if not dsn:
        print("Failed to prepare dockerized PostgreSQL. See logs.")
        return

    cfg = UltraMemoryConfig(mode="mixed", backend="postgres_pgvector", dsn=dsn)
    um = UltraMemory(cfg)
    um.create_collection(CollectionSpec(name="papers", mode="vector", vector_dimensions={"text_vec": 3}, metric="cosine", index_params={"lists": 50}))

    ids = um.upsert(
        "papers",
        [
            Record(id="d1", attributes={"categories": "cs.AI"}, content="delta", vector=[1, 0, 0]),
            Record(id="e2", attributes={"categories": "cs.LG"}, content="echo", vector=[0.9, 0.1, 0]),
        ],
    )
    print("upsert ids:", ids)

    rows = um.query("papers", Filter(expr={"eq": ["categories", "cs.LG"]}, limit=10))
    print("query rows:", rows)

    res = um.search_vectors("papers", [1, 0, 0], top_k=2, flt=Filter(expr={"in": ["categories", ["cs.AI", "cs.LG"]]}))
    print("search:", res)

    print("stats:", um.stats())
    print("health:", um.health())


if __name__ == "__main__":
    main()


