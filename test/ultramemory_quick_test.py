from __future__ import annotations

from agent_engine.memory.ultra_memory import UltraMemory, UltraMemoryConfig, CollectionSpec, Record, Filter


def main() -> None:
    cfg = UltraMemoryConfig(
        mode="mixed",
        backend="postgres_pgvector",
        dsn="postgresql://placeholder",  # not used in in-memory placeholder
    )
    um = UltraMemory(cfg)
    um.create_collection(CollectionSpec(name="papers", mode="vector", vector_dimensions={"text_vec": 3}))

    ids = um.upsert(
        "papers",
        [
            Record(id="a1", attributes={"categories": "cs.AI"}, content="alpha", vector=[1, 0, 0]),
            Record(id="b2", attributes={"categories": "cs.LG"}, content="bravo", vector=[0.9, 0.1, 0]),
            Record(id="c3", attributes={"categories": "math"}, content="charlie", vector=[0, 1, 0]),
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


