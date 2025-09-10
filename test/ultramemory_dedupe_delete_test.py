from __future__ import annotations

from agent_engine.memory.ultra_memory import UltraMemory, UltraMemoryConfig, CollectionSpec, Record, Filter
from agent_engine.memory.ultra_memory.docker_pg import ensure_docker_postgres


def main() -> None:
    dsn = ensure_docker_postgres()
    um = UltraMemory(UltraMemoryConfig(backend="postgres_pgvector", dsn=dsn))
    um.create_collection(CollectionSpec(name="dedupe_demo", mode="vector", vector_dimensions={"text_vec": 3}))

    # 1) dedupe by id (explicit)
    ids = um.upsert("dedupe_demo", [Record(id="u1", attributes={"k": "v1"}, content="one", vector=[1, 0, 0])])
    print("insert u1:", ids)
    ids = um.upsert("dedupe_demo", [Record(id="u1", attributes={"k": "v2"}, content="one-updated", vector=[1, 0.1, 0])])
    print("update u1:", ids)
    print("after update:", um.query("dedupe_demo", Filter(expr={"eq": ["id", "u1"]})))

    # 2) dedupe by attributes key (dedupe_key)
    ids = um.upsert("dedupe_demo", [Record(id=None, attributes={"biz": "ord-100", "k": "v"}, content="order", vector=[0, 1, 0])], dedupe_key="biz")
    print("insert ord-100", ids)
    ids = um.upsert("dedupe_demo", [Record(id=None, attributes={"biz": "ord-100", "k": "v-up"}, content="order-up", vector=[0, 1, 0.1])], dedupe_key="biz")
    print("update ord-100", ids)
    print("after ord-100 update:", um.query("dedupe_demo", Filter(expr={"eq": ["biz", "ord-100"]})))

    # 3) delete by id
    deleted = um.delete_by_id("dedupe_demo", "u1")
    print("deleted u1 count:", deleted)
    print("after delete u1:", um.query("dedupe_demo", Filter(expr={"eq": ["id", "u1"]})))

    # 4) delete by filter
    deleted = um.delete_by_filter("dedupe_demo", Filter(expr={"eq": ["biz", "ord-100"]}))
    print("deleted ord-100 count:", deleted)
    print("after filter delete:", um.query("dedupe_demo", Filter(expr={"eq": ["biz", "ord-100"]})))


if __name__ == "__main__":
    main()


