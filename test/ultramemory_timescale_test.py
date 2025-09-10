from __future__ import annotations

from agent_engine.memory.ultra_memory import UltraMemory, UltraMemoryConfig, CollectionSpec, Filter
from agent_engine.memory.ultra_memory.docker_pg import ensure_docker_timescale
from agent_engine.memory.ultra_memory.models import Point


def main() -> None:
    dsn = ensure_docker_timescale()
    if not dsn:
        print("Failed to prepare timescale container")
        return

    # Wire adapter manually for now
    um = UltraMemory(UltraMemoryConfig(backend="timescaledb", mode="timeseries", dsn=dsn))
    # Create a collection representing the metric namespace
    um.create_collection(CollectionSpec(name="cpu", mode="timeseries"))

    metric = "cpu"
    appended = um.append_points(metric, [
        Point(metric=metric, timestamp="2025-01-01T00:00:00Z", tags={"host": "h1"}, fields={"usage": 0.5}),
        Point(metric=metric, timestamp="2025-01-01T00:05:00Z", tags={"host": "h1"}, fields={"usage": 0.6}),
    ])
    print("appended:", appended)

    rows = um.query_timeseries(metric, Filter(expr={"range": {"timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"]}}))
    print("rows:", rows)


if __name__ == "__main__":
    main()


