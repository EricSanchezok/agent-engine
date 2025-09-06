from __future__ import annotations

import asyncio
import math
from typing import Dict, Any, List, Optional, Tuple

from agent_engine.agent_logger import AgentLogger

from agents.ICUDataIngestionAgent.agent import ICUDataIngestionAgent
from agents.ICUMemoryAgent.agent import ICUMemoryAgent


PATIENT_ID = "1125112810"
NUM_EVENTS = 100


async def ingest_n_events(ingestion: ICUDataIngestionAgent, memory: ICUMemoryAgent, target_n: int) -> Tuple[str, int]:
    logger = AgentLogger("ICUMemoryAgentTest")
    patient_json_path = f"database/icu_raw/{PATIENT_ID}.json"
    ingestion.load_patient(patient_json_path)
    patient_id = ingestion.patient_id or PATIENT_ID

    # Reset memory storage for a clean test
    memory.delete_patient_memory(patient_id)

    total = 0
    while total < target_n:
        batch = await ingestion.update()
        if not batch:
            logger.info("No more events from ingestion; stopping early at total=%s", total)
            break
        # Trim batch if it would exceed target
        remain = target_n - total
        if len(batch) > remain:
            batch = batch[:remain]
        ids = await memory.add_events(patient_id, batch)
        total += len(ids)
        logger.info("Ingested %s/%s events", total, target_n)

    return patient_id, total


def choose_existing_subtype(events: List[Dict[str, Any]]) -> Optional[str]:
    counts: Dict[str, int] = {}
    for e in events:
        st = e.get("sub_type") or (e.get("metadata") or {}).get("sub_type")
        if st:
            counts[st] = counts.get(st, 0) + 1
    if not counts:
        return None
    # pick the most frequent subtype
    return max(counts.items(), key=lambda kv: kv[1])[0]


def get_earliest_latest(events: List[Dict[str, Any]]) -> Tuple[str, str]:
    # events are dicts with timestamp strings
    sorted_ev = sorted(events, key=lambda x: x.get("timestamp") or "")
    return (sorted_ev[0]["timestamp"], sorted_ev[-1]["timestamp"]) if sorted_ev else ("", "")


async def main() -> int:
    logger = AgentLogger("ICUMemoryAgentTest")
    ingestion = ICUDataIngestionAgent()
    memory = ICUMemoryAgent()

    patient_id, written = await ingest_n_events(ingestion, memory, NUM_EVENTS)
    assert written > 0, "No events ingested for testing"

    # Baseline: pull the latest N events and derive a subtype that surely exists
    latest_n = memory.get_recent_events(patient_id, n=NUM_EVENTS)
    assert latest_n, "Expected some recent events"
    existing_subtype = choose_existing_subtype(latest_n)
    assert existing_subtype, "Could not find an existing sub_type among recent events"
    logger.info("Chosen sub_type for filtering: %s", existing_subtype)

    # 1) Test get_recent_events with sub_types
    recent_filtered = memory.get_recent_events(patient_id, n=NUM_EVENTS, sub_types=[existing_subtype])
    assert all(ev.get("sub_type") == existing_subtype for ev in recent_filtered), "Sub-type filter failed for get_recent_events"
    # Verify sorting order (descending timestamps): first >= last
    if len(recent_filtered) >= 2:
        assert (recent_filtered[0]["timestamp"] >= recent_filtered[-1]["timestamp"]) , "Order failed for get_recent_events"
    logger.info("get_recent_events passed with %s results", len(recent_filtered))

    # 2) Test get_events_within_hours covering the entire window, then sub_types filter
    earliest_ts, latest_ts = get_earliest_latest(latest_n)
    # Compute hours to cover full range
    # Use ICUMemoryAgent._to_datetime via a small helper instance
    earliest_dt = memory._to_datetime(earliest_ts)  # type: ignore[attr-defined]
    latest_dt = memory._to_datetime(latest_ts)  # type: ignore[attr-defined]
    assert earliest_dt is not None and latest_dt is not None
    span_hours = max(1, int(math.ceil((latest_dt - earliest_dt).total_seconds() / 3600.0)) + 1)
    within_all = memory.get_events_within_hours(patient_id, ref_time=latest_ts, hours=span_hours, sub_types=[existing_subtype])
    # Expected count: how many of latest_n belong to that sub_type
    expected_count = sum(1 for e in latest_n if e.get("sub_type") == existing_subtype)
    assert len(within_all) == expected_count, (f"get_events_within_hours count mismatch: got {len(within_all)} expected {expected_count}")
    # Verify ascending order
    if len(within_all) >= 2:
        assert within_all[0]["timestamp"] <= within_all[-1]["timestamp"], "Order failed for get_events_within_hours"
    logger.info("get_events_within_hours passed with %s results", len(within_all))

    # 3) Test get_events_between with a middle window and sub_types filter
    # Get a sorted list by timestamp ascending
    sorted_all = sorted(latest_n, key=lambda x: x.get("timestamp") or "")
    if len(sorted_all) >= 10:
        start_idx = max(0, len(sorted_all) // 4)
        end_idx = min(len(sorted_all) - 1, (len(sorted_all) * 3) // 4)
    else:
        start_idx = 0
        end_idx = len(sorted_all) - 1
    range_start = sorted_all[start_idx]["timestamp"]
    range_end = sorted_all[end_idx]["timestamp"]

    between_filtered = memory.get_events_between(
        patient_id,
        start_time=range_start,
        end_time=range_end,
        sub_types=[existing_subtype],
    )
    # Expected subset from sorted_all in the range with subtype
    expected_ids = {
        e["id"]
        for e in sorted_all
        if (range_start <= (e.get("timestamp") or "") <= range_end) and (e.get("sub_type") == existing_subtype)
    }
    got_ids = {e["id"] for e in between_filtered}
    assert got_ids == expected_ids, "get_events_between ids mismatch"
    if len(between_filtered) >= 2:
        assert between_filtered[0]["timestamp"] <= between_filtered[-1]["timestamp"], "Order failed for get_events_between"
    logger.info("get_events_between passed with %s results", len(between_filtered))

    # 4) Test query_search with and without sub_types
    # Choose a query content from an event of the chosen subtype
    candidate = next((e for e in latest_n if e.get("sub_type") == existing_subtype and e.get("content")), None)
    if candidate is None:
        candidate = latest_n[0]
    query_text = candidate.get("content") or ""
    search_res = await memory.query_search(
        patient_id,
        query=query_text,
        top_k=5,
        threshold=0.0,
        include_vectors=True,
        sub_types=[existing_subtype],
        ef_search=None,
        near_duplicate_delta=0.0,
    )
    assert search_res, "query_search returned no results"
    # Expect at least one result with the same id
    ids = [r.get("id") for r in search_res]
    assert candidate.get("id") in ids, "query_search did not return the expected id"
    # Vector should be present when include_vectors=True
    assert "vector" in search_res[0], "Vector not included in query_search result"
    logger.info("query_search passed with %s results", len(search_res))

    print({
        "recent_filtered": len(recent_filtered),
        "within_all": len(within_all),
        "between_filtered": len(between_filtered),
        "search_results": len(search_res),
        "sub_type": existing_subtype,
    })

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


