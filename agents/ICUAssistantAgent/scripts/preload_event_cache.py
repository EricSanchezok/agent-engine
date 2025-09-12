import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import time
from collections import defaultdict

from dotenv import load_dotenv

from agent_engine.agent_logger import AgentLogger
from agent_engine.memory.e_memory import PodEMemory, Record
from agent_engine.utils import get_current_file_dir
from agent_engine.llm_client import QzClient

CONCURRENCY = 32
sem = asyncio.Semaphore(CONCURRENCY)
data_dir = "agents/ICUAssistantAgent/database/icu_patients"

load_dotenv()
logger = AgentLogger(__name__)

event_cache = PodEMemory(
    name="event_cache",
    persist_dir=Path(get_current_file_dir().parent / "database"),
    dimension=3072
)

api_key = os.getenv("INF_API_KEY", "")
if not api_key:
    logger.error("INF_API_KEY not found in environment variables")
    raise ValueError("INF_API_KEY is required")

llm_client = QzClient(api_key=api_key, base_url="http://eric-vpn.cpolar.top/r/eric_qwen3_embedding_8b")
embed_model = "eric-qwen3-embedding-8b"

class EventInfo:
    """Event information for preloading"""
    def __init__(self, event_id: str, event_content: str, patient_id: str):
        self.event_id = event_id
        self.event_content = event_content
        self.patient_id = patient_id

def load_all_events() -> List[EventInfo]:
    """Load all events from all patient files into a single list"""
    all_events = []
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    
    logger.info(f"Loading events from {len(json_files)} patient files")
    
    for file in json_files:
        try:
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)
            patient_id = file.split(".")[0]
            sequence = data.get("sequence", [])
            
            for event in sequence:
                event_id = event.get("id")
                event_content = event.get("event_content", "")
                if event_id and event_content:
                    all_events.append(EventInfo(event_id, event_content, patient_id))
                    
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
    
    # Remove duplicates based on event_id
    unique_events = {}
    for event in all_events:
        if event.event_id not in unique_events:
            unique_events[event.event_id] = event
    
    final_events = list(unique_events.values())
    
    logger.info(f"Total events loaded: {len(all_events)}")
    logger.info(f"Unique events: {len(final_events)}")
    logger.info(f"Duplication ratio: {len(all_events) / len(final_events):.2f}")
    
    return final_events

async def preload_single_event(event_info: EventInfo) -> Tuple[str, bool, str]:
    """Preload a single event with embedding"""
    async with sem:
        try:
            # Check if already exists
            if event_cache.get(event_info.event_id) is not None:
                return event_info.event_id, False, "already_cached"
            
            # Generate embedding
            vector = await llm_client.embedding(event_info.event_content, model_name=embed_model)
            
            # Create and add record to cache
            record = Record(
                id=event_info.event_id,
                vector=vector
            )
            
            event_cache.add(record)
            return event_info.event_id, True, "success"
            
        except Exception as e:
            logger.error(f"Error preloading event {event_info.event_id}: {e}")
            return event_info.event_id, False, f"error: {str(e)}"

async def preload_batch(events_batch: List[EventInfo]) -> Tuple[int, int, int, List[str]]:
    """Preload a batch of events"""
    batch_cached = 0
    batch_preloaded = 0
    batch_errors = 0
    batch_failed_ids = []
    
    # Process batch concurrently
    tasks = [preload_single_event(event) for event in events_batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            batch_errors += 1
            batch_failed_ids.append(events_batch[i].event_id)
        else:
            event_id, success, status = result
            if status == "already_cached":
                batch_cached += 1
            elif status == "success":
                batch_preloaded += 1
            else:
                batch_errors += 1
                batch_failed_ids.append(event_id)
    
    return batch_cached, batch_preloaded, batch_errors, batch_failed_ids

async def concurrent_preload():
    """Concurrent preloading of all events with batch processing"""
    start_time = time.time()
    
    # Get initial cache count
    initial_cache_count = event_cache.count()
    logger.info(f"Initial cache count: {initial_cache_count}")
    
    # Load all events
    all_events = load_all_events()
    if not all_events:
        logger.warning("No events found for preloading")
        return
    
    total_events = len(all_events)
    logger.info(f"Total events to process: {total_events}")
    
    # Calculate how many events need to be processed (not already cached)
    events_to_process = total_events - initial_cache_count
    expected_final_count = total_events  # All events should be cached
    logger.info(f"Events already cached: {initial_cache_count}")
    logger.info(f"Events to process: {events_to_process}")
    logger.info(f"Expected final cache count: {expected_final_count}")
    
    # Process in batches
    BATCH_SIZE = 64
    total_batches = (total_events + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Processing {total_events} events in {total_batches} batches of {BATCH_SIZE}")
    
    # Track progress
    total_cached = 0
    total_preloaded = 0
    total_errors = 0
    processed_events = 0
    failed_event_ids = []  # Store failed event IDs
    
    # Track timing for preloaded events only
    preload_start_time = None
    total_preload_time = 0
    
    batch_start_time = start_time
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total_events)
        events_batch = all_events[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} "
                   f"(events {batch_start + 1}-{batch_end})")
        
        # Process batch
        batch_start_time = time.time()
        batch_cached, batch_preloaded, batch_errors, batch_failed_ids = await preload_batch(events_batch)
        
        # Update totals
        total_cached += batch_cached
        total_preloaded += batch_preloaded
        total_errors += batch_errors
        processed_events += len(events_batch)
        failed_event_ids.extend(batch_failed_ids)
        
        # Track timing for preloaded events only
        if batch_preloaded > 0:
            if preload_start_time is None:
                preload_start_time = batch_start_time
            total_preload_time += time.time() - batch_start_time
        
        # Calculate batch timing and rate (only for actual preloaded events)
        batch_elapsed = time.time() - batch_start_time
        batch_rate = batch_preloaded / batch_elapsed if batch_elapsed > 0 and batch_preloaded > 0 else 0
        
        # Calculate overall progress
        current_cache_count = event_cache.count()
        new_events_added = current_cache_count - initial_cache_count
        progress_percentage = (new_events_added / events_to_process * 100) if events_to_process > 0 else 100
        
        # Calculate preload rate (only for events that were actually preloaded)
        preload_rate = total_preloaded / total_preload_time if total_preload_time > 0 and total_preloaded > 0 else 0
        
        logger.info(f"Batch {batch_idx + 1} completed in {batch_elapsed:.2f}s")
        if batch_preloaded > 0:
            logger.info(f"Batch preload rate: {batch_rate:.1f} events/sec")
        logger.info(f"Batch results - Cached: {batch_cached}, Preloaded: {batch_preloaded}, Errors: {batch_errors}")
        logger.info(f"Overall progress: {new_events_added}/{events_to_process} events processed "
                   f"({progress_percentage:.1f}%) - Total cached: {current_cache_count}/{expected_final_count}")
        if total_preloaded > 0:
            logger.info(f"Preload rate: {preload_rate:.1f} events/sec")
        logger.info("-" * 60)
    
    # Save failed event IDs to JSON file
    if failed_event_ids:
        failed_events_file = Path(get_current_file_dir().parent / "database" / "failed_events.json")
        failed_events_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_failed_events": len(failed_event_ids),
            "failed_event_ids": failed_event_ids
        }
        
        try:
            with open(failed_events_file, "w", encoding="utf-8") as f:
                json.dump(failed_events_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Failed event IDs saved to: {failed_events_file}")
        except Exception as e:
            logger.error(f"Failed to save failed events file: {e}")
    
    # Final statistics
    final_cache_count = event_cache.count()
    total_elapsed = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("PRELOAD COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total events processed: {processed_events}")
    logger.info(f"Events already cached: {total_cached}")
    logger.info(f"Events preloaded: {total_preloaded}")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Initial cache count: {initial_cache_count}")
    logger.info(f"Final cache count: {final_cache_count}")
    logger.info(f"New events added: {final_cache_count - initial_cache_count}")
    logger.info(f"Total time: {total_elapsed:.2f} seconds")
    if total_preloaded > 0:
        logger.info(f"Preload rate: {total_preloaded / total_preload_time:.1f} events/second")
    logger.info(f"Success rate: {(processed_events - total_errors) / processed_events * 100:.1f}%")
    if failed_event_ids:
        logger.info(f"Failed events saved to: {failed_events_file}")

if __name__ == "__main__":
    # Configuration: manually set the action here
    # Options: "preload"
    ACTION = "preload"  # Change this to run preloading
    
    if ACTION == "preload":
        asyncio.run(concurrent_preload())
    else:
        logger.error(f"Invalid ACTION: {ACTION}. Only 'preload' is supported.")