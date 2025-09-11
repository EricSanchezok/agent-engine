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
from agent_engine.llm_client import AzureClient

CONCURRENCY = 128
sem = asyncio.Semaphore(CONCURRENCY)
data_dir = "agents/ICUAssistantAgent/database/icu_patients"

load_dotenv()
logger = AgentLogger(__name__)

event_cache = PodEMemory(
    name="event_cache",
    persist_dir=Path(get_current_file_dir().parent / "database"),
    dimension=3072
)

api_key = os.getenv("AZURE_API_KEY", "")
if not api_key:
    logger.error("AZURE_API_KEY not found in environment variables")
    raise ValueError("AZURE_API_KEY is required")

llm_client = AzureClient(api_key=api_key)
embed_model = "text-embedding-3-large"

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

async def concurrent_preload():
    """Concurrent preloading of all events with detailed progress logging"""
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
    
    # Create tasks for concurrent execution
    tasks = [preload_single_event(event) for event in all_events]
    
    logger.info(f"Starting concurrent preload with {CONCURRENCY} concurrent workers")
    logger.info("Progress will be logged every 50 completed events")
    
    # Execute all tasks concurrently and track progress
    completed_count = 0
    cached_count = 0
    preloaded_count = 0
    error_count = 0
    last_log_time = time.time()
    
    # Process results as they complete
    for task in asyncio.as_completed(tasks):
        event_id, success, status = await task
        completed_count += 1
        
        if status == "already_cached":
            cached_count += 1
        elif status == "success":
            preloaded_count += 1
        else:
            error_count += 1
        
        # Log progress every 50 events or every 10 seconds
        current_time = time.time()
        if completed_count % 50 == 0 or (current_time - last_log_time) >= 10:
            elapsed = current_time - start_time
            rate = completed_count / elapsed if elapsed > 0 else 0
            
            logger.info(f"Progress: {completed_count}/{total_events} ({completed_count/total_events*100:.1f}%) "
                       f"- Cached: {cached_count}, Preloaded: {preloaded_count}, Errors: {error_count} "
                       f"- Rate: {rate:.1f} events/sec")
            
            last_log_time = current_time
    
    # Final statistics
    final_cache_count = event_cache.count()
    total_elapsed = time.time() - start_time
    overall_rate = completed_count / total_elapsed if total_elapsed > 0 else 0
    
    logger.info("=" * 80)
    logger.info("PRELOAD COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total events processed: {completed_count}")
    logger.info(f"Events already cached: {cached_count}")
    logger.info(f"Events preloaded: {preloaded_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Initial cache count: {initial_cache_count}")
    logger.info(f"Final cache count: {final_cache_count}")
    logger.info(f"New events added: {final_cache_count - initial_cache_count}")
    logger.info(f"Total time: {total_elapsed:.2f} seconds")
    logger.info(f"Average rate: {overall_rate:.1f} events/second")
    logger.info(f"Success rate: {(completed_count - error_count) / completed_count * 100:.1f}%")

if __name__ == "__main__":
    # Configuration: manually set the action here
    # Options: "preload"
    ACTION = "preload"  # Change this to run preloading
    
    if ACTION == "preload":
        asyncio.run(concurrent_preload())
    else:
        logger.error(f"Invalid ACTION: {ACTION}. Only 'preload' is supported.")