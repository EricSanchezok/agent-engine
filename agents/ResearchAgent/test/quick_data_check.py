from __future__ import annotations

import time
from typing import Dict, List

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ResearchAgent.config import PAPER_DSN_TEMPLATE
from agents.ResearchAgent.paper_memory import PaperMemory, PaperMemoryConfig


logger = AgentLogger("QuickDataCheck")


def quick_check() -> None:
    """Quick check to see if data exists in remote database."""
    logger.info("Starting quick data check...")
    
    # Check configuration
    if not PAPER_DSN_TEMPLATE or "USER:PASS@HOST:PORT" in PAPER_DSN_TEMPLATE:
        logger.error("PAPER_DSN_TEMPLATE is not properly configured")
        logger.error("Please check agents/ResearchAgent/config.py")
        return
    
    # Initialize PaperMemory
    pm = PaperMemory(PaperMemoryConfig(
        dsn_template=PAPER_DSN_TEMPLATE,
        collection_name="papers",
        vector_field="text_vec",
        vector_dim=3072,
        metric="cosine",
        index_params={"lists": 100},
    ))
    
    # Segments to check
    segments = [
        "2022H1", "2022H2",
        "2023H1", "2023H2", 
        "2024H1", "2024H2",
        "2025H1", "2025H2",
        "undated"
    ]
    
    logger.info("Checking data in each segment...")
    logger.info("-" * 50)
    
    total_records = 0
    segments_with_data = 0
    
    for seg in segments:
        try:
            logger.info(f"Checking segment: {seg}")
            
            # Get UltraMemory for this segment
            um = pm._get_segment_um(seg)
            
            # Count records by querying with empty filter
            from agent_engine.memory.ultra_memory import Filter
            records = um.query("papers", Filter())
            count = len(records)
            total_records += count
            
            if count > 0:
                segments_with_data += 1
                logger.info(f"  ✓ {count:,} records found")
                
                # Get a sample record to verify content
                from agent_engine.memory.ultra_memory import Filter
                sample_records = um.query("papers", Filter(limit=1))
                if sample_records:
                    record_data = sample_records[0]
                    # Handle both Record objects and dict results
                    if hasattr(record_data, 'content'):
                        record = record_data
                    else:
                        # Convert dict to Record-like object
                        class RecordLike:
                            def __init__(self, data):
                                self.id = data.get('id')
                                self.content = data.get('content')
                                self.vector = data.get('vector')
                                self.attributes = data.get('attributes', {})
                                self.timestamp = data.get('timestamp')
                        record = RecordLike(record_data)
                    
                    has_content = bool(record.content and len(str(record.content).strip()) > 0)
                    has_vector = bool(record.vector and len(record.vector) == 3072)
                    has_attributes = bool(record.attributes)
                    has_timestamp = bool(record.timestamp)
                    
                    logger.info(f"    Sample record ID: {record.id}")
                    logger.info(f"    Has content: {has_content}")
                    logger.info(f"    Has vector: {has_vector}")
                    logger.info(f"    Has attributes: {has_attributes}")
                    logger.info(f"    Has timestamp: {has_timestamp}")
                    
                    if record.attributes:
                        logger.info(f"    Attribute keys: {list(record.attributes.keys())[:5]}")
            else:
                logger.info(f"  ✗ No records found")
                
        except Exception as e:
            logger.error(f"  ✗ Error checking segment {seg}: {e}")
    
    logger.info("-" * 50)
    logger.info("SUMMARY:")
    logger.info(f"  Segments with data: {segments_with_data}/{len(segments)}")
    logger.info(f"  Total records: {total_records:,}")
    
    if total_records > 0:
        logger.info("✓ Data migration appears to be successful!")
        logger.info("  Your migration script has transferred data to the remote database.")
    else:
        logger.info("✗ No data found in remote database.")
        logger.info("  The migration may still be running or may have failed.")
    
    logger.info("-" * 50)


if __name__ == "__main__":
    quick_check()
