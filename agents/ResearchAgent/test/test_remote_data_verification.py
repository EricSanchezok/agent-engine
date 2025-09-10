from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ResearchAgent.config import PAPER_DSN_TEMPLATE
from agents.ResearchAgent.paper_memory import PaperMemory, PaperMemoryConfig
from agent_engine.memory.ultra_memory import Record, Filter
from agent_engine.memory.ultra_memory.adapters.postgres_pgvector import PostgresPgvectorAdapter


logger = AgentLogger("RemoteDataVerification")


class RemoteDataVerifier:
    """Test script to verify data migration to remote database."""
    
    def __init__(self):
        self.dsn_template = PAPER_DSN_TEMPLATE
        self.collection_name = "papers"
        self.vector_field = "text_vec"
        self.vector_dim = 3072
        self.metric = "cosine"
        
        # Segments to test (matching migration script)
        self.segments = [
            "2022H1", "2022H2",
            "2023H1", "2023H2", 
            "2024H1", "2024H2",
            "2025H1", "2025H2",
            "undated"
        ]
        
        self.pm = PaperMemory(PaperMemoryConfig(
            dsn_template=self.dsn_template,
            collection_name=self.collection_name,
            vector_field=self.vector_field,
            vector_dim=self.vector_dim,
            metric=self.metric,
            index_params={"lists": 100},
        ))
        
        self.results: Dict[str, Dict] = {}
    
    def test_connectivity(self) -> Dict[str, bool]:
        """Test connectivity to each segment database."""
        logger.info("Testing connectivity to remote databases...")
        connectivity_results = {}
        
        for seg in self.segments:
            try:
                logger.info(f"Testing connectivity to segment: {seg}")
                
                # Get the UltraMemory instance for this segment
                um = self.pm._get_segment_um(seg)
                adapter = getattr(um, "adapter", None)
                
                if not isinstance(adapter, PostgresPgvectorAdapter):
                    logger.error(f"Segment {seg}: adapter is not PostgresPgvectorAdapter")
                    connectivity_results[seg] = False
                    continue
                
                conn = getattr(adapter, "_conn", None)
                if conn is None:
                    logger.error(f"Segment {seg}: connection not established")
                    connectivity_results[seg] = False
                    continue
                
                # Test basic connectivity
                cur = conn.cursor()
                try:
                    cur.execute("SELECT 1")
                    row = cur.fetchone()
                    if row and row[0] == 1:
                        logger.info(f"Segment {seg}: Connectivity OK")
                        connectivity_results[seg] = True
                    else:
                        logger.error(f"Segment {seg}: Unexpected query result: {row}")
                        connectivity_results[seg] = False
                finally:
                    cur.close()
                    
            except Exception as e:
                logger.error(f"Segment {seg}: Connectivity test failed: {e}")
                connectivity_results[seg] = False
        
        return connectivity_results
    
    def count_records(self) -> Dict[str, int]:
        """Count records in each segment."""
        logger.info("Counting records in each segment...")
        record_counts = {}
        
        for seg in self.segments:
            try:
                logger.info(f"Counting records in segment: {seg}")
                
                # Get records count by querying with empty filter
                um = self.pm._get_segment_um(seg)
                from agent_engine.memory.ultra_memory import Filter
                records = um.query("papers", Filter())
                count = len(records)
                record_counts[seg] = count
                logger.info(f"Segment {seg}: {count} records")
                
            except Exception as e:
                logger.error(f"Segment {seg}: Failed to count records: {e}")
                record_counts[seg] = 0
        
        return record_counts
    
    def test_vector_search(self, sample_size: int = 5) -> Dict[str, bool]:
        """Test vector search functionality on each segment."""
        logger.info("Testing vector search functionality...")
        search_results = {}
        
        # Create a dummy vector for testing
        dummy_vector = [0.1] * self.vector_dim
        
        for seg in self.segments:
            try:
                logger.info(f"Testing vector search in segment: {seg}")
                
                um = self.pm._get_segment_um(seg)
                
                # Try to search with the dummy vector
                results = um.search_records(
                    vector=dummy_vector,
                    limit=sample_size,
                    metric=self.metric
                )
                
                if results is not None:
                    logger.info(f"Segment {seg}: Vector search OK, found {len(results)} results")
                    search_results[seg] = True
                else:
                    logger.warning(f"Segment {seg}: Vector search returned None")
                    search_results[seg] = False
                    
            except Exception as e:
                logger.error(f"Segment {seg}: Vector search failed: {e}")
                search_results[seg] = False
        
        return search_results
    
    def get_sample_records(self, segment: str, limit: int = 3) -> List[Record]:
        """Get sample records from a specific segment."""
        try:
            um = self.pm._get_segment_um(segment)
            records = um.get_records(limit=limit)
            return records if records else []
        except Exception as e:
            logger.error(f"Failed to get sample records from {segment}: {e}")
            return []
    
    def analyze_record_content(self, segment: str) -> Dict[str, any]:
        """Analyze content of records in a segment."""
        logger.info(f"Analyzing record content in segment: {segment}")
        
        try:
            sample_records = self.get_sample_records(segment, limit=5)
            if not sample_records:
                return {"error": "No records found"}
            
            analysis = {
                "total_samples": len(sample_records),
                "has_content": 0,
                "has_vector": 0,
                "has_attributes": 0,
                "has_timestamp": 0,
                "content_lengths": [],
                "attribute_keys": set(),
                "sample_ids": []
            }
            
            for record in sample_records:
                analysis["sample_ids"].append(record.id)
                
                if record.content and len(record.content.strip()) > 0:
                    analysis["has_content"] += 1
                    analysis["content_lengths"].append(len(record.content))
                
                if record.vector and len(record.vector) == self.vector_dim:
                    analysis["has_vector"] += 1
                
                if record.attributes:
                    analysis["has_attributes"] += 1
                    analysis["attribute_keys"].update(record.attributes.keys())
                
                if record.timestamp:
                    analysis["has_timestamp"] += 1
            
            # Convert set to list for JSON serialization
            analysis["attribute_keys"] = list(analysis["attribute_keys"])
            analysis["avg_content_length"] = (
                sum(analysis["content_lengths"]) / len(analysis["content_lengths"])
                if analysis["content_lengths"] else 0
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze records in {segment}: {e}")
            return {"error": str(e)}
    
    def run_full_verification(self) -> Dict[str, any]:
        """Run complete verification of remote data."""
        logger.info("Starting full verification of remote database...")
        start_time = time.time()
        
        # Test connectivity
        connectivity_results = self.test_connectivity()
        
        # Count records
        record_counts = self.count_records()
        
        # Test vector search
        search_results = self.test_vector_search()
        
        # Analyze content for segments with data
        content_analysis = {}
        for seg in self.segments:
            if record_counts.get(seg, 0) > 0:
                content_analysis[seg] = self.analyze_record_content(seg)
        
        end_time = time.time()
        
        # Compile results
        verification_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(end_time - start_time, 2),
            "connectivity": connectivity_results,
            "record_counts": record_counts,
            "vector_search": search_results,
            "content_analysis": content_analysis,
            "summary": self._generate_summary(connectivity_results, record_counts, search_results)
        }
        
        return verification_results
    
    def _generate_summary(self, connectivity: Dict[str, bool], 
                         counts: Dict[str, int], 
                         search: Dict[str, bool]) -> Dict[str, any]:
        """Generate summary statistics."""
        total_segments = len(self.segments)
        connected_segments = sum(1 for v in connectivity.values() if v)
        segments_with_data = sum(1 for v in counts.values() if v > 0)
        total_records = sum(counts.values())
        working_search = sum(1 for v in search.values() if v)
        
        return {
            "total_segments": total_segments,
            "connected_segments": connected_segments,
            "segments_with_data": segments_with_data,
            "total_records": total_records,
            "working_vector_search": working_search,
            "connection_rate": f"{connected_segments}/{total_segments} ({connected_segments/total_segments*100:.1f}%)",
            "data_coverage": f"{segments_with_data}/{total_segments} ({segments_with_data/total_segments*100:.1f}%)",
            "search_success_rate": f"{working_search}/{total_segments} ({working_search/total_segments*100:.1f}%)"
        }
    
    def print_results(self, results: Dict[str, any]) -> None:
        """Print verification results in a readable format."""
        logger.info("=" * 80)
        logger.info("REMOTE DATABASE VERIFICATION RESULTS")
        logger.info("=" * 80)
        
        summary = results["summary"]
        logger.info(f"Verification completed at: {results['timestamp']}")
        logger.info(f"Duration: {results['duration_seconds']} seconds")
        logger.info("")
        
        logger.info("SUMMARY:")
        logger.info(f"  Total segments: {summary['total_segments']}")
        logger.info(f"  Connected segments: {summary['connection_rate']}")
        logger.info(f"  Segments with data: {summary['data_coverage']}")
        logger.info(f"  Total records: {summary['total_records']:,}")
        logger.info(f"  Working vector search: {summary['search_success_rate']}")
        logger.info("")
        
        logger.info("DETAILED RESULTS:")
        logger.info("-" * 40)
        
        for seg in self.segments:
            connected = "✓" if results["connectivity"].get(seg, False) else "✗"
            count = results["record_counts"].get(seg, 0)
            search_ok = "✓" if results["vector_search"].get(seg, False) else "✗"
            
            logger.info(f"  {seg:>8}: Connect {connected} | Records {count:>8,} | Search {search_ok}")
        
        logger.info("")
        
        # Show content analysis for segments with data
        if results["content_analysis"]:
            logger.info("CONTENT ANALYSIS:")
            logger.info("-" * 40)
            
            for seg, analysis in results["content_analysis"].items():
                if "error" in analysis:
                    logger.info(f"  {seg}: Error - {analysis['error']}")
                    continue
                
                logger.info(f"  {seg}:")
                logger.info(f"    Sample records: {analysis['total_samples']}")
                logger.info(f"    Has content: {analysis['has_content']}/{analysis['total_samples']}")
                logger.info(f"    Has vectors: {analysis['has_vector']}/{analysis['total_samples']}")
                logger.info(f"    Has attributes: {analysis['has_attributes']}/{analysis['total_samples']}")
                logger.info(f"    Has timestamps: {analysis['has_timestamp']}/{analysis['total_samples']}")
                if analysis['avg_content_length'] > 0:
                    logger.info(f"    Avg content length: {analysis['avg_content_length']:.0f} chars")
                if analysis['attribute_keys']:
                    logger.info(f"    Attribute keys: {', '.join(analysis['attribute_keys'][:5])}")
                    if len(analysis['attribute_keys']) > 5:
                        logger.info(f"    ... and {len(analysis['attribute_keys']) - 5} more")
        
        logger.info("=" * 80)


def main():
    """Main function to run the verification."""
    logger.info("Starting remote database verification...")
    
    # Check if DSN template is properly configured
    if not PAPER_DSN_TEMPLATE or "USER:PASS@HOST:PORT" in PAPER_DSN_TEMPLATE:
        logger.error("PAPER_DSN_TEMPLATE is not properly configured in config.py")
        logger.error("Please set the correct database connection string")
        return
    
    try:
        verifier = RemoteDataVerifier()
        results = verifier.run_full_verification()
        verifier.print_results(results)
        
        # Save results to file
        import json
        results_file = "remote_verification_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise


if __name__ == "__main__":
    main()
