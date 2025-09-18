"""
PodEMemory - A sharded memory implementation using multiple EMemory instances.

This module provides a high-capacity memory implementation that manages multiple EMemory
instances to handle large datasets efficiently by distributing records across multiple shards.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

import hashlib
import sqlite3
import shutil
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time

from .core import EMemory
from .models import Record
from .safe_operations import SafeOperationManager, SafeBatchProcessor
from ...agent_logger.agent_logger import AgentLogger

logger = AgentLogger(__name__)


class ShardHealthStatus(Enum):
    """Shard health status enumeration"""
    HEALTHY = "healthy"
    CORRUPTED = "corrupted"
    MISSING = "missing"
    ERROR = "error"


class ShardHealthInfo:
    """Information about shard health status"""
    def __init__(self, shard_id: int, status: ShardHealthStatus, 
                 record_count: int = 0, error_message: str = "", 
                 file_size: int = 0, last_modified: Optional[datetime] = None):
        self.shard_id = shard_id
        self.status = status
        self.record_count = record_count
        self.error_message = error_message
        self.file_size = file_size
        self.last_modified = last_modified
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/reporting"""
        return {
            "shard_id": self.shard_id,
            "status": self.status.value,
            "record_count": self.record_count,
            "error_message": self.error_message,
            "file_size": self.file_size,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None
        }


class PodEMemory:
    """
    A sharded memory implementation using multiple EMemory instances.
    
    This class manages multiple EMemory instances to handle large datasets
    efficiently by distributing records across multiple shards. Each shard
    is an independent EMemory instance with its own SQLite + ChromaDB files.
    """
    
    def __init__(
        self,
        name: str,
        persist_dir: Optional[str] = None,
        max_elements_per_shard: int = 100000,
        distance_metric: str = "cosine",
        num_shards: Optional[int] = None,
        use_hash_sharding: bool = True
    ):
        """
        Initialize PodEMemory with multiple EMemory shards.
        
        Args:
            name: Pod name (used for subdirectory creation)
            persist_dir: Storage directory (optional, defaults to root/.memory/name)
            max_elements_per_shard: Maximum elements per EMemory shard (used for legacy mode)
            distance_metric: Distance metric ('cosine', 'l2', 'ip')
            num_shards: Fixed number of shards for hash-based sharding (recommended: 10-100)
            use_hash_sharding: Whether to use hash-based sharding (True) or legacy sequential sharding (False)
        """
        self.name = name
        self.max_elements_per_shard = max_elements_per_shard
        self.distance_metric = distance_metric
        self.use_hash_sharding = use_hash_sharding
        
        # Initialize safe operation manager
        self.safe_ops = SafeOperationManager()
        self.batch_processor = SafeBatchProcessor()
        
        # Determine storage directory
        if persist_dir is None:
            from ...utils.project_root import get_project_root
            root = get_project_root()
            self.persist_dir = root / ".memory" / name
        else:
            self.persist_dir = Path(persist_dir) / name
        
        # Create directory if it doesn't exist
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Shard management
        self._shards: Dict[int, EMemory] = {}
        
        if use_hash_sharding:
            # Hash-based sharding: fixed number of shards
            if num_shards is None:
                num_shards = 8  # Default number of shards
            self.num_shards = num_shards
            self._shard_count = num_shards
            self._current_shard = None  # Not used in hash sharding
            
            # Create all shards upfront for hash-based sharding
            self._create_all_shards()
        else:
            # Legacy sequential sharding: dynamic shard creation
            self.num_shards = None
            self._shard_count = 0
            self._current_shard = 0
            
            # Load existing shards or create first shard
            self._load_existing_shards()
        
        logger.info(f"PodEMemory '{name}' initialized at {self.persist_dir}")
        logger.info(f"Sharding strategy: {'Hash-based' if use_hash_sharding else 'Sequential'}")
        if use_hash_sharding:
            logger.info(f"Number of shards: {self.num_shards}")
        else:
            logger.info(f"Max elements per shard: {max_elements_per_shard}")
    
    def _create_all_shards(self) -> None:
        """Create all shards upfront for hash-based sharding."""
        for shard_id in range(self.num_shards):
            shard_name = f"{self.name}_shard_{shard_id}"
            shard_dir = self.persist_dir / shard_name
            
            # Create EMemory instance for this shard
            shard = EMemory(
                name=shard_name,
                persist_dir=str(shard_dir),
                distance_metric=self.distance_metric
            )
            
            self._shards[shard_id] = shard
            logger.debug(f"Created shard {shard_id} with name '{shard_name}'")
        
        logger.info(f"Created {self.num_shards} shards for hash-based sharding")
    
    def _load_existing_shards(self) -> None:
        """Load existing shards from the persist directory."""
        if not self.persist_dir.exists():
            # No existing directory, create first shard
            self._create_new_shard()
            return
        
        # Find existing shard directories
        shard_dirs = []
        for item in self.persist_dir.iterdir():
            if item.is_dir() and item.name.startswith(f"{self.name}_shard_"):
                try:
                    # Extract shard ID from directory name
                    shard_id = int(item.name.split("_shard_")[1])
                    shard_dirs.append((shard_id, item))
                except (ValueError, IndexError):
                    # Skip directories that don't match expected pattern
                    continue
        
        if not shard_dirs:
            # No existing shards found, create first shard
            self._create_new_shard()
            return
        
        # Sort by shard ID and load existing shards
        shard_dirs.sort(key=lambda x: x[0])
        for shard_id, shard_dir in shard_dirs:
            try:
                shard_name = f"{self.name}_shard_{shard_id}"
                shard = EMemory(
                    name=shard_name,
                    persist_dir=str(shard_dir),
                    distance_metric=self.distance_metric
                )
                
                self._shards[shard_id] = shard
                self._shard_count = max(self._shard_count, shard_id + 1)
                self._current_shard = shard_id  # Set current shard to the last loaded one
                
                logger.info(f"Loaded existing shard {shard_id} with {shard.count()} records")
                
            except Exception as e:
                logger.error(f"Failed to load shard {shard_id} from {shard_dir}: {e}")
                continue
        
        if not self._shards:
            # No shards could be loaded, create first shard
            logger.warning("No existing shards could be loaded, creating new shard")
            self._create_new_shard()
        else:
            logger.info(f"Loaded {len(self._shards)} existing shards")
    
    def _create_new_shard(self) -> int:
        """Create a new EMemory shard."""
        shard_id = self._shard_count
        shard_name = f"{self.name}_shard_{shard_id}"
        shard_dir = self.persist_dir / shard_name
        
        # Create EMemory instance for this shard
        shard = EMemory(
            name=shard_name,
            persist_dir=str(shard_dir),
            distance_metric=self.distance_metric
        )
        
        self._shards[shard_id] = shard
        self._shard_count += 1
        self._current_shard = shard_id  # Update current shard pointer
        
        logger.info(f"Created new shard {shard_id} with name '{shard_name}'")
        return shard_id
    
    def _get_shard_for_record(self, record_id: str) -> int:
        """Get the shard ID for a given record ID."""
        if self.use_hash_sharding:
            return self._get_hash_shard_for_record(record_id)
        else:
            return self._get_sequential_shard_for_record(record_id)
    
    def _get_hash_shard_for_record(self, record_id: str) -> int:
        """Get the shard ID for a given record ID using hash-based sharding."""
        # Use consistent hashing to determine shard
        hasher = hashlib.sha256(record_id.encode('utf-8'))
        shard_id = int(hasher.hexdigest(), 16) % self.num_shards
        return shard_id
    
    def _get_sequential_shard_for_record(self, record_id: str) -> int:
        """Get the shard ID for a given record ID using sequential sharding."""
        shard_id = self._find_shard_with_record(record_id)
        if shard_id is not None:
            return shard_id
        
        if self._shards[self._current_shard].count() < self.max_elements_per_shard:
            logger.debug(f"Record {record_id} will be added to shard {self._current_shard} (not full)")
            return self._current_shard
        
        logger.debug(f"All shards are full, creating new shard for record {record_id}")
        return self._create_new_shard()
    
    def _find_shard_with_record(self, record_id: str) -> Optional[int]:
        """Find which shard contains the given record ID."""
        for shard_id in sorted(self._shards.keys()):
            shard = self._shards[shard_id]
            if shard.exists(record_id):
                return shard_id
        return None
    
    def _find_shards_for_records_batch(self, record_ids: List[str]) -> Dict[str, int]:
        """
        Batch find which shard contains each record ID.
        
        This method efficiently finds the shard for multiple records by querying
        each shard once instead of querying each record individually.
        
        Args:
            record_ids: List of record IDs to find
            
        Returns:
            Dictionary mapping record_id to shard_id (only for existing records)
        """
        if not record_ids:
            return {}
        
        # Initialize result mapping
        existing_map = {}
        remaining_ids = set(record_ids)
        
        # Query each shard once for all remaining IDs
        for shard_id in sorted(self._shards.keys()):
            if not remaining_ids:
                break  # All IDs found, early exit
                
            shard = self._shards[shard_id]
            
            # Get batch existence check for remaining IDs
            found_in_shard = shard.exists_batch(list(remaining_ids))
            
            # Update mapping for found records
            for record_id, exists in found_in_shard.items():
                if exists:
                    existing_map[record_id] = shard_id
                    remaining_ids.remove(record_id)
        
        return existing_map
    
    def add(self, record: Record) -> bool:
        """
        Add a record to the pod.
        
        Args:
            record: Record object to add
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Generate ID if not provided
            if not record.id:
                import uuid
                record.id = str(uuid.uuid4())
            
            # Get the appropriate shard (may create new shard if needed)
            shard_id = self._get_shard_for_record(record.id)
            
            # Add to appropriate shard
            shard = self._shards[shard_id]
            success = shard.add(record)
            
            if success:
                logger.debug(f"Added record {record.id} to shard {shard_id}")
            else:
                logger.error(f"Failed to add record {record.id} to shard {shard_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add record to pod: {e}")
            return False

    def add_batch(self, records: List[Record]) -> bool:
        """
        Add multiple records to the pod in batch with safe operation handling.
        
        Optimized implementation that reduces database queries from O(N*M) to O(M)
        where N is number of records and M is number of shards.
        
        Args:
            records: List of Record objects to add
            
        Returns:
            True if all records were successfully added, False otherwise
        """
        if not records:
            return True
        
        operation_id = f"pod_add_batch_{self.name}_{int(time.time())}"
        
        try:
            with self.safe_ops.safe_operation(operation_id, "pod_batch_add"):
                # Check for interruption before starting
                if self.safe_ops.is_interrupted():
                    logger.warning("Pod batch add operation interrupted before starting")
                    return False
                
                # Generate IDs for records that don't have them
                for record in records:
                    if not record.id:
                        import uuid
                        record.id = str(uuid.uuid4())
                
                # Step 1: Batch find existing records and their shards (O(M) queries)
                existing_map = self._find_shards_for_records_batch([r.id for r in records])
                
                # Step 2: Group records by target shard
                shard_groups: Dict[int, List[Record]] = {}
                new_records = []
                
                for record in records:
                    if record.id in existing_map:
                        # Record exists, add to existing shard
                        shard_id = existing_map[record.id]
                        if shard_id not in shard_groups:
                            shard_groups[shard_id] = []
                        shard_groups[shard_id].append(record)
                    else:
                        # New record, will be allocated later
                        new_records.append(record)
                
                # Step 3: Allocate new records to appropriate shards
                if new_records:
                    if self.use_hash_sharding:
                        # Hash sharding: use hash function to determine shard for each record
                        for record in new_records:
                            shard_id = self._get_hash_shard_for_record(record.id)
                            if shard_id not in shard_groups:
                                shard_groups[shard_id] = []
                            shard_groups[shard_id].append(record)
                    else:
                        # Sequential sharding: use existing logic
                        shard_counts = {sid: shard.count() for sid, shard in self._shards.items()}
                        current_shard = self._current_shard
                        
                        for record in new_records:
                            # Check for interruption during allocation
                            if self.safe_ops.is_interrupted():
                                logger.warning("Pod batch add interrupted during shard allocation")
                                return False
                            
                            # Find next available shard
                            while shard_counts.get(current_shard, 0) >= self.max_elements_per_shard:
                                current_shard += 1
                                if current_shard not in self._shards:
                                    self._create_new_shard()
                                    shard_counts[current_shard] = 0
                            
                            shard_id = current_shard
                            if shard_id not in shard_groups:
                                shard_groups[shard_id] = []
                            shard_groups[shard_id].append(record)
                            shard_counts[shard_id] += 1
                
                # Step 4: Batch write to each shard with progress tracking
                total_processed = 0
                for shard_id, recs in shard_groups.items():
                    # Check for interruption before processing each shard
                    if self.safe_ops.is_interrupted():
                        logger.warning(f"Pod batch add interrupted at shard {shard_id}")
                        return False
                    
                    shard_operation_id = f"{operation_id}_shard_{shard_id}"
                    with self.safe_ops.safe_operation(shard_operation_id, "shard_batch_add", shard_id):
                        shard = self._shards[shard_id]
                        success = shard.add_batch(recs)
                        if not success:
                            logger.error(f"Failed to add batch to shard {shard_id}")
                            return False
                        
                        total_processed += len(recs)
                        logger.debug(f"Added {len(recs)} records to shard {shard_id}")
                        
                        # Log progress
                        if total_processed % 10000 == 0:
                            progress_pct = (total_processed / len(records)) * 100
                            logger.info(f"Pod batch add progress: {total_processed}/{len(records)} ({progress_pct:.1f}%)")
                
                logger.info(f"Successfully added {len(records)} records to PodEMemory in batch across {len(shard_groups)} shards")
                return True
            
        except Exception as e:
            logger.error(f"Failed to add batch records to pod: {e}")
            return False

    def get(self, record_id: str) -> Optional[Record]:
        """
        Get a record by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            Record if found, None otherwise
        """
        if self.use_hash_sharding:
            # O(1) lookup using hash-based sharding
            shard_id = self._get_hash_shard_for_record(record_id)
            shard = self._shards[shard_id]
            return shard.get(record_id)
        else:
            # O(M) lookup using sequential sharding
            shard_id = self._find_shard_with_record(record_id)
            if shard_id is None:
                return None
            shard = self._shards[shard_id]
            return shard.get(record_id)
    
    def get_vector(self, record_id: str) -> Optional[List[float]]:
        """
        Get the vector embedding for a record by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            Vector embedding as List[float] if found and has vector, None otherwise
        """
        if self.use_hash_sharding:
            # O(1) lookup using hash-based sharding
            shard_id = self._get_hash_shard_for_record(record_id)
            shard = self._shards[shard_id]
            return shard.get_vector(record_id)
        else:
            # O(M) lookup using sequential sharding
            shard_id = self._find_shard_with_record(record_id)
            if shard_id is None:
                return None
            shard = self._shards[shard_id]
            return shard.get_vector(record_id)
    
    def delete(self, record_id: str) -> bool:
        """
        Delete a record by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            True if deleted, False if not found
        """
        if self.use_hash_sharding:
            # O(1) lookup using hash-based sharding
            shard_id = self._get_hash_shard_for_record(record_id)
            shard = self._shards[shard_id]
            return shard.delete(record_id)
        else:
            # O(M) lookup using sequential sharding
            shard_id = self._find_shard_with_record(record_id)
            if shard_id is None:
                return False
            shard = self._shards[shard_id]
            return shard.delete(record_id)
    
    def update(self, record: Record) -> bool:
        """
        Update an existing record in the pod.
        
        Args:
            record: Record object to update (must have an ID)
            
        Returns:
            True if successfully updated, False otherwise
        """
        if not record.id:
            logger.warning("Cannot update record without ID")
            return False
        
        if self.use_hash_sharding:
            # O(1) lookup using hash-based sharding
            shard_id = self._get_hash_shard_for_record(record.id)
            shard = self._shards[shard_id]
            return shard.update(record)
        else:
            # O(M) lookup using sequential sharding
            shard_id = self._find_shard_with_record(record.id)
            if shard_id is None:
                logger.warning(f"Record {record.id} not found, cannot update")
                return False
            shard = self._shards[shard_id]
            return shard.update(record)

    def update_batch(self, records: List[Record]) -> bool:
        """
        Update multiple existing records in the pod in batch.
        
        Optimized implementation that reduces database queries from O(N*M) to O(M)
        where N is number of records and M is number of shards.
        
        Args:
            records: List of Record objects to update (all must have IDs)
            
        Returns:
            True if all records were successfully updated, False otherwise
        """
        if not records:
            return True
        
        try:
            # Step 1: Validate all records have IDs
            record_ids = [record.id for record in records if record.id]
            if len(record_ids) != len(records):
                logger.error("All records must have IDs for batch update")
                return False
            
            # Step 2: Batch find existing records and their shards (O(M) queries)
            existing_map = self._find_shards_for_records_batch(record_ids)
            
            # Step 3: Check for missing records
            missing_ids = [rid for rid in record_ids if rid not in existing_map]
            if missing_ids:
                logger.error(f"Records not found for batch update: {missing_ids}")
                return False
            
            # Step 4: Group records by target shard
            shard_groups: Dict[int, List[Record]] = {}
            for record in records:
                shard_id = existing_map[record.id]
                if shard_id not in shard_groups:
                    shard_groups[shard_id] = []
                shard_groups[shard_id].append(record)
            
            # Step 5: Batch update each shard
            for shard_id, recs in shard_groups.items():
                shard = self._shards[shard_id]
                success = shard.update_batch(recs)
                if not success:
                    logger.error(f"Failed to update batch in shard {shard_id}")
                    return False
                logger.debug(f"Updated {len(recs)} records in shard {shard_id}")
            
            logger.info(f"Updated {len(records)} records in PodEMemory batch across {len(shard_groups)} shards")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update batch records in pod: {e}")
            return False
    
    def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Record]:
        """
        List all records across all shards.
        
        ⚠️  WARNING: This method loads ALL records into memory at once!
        For large databases (millions of records), this can cause memory explosion.
        Consider using more specific query methods or pagination with small limits.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of records
        """
        import warnings
        warnings.warn(
            "list_all() loads ALL records into memory. For large databases, "
            "this can cause memory explosion. Consider using specific query methods "
            "or pagination with small limits.",
            UserWarning,
            stacklevel=2
        )
        
        all_records = []
        current_offset = offset
        
        # Collect records from all shards
        for shard_id in sorted(self._shards.keys()):
            shard = self._shards[shard_id]
            shard_records = shard.list_all(limit=None, offset=0)
            all_records.extend(shard_records)
        
        # Sort by timestamp (most recent first)
        all_records.sort(key=lambda r: r.timestamp or "", reverse=True)
        
        # Apply offset and limit
        if offset > 0:
            all_records = all_records[offset:]
        
        if limit is not None:
            all_records = all_records[:limit]
        
        return all_records
    
    def count(self) -> int:
        """
        Get total number of records across all shards.
        
        Returns:
            Total number of records
        """
        total = 0
        for shard in self._shards.values():
            total += shard.count()
        return total
    
    def search_similar(
        self,
        query_vector: List[float],
        k: int = 10,
        ef: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors across all shards.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            ef: Search parameter (optional)
            
        Returns:
            List of (record_id, distance) tuples
        """
        all_results = []
        
        # Search in all shards
        for shard in self._shards.values():
            shard_results = shard.search_similar(query_vector, k, ef)
            all_results.extend(shard_results)
        
        # Sort by distance and return top k
        all_results.sort(key=lambda x: x[1])
        return all_results[:k]
    
    def search_similar_records(
        self,
        query_vector: List[float],
        k: int = 10,
        ef: Optional[int] = None
    ) -> List[Tuple[Record, float]]:
        """
        Search for similar records across all shards.
        
        Optimized implementation that eliminates N+1 query problem by using
        batch operations to retrieve records efficiently.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            ef: Search parameter (optional)
            
        Returns:
            List of (Record, distance) tuples
        """
        # Step 1: Get similar record IDs and distances from all shards
        similar_ids_with_dist = self.search_similar(query_vector, k, ef)
        if not similar_ids_with_dist:
            return []

        record_ids = [item[0] for item in similar_ids_with_dist]
        dist_map = {item[0]: item[1] for item in similar_ids_with_dist}

        if self.use_hash_sharding:
            # Hash sharding: group by shard and batch retrieve
            shard_groups: Dict[int, List[str]] = {}
            for record_id in record_ids:
                shard_id = self._get_hash_shard_for_record(record_id)
                if shard_id not in shard_groups:
                    shard_groups[shard_id] = []
                shard_groups[shard_id].append(record_id)
            
            # Batch retrieve from each shard
            all_records_map: Dict[str, Record] = {}
            for shard_id, rids_in_shard in shard_groups.items():
                shard = self._shards[shard_id]
                records_from_shard = shard.get_batch(rids_in_shard)
                all_records_map.update(records_from_shard)
        else:
            # Sequential sharding: use existing batch method
            shard_map = self._find_shards_for_records_batch(record_ids)
            
            # Group by shard
            shard_groups: Dict[int, List[str]] = {}
            for rid, sid in shard_map.items():
                if sid not in shard_groups:
                    shard_groups[sid] = []
                shard_groups[sid].append(rid)
            
            # Batch retrieve from each shard
            all_records_map: Dict[str, Record] = {}
            for shard_id, rids_in_shard in shard_groups.items():
                shard = self._shards[shard_id]
                records_from_shard = shard.get_batch(rids_in_shard)
                all_records_map.update(records_from_shard)
        
        # Step 4: Reassemble results maintaining original order
        results = []
        for record_id, distance in similar_ids_with_dist:
            record = all_records_map.get(record_id)
            if record:
                results.append((record, distance))
        
        return results
    
    def query_by_date_range(
        self, 
        start_date: str, 
        end_date: str, 
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Record]:
        """
        Query records by date range across all shards with optimized performance.
        
        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of records within the date range
        """
        logger.info(f"Querying records by date range: {start_date} to {end_date}")
        
        # If limit is specified and small, we can optimize by querying fewer shards
        # and stopping early when we have enough records
        if limit is not None and limit <= 100:
            return self._query_by_date_range_optimized(start_date, end_date, limit, offset)
        
        # For larger queries, use the original approach but with better error handling
        return self._query_by_date_range_full(start_date, end_date, limit, offset)
    
    def _query_by_date_range_optimized(
        self, 
        start_date: str, 
        end_date: str, 
        limit: int,
        offset: int
    ) -> List[Record]:
        """Optimized query for small result sets."""
        all_records = []
        target_count = limit + offset
        
        # Query shards in order until we have enough records
        for shard_id in sorted(self._shards.keys()):
            if len(all_records) >= target_count:
                break
                
            try:
                shard = self._shards[shard_id]
                # Query with a reasonable limit to avoid loading too much data
                shard_limit = min(limit * 2, 1000)  # Reasonable upper bound
                shard_records = shard.query_by_date_range(start_date, end_date, limit=shard_limit, offset=0)
                all_records.extend(shard_records)
                
                logger.debug(f"Shard {shard_id}: found {len(shard_records)} records")
                
            except Exception as e:
                logger.error(f"Error querying shard {shard_id}: {e}")
                continue
        
        # Sort by timestamp (most recent first)
        all_records.sort(key=lambda r: r.timestamp or "", reverse=True)
        
        # Apply offset and limit
        if offset > 0:
            all_records = all_records[offset:]
        
        if limit is not None:
            all_records = all_records[:limit]
        
        logger.info(f"Optimized query returned {len(all_records)} records")
        return all_records
    
    def _query_by_date_range_full(
        self, 
        start_date: str, 
        end_date: str, 
        limit: Optional[int],
        offset: int
    ) -> List[Record]:
        """Full query for larger result sets with timeout protection."""
        all_records = []
        failed_shards = []
        
        # Query each shard with timeout protection
        for shard_id in sorted(self._shards.keys()):
            try:
                shard = self._shards[shard_id]
                
                # Use a reasonable limit per shard to prevent memory issues
                shard_limit = min(limit or 10000, 10000) if limit else 10000
                
                shard_records = shard.query_by_date_range(start_date, end_date, limit=shard_limit, offset=0)
                all_records.extend(shard_records)
                
                logger.debug(f"Shard {shard_id}: found {len(shard_records)} records")
                
            except Exception as e:
                logger.error(f"Error querying shard {shard_id}: {e}")
                failed_shards.append(shard_id)
                continue
        
        if failed_shards:
            logger.warning(f"Failed to query {len(failed_shards)} shards: {failed_shards}")
        
        # Sort by timestamp (most recent first)
        all_records.sort(key=lambda r: r.timestamp or "", reverse=True)
        
        # Apply offset and limit
        if offset > 0:
            all_records = all_records[offset:]
        
        if limit is not None:
            all_records = all_records[:limit]
        
        logger.info(f"Full query returned {len(all_records)} records from {len(self._shards) - len(failed_shards)} shards")
        return all_records
    
    def clear(self, confirm: bool = True) -> None:
        """
        Clear all records from all shards.
        
        Args:
            confirm: If True, requires user confirmation before clearing.
                    If False, clears immediately without confirmation.
        """
        if confirm:
            # Get total count for confirmation message
            total_count = self.count()
            
            if total_count == 0:
                logger.info("No records to clear in PodEMemory")
                return
            
            print(f"\n⚠️  WARNING: You are about to delete ALL {total_count} records from PodEMemory '{self.name}'!")
            print("This action cannot be undone!")
            print("\nTo confirm deletion, type 'DELETE' and press Enter.")
            print("To cancel, press Enter or type anything else.")
            
            try:
                user_input = input("Confirmation: ").strip()
                if user_input != "DELETE":
                    print("Deletion cancelled.")
                    logger.info("PodEMemory clear operation cancelled by user")
                    return
            except KeyboardInterrupt:
                print("\nDeletion cancelled.")
                logger.info("PodEMemory clear operation cancelled by user (Ctrl+C)")
                return
            except EOFError:
                print("\nDeletion cancelled.")
                logger.info("PodEMemory clear operation cancelled by user (EOF)")
                return
        
        # Proceed with clearing
        for shard in self._shards.values():
            shard.clear(confirm=False)  # Don't ask for confirmation again
        logger.info("Cleared all records from PodEMemory")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pod statistics.
        
        Returns:
            Statistics dictionary
        """
        total_records = self.count()
        shard_stats = []
        
        for shard_id, shard in self._shards.items():
            shard_stat = shard.get_stats()
            shard_stat["shard_id"] = shard_id
            shard_stats.append(shard_stat)
        
        return {
            "name": self.name,
            "total_records": total_records,
            "shard_count": self._shard_count,
            "max_elements_per_shard": self.max_elements_per_shard,
            "persist_dir": str(self.persist_dir),
            "shards": shard_stats
        }
    
    def exists(self, record_id: str) -> bool:
        """
        Check if a record exists by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            True if record exists, False otherwise
        """
        if self.use_hash_sharding:
            # O(1) lookup using hash-based sharding
            shard_id = self._get_hash_shard_for_record(record_id)
            shard = self._shards[shard_id]
            return shard.exists(record_id)
        else:
            # O(M) lookup using sequential sharding
            shard_id = self._find_shard_with_record(record_id)
            return shard_id is not None
    
    def has_vector(self, record_id: str) -> bool:
        """
        Check if a record has a vector embedding.
        
        Args:
            record_id: Record ID
            
        Returns:
            True if record has vector, False otherwise
        """
        if self.use_hash_sharding:
            # O(1) lookup using hash-based sharding
            shard_id = self._get_hash_shard_for_record(record_id)
            shard = self._shards[shard_id]
            return shard.has_vector(record_id)
        else:
            # O(M) lookup using sequential sharding
            shard_id = self._find_shard_with_record(record_id)
            if shard_id is None:
                return False
            shard = self._shards[shard_id]
            return shard.has_vector(record_id)
    
    def exists_batch(self, record_ids: List[str]) -> Dict[str, bool]:
        """
        Check existence of multiple records by IDs.
        
        Optimized implementation that reduces database queries from O(N*M) to O(M)
        where N is number of records and M is number of shards.
        
        Args:
            record_ids: List of record IDs
            
        Returns:
            Dictionary mapping ID to existence status
        """
        if not record_ids:
            return {}

        if self.use_hash_sharding:
            # 🚀 Efficient hash sharding path
            results = {record_id: False for record_id in record_ids}
            
            # 1. Group record_ids by shard
            shard_groups: Dict[int, List[str]] = {}
            for record_id in record_ids:
                shard_id = self._get_hash_shard_for_record(record_id)
                if shard_id not in shard_groups:
                    shard_groups[shard_id] = []
                shard_groups[shard_id].append(record_id)

            # 2. Execute batch query on each relevant shard
            for shard_id, rids_in_shard in shard_groups.items():
                shard = self._shards[shard_id]
                found_in_shard = shard.exists_batch(rids_in_shard)
                # Update results
                for record_id, exists in found_in_shard.items():
                    if exists:
                        results[record_id] = True
            return results
        
        else:
            # Compatible sequential sharding path (unchanged)
            # Initialize all results as False
            results = {record_id: False for record_id in record_ids}
            remaining_ids = set(record_ids)
            
            # Query each shard once for all remaining IDs
            for shard in self._shards.values():
                if not remaining_ids:
                    break  # All IDs found, early exit
                
                # Get batch existence check for remaining IDs
                found_in_shard = shard.exists_batch(list(remaining_ids))
                
                # Update results for found records
                for record_id, exists in found_in_shard.items():
                    if exists:
                        results[record_id] = True
                        remaining_ids.remove(record_id)
            
            return results
    
    def has_vector_batch(self, record_ids: List[str]) -> Dict[str, bool]:
        """
        Check if multiple records have vector embeddings.
        
        Optimized implementation that reduces database queries from O(N*M) to O(M)
        where N is number of records and M is number of shards.
        
        Args:
            record_ids: List of record IDs
            
        Returns:
            Dictionary mapping ID to vector existence status
        """
        if not record_ids:
            return {}

        if self.use_hash_sharding:
            # 🚀 Efficient hash sharding path
            results = {record_id: False for record_id in record_ids}
            
            # 1. Group record_ids by shard
            shard_groups: Dict[int, List[str]] = {}
            for record_id in record_ids:
                shard_id = self._get_hash_shard_for_record(record_id)
                if shard_id not in shard_groups:
                    shard_groups[shard_id] = []
                shard_groups[shard_id].append(record_id)

            # 2. Execute batch query on each relevant shard
            for shard_id, rids_in_shard in shard_groups.items():
                shard = self._shards[shard_id]
                has_vector_in_shard = shard.has_vector_batch(rids_in_shard)
                # Update results
                results.update(has_vector_in_shard)
            return results
        
        else:
            # Compatible sequential sharding path (unchanged)
            # Initialize all results as False
            results = {record_id: False for record_id in record_ids}
            remaining_ids = set(record_ids)
            
            # Query each shard once for all remaining IDs
            for shard in self._shards.values():
                if not remaining_ids:
                    break  # All IDs found, early exit
                
                # Get batch existence check for remaining IDs to know which records exist in this shard
                exists_in_shard = shard.exists_batch(list(remaining_ids))
                
                # Get batch vector existence check for records that actually exist in this shard
                existing_ids = [rid for rid in remaining_ids if exists_in_shard.get(rid, False)]
                if existing_ids:
                    has_vector_in_shard = shard.has_vector_batch(existing_ids)
                else:
                    has_vector_in_shard = {}
                
                # Update results for records that exist in this shard
                for record_id in list(remaining_ids):
                    if exists_in_shard.get(record_id, False):
                        # Record exists in this shard
                        results[record_id] = has_vector_in_shard.get(record_id, False)
                        remaining_ids.remove(record_id)
            
            return results

    def get_shard_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all shards.
        
        Returns:
            List of shard information dictionaries
        """
        shard_info = []
        for shard_id, shard in self._shards.items():
            info = {
                "shard_id": shard_id,
                "name": shard.name,
                "record_count": shard.count(),
                "persist_dir": str(shard.persist_dir)
            }
            shard_info.append(info)
        
        return shard_info
    
    def check_shard_health(self, shard_id: int) -> ShardHealthInfo:
        """
        Check the health status of a specific shard.
        
        Args:
            shard_id: Shard ID to check
            
        Returns:
            ShardHealthInfo object with health status
        """
        if shard_id not in self._shards:
            return ShardHealthInfo(
                shard_id=shard_id,
                status=ShardHealthStatus.MISSING,
                error_message=f"Shard {shard_id} not found in PodEMemory"
            )
        
        shard = self._shards[shard_id]
        shard_name = shard.name
        sqlite_file = shard.persist_dir / f"{shard_name}.sqlite"
        
        # Check if SQLite file exists
        if not sqlite_file.exists():
            return ShardHealthInfo(
                shard_id=shard_id,
                status=ShardHealthStatus.MISSING,
                error_message=f"SQLite file not found: {sqlite_file}"
            )
        
        # Get file info
        file_size = sqlite_file.stat().st_size
        last_modified = datetime.fromtimestamp(sqlite_file.stat().st_mtime)
        
        try:
            # Test SQLite database integrity
            conn = sqlite3.connect(str(sqlite_file))
            cursor = conn.cursor()
            
            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()
            
            if integrity_result[0] != "ok":
                conn.close()
                return ShardHealthInfo(
                    shard_id=shard_id,
                    status=ShardHealthStatus.CORRUPTED,
                    error_message=f"Database integrity check failed: {integrity_result[0]}",
                    file_size=file_size,
                    last_modified=last_modified
                )
            
            # Count records
            cursor.execute("SELECT COUNT(*) FROM records")
            record_count = cursor.fetchone()[0]
            
            conn.close()
            
            return ShardHealthInfo(
                shard_id=shard_id,
                status=ShardHealthStatus.HEALTHY,
                record_count=record_count,
                file_size=file_size,
                last_modified=last_modified
            )
            
        except sqlite3.Error as e:
            return ShardHealthInfo(
                shard_id=shard_id,
                status=ShardHealthStatus.ERROR,
                error_message=f"SQLite error: {str(e)}",
                file_size=file_size,
                last_modified=last_modified
            )
        except Exception as e:
            return ShardHealthInfo(
                shard_id=shard_id,
                status=ShardHealthStatus.ERROR,
                error_message=f"Unexpected error: {str(e)}",
                file_size=file_size,
                last_modified=last_modified
            )
    
    def check_all_shards_health(self) -> Dict[int, ShardHealthInfo]:
        """
        Check the health status of all shards.
        
        Returns:
            Dictionary mapping shard_id to ShardHealthInfo
        """
        health_info = {}
        
        for shard_id in self._shards.keys():
            health_info[shard_id] = self.check_shard_health(shard_id)
        
        return health_info
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive health summary of the PodEMemory.
        
        Returns:
            Dictionary with health summary information
        """
        shard_health = self.check_all_shards_health()
        
        # Count shards by status
        status_counts = {}
        total_records = 0
        total_size = 0
        corrupted_shards = []
        missing_shards = []
        error_shards = []
        
        for shard_id, health_info in shard_health.items():
            status = health_info.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            total_records += health_info.record_count
            total_size += health_info.file_size
            
            if health_info.status == ShardHealthStatus.CORRUPTED:
                corrupted_shards.append(shard_id)
            elif health_info.status == ShardHealthStatus.MISSING:
                missing_shards.append(shard_id)
            elif health_info.status == ShardHealthStatus.ERROR:
                error_shards.append(shard_id)
        
        # Determine overall health
        overall_health = "healthy"
        if corrupted_shards or error_shards:
            overall_health = "unhealthy"
        elif missing_shards:
            overall_health = "degraded"
        
        return {
            "overall_health": overall_health,
            "total_shards": len(self._shards),
            "status_counts": status_counts,
            "total_records": total_records,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "corrupted_shards": corrupted_shards,
            "missing_shards": missing_shards,
            "error_shards": error_shards,
            "shard_details": {shard_id: info.to_dict() for shard_id, info in shard_health.items()}
        }
    
    def repair_corrupted_shard(self, shard_id: int, backup_dir: Optional[str] = None) -> bool:
        """
        Attempt to repair a corrupted shard.
        
        Args:
            shard_id: Shard ID to repair
            backup_dir: Optional backup directory for corrupted files
            
        Returns:
            True if repair was successful, False otherwise
        """
        if shard_id not in self._shards:
            logger.error(f"Shard {shard_id} not found in PodEMemory")
            return False
        
        shard = self._shards[shard_id]
        shard_name = shard.name
        sqlite_file = shard.persist_dir / f"{shard_name}.sqlite"
        
        logger.info(f"Attempting to repair corrupted shard {shard_id}")
        
        # Check if shard is actually corrupted
        health_info = self.check_shard_health(shard_id)
        if health_info.status != ShardHealthStatus.CORRUPTED:
            logger.warning(f"Shard {shard_id} is not corrupted, status: {health_info.status.value}")
            return False
        
        try:
            # Create backup if backup_dir is provided
            if backup_dir:
                backup_path = Path(backup_dir) / f"{shard_name}_corrupted_backup.sqlite"
                shutil.copy2(sqlite_file, backup_path)
                logger.info(f"Created backup of corrupted shard at: {backup_path}")
            
            # Try to recover data using SQLite's recovery mechanisms
            recovered_file = shard.persist_dir / f"{shard_name}_recovered.sqlite"
            
            # Use sqlite3 command line tool for recovery if available
            import subprocess
            try:
                # Try to dump and restore the database
                dump_cmd = ["sqlite3", str(sqlite_file), ".dump"]
                restore_cmd = ["sqlite3", str(recovered_file)]
                
                # Run dump command
                dump_result = subprocess.run(dump_cmd, capture_output=True, text=True, timeout=300)
                
                if dump_result.returncode == 0:
                    # Run restore command
                    restore_result = subprocess.run(restore_cmd, input=dump_result.stdout, 
                                                   capture_output=True, text=True, timeout=300)
                    
                    if restore_result.returncode == 0:
                        # Replace original file with recovered file
                        sqlite_file.unlink()
                        recovered_file.rename(sqlite_file)
                        
                        # Verify the repair
                        new_health = self.check_shard_health(shard_id)
                        if new_health.status == ShardHealthStatus.HEALTHY:
                            logger.info(f"Successfully repaired shard {shard_id}")
                            return True
                        else:
                            logger.error(f"Repair failed for shard {shard_id}: {new_health.error_message}")
                            return False
                    else:
                        logger.error(f"Restore failed for shard {shard_id}: {restore_result.stderr}")
                        return False
                else:
                    logger.error(f"Dump failed for shard {shard_id}: {dump_result.stderr}")
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Repair timeout for shard {shard_id}")
                return False
            except FileNotFoundError:
                logger.warning("sqlite3 command line tool not available, trying Python recovery")
                
                # Fallback: try Python-based recovery
                return self._repair_shard_python(shard_id, sqlite_file)
                
        except Exception as e:
            logger.error(f"Failed to repair shard {shard_id}: {str(e)}")
            return False
    
    def _repair_shard_python(self, shard_id: int, sqlite_file: Path) -> bool:
        """
        Attempt Python-based shard repair with improved error handling.
        
        Args:
            shard_id: Shard ID to repair
            sqlite_file: Path to SQLite file
            
        Returns:
            True if repair was successful, False otherwise
        """
        recovered_file = sqlite_file.parent / f"{sqlite_file.stem}_recovered.sqlite"
        
        try:
            # Step 1: Try to read what we can from the corrupted database
            logger.info(f"Attempting Python repair for shard {shard_id}")
            
            # Use WAL mode and PRAGMA settings for better recovery
            conn = sqlite3.connect(str(sqlite_file))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=OFF")  # Faster but less safe
            cursor = conn.cursor()
            
            # Try to get table schema
            try:
                cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
                schemas = cursor.fetchall()
                
                if not schemas:
                    logger.error(f"No table schemas found in shard {shard_id}")
                    conn.close()
                    return False
                
                logger.info(f"Found {len(schemas)} table schemas in corrupted database")
                
            except sqlite3.Error as e:
                logger.error(f"Failed to read schemas from shard {shard_id}: {str(e)}")
                conn.close()
                return False
            
            # Step 2: Create new clean database
            if recovered_file.exists():
                recovered_file.unlink()  # Remove any existing recovered file
            
            new_conn = sqlite3.connect(str(recovered_file))
            new_cursor = new_conn.cursor()
            
            # Enable WAL mode for better performance
            new_cursor.execute("PRAGMA journal_mode=WAL")
            new_cursor.execute("PRAGMA synchronous=NORMAL")
            
            # Step 3: Recreate tables with error handling
            for schema in schemas:
                try:
                    # Use IF NOT EXISTS to avoid conflicts
                    schema_sql = schema[0]
                    if "CREATE TABLE" in schema_sql.upper():
                        # Add IF NOT EXISTS to avoid table already exists error
                        if "IF NOT EXISTS" not in schema_sql.upper():
                            schema_sql = schema_sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
                    
                    new_cursor.execute(schema_sql)
                    logger.debug(f"Created table from schema: {schema_sql[:50]}...")
                    
                except sqlite3.Error as e:
                    logger.warning(f"Failed to create table from schema: {str(e)}")
                    continue
            
            # Step 4: Try to copy data with multiple strategies
            data_copied = False
            
            # Strategy 1: Try to copy all records at once
            try:
                cursor.execute("SELECT COUNT(*) FROM records")
                total_records = cursor.fetchone()[0]
                logger.info(f"Attempting to copy {total_records} records from corrupted database")
                
                cursor.execute("SELECT * FROM records")
                records = cursor.fetchall()
                
                # Get column names
                cursor.execute("PRAGMA table_info(records)")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Insert records in batches
                placeholders = ", ".join(["?" for _ in columns])
                insert_sql = f"INSERT INTO records ({', '.join(columns)}) VALUES ({placeholders})"
                
                batch_size = 1000
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    new_cursor.executemany(insert_sql, batch)
                    logger.debug(f"Copied batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}")
                
                new_conn.commit()
                data_copied = True
                logger.info(f"Successfully copied {len(records)} records")
                
            except sqlite3.Error as e:
                logger.warning(f"Strategy 1 failed: {str(e)}")
                
                # Strategy 2: Try to copy records one by one
                try:
                    logger.info("Trying strategy 2: copy records one by one")
                    
                    cursor.execute("SELECT * FROM records")
                    cursor.execute("PRAGMA table_info(records)")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    placeholders = ", ".join(["?" for _ in columns])
                    insert_sql = f"INSERT INTO records ({', '.join(columns)}) VALUES ({placeholders})"
                    
                    records_copied = 0
                    while True:
                        try:
                            record = cursor.fetchone()
                            if record is None:
                                break
                            
                            new_cursor.execute(insert_sql, record)
                            records_copied += 1
                            
                            if records_copied % 1000 == 0:
                                new_conn.commit()
                                logger.debug(f"Copied {records_copied} records")
                                
                        except sqlite3.Error as record_error:
                            logger.warning(f"Skipping corrupted record: {str(record_error)}")
                            continue
                    
                    new_conn.commit()
                    data_copied = True
                    logger.info(f"Successfully copied {records_copied} records using strategy 2")
                    
                except sqlite3.Error as e2:
                    logger.warning(f"Strategy 2 failed: {str(e2)}")
                    
                    # Strategy 3: Try to recover what we can
                    try:
                        logger.info("Trying strategy 3: recover what we can")
                        
                        # Try to get any readable data
                        cursor.execute("SELECT * FROM records LIMIT 1")
                        sample_record = cursor.fetchone()
                        
                        if sample_record:
                            # Create a minimal database with what we can recover
                            cursor.execute("SELECT COUNT(*) FROM records")
                            total_count = cursor.fetchone()[0]
                            
                            logger.info(f"Database has {total_count} records, attempting partial recovery")
                            
                            # Try to recover in smaller chunks
                            chunk_size = 100
                            recovered_count = 0
                            
                            for offset in range(0, total_count, chunk_size):
                                try:
                                    cursor.execute(f"SELECT * FROM records LIMIT {chunk_size} OFFSET {offset}")
                                    chunk_records = cursor.fetchall()
                                    
                                    if chunk_records:
                                        new_cursor.executemany(insert_sql, chunk_records)
                                        recovered_count += len(chunk_records)
                                        
                                except sqlite3.Error:
                                    logger.warning(f"Skipping chunk at offset {offset}")
                                    continue
                            
                            new_conn.commit()
                            data_copied = True
                            logger.info(f"Partially recovered {recovered_count} records")
                        
                    except sqlite3.Error as e3:
                        logger.error(f"Strategy 3 failed: {str(e3)}")
            
            # Clean up connections
            new_cursor.close()
            new_conn.close()
            cursor.close()
            conn.close()
            
            # Step 5: Verify the recovered database
            if data_copied:
                try:
                    verify_conn = sqlite3.connect(str(recovered_file))
                    verify_cursor = verify_conn.cursor()
                    
                    # Check integrity
                    verify_cursor.execute("PRAGMA integrity_check")
                    integrity_result = verify_cursor.fetchone()
                    
                    if integrity_result[0] == "ok":
                        # Count records
                        verify_cursor.execute("SELECT COUNT(*) FROM records")
                        record_count = verify_cursor.fetchone()[0]
                        
                        verify_cursor.close()
                        verify_conn.close()
                        
                        # Replace original file
                        sqlite_file.unlink()
                        recovered_file.rename(sqlite_file)
                        
                        logger.info(f"Successfully repaired shard {shard_id} with {record_count} records")
                        return True
                    else:
                        logger.error(f"Recovered database failed integrity check: {integrity_result[0]}")
                        verify_cursor.close()
                        verify_conn.close()
                        return False
                        
                except Exception as verify_error:
                    logger.error(f"Failed to verify recovered database: {str(verify_error)}")
                    return False
            else:
                logger.error(f"No data could be recovered from shard {shard_id}")
                return False
                
        except Exception as e:
            logger.error(f"Python repair failed for shard {shard_id}: {str(e)}")
            return False
        finally:
            # Clean up recovered file if it exists
            if recovered_file.exists():
                try:
                    recovered_file.unlink()
                except:
                    pass
    
    def log_health_status(self) -> None:
        """
        Log the current health status of all shards.
        """
        health_summary = self.get_health_summary()
        
        logger.info("=== PodEMemory Health Status ===")
        logger.info(f"Overall Health: {health_summary['overall_health']}")
        logger.info(f"Total Records: {health_summary['total_records']:,}")
        logger.info(f"Total Size: {health_summary['total_size_mb']} MB")
        logger.info(f"Status Counts: {health_summary['status_counts']}")
        
        if health_summary['corrupted_shards']:
            logger.warning(f"Corrupted Shards: {health_summary['corrupted_shards']}")
        if health_summary['missing_shards']:
            logger.warning(f"Missing Shards: {health_summary['missing_shards']}")
        if health_summary['error_shards']:
            logger.error(f"Error Shards: {health_summary['error_shards']}")
        
        # Log detailed shard information
        for shard_id, details in health_summary['shard_details'].items():
            if details['status'] != 'healthy':
                logger.warning(f"Shard {shard_id}: {details['status']} - {details['error_message']}")
    
    def close(self):
        """
        Properly close all shard connections and release file handles.
        
        This method should be called when the PodEMemory instance is no longer needed
        to ensure proper cleanup of database connections and file handles.
        """
        logger.info(f"Closing PodEMemory '{self.name}' and all shards")
        
        for shard_id, shard in self._shards.items():
            try:
                # Close ChromaDB connection
                if hasattr(shard, 'chroma_client') and shard.chroma_client:
                    try:
                        shard.chroma_client = None
                        logger.debug(f"Closed ChromaDB connection for shard {shard_id}")
                    except Exception as e:
                        logger.warning(f"Error closing ChromaDB connection for shard {shard_id}: {e}")
                
                # Close SQLite connection (if any explicit connection exists)
                if hasattr(shard, '_connection') and shard._connection:
                    try:
                        shard._connection.close()
                        shard._connection = None
                        logger.debug(f"Closed SQLite connection for shard {shard_id}")
                    except Exception as e:
                        logger.warning(f"Error closing SQLite connection for shard {shard_id}: {e}")
                        
            except Exception as e:
                logger.error(f"Error closing shard {shard_id}: {e}")
        
        # Clear shards dictionary
        self._shards.clear()
        
        # Force garbage collection to release file handles
        import gc
        gc.collect()
        
        logger.info(f"PodEMemory '{self.name}' closed successfully")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.close()
