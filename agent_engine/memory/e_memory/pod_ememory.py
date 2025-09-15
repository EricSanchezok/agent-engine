"""
PodEMemory - A sharded memory implementation using multiple EMemory instances.

This module provides a high-capacity memory implementation that manages multiple EMemory
instances to handle large datasets efficiently by distributing records across multiple shards.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .core import EMemory
from .models import Record
from ...agent_logger.agent_logger import AgentLogger

logger = AgentLogger(__name__)


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
        distance_metric: str = "cosine"
    ):
        """
        Initialize PodEMemory with multiple EMemory shards.
        
        Args:
            name: Pod name (used for subdirectory creation)
            persist_dir: Storage directory (optional, defaults to root/.memory/name)
            max_elements_per_shard: Maximum elements per EMemory shard
            distance_metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.name = name
        self.max_elements_per_shard = max_elements_per_shard
        self.distance_metric = distance_metric
        
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
        self._shard_count = 0
        self._current_shard = 0
        
        # Load existing shards or create first shard
        self._load_existing_shards()
        
        logger.info(f"PodEMemory '{name}' initialized at {self.persist_dir}")
        logger.info(f"Max elements per shard: {max_elements_per_shard}")
    
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
        """Get the shard ID for a given record ID using existence-based sharding."""
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
        Add multiple records to the pod in batch.
        
        Optimized implementation that reduces database queries from O(N*M) to O(M)
        where N is number of records and M is number of shards.
        
        Args:
            records: List of Record objects to add
            
        Returns:
            True if all records were successfully added, False otherwise
        """
        if not records:
            return True
        
        try:
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
                # Get current shard counts
                shard_counts = {sid: shard.count() for sid, shard in self._shards.items()}
                current_shard = self._current_shard
                
                for record in new_records:
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
            
            # Step 4: Batch write to each shard
            for shard_id, recs in shard_groups.items():
                shard = self._shards[shard_id]
                success = shard.add_batch(recs)
                if not success:
                    logger.error(f"Failed to add batch to shard {shard_id}")
                    return False
                logger.debug(f"Added {len(recs)} records to shard {shard_id}")
            
            logger.info(f"Added {len(records)} records to PodEMemory in batch across {len(shard_groups)} shards")
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
        # Find which shard contains this record
        shard_id = self._find_shard_with_record(record_id)
        
        if shard_id is None:
            return None
        
        shard = self._shards[shard_id]
        return shard.get(record_id)
    
    def delete(self, record_id: str) -> bool:
        """
        Delete a record by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            True if deleted, False if not found
        """
        # Find which shard contains this record
        shard_id = self._find_shard_with_record(record_id)
        
        if shard_id is None:
            return False
        
        shard = self._shards[shard_id]
        return shard.delete(record_id)
    
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
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            ef: Search parameter (optional)
            
        Returns:
            List of (Record, distance) tuples
        """
        similar_ids = self.search_similar(query_vector, k, ef)
        results = []
        
        for record_id, distance in similar_ids:
            record = self.get(record_id)
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
        Query records by date range across all shards.
        
        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of records within the date range
        """
        all_records = []
        
        # Query each shard
        for shard_id in sorted(self._shards.keys()):
            shard = self._shards[shard_id]
            shard_records = shard.query_by_date_range(start_date, end_date, limit=None, offset=0)
            all_records.extend(shard_records)
        
        # Sort by timestamp (most recent first)
        all_records.sort(key=lambda r: r.timestamp or "", reverse=True)
        
        # Apply offset and limit
        if offset > 0:
            all_records = all_records[offset:]
        
        if limit is not None:
            all_records = all_records[:limit]
        
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
        # Find which shard contains this record
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
        # Find which shard contains this record
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
