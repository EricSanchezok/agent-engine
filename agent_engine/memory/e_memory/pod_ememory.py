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
        """Get the shard ID for a given record ID using consistent hashing."""
        # Use hash of record ID to determine shard
        hash_value = int(hashlib.md5(record_id.encode()).hexdigest(), 16)
        # Use a large number for consistent hashing, then map to available shards
        return hash_value % max(1, self._shard_count)
    
    def _get_available_shard(self) -> int:
        """Get an available shard for new records using consistent hashing."""
        # For new records without ID, we need to generate one first
        # But since we don't have the ID yet, we'll use a capacity-based approach
        # Check if current shard has space
        current_shard = self._shards[self._current_shard]
        if current_shard.count() < self.max_elements_per_shard:
            return self._current_shard
        
        # Current shard is full, create a new one
        return self._create_new_shard()
    
    def add(self, record: Record) -> str:
        """
        Add a record to the pod.
        
        Args:
            record: Record object to add
            
        Returns:
            Record ID
        """
        # Generate ID if not provided
        if not record.id:
            import uuid
            record.id = str(uuid.uuid4())
        
        # Use consistent hashing to determine shard
        shard_id = self._get_shard_for_record(record.id)
        
        # Ensure the target shard exists, create if necessary
        while shard_id >= self._shard_count:
            self._create_new_shard()
        
        # Add to appropriate shard
        shard = self._shards[shard_id]
        result_id = shard.add(record)
        
        logger.debug(f"Added record {result_id} to shard {shard_id}")
        return result_id

    def add_batch(self, records: List[Record]) -> List[str]:
        """
        Add multiple records to the pod in batch.
        
        Args:
            records: List of Record objects to add
            
        Returns:
            List of record IDs
        """
        if not records:
            return []
        
        # Generate IDs for records that don't have them
        for record in records:
            if not record.id:
                import uuid
                record.id = str(uuid.uuid4())
        
        # Group records by shard using consistent hashing
        shard_groups: Dict[int, List[Record]] = {}
        
        for record in records:
            # Use consistent hashing to determine shard
            shard_id = self._get_shard_for_record(record.id)
            
            # Ensure the target shard exists, create if necessary
            while shard_id >= self._shard_count:
                self._create_new_shard()
            
            if shard_id not in shard_groups:
                shard_groups[shard_id] = []
            shard_groups[shard_id].append(record)
        
        # Add records to each shard in batch
        all_record_ids = []
        for shard_id, shard_records in shard_groups.items():
            shard = self._shards[shard_id]
            record_ids = shard.add_batch(shard_records)
            all_record_ids.extend(record_ids)
            logger.debug(f"Added {len(shard_records)} records to shard {shard_id}")
        
        logger.info(f"Added {len(records)} records to PodEMemory in batch across {len(shard_groups)} shards")
        return all_record_ids

    def get(self, record_id: str) -> Optional[Record]:
        """
        Get a record by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            Record if found, None otherwise
        """
        # Determine which shard contains this record
        shard_id = self._get_shard_for_record(record_id)
        
        if shard_id not in self._shards:
            return None
        
        shard = self._shards[shard_id]
        return shard.get(record_id)
    
    def update(self, record: Record) -> bool:
        """
        Update an existing record.
        
        Args:
            record: Record object with updated data (must have id)
            
        Returns:
            True if updated, False if record not found
        """
        if not record.id:
            raise ValueError("Record must have an ID for update")
        
        # Determine which shard contains this record
        shard_id = self._get_shard_for_record(record.id)
        
        if shard_id not in self._shards:
            return False
        
        shard = self._shards[shard_id]
        return shard.update(record)
    
    def delete(self, record_id: str) -> bool:
        """
        Delete a record by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            True if deleted, False if not found
        """
        # Determine which shard contains this record
        shard_id = self._get_shard_for_record(record_id)
        
        if shard_id not in self._shards:
            return False
        
        shard = self._shards[shard_id]
        return shard.delete(record_id)
    
    def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Record]:
        """
        List all records across all shards.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of records
        """
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
    
    def clear(self) -> None:
        """Clear all records from all shards."""
        for shard in self._shards.values():
            shard.clear()
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
        # Determine which shard contains this record
        shard_id = self._get_shard_for_record(record_id)
        
        if shard_id not in self._shards:
            return False
        
        shard = self._shards[shard_id]
        return shard.exists(record_id)
    
    def has_vector(self, record_id: str) -> bool:
        """
        Check if a record has a vector embedding.
        
        Args:
            record_id: Record ID
            
        Returns:
            True if record has vector, False otherwise
        """
        # Determine which shard contains this record
        shard_id = self._get_shard_for_record(record_id)
        
        if shard_id not in self._shards:
            return False
        
        shard = self._shards[shard_id]
        return shard.has_vector(record_id)
    
    def exists_batch(self, record_ids: List[str]) -> Dict[str, bool]:
        """
        Check existence of multiple records by IDs.
        
        Args:
            record_ids: List of record IDs
            
        Returns:
            Dictionary mapping ID to existence status
        """
        if not record_ids:
            return {}
        
        # Group IDs by shard
        shard_groups: Dict[int, List[str]] = {}
        for record_id in record_ids:
            shard_id = self._get_shard_for_record(record_id)
            if shard_id not in shard_groups:
                shard_groups[shard_id] = []
            shard_groups[shard_id].append(record_id)
        
        # Check existence in each shard
        results = {}
        for shard_id, ids in shard_groups.items():
            if shard_id in self._shards:
                shard = self._shards[shard_id]
                shard_results = shard.exists_batch(ids)
                results.update(shard_results)
            else:
                # Shard doesn't exist, all IDs are False
                for record_id in ids:
                    results[record_id] = False
        
        return results
    
    def has_vector_batch(self, record_ids: List[str]) -> Dict[str, bool]:
        """
        Check if multiple records have vector embeddings.
        
        Args:
            record_ids: List of record IDs
            
        Returns:
            Dictionary mapping ID to vector existence status
        """
        if not record_ids:
            return {}
        
        # Group IDs by shard
        shard_groups: Dict[int, List[str]] = {}
        for record_id in record_ids:
            shard_id = self._get_shard_for_record(record_id)
            if shard_id not in shard_groups:
                shard_groups[shard_id] = []
            shard_groups[shard_id].append(record_id)
        
        # Check vector existence in each shard
        results = {}
        for shard_id, ids in shard_groups.items():
            if shard_id in self._shards:
                shard = self._shards[shard_id]
                shard_results = shard.has_vector_batch(ids)
                results.update(shard_results)
            else:
                # Shard doesn't exist, all IDs are False
                for record_id in ids:
                    results[record_id] = False
        
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
