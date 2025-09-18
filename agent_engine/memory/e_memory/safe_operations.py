"""
Safe database operations with signal handling and transaction management.

This module provides robust database operations that can handle interruptions
gracefully, including Ctrl+C (SIGINT) and other termination signals.
"""

import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sqlite3
import json
from datetime import datetime
import uuid

from ...agent_logger.agent_logger import AgentLogger
from .models import Record

logger = AgentLogger(__name__)


class SafeOperationManager:
    """
    Manages safe database operations with signal handling and transaction management.
    Uses singleton pattern to ensure only one signal handler is registered.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._interrupted = False
        self._operation_lock = threading.Lock()
        self._active_operations: Dict[str, Dict[str, Any]] = {}
        self._signal_handlers = {}
        self._setup_signal_handlers()
        self._initialized = True
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            self._interrupted = True
            
            # Log active operations
            if self._active_operations:
                logger.warning(f"Active operations: {list(self._active_operations.keys())}")
            
            # Call the original handler if it exists and is not the default handler
            original_handler = self._signal_handlers.get(signum)
            if original_handler and original_handler != signal.SIG_DFL and original_handler != signal.SIG_IGN:
                logger.debug(f"Calling original signal handler for {signum}")
                try:
                    original_handler(signum, frame)
                except Exception as e:
                    logger.error(f"Error calling original signal handler: {e}")
            else:
                # If no original handler or it's default, force exit after a short delay
                logger.warning("No original handler found, forcing exit in 2 seconds...")
                import threading
                def force_exit():
                    time.sleep(2)
                    logger.warning("Forcing exit due to signal")
                    sys.exit(1)
                threading.Thread(target=force_exit, daemon=True).start()
        
        # Register handlers for common termination signals
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                old_handler = signal.signal(sig, signal_handler)
                self._signal_handlers[sig] = old_handler
            except (OSError, ValueError):
                # Signal not available on this platform
                pass
    
    def is_interrupted(self) -> bool:
        """Check if an interrupt signal has been received."""
        return self._interrupted
    
    def force_shutdown(self, timeout: float = 5.0):
        """
        Force shutdown after a timeout period.
        
        Args:
            timeout: Time to wait before forcing exit (seconds)
        """
        logger.warning(f"Forcing shutdown in {timeout} seconds...")
        self._interrupted = True
        
        def delayed_exit():
            time.sleep(timeout)
            logger.warning("Timeout reached, forcing exit")
            sys.exit(1)
        
        import threading
        threading.Thread(target=delayed_exit, daemon=True).start()
    
    def register_operation(self, operation_id: str, operation_type: str, 
                          shard_id: Optional[int] = None) -> None:
        """Register an active operation."""
        with self._operation_lock:
            self._active_operations[operation_id] = {
                'type': operation_type,
                'shard_id': shard_id,
                'start_time': time.time(),
                'status': 'running'
            }
    
    def unregister_operation(self, operation_id: str) -> None:
        """Unregister a completed operation."""
        with self._operation_lock:
            if operation_id in self._active_operations:
                del self._active_operations[operation_id]
    
    def get_active_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active operations."""
        with self._operation_lock:
            return self._active_operations.copy()
    
    @contextmanager
    def safe_operation(self, operation_id: str, operation_type: str, 
                      shard_id: Optional[int] = None):
        """
        Context manager for safe database operations.
        
        Args:
            operation_id: Unique identifier for this operation
            operation_type: Type of operation (e.g., 'batch_add', 'repair', 'search')
            shard_id: Optional shard ID if operation is shard-specific
        """
        self.register_operation(operation_id, operation_type, shard_id)
        
        try:
            logger.debug(f"Starting safe operation: {operation_id} ({operation_type})")
            
            # Check for interruption before starting
            if self.is_interrupted():
                logger.warning(f"Operation {operation_id} skipped due to interruption")
                raise KeyboardInterrupt("Operation interrupted before starting")
            
            yield self
            
            if self.is_interrupted():
                logger.warning(f"Operation {operation_id} completed but system was interrupted")
            else:
                logger.debug(f"Operation {operation_id} completed successfully")
                
        except KeyboardInterrupt:
            logger.warning(f"Operation {operation_id} interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Operation {operation_id} failed: {e}")
            raise
        finally:
            self.unregister_operation(operation_id)


class SafeSQLiteConnection:
    """
    Safe SQLite connection with automatic retry and transaction management.
    """
    
    def __init__(self, db_path: Union[str, Path], timeout: float = 30.0, 
                 max_retries: int = 3):
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.max_retries = max_retries
        self._connection = None
    
    def __enter__(self):
        """Enter context manager."""
        for attempt in range(self.max_retries):
            try:
                self._connection = sqlite3.connect(
                    str(self.db_path),
                    timeout=self.timeout,
                    isolation_level=None  # Use autocommit mode for better control
                )
                
                # Enable WAL mode for better concurrency
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=NORMAL")
                self._connection.execute("PRAGMA temp_store=MEMORY")
                
                return self._connection
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < self.max_retries - 1:
                    logger.warning(f"Database locked, retrying in 1 second (attempt {attempt + 1})")
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Failed to connect to database after {attempt + 1} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error connecting to database: {e}")
                raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")
            finally:
                self._connection = None


class SafeBatchProcessor:
    """
    Safe batch processing with progress tracking and interruption handling.
    """
    
    def __init__(self, batch_size: int = 1000, progress_interval: int = 10000):
        self.batch_size = batch_size
        self.progress_interval = progress_interval
        self.operation_manager = SafeOperationManager()
    
    def process_batch_safe(self, 
                          items: List[Any],
                          process_func: Callable[[List[Any]], bool],
                          operation_id: str,
                          operation_type: str = "batch_process") -> Tuple[bool, int]:
        """
        Process items in batches with safe interruption handling.
        
        Args:
            items: List of items to process
            process_func: Function to process each batch
            operation_id: Unique operation identifier
            operation_type: Type of operation for logging
            
        Returns:
            Tuple of (success, processed_count)
        """
        total_items = len(items)
        processed_count = 0
        
        with self.operation_manager.safe_operation(operation_id, operation_type):
            logger.info(f"Starting batch processing: {total_items} items in batches of {self.batch_size}")
            
            for i in range(0, total_items, self.batch_size):
                # Check for interruption
                if self.operation_manager.is_interrupted():
                    logger.warning(f"Batch processing interrupted at {processed_count}/{total_items} items")
                    return False, processed_count
                
                batch = items[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (total_items + self.batch_size - 1) // self.batch_size
                
                try:
                    logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
                    
                    success = process_func(batch)
                    if not success:
                        logger.error(f"Batch {batch_num} processing failed")
                        return False, processed_count
                    
                    processed_count += len(batch)
                    
                    # Log progress
                    if processed_count % self.progress_interval == 0:
                        progress_pct = (processed_count / total_items) * 100
                        logger.info(f"Progress: {processed_count}/{total_items} ({progress_pct:.1f}%)")
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {e}")
                    return False, processed_count
            
            logger.info(f"Batch processing completed: {processed_count}/{total_items} items")
            return True, processed_count


class SafeTransactionManager:
    """
    Manages safe transactions across SQLite and ChromaDB with rollback capability.
    """
    
    def __init__(self, sqlite_path: Union[str, Path], chroma_collection=None):
        self.sqlite_path = Path(sqlite_path)
        self.chroma_collection = chroma_collection
        self._sqlite_backup_path = None
        self._chroma_backup_data = None
    
    @contextmanager
    def safe_transaction(self, operation_id: str):
        """
        Context manager for safe transactions with automatic rollback.
        """
        # Create backup before transaction
        self._create_backup()
        
        try:
            logger.debug(f"Starting safe transaction: {operation_id}")
            yield self
            
            # Transaction completed successfully
            logger.debug(f"Transaction {operation_id} completed successfully")
            
        except Exception as e:
            # Transaction failed, attempt rollback
            logger.error(f"Transaction {operation_id} failed, attempting rollback: {e}")
            self._rollback()
            raise
        finally:
            # Clean up backup
            self._cleanup_backup()
    
    def _create_backup(self):
        """Create backup of current state."""
        try:
            # Backup SQLite data
            import shutil
            self._sqlite_backup_path = self.sqlite_path.parent / f"{self.sqlite_path.stem}_backup.sqlite"
            shutil.copy2(self.sqlite_path, self._sqlite_backup_path)
            
            # Backup ChromaDB data if collection exists
            if self.chroma_collection:
                try:
                    all_data = self.chroma_collection.get()
                    self._chroma_backup_data = all_data
                except Exception as e:
                    logger.warning(f"Could not backup ChromaDB data: {e}")
                    self._chroma_backup_data = None
            
            logger.debug("Backup created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def _rollback(self):
        """Rollback to backup state."""
        try:
            # Restore SQLite
            if self._sqlite_backup_path and self._sqlite_backup_path.exists():
                import shutil
                shutil.copy2(self._sqlite_backup_path, self.sqlite_path)
                logger.info("SQLite rollback completed")
            
            # Restore ChromaDB
            if self._chroma_backup_data and self.chroma_collection:
                try:
                    # Clear current collection
                    self.chroma_collection.delete()
                    
                    # Restore backup data
                    if self._chroma_backup_data.get('ids'):
                        self.chroma_collection.add(
                            ids=self._chroma_backup_data['ids'],
                            embeddings=self._chroma_backup_data.get('embeddings', []),
                            metadatas=self._chroma_backup_data.get('metadatas', [])
                        )
                    logger.info("ChromaDB rollback completed")
                except Exception as e:
                    logger.error(f"ChromaDB rollback failed: {e}")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
    
    def _cleanup_backup(self):
        """Clean up backup files."""
        try:
            if self._sqlite_backup_path and self._sqlite_backup_path.exists():
                self._sqlite_backup_path.unlink()
            self._sqlite_backup_path = None
            self._chroma_backup_data = None
        except Exception as e:
            logger.warning(f"Failed to cleanup backup: {e}")


class SafeMemoryOperations:
    """
    Safe memory operations with comprehensive error handling and signal management.
    """
    
    def __init__(self):
        self.operation_manager = SafeOperationManager()
        self.batch_processor = SafeBatchProcessor()
    
    def safe_add_batch(self, 
                      records: List[Record],
                      sqlite_path: Union[str, Path],
                      chroma_collection,
                      operation_id: Optional[str] = None) -> Tuple[bool, int]:
        """
        Safely add a batch of records with transaction management.
        """
        if not operation_id:
            operation_id = f"batch_add_{int(time.time())}"
        
        if not records:
            return True, 0
        
        def process_batch(batch_records: List[Record]) -> bool:
            """Process a batch of records."""
            try:
                with SafeTransactionManager(sqlite_path, chroma_collection).safe_transaction(f"{operation_id}_batch"):
                    # Process SQLite batch
                    sqlite_success = self._add_batch_sqlite(batch_records, sqlite_path)
                    
                    # Process ChromaDB batch
                    chroma_success = self._add_batch_chroma(batch_records, chroma_collection)
                    
                    return sqlite_success and chroma_success
                    
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                return False
        
        return self.batch_processor.process_batch_safe(
            records, process_batch, operation_id, "batch_add"
        )
    
    def _add_batch_sqlite(self, records: List[Record], sqlite_path: Union[str, Path]) -> bool:
        """Add batch to SQLite database."""
        try:
            with SafeSQLiteConnection(sqlite_path) as conn:
                conn.execute("BEGIN TRANSACTION")
                
                db_data = []
                for record in records:
                    record_id = record.id or str(uuid.uuid4())
                    record_timestamp = record.timestamp or datetime.utcnow().isoformat() + "Z"
                    record_attributes = record.attributes or {}
                    has_vector = 1 if record.vector else 0
                    
                    db_data.append((
                        record_id,
                        json.dumps(record_attributes),
                        record.content,
                        record_timestamp,
                        has_vector
                    ))
                
                conn.executemany("""
                    INSERT OR REPLACE INTO records (id, attributes, content, timestamp, has_vector)
                    VALUES (?, ?, ?, ?, ?)
                """, db_data)
                
                conn.execute("COMMIT")
                return True
                
        except Exception as e:
            logger.error(f"SQLite batch add failed: {e}")
            return False
    
    def _add_batch_chroma(self, records: List[Record], chroma_collection) -> bool:
        """Add batch to ChromaDB collection."""
        try:
            vectors_to_add = []
            ids_for_vectors = []
            
            for record in records:
                if record.vector:
                    vectors_to_add.append(record.vector)
                    ids_for_vectors.append(record.id or str(uuid.uuid4()))
            
            if vectors_to_add:
                chroma_collection.add(
                    embeddings=vectors_to_add,
                    ids=ids_for_vectors
                )
            
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB batch add failed: {e}")
            return False


# Global instance for easy access
safe_operations = SafeOperationManager()
safe_memory_ops = SafeMemoryOperations()
