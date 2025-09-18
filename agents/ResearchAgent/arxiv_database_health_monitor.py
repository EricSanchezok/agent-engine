"""
ArxivDatabase Health Monitor and Auto-Repair System

This module provides comprehensive health monitoring and automatic repair
capabilities specifically for ArxivDatabase.
"""

import asyncio
import json
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
from dataclasses import dataclass, asdict

from agent_engine.agent_logger import AgentLogger
from agent_engine.memory.e_memory.pod_ememory import PodEMemory, ShardHealthStatus, ShardHealthInfo
from agent_engine.memory.e_memory.safe_operations import SafeOperationManager
from agent_engine.utils.project_root import get_project_root

logger = AgentLogger(__name__)


@dataclass
class ArxivHealthCheckResult:
    """Result of an ArxivDatabase health check operation."""
    timestamp: str
    overall_health: str  # "healthy", "degraded", "critical"
    total_shards: int
    healthy_shards: int
    degraded_shards: int
    critical_shards: int
    total_papers: int
    papers_with_vectors: int
    corrupted_files: List[str]
    repair_attempts: List[str]
    recommendations: List[str]


@dataclass
class ArxivRepairConfig:
    """Configuration for ArxivDatabase automatic repair operations."""
    enable_auto_repair: bool = True
    max_repair_attempts: int = 3
    repair_timeout_seconds: int = 300
    backup_before_repair: bool = True
    health_check_interval_minutes: int = 30
    critical_threshold_percent: float = 50.0  # If >50% shards are critical, trigger repair


class ArxivDatabaseHealthMonitor:
    """
    Comprehensive health monitoring and auto-repair system for ArxivDatabase.
    
    Features:
    1. Continuous health monitoring of ArxivDatabase
    2. Automatic corruption detection
    3. Self-healing capabilities
    4. Backup and restore functionality
    5. Performance optimization
    6. Long-running operation support
    """
    
    def __init__(
        self,
        arxiv_database,
        config: Optional[ArxivRepairConfig] = None
    ):
        """
        Initialize the ArxivDatabase health monitor.
        
        Args:
            arxiv_database: ArxivDatabase instance to monitor
            config: Repair configuration
        """
        self.arxiv_database = arxiv_database
        self.pod_ememory = arxiv_database.pod_ememory
        self.config = config or ArxivRepairConfig()
        self.safe_ops = SafeOperationManager()
        
        # Health monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.last_health_check: Optional[ArxivHealthCheckResult] = None
        self.repair_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_stats = {
            "total_checks": 0,
            "successful_repairs": 0,
            "failed_repairs": 0,
            "last_check_duration": 0,
            "average_check_duration": 0
        }
        
        logger.info("ArxivDatabaseHealthMonitor initialized")
        logger.info(f"Auto-repair enabled: {self.config.enable_auto_repair}")
        logger.info(f"Health check interval: {self.config.health_check_interval_minutes} minutes")
    
    def _calculate_overall_health(self, shard_healths: Dict[int, ShardHealthInfo]) -> str:
        """Calculate overall health status."""
        total_shards = len(shard_healths)
        if total_shards == 0:
            return "critical"
        
        critical_count = sum(1 for info in shard_healths.values() if info.status == ShardHealthStatus.CRITICAL)
        degraded_count = sum(1 for info in shard_healths.values() if info.status == ShardHealthStatus.DEGRADED)
        
        critical_percent = (critical_count / total_shards) * 100
        degraded_percent = (degraded_count / total_shards) * 100
        
        if critical_percent >= self.config.critical_threshold_percent:
            return "critical"
        elif degraded_percent >= 30 or critical_percent > 0:
            return "degraded"
        else:
            return "healthy"
    
    def _get_paper_statistics(self) -> Tuple[int, int]:
        """Get paper statistics from ArxivDatabase."""
        try:
            # Get total papers count
            total_papers = self.pod_ememory.count()
            
            # Get papers with vectors count (approximate)
            papers_with_vectors = 0
            for shard_id in self.pod_ememory._shards.keys():
                try:
                    shard = self.pod_ememory._shards[shard_id]
                    # Count records with has_vector=1
                    with sqlite3.connect(shard.sqlite_path) as conn:
                        cursor = conn.execute("SELECT COUNT(*) FROM records WHERE has_vector = 1")
                        papers_with_vectors += cursor.fetchone()[0]
                except Exception as e:
                    logger.warning(f"Error counting vectors in shard {shard_id}: {e}")
                    continue
            
            return total_papers, papers_with_vectors
            
        except Exception as e:
            logger.error(f"Error getting paper statistics: {e}")
            return 0, 0
    
    def _generate_recommendations(self, health_result: ArxivHealthCheckResult) -> List[str]:
        """Generate recommendations based on health status."""
        recommendations = []
        
        if health_result.overall_health == "critical":
            recommendations.append("CRITICAL: Immediate repair required for ArxivDatabase")
            recommendations.append("Consider stopping operations and running full repair")
        elif health_result.overall_health == "degraded":
            recommendations.append("Schedule maintenance window for ArxivDatabase repair")
            recommendations.append("Monitor performance closely")
        
        if health_result.corrupted_files:
            recommendations.append(f"Found {len(health_result.corrupted_files)} corrupted shard files")
        
        if health_result.total_papers == 0:
            recommendations.append("WARNING: No papers found in ArxivDatabase")
        
        if health_result.papers_with_vectors == 0 and health_result.total_papers > 0:
            recommendations.append("WARNING: No papers have vector embeddings")
        
        vector_coverage = (health_result.papers_with_vectors / health_result.total_papers * 100) if health_result.total_papers > 0 else 0
        if vector_coverage < 50:
            recommendations.append(f"Low vector coverage: {vector_coverage:.1f}% of papers have embeddings")
        
        return recommendations
    
    async def perform_health_check(self) -> ArxivHealthCheckResult:
        """
        Perform a comprehensive health check on ArxivDatabase.
        
        Returns:
            ArxivHealthCheckResult with detailed health information
        """
        start_time = time.time()
        
        logger.info("Starting comprehensive ArxivDatabase health check...")
        
        try:
            # Get shard health information
            shard_healths = self.pod_ememory.check_all_shards_health()
            
            # Calculate overall health
            overall_health = self._calculate_overall_health(shard_healths)
            
            # Count shards by status
            healthy_count = sum(1 for info in shard_healths.values() if info.status == ShardHealthStatus.HEALTHY)
            degraded_count = sum(1 for info in shard_healths.values() if info.status == ShardHealthStatus.DEGRADED)
            critical_count = sum(1 for info in shard_healths.values() if info.status == ShardHealthStatus.CRITICAL)
            
            # Get paper statistics
            total_papers, papers_with_vectors = self._get_paper_statistics()
            
            # Identify corrupted files
            corrupted_files = []
            for shard_id, health_info in shard_healths.items():
                if health_info.status == ShardHealthStatus.CRITICAL:
                    corrupted_files.append(f"arxiv_papers_shard_{shard_id}")
            
            # Create health result
            health_result = ArxivHealthCheckResult(
                timestamp=datetime.now().isoformat(),
                overall_health=overall_health,
                total_shards=len(shard_healths),
                healthy_shards=healthy_count,
                degraded_shards=degraded_count,
                critical_shards=critical_count,
                total_papers=total_papers,
                papers_with_vectors=papers_with_vectors,
                corrupted_files=corrupted_files,
                repair_attempts=[],
                recommendations=[]
            )
            
            # Generate recommendations
            health_result.recommendations = self._generate_recommendations(health_result)
            
            # Update performance stats
            check_duration = time.time() - start_time
            self.performance_stats["total_checks"] += 1
            self.performance_stats["last_check_duration"] = check_duration
            self.performance_stats["average_check_duration"] = (
                (self.performance_stats["average_check_duration"] * (self.performance_stats["total_checks"] - 1) + check_duration) 
                / self.performance_stats["total_checks"]
            )
            
            self.last_health_check = health_result
            
            logger.info(f"ArxivDatabase health check completed in {check_duration:.2f}s")
            logger.info(f"Overall health: {overall_health}")
            logger.info(f"Healthy shards: {healthy_count}/{len(shard_healths)}")
            logger.info(f"Total papers: {total_papers:,}, With vectors: {papers_with_vectors:,}")
            
            return health_result
            
        except Exception as e:
            logger.error(f"ArxivDatabase health check failed: {e}")
            # Return critical health result on failure
            return ArxivHealthCheckResult(
                timestamp=datetime.now().isoformat(),
                overall_health="critical",
                total_shards=0,
                healthy_shards=0,
                degraded_shards=0,
                critical_shards=0,
                total_papers=0,
                papers_with_vectors=0,
                corrupted_files=["health_check_failed"],
                repair_attempts=[f"Health check error: {str(e)}"],
                recommendations=["CRITICAL: ArxivDatabase health check system failure"]
            )
    
    async def repair_corrupted_shards(self, health_result: ArxivHealthCheckResult) -> bool:
        """
        Repair corrupted shards in ArxivDatabase based on health check results.
        
        Args:
            health_result: Health check result containing corruption information
            
        Returns:
            True if repair was successful, False otherwise
        """
        if not self.config.enable_auto_repair:
            logger.info("Auto-repair disabled, skipping ArxivDatabase repair")
            return False
        
        if not health_result.corrupted_files:
            logger.info("No corrupted files found in ArxivDatabase, no repair needed")
            return True
        
        logger.warning(f"Starting ArxivDatabase repair for {len(health_result.corrupted_files)} corrupted shards")
        
        repair_success = True
        repair_attempts = []
        
        for corrupted_file in health_result.corrupted_files:
            if not corrupted_file.startswith("arxiv_papers_shard_"):
                continue
            
            try:
                shard_id = int(corrupted_file.split("_")[-1])
                logger.info(f"Attempting to repair ArxivDatabase shard {shard_id}")
                
                # Create backup if configured
                backup_dir = None
                if self.config.backup_before_repair:
                    backup_dir = self.pod_ememory.persist_dir / "arxiv_repair_backups" / f"shard_{shard_id}_{int(time.time())}"
                    backup_dir.mkdir(parents=True, exist_ok=True)
                
                # Attempt repair
                repair_success_shard = self.pod_ememory.repair_corrupted_shard(shard_id, str(backup_dir))
                
                if repair_success_shard:
                    logger.info(f"Successfully repaired ArxivDatabase shard {shard_id}")
                    repair_attempts.append(f"ArxivDatabase Shard {shard_id}: SUCCESS")
                else:
                    logger.error(f"Failed to repair ArxivDatabase shard {shard_id}")
                    repair_attempts.append(f"ArxivDatabase Shard {shard_id}: FAILED")
                    repair_success = False
                
            except Exception as e:
                logger.error(f"Error repairing ArxivDatabase shard {corrupted_file}: {e}")
                repair_attempts.append(f"ArxivDatabase Shard {corrupted_file}: ERROR - {str(e)}")
                repair_success = False
        
        # Update repair history
        repair_record = {
            "timestamp": datetime.now().isoformat(),
            "database_type": "ArxivDatabase",
            "corrupted_files": health_result.corrupted_files,
            "repair_attempts": repair_attempts,
            "success": repair_success
        }
        self.repair_history.append(repair_record)
        
        # Update performance stats
        if repair_success:
            self.performance_stats["successful_repairs"] += 1
        else:
            self.performance_stats["failed_repairs"] += 1
        
        logger.info(f"ArxivDatabase repair completed: {'SUCCESS' if repair_success else 'FAILED'}")
        return repair_success
    
    async def _monitoring_loop(self):
        """Main monitoring loop for continuous ArxivDatabase health checking."""
        logger.info("Starting continuous ArxivDatabase health monitoring")
        
        while self.is_monitoring:
            try:
                # Perform health check
                health_result = await self.perform_health_check()
                
                # Log health status
                logger.info(f"ArxivDatabase health status: {health_result.overall_health}")
                logger.info(f"Shards: {health_result.healthy_shards}H/{health_result.degraded_shards}D/{health_result.critical_shards}C")
                logger.info(f"Papers: {health_result.total_papers:,} total, {health_result.papers_with_vectors:,} with vectors")
                
                # Auto-repair if needed
                if health_result.overall_health in ["critical", "degraded"]:
                    logger.warning(f"ArxivDatabase health status is {health_result.overall_health}, attempting repair...")
                    await self.repair_corrupted_shards(health_result)
                
                # Wait for next check
                await asyncio.sleep(self.config.health_check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in ArxivDatabase monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def start_monitoring(self):
        """Start continuous ArxivDatabase health monitoring."""
        if self.is_monitoring:
            logger.warning("ArxivDatabase health monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=lambda: asyncio.run(self._monitoring_loop()))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("ArxivDatabase health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous ArxivDatabase health monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("ArxivDatabase health monitoring stopped")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current ArxivDatabase health summary."""
        summary = {
            "monitoring_active": self.is_monitoring,
            "last_check": asdict(self.last_health_check) if self.last_health_check else None,
            "performance_stats": self.performance_stats.copy(),
            "repair_history_count": len(self.repair_history),
            "config": asdict(self.config)
        }
        
        return summary
    
    def save_health_report(self, filepath: Optional[Path] = None) -> Path:
        """Save comprehensive ArxivDatabase health report to file."""
        if filepath is None:
            # Get project root and create health reports directory
            project_root = get_project_root()
            reports_dir = project_root / "agents" / "ResearchAgent" / "database" / "database_health_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = reports_dir / f"arxiv_database_health_report_{timestamp}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "database_type": "ArxivDatabase",
            "health_summary": self.get_health_summary(),
            "repair_history": self.repair_history[-10:],  # Last 10 repairs
            "arxiv_database_stats": self.arxiv_database.get_stats()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"ArxivDatabase health report saved to {filepath}")
        return filepath


class SafeArxivDatabaseOperations:
    """
    Wrapper for safe ArxivDatabase operations with automatic health monitoring.
    """
    
    def __init__(self, arxiv_database, enable_monitoring: bool = True):
        """
        Initialize safe ArxivDatabase operations.
        
        Args:
            arxiv_database: ArxivDatabase instance
            enable_monitoring: Whether to enable health monitoring
        """
        self.arxiv_database = arxiv_database
        
        # Initialize health monitor
        self.health_monitor = ArxivDatabaseHealthMonitor(arxiv_database) if enable_monitoring else None
        
        if self.health_monitor:
            self.health_monitor.start_monitoring()
        
        logger.info("SafeArxivDatabaseOperations initialized")
    
    def add_papers_safe(self, papers, embeddings=None) -> List[str]:
        """Safely add papers with ArxivDatabase health monitoring."""
        try:
            # Check health before operation
            if self.health_monitor and self.health_monitor.last_health_check:
                if self.health_monitor.last_health_check.overall_health == "critical":
                    logger.warning("ArxivDatabase health is critical, proceeding with caution")
            
            # Perform operation
            result = self.arxiv_database.add_papers(papers, embeddings)
            
            # Log operation
            logger.info(f"Successfully added {len(result)} papers to ArxivDatabase safely")
            return result
            
        except Exception as e:
            logger.error(f"Safe ArxivDatabase add_papers failed: {e}")
            # Trigger health check on failure
            if self.health_monitor:
                asyncio.create_task(self.health_monitor.perform_health_check())
            raise
    
    def update_papers_safe(self, papers, embeddings) -> bool:
        """Safely update papers with ArxivDatabase health monitoring."""
        try:
            # Check health before operation
            if self.health_monitor and self.health_monitor.last_health_check:
                if self.health_monitor.last_health_check.overall_health == "critical":
                    logger.warning("ArxivDatabase health is critical, proceeding with caution")
            
            # Perform operation
            result = self.arxiv_database.update_papers(papers, embeddings)
            
            # Log operation
            logger.info(f"Successfully updated {len(papers)} papers in ArxivDatabase safely")
            return result
            
        except Exception as e:
            logger.error(f"Safe ArxivDatabase update_papers failed: {e}")
            # Trigger health check on failure
            if self.health_monitor:
                asyncio.create_task(self.health_monitor.perform_health_check())
            raise
    
    def get_health_status(self) -> Optional[Dict[str, Any]]:
        """Get current ArxivDatabase health status."""
        if self.health_monitor:
            return self.health_monitor.get_health_summary()
        return None
    
    def close(self):
        """Close ArxivDatabase health monitoring."""
        if self.health_monitor:
            self.health_monitor.stop_monitoring()
