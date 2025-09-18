"""
Arxiv Papers Embedding Generator

This script generates embeddings for arXiv papers by:
1. Reading papers from database in daily batches
2. Filtering papers that don't have embeddings yet
3. Generating embeddings concurrently using QzClient
4. Updating database with new embeddings
5. Logging failed paper IDs to JSON files

This is the second step of the two-step preloading process.
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

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client.qz_client import QzClient

from core.arxiv_fetcher.arxiv_paper import ArxivPaper
from agents.ResearchAgent.arxiv_database import ArxivDatabase
from agents.ResearchAgent.arxiv_database_health_monitor import SafeArxivDatabaseOperations, ArxivRepairConfig


class ArxivEmbeddingGenerator:
    """Arxiv embedding generator with batch processing and concurrent embedding generation."""
    
    def __init__(
        self,
        qz_api_key: str,
        qz_base_url: str,
        embedding_model: str = "text-embedding-3-small",
        max_concurrent_embeddings: int = 10,
        database_name: str = "arxiv_papers",
        database_dir: Optional[str] = None,
        batch_size: int = 1000,
        enable_health_monitoring: bool = True
    ):
        """
        Initialize the Arxiv embedding generator.
        
        Args:
            qz_api_key: Qz API key for embedding generation
            qz_base_url: Qz API base URL
            embedding_model: Model name for embeddings
            max_concurrent_embeddings: Maximum concurrent embedding requests
            database_name: Name for the ArxivDatabase
            database_dir: Directory for database storage
            batch_size: Number of papers to process in each batch
            enable_health_monitoring: Whether to enable database health monitoring
        """
        self.logger = AgentLogger(self.__class__.__name__)
        
        # Initialize components
        self.qz_client = QzClient(api_key=qz_api_key, base_url=qz_base_url)
        self.arxiv_database = ArxivDatabase(
            name=database_name,
            persist_dir=database_dir
        )
        
        # Initialize safe ArxivDatabase operations with health monitoring
        self.safe_db = SafeArxivDatabaseOperations(
            self.arxiv_database, 
            enable_monitoring=enable_health_monitoring
        )
        
        # Configuration
        self.embedding_model = embedding_model
        self.max_concurrent_embeddings = max_concurrent_embeddings
        self.batch_size = batch_size
        
        # Create failed IDs directory
        self.failed_ids_dir = Path(__file__).parent.parent / "database" / "embed_failed_ids"
        self.failed_ids_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ArxivEmbeddingGenerator initialized with embedding model: {embedding_model}")
        self.logger.info(f"Batch size: {batch_size}, Max concurrent embeddings: {max_concurrent_embeddings}")
        self.logger.info(f"Health monitoring enabled: {enable_health_monitoring}")
        self.logger.info(f"Failed IDs will be saved to: {self.failed_ids_dir}")
    
    async def close(self):
        """Close all connections."""
        await self.qz_client.close()
        self.safe_db.close()
        self.logger.info("ArxivEmbeddingGenerator connections closed")
    
    def get_health_status(self):
        """Get current database health status."""
        return self.safe_db.get_health_status()
    
    def _get_day_range(self, target_date: datetime) -> Tuple[datetime, datetime]:
        """
        Get the day range for a given target date.
        
        Args:
            target_date: Target date to search papers for
            
        Returns:
            Tuple of (day_start, day_end) dates
        """
        # Start of the target day
        day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        # End of the target day (start of next day)
        day_end = day_start + timedelta(days=1)
        
        return day_start, day_end
    
    def _format_date_range_name(self, start_date: datetime, end_date: datetime) -> str:
        """Format date range for file naming."""
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return f"{start_str}_{end_str}"
    
    def _get_papers_for_day_range(self, start_date: datetime, end_date: datetime) -> List[ArxivPaper]:
        """
        Get all papers from database for a given day range.
        
        Args:
            start_date: Day start date
            end_date: Day end date (next day)
            
        Returns:
            List of ArxivPaper objects
        """
        day_name = self._format_date_range_name(start_date, end_date)
        
        self.logger.info(f"Loading papers from database for day range {day_name}")
        
        try:
            # Get papers from database for the specific date (start_date)
            # Note: We use start_date since ArxivDatabase.get_papers_by_date takes a single date
            papers = self.arxiv_database.get_papers_by_date(start_date)
            
            self.logger.info(f"Found {len(papers)} papers in database for day range {day_name}")
            return papers
            
        except Exception as e:
            self.logger.error(f"Failed to load papers for day range {day_name}: {e}")
            return []
    
    def _filter_papers_without_embeddings(self, papers: List[ArxivPaper]) -> List[ArxivPaper]:
        """
        Filter papers that don't have embeddings yet.
        
        Args:
            papers: List of ArxivPaper objects
            
        Returns:
            List of papers that need embeddings
        """
        if not papers:
            return []
        
        # Use batch method for efficient checking
        self.logger.info(f"Checking embeddings for {len(papers)} papers in batch")
        vector_map = self.arxiv_database.has_vector_batch(papers)
        
        # Filter papers that don't have vectors
        papers_needing_embeddings = []
        for paper in papers:
            paper_id = paper.full_id
            has_vector = vector_map.get(paper_id, False)
            
            if not has_vector:
                papers_needing_embeddings.append(paper)
        
        self.logger.info(f"Filtered {len(papers)} papers to {len(papers_needing_embeddings)} papers needing embeddings")
        return papers_needing_embeddings
    
    async def _generate_embedding(self, paper: ArxivPaper) -> Optional[List[float]]:
        """
        Generate embedding for a single paper's summary.
        
        Args:
            paper: ArxivPaper object
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            if not paper.summary.strip():
                self.logger.warning(f"Paper {paper.full_id} has empty summary")
                return None
            
            embedding = await self.qz_client.get_embeddings(
                model_name=self.embedding_model,
                text=paper.summary
            )
            
            if embedding is None:
                self.logger.warning(f"Failed to generate embedding for paper {paper.full_id}")
                return None
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding for paper {paper.full_id}: {e}")
            return None
    
    async def _generate_embeddings_concurrent(
        self, 
        papers: List[ArxivPaper]
    ) -> Tuple[List[Tuple[ArxivPaper, List[float]]], List[ArxivPaper]]:
        """
        Generate embeddings for multiple papers concurrently.
        
        Args:
            papers: List of ArxivPaper objects
            
        Returns:
            Tuple of (successful_papers_with_embeddings, failed_papers)
        """
        if not papers:
            return [], []
        
        self.logger.info(f"Generating embeddings for {len(papers)} papers")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_embeddings)
        
        async def process_paper(paper: ArxivPaper) -> Tuple[ArxivPaper, Optional[List[float]]]:
            async with semaphore:
                embedding = await self._generate_embedding(paper)
                return paper, embedding
        
        # Process all papers concurrently
        tasks = [process_paper(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful and failed results
        successful_papers = []
        failed_papers = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Exception processing paper {papers[i].full_id}: {result}")
                failed_papers.append(papers[i])
            else:
                paper, embedding = result
                if embedding is not None:
                    successful_papers.append((paper, embedding))
                else:
                    failed_papers.append(paper)
        
        self.logger.info(f"Embedding generation completed: {len(successful_papers)} successful, {len(failed_papers)} failed")
        return successful_papers, failed_papers
    
    async def _update_papers_embeddings(
        self, 
        papers_with_embeddings: List[Tuple[ArxivPaper, List[float]]]
    ) -> List[str]:
        """
        Update papers with their embeddings in the database using batch update.
        
        Args:
            papers_with_embeddings: List of (paper, embedding) tuples
            
        Returns:
            List of successfully updated paper IDs
        """
        if not papers_with_embeddings:
            return []
        
        self.logger.info(f"Updating {len(papers_with_embeddings)} papers with embeddings in database (batch)")
        
        try:
            # Separate papers and embeddings for batch update
            papers = [paper for paper, _ in papers_with_embeddings]
            embeddings = [embedding for _, embedding in papers_with_embeddings]
            
            # Use batch update method for better performance with safe operations
            success = self.safe_db.update_papers_safe(papers, embeddings)
            
            if success:
                successful_ids = [paper.full_id for paper in papers]
                self.logger.info(f"Successfully updated {len(successful_ids)} papers with embeddings in batch")
                return successful_ids
            else:
                self.logger.error("Failed to update papers with embeddings in batch")
                return []
            
        except Exception as e:
            self.logger.error(f"Failed to update papers with embeddings: {e}")
            return []
    
    def _save_failed_ids(self, failed_papers: List[ArxivPaper], batch_name: str):
        """
        Save failed paper IDs to JSON file.
        
        Args:
            failed_papers: List of failed ArxivPaper objects
            batch_name: Batch identifier for filename
        """
        if not failed_papers:
            return
        
        failed_ids = [paper.full_id for paper in failed_papers]
        
        # Create filename
        filename = f"{batch_name}.json"
        filepath = self.failed_ids_dir / filename
        
        # Save to JSON
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "batch_range": batch_name,
                    "failed_count": len(failed_ids),
                    "failed_ids": failed_ids,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(failed_ids)} failed IDs to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save failed IDs to {filepath}: {e}")
    
    async def _process_batch(self, papers: List[ArxivPaper], batch_name: str) -> dict:
        """
        Process a batch of papers for embedding generation.
        
        Args:
            papers: List of ArxivPaper objects
            batch_name: Name identifier for this batch
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing batch: {batch_name} with {len(papers)} papers")
        
        # Perform health check before processing batch
        if self.safe_db.health_monitor:
            health_result = await self.safe_db.health_monitor.perform_health_check()
            if health_result.overall_health == "critical":
                self.logger.error(f"ArxivDatabase health is critical, skipping batch {batch_name}")
                return {
                    "batch": batch_name,
                    "total_papers": len(papers),
                    "filtered_papers": 0,
                    "successful_embeddings": 0,
                    "failed_embeddings": 0,
                    "updated_papers": 0,
                    "health_status": "critical"
                }
            elif health_result.overall_health == "degraded":
                self.logger.warning(f"ArxivDatabase health is degraded, proceeding with caution for batch {batch_name}")
        
        if not papers:
            return {
                "batch": batch_name,
                "total_papers": 0,
                "filtered_papers": 0,
                "successful_embeddings": 0,
                "failed_embeddings": 0,
                "updated_papers": 0
            }
        
        # Filter papers that don't have embeddings yet
        papers_needing_embeddings = self._filter_papers_without_embeddings(papers)
        filtered_count = len(papers) - len(papers_needing_embeddings)
        
        if not papers_needing_embeddings:
            self.logger.info(f"All papers in batch {batch_name} already have embeddings")
            return {
                "batch": batch_name,
                "total_papers": len(papers),
                "filtered_papers": filtered_count,
                "successful_embeddings": 0,
                "failed_embeddings": 0,
                "updated_papers": 0
            }
        
        # Generate embeddings concurrently for papers that need them
        successful_papers, failed_papers = await self._generate_embeddings_concurrent(papers_needing_embeddings)
        
        # Update papers with embeddings in database
        updated_paper_ids = await self._update_papers_embeddings(successful_papers)
        
        # Save failed IDs to file
        if failed_papers:
            self._save_failed_ids(failed_papers, batch_name)
        
        result = {
            "batch": batch_name,
            "total_papers": len(papers),
            "filtered_papers": filtered_count,
            "successful_embeddings": len(successful_papers),
            "failed_embeddings": len(failed_papers),
            "updated_papers": len(updated_paper_ids)
        }
        
        self.logger.info(f"Batch {batch_name} processed: {result}")
        return result
    
    async def embed_days(self, num_days: int = 10) -> List[dict]:
        """
        Generate embeddings for papers in the specified number of days, rolling back from today.
        
        Args:
            num_days: Number of days to process
            
        Returns:
            List of processing results for each day
        """
        from preload_config import PreloadConfig
        
        self.logger.info(f"Starting embedding generation for {num_days} days")
        
        results = []
        current_date = datetime.now()
        
        for day_offset in range(num_days):
            # Calculate the day range
            target_date = current_date - timedelta(days=day_offset)
            day_start, day_end = self._get_day_range(target_date)
            day_name = self._format_date_range_name(day_start, day_end)
            
            # Get papers for this day from database
            papers = self._get_papers_for_day_range(day_start, day_end)
            
            if not papers:
                self.logger.info(f"No papers found in database for day {day_name}")
                results.append({
                    "day": day_name,
                    "total_papers": 0,
                    "filtered_papers": 0,
                    "successful_embeddings": 0,
                    "failed_embeddings": 0,
                    "updated_papers": 0
                })
                continue
            
            # Process papers in batches if there are too many
            if len(papers) <= self.batch_size:
                # Process all papers as one batch
                result = await self._process_batch(papers, day_name)
                result["day"] = day_name
                results.append(result)
            else:
                # Process papers in multiple batches
                batch_results = []
                for i in range(0, len(papers), self.batch_size):
                    batch_papers = papers[i:i + self.batch_size]
                    batch_name = f"{day_name}_batch_{i//self.batch_size + 1}"
                    batch_result = await self._process_batch(batch_papers, batch_name)
                    batch_results.append(batch_result)
                
                # Combine batch results
                combined_result = {
                    "day": day_name,
                    "total_papers": sum(r["total_papers"] for r in batch_results),
                    "filtered_papers": sum(r["filtered_papers"] for r in batch_results),
                    "successful_embeddings": sum(r["successful_embeddings"] for r in batch_results),
                    "failed_embeddings": sum(r["failed_embeddings"] for r in batch_results),
                    "updated_papers": sum(r["updated_papers"] for r in batch_results)
                }
                results.append(combined_result)
            
            # Small delay between days to be respectful to APIs
            await asyncio.sleep(PreloadConfig.DELAY_BETWEEN_DAYS)
        
        # Summary
        total_papers = sum(r["total_papers"] for r in results)
        total_filtered = sum(r["filtered_papers"] for r in results)
        total_successful = sum(r["successful_embeddings"] for r in results)
        total_failed = sum(r["failed_embeddings"] for r in results)
        total_updated = sum(r["updated_papers"] for r in results)
        
        self.logger.info(f"Embedding generation completed:")
        self.logger.info(f"  Total papers processed: {total_papers}")
        self.logger.info(f"  Papers already have embeddings: {total_filtered}")
        self.logger.info(f"  Successful embeddings: {total_successful}")
        self.logger.info(f"  Failed embeddings: {total_failed}")
        self.logger.info(f"  Updated in database: {total_updated}")
        
        return results


async def main():
    """Main execution function."""
    from preload_config import PreloadConfig
    
    # Validate configuration
    if not PreloadConfig.validate():
        return
    
    PreloadConfig.print_config()
    
    # Initialize embedding generator
    generator = ArxivEmbeddingGenerator(
        qz_api_key=PreloadConfig.QZ_API_KEY,
        qz_base_url=PreloadConfig.QZ_BASE_URL,
        embedding_model=PreloadConfig.EMBEDDING_MODEL,
        max_concurrent_embeddings=PreloadConfig.MAX_CONCURRENT_EMBEDDINGS,
        database_name=PreloadConfig.DATABASE_NAME,
        database_dir=PreloadConfig.DATABASE_DIR
    )
    
    try:
        # Generate embeddings for specified number of days
        results = await generator.embed_days(num_days=PreloadConfig.DEFAULT_NUM_DAYS)
        
        # Print summary
        print("\n=== Embedding Generation Summary ===")
        for result in results:
            print(f"Day {result['day']}: {result['total_papers']} papers processed, "
                  f"{result['filtered_papers']} already have embeddings, "
                  f"{result['successful_embeddings']} successful, "
                  f"{result['failed_embeddings']} failed, "
                  f"{result['updated_papers']} updated")
        
        # Print health status and save report
        health_status = generator.get_health_status()
        if health_status:
            print("\n=== ArxivDatabase Health Status ===")
            print(f"Monitoring Active: {health_status['monitoring_active']}")
            if health_status['last_check']:
                last_check = health_status['last_check']
                print(f"Overall Health: {last_check['overall_health']}")
                print(f"Healthy Shards: {last_check['healthy_shards']}/{last_check['total_shards']}")
                print(f"Total Papers: {last_check['total_papers']:,}")
                print(f"Papers with Vectors: {last_check['papers_with_vectors']:,}")
            
            # Save health report to file
            if generator.safe_db.health_monitor:
                report_path = generator.safe_db.health_monitor.save_health_report()
                print(f"Health report saved to: {report_path}")
        
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await generator.close()


if __name__ == "__main__":
    asyncio.run(main())
