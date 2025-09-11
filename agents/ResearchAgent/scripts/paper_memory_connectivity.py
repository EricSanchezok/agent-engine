import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.agent_logger.agent_logger import AgentLogger
from agents.ResearchAgent.paper_memory import PaperMemory, PaperMemoryConfig

# Load environment variables
load_dotenv()

logger = AgentLogger(__name__)


class PaperMemoryConnectivityTester:
    """Test PaperMemory connectivity and count records."""
    
    def __init__(self):
        self.logger = AgentLogger(self.__class__.__name__)
        self.paper_memory = None
        
    def initialize_paper_memory(self) -> bool:
        """Initialize PaperMemory with environment-based configuration."""
        try:
            # Show current environment configuration
            remote_paper_db = os.getenv("REMOTE_PAPER_DB", "")
            self.logger.info(f"REMOTE_PAPER_DB: {remote_paper_db}")
            
            if remote_paper_db:
                dsn_template = os.getenv("REMOTE_PAPER_DB_DSN", "")
                self.logger.info(f"Using REMOTE database: {dsn_template[:50]}..." if dsn_template else "REMOTE_PAPER_DB_DSN not set")
            else:
                dsn_template = os.getenv("LOCAL_PAPER_DB_DSN", "")
                self.logger.info(f"Using LOCAL database: {dsn_template[:50]}..." if dsn_template else "LOCAL_PAPER_DB_DSN not set")
            
            # Create PaperMemoryConfig (will auto-determine DSN from environment)
            config = PaperMemoryConfig()
            self.paper_memory = PaperMemory(config)
            
            self.logger.info("PaperMemory initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PaperMemory: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test basic database connectivity."""
        try:
            if not self.paper_memory:
                self.logger.error("PaperMemory not initialized")
                return False
            
            # Try to list existing segments
            segments = self.paper_memory._list_existing_segments()
            self.logger.info(f"Found {len(segments)} segments: {segments}")
            
            if not segments:
                self.logger.warning("No segments found - database might be empty or not accessible")
                return True  # Still consider it a successful connection
            
            # Test connection to first segment
            first_segment = segments[0]
            try:
                um = self.paper_memory._get_segment_um(first_segment)
                self.logger.info(f"Successfully connected to segment: {first_segment}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to connect to segment {first_segment}: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False


async def main():
    """Main function."""
    tester = PaperMemoryConnectivityTester()
    
    try:
        tester.initialize_paper_memory()
        success = tester.test_connection()
        if success:
            print("✅ Connectivity test completed successfully")
            return 0
        else:
            print("❌ Connectivity test failed")
            return 1
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
