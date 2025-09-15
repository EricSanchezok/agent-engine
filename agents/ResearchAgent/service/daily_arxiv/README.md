# Daily arXiv Service

This service provides automated daily processing of arXiv papers, including filtering, downloading, and analysis.

## Overview

The daily arXiv service consists of multiple steps:

1. **Step 1: Filter and Download** - Filter today's papers based on similarity to qiji library and download the most relevant ones
2. **Step 2: Analysis** - Analyze downloaded papers (to be implemented)
3. **Step 3: Report Generation** - Generate reports and summaries (to be implemented)

## Configuration

The service uses a configuration-based approach similar to `arxiv_sync_service`. All parameters are controlled through environment variables and the `config.py` file.

### Environment Variables

You can set these environment variables to configure the service:

```bash
# API Configuration
QZ_API_KEY=your_api_key_here
USE_ERIC_VPN=true
ERIC_VPN_URL=http://eric-vpn.cpolar.top/r/

# Model URLs (used when USE_ERIC_VPN=false)
QWEN3_EMBEDDING_8B_H100_URL=https://your-embedding-url
QWEN3_RERANKER_8B_H100_URL=https://your-reranker-url

# Service Parameters
DAILY_ARXIV_TOP_K=16                    # Number of top papers to select
DAILY_ARXIV_MAX_CONCURRENT=5            # Max concurrent downloads
DAILY_ARXIV_MAX_CONCURRENT_EMBEDDINGS=16 # Max concurrent embeddings
DAILY_ARXIV_TARGET_DATE=2025-09-15      # Target date (YYYY-MM-DD format)

# Filtering Parameters
DAILY_ARXIV_MIN_PAPERS=1               # Minimum papers required
DAILY_ARXIV_REQUIRE_VECTORS=false      # Whether to require vectors

# Download Parameters
DAILY_ARXIV_MAX_RETRIES=3              # Max download retries
DAILY_ARXIV_DOWNLOAD_TIMEOUT=300       # Download timeout in seconds
DAILY_ARXIV_DOWNLOAD_DELAY=1.0         # Delay between downloads

# Performance Parameters
DAILY_ARXIV_BATCH_SIZE=100             # Batch size for processing
DAILY_ARXIV_MEMORY_LIMIT_MB=1024       # Memory limit in MB
```

### Configuration Management

The service provides several commands to manage configuration:

```bash
# Show current configuration
python run_daily_arxiv.py --show-config

# Validate configuration
python run_daily_arxiv.py --validate-config

# Check service status
python run_daily_arxiv.py --status-only

# Run with custom parameters (overrides config)
python run_daily_arxiv.py --date 2025-09-15 --top-k 20 --max-concurrent 10
```

## Usage

### Basic Usage

```bash
# Run with default configuration
python run_daily_arxiv.py

# Run for a specific date
python run_daily_arxiv.py --date 2025-09-15

# Override specific parameters
python run_daily_arxiv.py --top-k 20 --max-concurrent 10
```

### Programmatic Usage

```python
import asyncio
from agents.ResearchAgent.service.daily_arxiv import DailyArxivService, DailyArxivConfig

async def main():
    # Show configuration
    DailyArxivConfig.print_config()
    
    # Validate configuration
    if not DailyArxivConfig.validate():
        print("Configuration validation failed")
        return
    
    # Initialize service
    service = DailyArxivService()
    
    # Check status
    status = await service.check_service_status()
    print(f"Service ready: {status['service_ready']}")
    
    # Run service
    result = await service.run_daily_service()
    print(f"Success: {result['service_success']}")

asyncio.run(main())
```

## Components

### DailyArxivConfig
Centralized configuration management with validation and environment variable support.

### DailyArxivFilterAndDownload
Handles the first step of the daily arXiv service:
- Checks if ArxivDatabase has updated papers for today
- Gets vectors for today's papers
- Calculates minimum distances using qiji_library
- Selects top K papers and downloads them

### DailyArxivService
Main service coordinator that orchestrates all steps of the daily processing pipeline.

## File Structure

```
daily_arxiv/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── filter_and_download.py   # Step 1: Filter and download
├── daily_arxiv_service.py   # Main service coordinator
├── run_daily_arxiv.py       # Command-line runner
└── README.md               # This file
```

## Dependencies

- ArxivDatabase: For retrieving papers and vectors
- QijiLibrary: For calculating similarity distances
- ArxivFetcher: For downloading PDFs
- AgentLogger: For logging

## Logging

The service uses AgentLogger for comprehensive logging. All operations are logged with appropriate levels:
- INFO: Normal operations and progress
- WARNING: Non-critical issues
- ERROR: Critical errors and failures

## Error Handling

The service includes comprehensive error handling:
- Database connection issues
- Network timeouts during downloads
- Invalid paper data
- File system errors
- Configuration validation errors

All errors are logged and reported in the service results.
