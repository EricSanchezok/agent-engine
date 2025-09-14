# Arxiv Sync Service

A robust service for synchronizing arXiv papers to the database. This service runs continuously and automatically syncs papers from a rolling 7-day window with embedding generation.

## Features

- **Continuous Operation**: Runs continuously and checks for new papers every 15 minutes
- **Rolling 7-Day Sync**: Syncs papers from today going back 7 days (rolling window)
- **Smart Filtering**: Only processes papers that don't exist or don't have embeddings
- **Concurrent Processing**: Generates embeddings concurrently for better performance
- **Robust Error Handling**: Includes retry logic and graceful error recovery
- **Comprehensive Logging**: Detailed logging for monitoring and debugging

## Configuration

The service uses environment variables for configuration. Create a `.env` file or set the following variables:

```bash
# Qz API Configuration
QZ_API_KEY=your_api_key_here
QWEN3_EMBEDDING_8B_H100_URL=https://your-endpoint.com

# VPN Configuration (optional)
USE_ERIC_VPN=false
ERIC_VPN_URL=http://eric-vpn.cpolar.top/r/
QWEN3_EMBEDDING_8B_H100_PROXY_ROUTE=eric_qwen3_embedding_8b_h100

# Database Configuration
DATABASE_NAME=arxiv_papers
DATABASE_DIR=path/to/database  # Optional, uses default if not set
```

## Usage

### Start Continuous Service

To start the service for continuous operation:

```bash
./run.bat service/arxiv_sync_service/start_service.py
```

The service will:
- Run continuously in the background
- Check for new papers every 15 minutes
- Sync papers from today going back 7 days (rolling window)
- Generate embeddings for new papers
- Handle errors gracefully with retries

### Run Once (Testing)

To run the sync once for testing or manual execution:

```bash
./run.bat service/arxiv_sync_service/run_once.py
```

### Configuration Validation

To validate your configuration:

```bash
./run.bat service/arxiv_sync_service/config.py
```

## Service Architecture

### Components

1. **ArxivSyncService**: Main service class that orchestrates the sync process
2. **ArxivFetcher**: Fetches papers from arXiv API
3. **QzClient**: Generates embeddings using Qz API
4. **ArxivDatabase**: Stores papers and embeddings using PodEMemory

### Sync Process

1. **Get Rolling 7-Day Window**: Determines dates from today going back 7 days
2. **Fetch Papers**: Retrieves all papers for each day in the rolling window
3. **Filter New Papers**: Removes papers that already exist with embeddings
4. **Generate Embeddings**: Creates embeddings for new papers concurrently
5. **Save to Database**: Stores papers and embeddings in the database
6. **Error Handling**: Retries failed operations with exponential backoff

### Error Handling

The service includes robust error handling:

- **Retry Logic**: Automatically retries failed operations up to 3 times
- **Graceful Degradation**: Continues operation even if individual papers fail
- **Comprehensive Logging**: Logs all errors and operations for debugging
- **Service Recovery**: Automatically recovers from temporary failures

## Monitoring

### Logs

Logs are saved to `service/arxiv_sync_service/logs/` and include:

- Service start/stop events
- Sync cycle information
- Paper processing statistics
- Error messages and stack traces
- Performance metrics

### Statistics

The service tracks various statistics:

- Total sync cycles
- Successful/failed syncs
- Total papers processed
- Total papers added
- Database statistics

### Health Checks

You can monitor the service by checking:

- Log files for errors or warnings
- Database statistics for growth
- Service process status

## Performance

### Concurrency

- **Embedding Generation**: Up to 16 concurrent embedding requests
- **Day Processing**: Sequential processing of days with small delays
- **Database Operations**: Batch operations for efficiency

### Optimization

- **Smart Filtering**: Only processes new papers
- **Concurrent Embeddings**: Parallel embedding generation
- **Batch Database Operations**: Efficient database writes
- **Configurable Delays**: Respectful API usage

## Troubleshooting

### Common Issues

1. **Configuration Errors**: Check environment variables and API keys
2. **Network Issues**: Verify VPN settings and network connectivity
3. **API Limits**: Monitor API usage and adjust concurrency if needed
4. **Database Issues**: Check database permissions and disk space

### Debug Mode

For debugging, you can:

1. Run `run_once.py` to test configuration
2. Check logs for detailed error information
3. Validate configuration with `config.py`
4. Monitor database statistics

### Service Recovery

If the service stops:

1. Check logs for error messages
2. Verify configuration and network connectivity
3. Restart the service
4. Monitor for recurring issues

## Development

### Code Structure

```
service/arxiv_sync_service/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── arxiv_sync_service.py    # Main service implementation
├── start_service.py         # Continuous service launcher
├── run_once.py              # Single run launcher
├── README.md                # This documentation
└── logs/                    # Log files directory
```

### Adding Features

To extend the service:

1. Modify `ArxivSyncService` class for new functionality
2. Update configuration in `config.py`
3. Add new command-line options if needed
4. Update documentation

### Testing

To test changes:

1. Use `run_once.py` for quick testing
2. Check logs for errors
3. Verify database updates
4. Test error scenarios

## API Reference

### ArxivSyncService Class

#### Methods

- `start()`: Start continuous service
- `stop()`: Stop service gracefully
- `run_once()`: Run sync once
- `get_service_stats()`: Get service statistics

#### Properties

- `is_running`: Service running status
- `last_sync_date`: Last successful sync time
- `sync_stats`: Sync statistics

### Configuration Options

- `SYNC_INTERVAL_MINUTES`: Sync frequency (default: 15)
- `MAX_CONCURRENT_EMBEDDINGS`: Concurrency limit (default: 16)
- `MAX_RETRY_ATTEMPTS`: Retry limit (default: 3)
- `RETRY_DELAY_SECONDS`: Retry delay (default: 5.0)

## License

This service is part of the Agent Engine project.
