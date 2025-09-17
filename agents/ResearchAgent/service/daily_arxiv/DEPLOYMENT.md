# Daily arXiv API Server - Deployment Guide

## Quick Start

### 1. Start the API Server

```bash
# From project root directory
uv run --extra opts agents/ResearchAgent/service/daily_arxiv/start_api_server.py
```

### 2. Start with Custom Configuration

```bash
# Custom host and port
uv run --extra opts agents/ResearchAgent/service/daily_arxiv/start_api_server.py --host 0.0.0.0 --port 8080

# Enable debug mode
uv run --extra opts agents/ResearchAgent/service/daily_arxiv/start_api_server.py --debug
```

### 3. Test the API

```bash
# Run the test script
uv run --extra opts agents/ResearchAgent/service/daily_arxiv/test_api.py

# Test with custom URL
uv run --extra opts agents/ResearchAgent/service/daily_arxiv/test_api.py http://your-server:5000
```

## Server Configuration

### Default Settings
- **Host**: 0.0.0.0 (accessible from network)
- **Port**: 5000
- **Debug**: False

### Environment Variables
The server will automatically use the following environment variables if available:
- `AZURE_API_KEY`: For LLM services (if needed)
- `ARXIV_DATABASE_DIR`: Override default database directory

### Logging
- Logs are written to: `agents/ResearchAgent/service/daily_arxiv/logs/`
- Log level: INFO (can be changed in code)

## Network Access

The server uses `agent_engine.utils.network_utils.get_local_ip()` to display the local IP address for easy access.

### Access URLs
- **Local**: http://localhost:5000
- **Network**: http://[your-ip]:5000

### Firewall Configuration
Make sure port 5000 (or your custom port) is open in your firewall.

## Dependencies

Required packages (automatically installed with `--extra opts`):
- `flask>=3.1.2`
- `flask-cors>=6.0.1`
- `requests>=2.32.3`

## Production Deployment

### Using a WSGI Server

For production, consider using a WSGI server like Gunicorn:

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 agents.ResearchAgent.service.daily_arxiv.api_server:app
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[opts]"

EXPOSE 5000

CMD ["python", "agents/ResearchAgent/service/daily_arxiv/start_api_server.py"]
```

### Using systemd (Linux)

Create `/etc/systemd/system/daily-arxiv-api.service`:

```ini
[Unit]
Description=Daily arXiv API Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/agent-engine
ExecStart=/path/to/uv run --extra opts agents/ResearchAgent/service/daily_arxiv/start_api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Monitoring

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Status Check
```bash
curl http://localhost:5000/api/status
```

### Log Monitoring
```bash
tail -f agents/ResearchAgent/service/daily_arxiv/logs/agent_logger_*.log
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Use a different port
   uv run --extra opts agents/ResearchAgent/service/daily_arxiv/start_api_server.py --port 8080
   ```

2. **Permission denied**
   ```bash
   # Make sure you have write permissions to the logs directory
   chmod 755 agents/ResearchAgent/service/daily_arxiv/logs/
   ```

3. **Module not found**
   ```bash
   # Make sure you're using --extra opts
   uv run --extra opts agents/ResearchAgent/service/daily_arxiv/start_api_server.py
   ```

4. **Database directory not found**
   - The server will create the directory structure automatically
   - Make sure the parent directory is writable

### Debug Mode

Enable debug mode for detailed error information:

```bash
uv run --extra opts agents/ResearchAgent/service/daily_arxiv/start_api_server.py --debug
```

## Security Considerations

1. **Network Access**: The server binds to 0.0.0.0 by default, making it accessible from the network
2. **Authentication**: Currently no authentication is implemented
3. **Rate Limiting**: No rate limiting is currently implemented
4. **HTTPS**: For production, consider using HTTPS with a reverse proxy like nginx

## Performance

- **Concurrent Requests**: Flask development server handles multiple requests
- **Memory Usage**: Minimal memory footprint
- **Response Time**: Typically < 100ms for most requests
- **File I/O**: Reads markdown files and JSON metadata on each request
- **Metadata Extraction**: Automatically extracts authors, categories, and other paper metadata from stored JSON files

For high-traffic scenarios, consider:
- Using a production WSGI server
- Implementing caching
- Adding rate limiting
- Using a reverse proxy
