## Eric Secure Reverse Proxy Usage

This document describes how to run and use the secure reverse proxy in `service/eric-vpn`.

### Run

Run with the project's launcher (Windows PowerShell):

```bash
./run.bat service/eric-vpn/proxy_server.py
```

The server listens on host/port configured in `service/eric-vpn/config.py` (default `0.0.0.0:3000`).

### Configuration

- Edit `service/eric-vpn/config.py` to define:
  - `server`: host/port
  - `auth.api_keys`: optional API keys for requests (`X-API-Key` header)
  - `cors`: CORS policy (disabled by default)
  - `security`: SSRF protections, size/time limits, request header allowlist, X-Forwarded-* injection
  - `routes`: logical route map to upstream `base_url` and per-route policies

Example route config (uncomment and adjust):

```python
PROXY_SETTINGS["routes"]["example_httpbin"] = {
    "base_url": "https://httpbin.org",
    "allowed_methods": ["GET", "POST"],
    "allow_private": False,
    "allowed_cidrs": [],
    "preserve_host": False,
    "request_headers_allowlist": None,
    "add_x_forwarded_headers": True,
}
```

### Request Pattern

- Path format: `/r/{route_name}/{path}`
- Query string is forwarded as-is.
- Allowed methods per route are enforced.
- Standard hop-by-hop headers are stripped; `X-Forwarded-*` added (configurable).

Example:

```bash
curl -H "X-API-Key: <your-key>" \
     "http://127.0.0.1:3000/r/example_httpbin/get?hello=world"
```

### Security

- SSRF protections: block loopback/link-local/multicast/unspecified/private networks globally (private may be allowed per-route with optional CIDR allowlist).
- Only `http`/`https` upstreams are allowed.
- Request/response size limits are enforced.
- Optional API key authentication via `X-API-Key` header.

### Logging

- Integrated with `agent_engine.agent_logger.AgentLogger` using name `EricVPNProxy`.
- Each request is tagged with `X-Request-ID` (incoming or generated) and sensitive headers are masked in logs.

### Health

- `GET /health` returns `{"status": "healthy"}`


