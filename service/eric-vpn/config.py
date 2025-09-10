"""
Configuration for the secure reverse proxy (Eric VPN).

All parameters are defined in code to match the project's preference of manual in-code configuration.
"""

from typing import Dict, Any, List


# Global supported HTTP methods for routing. Per-route methods can further restrict.
GLOBAL_SUPPORTED_METHODS: List[str] = [
    "GET",
    "POST",
    "PUT",
    "DELETE",
    "PATCH",
    "HEAD",
    "OPTIONS",
]


PROXY_SETTINGS: Dict[str, Any] = {
    "logger_name": "EricVPNProxy",

    "server": {
        "host": "0.0.0.0",
        "port": 3000,
    },

    # Authentication configuration
    # - If "api_keys" is non-empty, incoming requests must include header: X-API-Key: <key>
    # - If empty, authentication is disabled
    "auth": {
        "api_keys": [],  # e.g., ["change-me-123"]
        # Reserved for JWT in the future
        "jwt": {
            "enabled": False,
            "issuer": "",
            "audience": "",
            "jwks_url": "",
        },
    },

    # CORS settings (disabled by default)
    "cors": {
        "enabled": False,
        "allow_origins": ["*"],
        "allow_credentials": False,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "expose_headers": [],
        "max_age": 600,
    },

    # Security and limits
    "security": {
        # SSRF protections
        "block_private_networks": True,
        "block_loopback": True,
        "block_link_local": True,
        "block_multicast": True,
        "block_unspecified": True,
        # Explicitly blocked hostnames
        "blocked_hostnames": [
            "169.254.169.254",  # AWS/GCP metadata default
            "metadata.google.internal",
        ],
        # Default CIDR allowlist (empty means no restriction beyond the above blocks)
        "default_allowed_cidrs": [],

        # Size limits (in bytes)
        "max_request_body_bytes": 10 * 1024 * 1024,   # 10MB
        "max_response_body_bytes": 50 * 1024 * 1024,  # 50MB

        # Default timeouts (seconds)
        "timeouts": {
            "connect": 5.0,
            "read": 60.0,
            "write": 60.0,
            "pool": 60.0,
        },

        # Default connection limits
        "limits": {
            "max_connections": 128,
            "max_keepalive_connections": 32,
        },

        # Default request header allowlist (case-insensitive match)
        "request_headers_allowlist": [
            "accept",
            "accept-encoding",
            "accept-language",
            "content-type",
            "authorization",
            "user-agent",
            "x-request-id",
        ],

        # Whether to add X-Forwarded-* headers
        "add_x_forwarded_headers": True,
    },

    # Route table: map logical route name to upstream base URL and policies
    # Callers will request: /r/{route_name}/{path}
    "routes": {
        # Example route (disabled by default, adjust as needed)
        # "example_httpbin": {
        #     "base_url": "https://httpbin.org",
        #     "allowed_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        #     "allow_private": False,                # whether to allow private IPs for this route
        #     "allowed_cidrs": [],                   # optional CIDR allowlist for this route
        #     "preserve_host": False,                # whether to forward original Host header
        #     "request_headers_allowlist": None,     # None means use global allowlist
        #     "timeouts": None,                      # None means use global timeouts
        #     "limits": None,                        # None means use global limits
        #     "add_x_forwarded_headers": True,
        # },
    },

    # Global method set for router registration
    "global_supported_methods": GLOBAL_SUPPORTED_METHODS,
}


