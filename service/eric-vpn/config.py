# Proxy Server Configuration

# Server settings
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 3000       # Default port

# Request settings
REQUEST_TIMEOUT = 30  # Timeout in seconds for forwarded requests
MAX_REDIRECTS = 5     # Maximum number of redirects to follow

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Security settings (for future use)
ALLOWED_HOSTS = []  # Empty list means allow all hosts
BLOCKED_HOSTS = []  # List of hosts to block

# Headers to remove when forwarding requests
HEADERS_TO_REMOVE = [
    'Host',
    'Content-Length', 
    'Transfer-Encoding',
    'Connection'
]

# Supported HTTP methods
SUPPORTED_METHODS = [
    'GET', 'POST', 'PUT', 'DELETE', 
    'PATCH', 'HEAD', 'OPTIONS'
]
