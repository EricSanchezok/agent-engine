import requests
from flask import Flask, request, Response
import re
import logging
from urllib.parse import urljoin, urlparse
import time
from config import *

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ProxyServer:
    def __init__(self):
        self.session = requests.Session()
        # Set default headers for requests
        self.session.headers.update({
            'User-Agent': 'ProxyServer/1.0'
        })
    
    def parse_target_url(self, path):
        """
        Parse the target URL from the proxy path
        Expected format: /proxy/{target_host}/{target_port}/{remaining_path}
        """
        # Remove leading slash if present
        if path.startswith('/'):
            path = path[1:]
        
        # Split the path into components
        parts = path.split('/', 3)  # Split into max 4 parts
        
        if len(parts) < 3:
            raise ValueError("Invalid proxy path format. Expected: /proxy/{host}/{port}/{path}")
        
        if parts[0] != 'proxy':
            raise ValueError("Path must start with 'proxy'")
        
        target_host = parts[1]
        target_port = parts[2]
        remaining_path = parts[3] if len(parts) > 3 else ''
        
        # Construct target URL
        target_url = f"http://{target_host}:{target_port}"
        if remaining_path:
            target_url = urljoin(target_url + '/', remaining_path)
        
        return target_url
    
    def forward_request(self, target_url, method='GET', headers=None, data=None, params=None):
        """
        Forward the request to the target URL
        """
        try:
            logger.info(f"Forwarding {method} request to: {target_url}")
            
            # Prepare request parameters
            request_kwargs = {
                'method': method,
                'url': target_url,
                'headers': headers or {},
                'params': params,
                'timeout': REQUEST_TIMEOUT
            }
            
            # Add data/body for POST/PUT/PATCH requests
            if method in ['POST', 'PUT', 'PATCH'] and data is not None:
                request_kwargs['data'] = data
            
            # Add JSON data if content-type is application/json
            if headers and 'content-type' in headers.get('content-type', '').lower():
                if 'application/json' in headers.get('content-type', '').lower():
                    request_kwargs['json'] = data
                    if 'data' in request_kwargs:
                        del request_kwargs['data']
            
            # Make the request
            response = self.session.request(**request_kwargs)
            
            logger.info(f"Received response from {target_url}: {response.status_code}")
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error forwarding request to {target_url}: {str(e)}")
            raise

# Create proxy server instance
proxy_server = ProxyServer()

@app.route('/proxy/<path:proxy_path>', methods=SUPPORTED_METHODS)
def proxy_handler(proxy_path):
    """
    Main proxy handler for all proxy requests
    """
    try:
        # Parse the target URL from the proxy path
        target_url = proxy_server.parse_target_url(proxy_path)
        
        # Get request method
        method = request.method
        
        # Get request headers (excluding some headers that shouldn't be forwarded)
        headers = dict(request.headers)
        for header in HEADERS_TO_REMOVE:
            headers.pop(header, None)
        
        # Get request data
        data = None
        if method in ['POST', 'PUT', 'PATCH']:
            if request.is_json:
                data = request.get_json()
            else:
                data = request.get_data()
        
        # Get query parameters
        params = dict(request.args)
        
        # Forward the request
        response = proxy_server.forward_request(
            target_url=target_url,
            method=method,
            headers=headers,
            data=data,
            params=params
        )
        
        # Create Flask response
        flask_response = Response(
            response.content,
            status=response.status_code,
            headers=dict(response.headers)
        )
        
        return flask_response
        
    except ValueError as e:
        logger.error(f"Invalid proxy path: {str(e)}")
        return {'error': str(e)}, 400
    except Exception as e:
        logger.error(f"Proxy error: {str(e)}")
        return {'error': 'Internal proxy error'}, 500

@app.route('/health')
def health_check():
    """
    Health check endpoint
    """
    return {'status': 'healthy', 'timestamp': time.time()}

@app.route('/')
def index():
    """
    Root endpoint with usage instructions
    """
    return {
        'message': 'Dynamic Proxy Server',
        'usage': 'Use /proxy/{target_host}/{target_port}/{path} to proxy requests',
        'example': '/proxy/10.245.134.199/8000/api/users',
        'health': '/health'
    }

if __name__ == '__main__':
    logger.info("Starting Dynamic Proxy Server...")
    logger.info(f"Server will be available at http://{HOST}:{PORT}")
    logger.info(f"Example usage: http://{HOST}:{PORT}/proxy/10.245.134.199/8000/api/users")
    
    app.run(host=HOST, port=PORT, debug=True)
