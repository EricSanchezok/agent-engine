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
        [旧逻辑] 从 /proxy/... 格式的路径中解析目标URL
        Expected format: proxy/{target_host}/{target_port}/{remaining_path}
        """
        # Remove leading slash if present, as it's handled by the route
        parts = path.split('/', 3)  # Split into max 4 parts
        
        if len(parts) < 3:
            raise ValueError("Invalid proxy path format. Expected: proxy/{host}/{port}/{path}")
        
        # NOTE: The first part is now the host because the '/proxy/' part is consumed by the route
        target_host = parts[0]
        target_port = parts[1]
        remaining_path = parts[2] if len(parts) > 2 else ''
        
        # Construct target URL
        # 确保协议是http，因为这种格式没有指定协议
        target_url = f"http://{target_host}:{target_port}"
        if remaining_path:
            # urljoin 能够智能地处理斜杠
            target_url = urljoin(target_url + '/', remaining_path)
        
        return target_url
    
    def forward_request(self, target_url, method='GET', headers=None, data=None, params=None):
        """
        将请求转发到目标URL
        """
        try:
            logger.info(f"Forwarding {method} request to: {target_url} with params: {params}")
            
            # 准备请求参数
            request_kwargs = {
                'method': method,
                'url': target_url,
                'headers': headers or {},
                'params': params,
                'timeout': REQUEST_TIMEOUT,
                'allow_redirects': False # 代理通常不应自动处理重定向
            }
            
            # 为 POST/PUT/PATCH 请求添加请求体
            if method in ['POST', 'PUT', 'PATCH'] and data is not None:
                content_type = headers.get('content-type', '').lower()
                if 'application/json' in content_type:
                    request_kwargs['json'] = data
                else:
                    request_kwargs['data'] = data
            
            # 发出请求
            response = self.session.request(**request_kwargs)
            
            logger.info(f"Received response from {target_url}: {response.status_code}")
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error forwarding request to {target_url}: {str(e)}")
            raise

# 创建代理服务器实例
proxy_server = ProxyServer()

# --- 新增的通用代理处理逻辑 ---
# 这个函数会处理所有形式的代理请求，避免代码重复
def process_and_forward_request(target_url):
    try:
        method = request.method
        
        # 排除不应转发的请求头
        headers = {key: value for key, value in request.headers if key.lower() not in [h.lower() for h in HEADERS_TO_REMOVE]}
        
        data = request.get_data() # get_data() 可以同时处理json和form数据
        
        params = request.args.to_dict()
        
        # 转发请求
        response = proxy_server.forward_request(
            target_url=target_url,
            method=method,
            headers=headers,
            data=data,
            params=params
        )
        
        # 创建 Flask 响应，排除某些响应头
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        response_headers = [(name, value) for (name, value) in response.raw.headers.items() if name.lower() not in excluded_headers]
        
        flask_response = Response(
            response.content,
            status=response.status_code,
            headers=response_headers
        )
        
        return flask_response
        
    except ValueError as e:
        logger.error(f"Invalid proxy path: {str(e)}")
        return {'error': str(e)}, 400
    except Exception as e:
        logger.error(f"Proxy error: {str(e)}")
        return {'error': 'Internal proxy error'}, 500

# --- 新增的、更灵活的 "catch-all" 路由 ---
# 这个路由会捕获所有看起来像URL的路径
@app.route('/<path:full_url>', methods=SUPPORTED_METHODS)
def catch_all_proxy(full_url):
    """
    处理形如 /http://... 或 /https://... 的请求
    """
    # 检查捕获的路径是否以 http:// 或 https:// 开头
    if full_url.startswith('http://') or full_url.startswith('https://'):
        # 它本身就是一个完整的URL，直接用它作为目标地址
        target_url = full_url
        # 如果原始请求有查询参数，需要附加到目标URL上
        if request.query_string:
            target_url += '?' + request.query_string.decode('utf-8')
        
        logger.info(f"Catch-all route matched. Target URL: {target_url}")
        return process_and_forward_request(target_url)
    
    # 如果不是我们期望的URL格式，返回404
    logger.warning(f"Catch-all route received a non-URL path: {full_url}")
    return {'error': 'Not Found. The path is not a valid proxy request.'}, 404

# --- 保留的原有路由 ---
@app.route('/proxy/<path:proxy_path>', methods=SUPPORTED_METHODS)
def proxy_handler(proxy_path):
    """
    处理 /proxy/{host}/{port}/{path} 格式的请求
    """
    logger.info(f"Proxy route matched. Path: {proxy_path}")
    target_url = proxy_server.parse_target_url(proxy_path)
    return process_and_forward_request(target_url)

@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': time.time()}

@app.route('/')
def index():
    return {
        'message': 'Enhanced Dynamic Proxy Server',
        'usage': 'Use /full_target_url_with_http',
        'example_2': 'base_url/http://10.245.134.199:8000/api/users',
        'health': '/health'
    }

if __name__ == '__main__':
    logger.info("Starting Enhanced Dynamic Proxy Server...")
    logger.info(f"Server will be available at http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)