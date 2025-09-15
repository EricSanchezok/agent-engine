"""
AutoService module for automatic HTTP service generation

This module provides a decorator and class for automatically exposing
class methods as HTTP endpoints.
"""

import inspect
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

from ..agent_logger.agent_logger import AgentLogger
from ..utils.network_utils import get_local_ip

logger = AgentLogger('AutoService')

class ExposeMetadata:
    """Metadata for exposed methods"""
    
    def __init__(self, 
                 route: Optional[str] = None,
                 methods: List[str] = None,
                 description: Optional[str] = None,
                 tags: List[str] = None):
        self.route = route
        self.methods = methods or ['GET']
        self.description = description
        self.tags = tags or []

def expose(route: Optional[str] = None,
           methods: Union[str, List[str]] = 'GET',
           description: Optional[str] = None,
           tags: Optional[List[str]] = None):
    """
    Decorator to mark methods for HTTP exposure
    
    Args:
        route: Custom route path (defaults to method name)
        methods: HTTP methods allowed (default: 'GET')
        description: Description for API documentation
        tags: Tags for API documentation
    
    Example:
        @expose('/api/process', methods=['POST'], description='Process data')
        def process_data(self, data: dict):
            return {"result": "processed"}
    """
    if isinstance(methods, str):
        methods = [methods]
    
    def decorator(func: Callable) -> Callable:
        # Store metadata in function
        func._expose_metadata = ExposeMetadata(
            route=route,
            methods=methods,
            description=description,
            tags=tags or []
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Copy metadata to wrapper
        wrapper._expose_metadata = func._expose_metadata
        return wrapper
    
    return decorator

class AutoService:
    """
    Automatic HTTP service generator
    
    This class can wrap any object and automatically expose its decorated
    methods as HTTP endpoints.
    """
    
    def __init__(self, 
                 target_object: Any,
                 host: Optional[str] = None,
                 port: int = 8000,
                 title: str = "AutoService API",
                 description: str = "Automatically generated API",
                 version: str = "1.0.0"):
        """
        Initialize AutoService
        
        Args:
            target_object: Object whose methods will be exposed
            host: Host IP address (default: auto-detect)
            port: Port number (default: 8000)
            title: API title
            description: API description
            version: API version
        """
        self.target_object = target_object
        self.host = host or get_local_ip()
        self.port = port
        self.title = title
        self.description = description
        self.version = version
        
        self.app = FastAPI(
            title=self.title,
            description=self.description,
            version=self.version
        )
        
        self.server_thread = None
        self.server_running = False
        
        # Discover and register exposed methods
        self._discover_and_register_methods()
        
        logger.info(f"AutoService initialized for {type(target_object).__name__}")
        logger.info(f"Service will run on {self.host}:{self.port}")
    
    def _discover_and_register_methods(self):
        """Discover methods with @expose decorator and register them as routes"""
        exposed_methods = []
        
        # Get all methods from the target object
        for name, method in inspect.getmembers(self.target_object, predicate=inspect.ismethod):
            if hasattr(method, '_expose_metadata'):
                metadata = method._expose_metadata
                exposed_methods.append((name, method, metadata))
                logger.info(f"Found exposed method: {name}")
        
        # Register each exposed method as a route
        for name, method, metadata in exposed_methods:
            self._register_method_as_route(name, method, metadata)
    
    def _register_method_as_route(self, method_name: str, method: Callable, metadata: ExposeMetadata):
        """Register a method as a FastAPI route"""
        # Determine route path
        route_path = metadata.route or f"/{method_name}"
        
        # Ensure route starts with /
        if not route_path.startswith('/'):
            route_path = f"/{route_path}"
        
        # Get method signature for parameter handling
        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        
        # Remove 'self' parameter
        if params and params[0].name == 'self':
            params = params[1:]
        
        # Create route handler
        async def route_handler(request: Request):
            try:
                # Parse request data
                request_data = await self._parse_request_data(request, params)
                
                # Call the method
                if params:
                    result = method(**request_data)
                else:
                    result = method()
                
                # Handle async methods
                if inspect.iscoroutine(result):
                    result = await result
                
                return JSONResponse(content={"result": result})
                
            except Exception as e:
                logger.error(f"Error in route {route_path}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Register route for each HTTP method
        for http_method in metadata.methods:
            if http_method.upper() == 'GET':
                self.app.get(route_path, tags=metadata.tags, summary=metadata.description)(route_handler)
            elif http_method.upper() == 'POST':
                self.app.post(route_path, tags=metadata.tags, summary=metadata.description)(route_handler)
            elif http_method.upper() == 'PUT':
                self.app.put(route_path, tags=metadata.tags, summary=metadata.description)(route_handler)
            elif http_method.upper() == 'DELETE':
                self.app.delete(route_path, tags=metadata.tags, summary=metadata.description)(route_handler)
            elif http_method.upper() == 'PATCH':
                self.app.patch(route_path, tags=metadata.tags, summary=metadata.description)(route_handler)
        
        logger.info(f"Registered route: {route_path} [{', '.join(metadata.methods)}]")
    
    async def _parse_request_data(self, request: Request, params: List[inspect.Parameter]) -> Dict[str, Any]:
        """Parse request data based on method parameters"""
        request_data = {}
        
        if not params:
            return request_data
        
        # Get request body for POST/PUT/PATCH
        if request.method in ['POST', 'PUT', 'PATCH']:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    request_data.update(body)
            except Exception:
                pass
        
        # Get query parameters
        query_params = dict(request.query_params)
        request_data.update(query_params)
        
        # Get path parameters
        path_params = dict(request.path_params)
        request_data.update(path_params)
        
        # Filter parameters based on method signature
        filtered_data = {}
        for param in params:
            param_name = param.name
            if param_name in request_data:
                value = request_data[param_name]
                
                # Type conversion based on annotation
                if param.annotation != inspect.Parameter.empty:
                    try:
                        if param.annotation == int:
                            value = int(value)
                        elif param.annotation == float:
                            value = float(value)
                        elif param.annotation == bool:
                            value = value.lower() in ('true', '1', 'yes', 'on')
                        elif param.annotation == str:
                            value = str(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Type conversion failed for {param_name}: {e}")
                
                filtered_data[param_name] = value
            elif param.default == inspect.Parameter.empty:
                # Required parameter not provided
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required parameter: {param_name}"
                )
        
        return filtered_data
    
    def start(self, block: bool = False):
        """
        Start the HTTP server
        
        Args:
            block: If True, block the current thread. If False, start in background.
        """
        if self.server_running:
            logger.warning("Server is already running")
            return
        
        def run_server():
            try:
                logger.info(f"Starting server on {self.host}:{self.port}")
                uvicorn.run(
                    self.app,
                    host=self.host,
                    port=self.port,
                    log_level="info"
                )
            except Exception as e:
                logger.error(f"Server error: {e}")
            finally:
                self.server_running = False
        
        if block:
            run_server()
        else:
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            self.server_running = True
            logger.info("Server started in background thread")
    
    def stop(self):
        """Stop the HTTP server"""
        if not self.server_running:
            logger.warning("Server is not running")
            return
        
        # Note: uvicorn doesn't have a clean shutdown method
        # This is a limitation of the current implementation
        logger.info("Server stop requested (restart required for clean shutdown)")
        self.server_running = False
    
    def get_service_url(self) -> str:
        """Get the service URL"""
        return f"http://{self.host}:{self.port}"
    
    def get_docs_url(self) -> str:
        """Get the API documentation URL"""
        return f"{self.get_service_url()}/docs"
    
    def is_running(self) -> bool:
        """Check if the server is running"""
        return self.server_running
