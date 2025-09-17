"""
Research Agent Server

A FastAPI-based server that provides HTTP endpoints for the Research Agent.
Supports both regular and streaming chat responses.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent_engine.agent_logger.agent_logger import AgentLogger
from .researcher import Researcher

logger = AgentLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Research Agent Server",
    description="A research assistant powered by Azure OpenAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global researcher instance
researcher: Optional[Researcher] = None


# Pydantic models for request/response
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    session_id: str = Field(..., description="Unique identifier for the session")
    message: str = Field(..., description="User's message")
    max_tokens: int = Field(default=2000, description="Maximum tokens for response")
    temperature: float = Field(default=0.7, description="Temperature for response generation")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's response")
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    timestamp: str = Field(..., description="Response timestamp")


class SessionStatsResponse(BaseModel):
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    stats: Dict[str, Any] = Field(..., description="Session statistics")


class ClearSessionRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    session_id: str = Field(..., description="Unique identifier for the session")


class ClearSessionResponse(BaseModel):
    success: bool = Field(..., description="Whether the session was cleared successfully")
    message: str = Field(..., description="Response message")


class ActiveSessionsResponse(BaseModel):
    sessions: List[Dict[str, str]] = Field(..., description="List of active sessions")


@app.on_event("startup")
async def startup_event():
    """Initialize the researcher on startup"""
    global researcher
    
    try:
        # Get Azure credentials from environment variables
        azure_api_key = os.getenv("AZURE_API_KEY")
        azure_base_url = os.getenv("AZURE_BASE_URL", "https://gpt.yunstorm.com/")
        azure_api_version = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
        model_name = os.getenv("MODEL_NAME", "gpt-4.1")
        
        if not azure_api_key:
            raise ValueError("AZURE_API_KEY environment variable is required")
        
        # Initialize researcher
        researcher = Researcher(
            azure_api_key=azure_api_key,
            azure_base_url=azure_base_url,
            azure_api_version=azure_api_version,
            model_name=model_name
        )
        
        logger.info("Research Agent Server started successfully")
        logger.info(f"Model: {model_name}")
        logger.info(f"Azure endpoint: {azure_base_url}")
        
    except Exception as e:
        logger.error(f"Failed to start Research Agent Server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global researcher
    
    try:
        if researcher:
            await researcher.close()
        logger.info("Research Agent Server shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Research Agent Server is running",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global researcher
    
    if researcher is None:
        raise HTTPException(status_code=503, detail="Researcher not initialized")
    
    return {
        "status": "healthy",
        "researcher_initialized": researcher is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and return a response.
    
    Args:
        request: Chat request containing user_id, session_id, and message
        
    Returns:
        Chat response with assistant's reply
    """
    global researcher
    
    if researcher is None:
        raise HTTPException(status_code=503, detail="Researcher not initialized")
    
    try:
        logger.info(f"Processing chat request: user={request.user_id}, session={request.session_id}")
        
        # Process the chat message
        response = await researcher.chat(
            user_id=request.user_id,
            session_id=request.session_id,
            user_message=request.message,
            stream=False,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        if response is None:
            raise HTTPException(status_code=500, detail="Failed to generate response")
        
        return ChatResponse(
            response=response,
            user_id=request.user_id,
            session_id=request.session_id,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Process a chat message with streaming response.
    
    Args:
        request: Chat request containing user_id, session_id, and message
        
    Returns:
        Streaming response with assistant's reply chunks
    """
    global researcher
    
    if researcher is None:
        raise HTTPException(status_code=503, detail="Researcher not initialized")
    
    try:
        logger.info(f"Processing streaming chat request: user={request.user_id}, session={request.session_id}")
        
        async def generate_response():
            try:
                async for chunk in researcher.chat_stream(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    user_message=request.message,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                ):
                    # Send chunk as Server-Sent Events format
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing streaming chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{user_id}/{session_id}/stats", response_model=SessionStatsResponse)
async def get_session_stats(user_id: str, session_id: str):
    """
    Get statistics for a specific session.
    
    Args:
        user_id: Unique identifier for the user
        session_id: Unique identifier for the session
        
    Returns:
        Session statistics
    """
    global researcher
    
    if researcher is None:
        raise HTTPException(status_code=503, detail="Researcher not initialized")
    
    try:
        stats = researcher.get_session_stats(user_id, session_id)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return SessionStatsResponse(
            user_id=user_id,
            session_id=session_id,
            stats=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/clear", response_model=ClearSessionResponse)
async def clear_session(request: ClearSessionRequest):
    """
    Clear a specific session.
    
    Args:
        request: Clear session request containing user_id and session_id
        
    Returns:
        Clear session response
    """
    global researcher
    
    if researcher is None:
        raise HTTPException(status_code=503, detail="Researcher not initialized")
    
    try:
        success = researcher.clear_session(request.user_id, request.session_id)
        
        if success:
            return ClearSessionResponse(
                success=True,
                message=f"Session cleared successfully for user={request.user_id}, session={request.session_id}"
            )
        else:
            return ClearSessionResponse(
                success=False,
                message=f"Session not found for user={request.user_id}, session={request.session_id}"
            )
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_model=ActiveSessionsResponse)
async def list_active_sessions():
    """
    List all active sessions.
    
    Returns:
        List of active sessions
    """
    global researcher
    
    if researcher is None:
        raise HTTPException(status_code=503, detail="Researcher not initialized")
    
    try:
        sessions = researcher.list_active_sessions()
        
        return ActiveSessionsResponse(sessions=sessions)
        
    except Exception as e:
        logger.error(f"Error listing active sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting Research Agent Server on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
