#!/usr/bin/env python3
"""
Inno-Researcher Main Entry Point
A research assistant service for helping researchers with their work.
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
from pathlib import Path
from agent_engine.agent_logger import set_agent_log_directory

current_file_dir = Path(__file__).parent
log_dir = current_file_dir / 'logs'
set_agent_log_directory(str(log_dir))

from agent_engine.agent_logger import agent_logger

# Simple chat service without deep research functionality


class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str
    user_id: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    response: str
    session_id: str
    timestamp: str
    status: str = "success"


# Removed research-related models for simple chat functionality


# Initialize FastAPI app
app = FastAPI(
    title="Inno-Researcher",
    description="AI-powered research assistant for researchers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_sessions: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup"""
    try:
        agent_logger.info("Initializing Inno-Researcher chat service...")
        agent_logger.info("Inno-Researcher chat service initialized successfully")
        
    except Exception as e:
        agent_logger.error(f"Failed to initialize Inno-Researcher service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    agent_logger.info("Inno-Researcher chat service shutdown")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Inno-Researcher",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "Inno-Researcher",
        "active_sessions": len(user_sessions),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for interactive conversations with the research assistant
    """
    try:
        agent_logger.info(f"Chat request from user {request.user_id}: {request.message[:100]}...")
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{request.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize user session if not exists
        if request.user_id not in user_sessions:
            user_sessions[request.user_id] = {
                "sessions": {},
                "created_at": datetime.now().isoformat()
            }
        
        if session_id not in user_sessions[request.user_id]["sessions"]:
            user_sessions[request.user_id]["sessions"][session_id] = {
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }
        
        # Add user message to session
        user_sessions[request.user_id]["sessions"][session_id]["messages"].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate simple response
        response_text = f"Hello! I received your message: '{request.message}'. This is a simple chat response from Inno-Researcher."
        
        # Add assistant response to session
        user_sessions[request.user_id]["sessions"][session_id]["messages"].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update last activity
        user_sessions[request.user_id]["sessions"][session_id]["last_activity"] = datetime.now().isoformat()
        
        agent_logger.info(f"Chat response generated for user {request.user_id}")
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        agent_logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Removed research endpoint for simple chat functionality


@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    """Get all sessions for a user"""
    if user_id not in user_sessions:
        return {"sessions": [], "message": "No sessions found for this user"}
    
    return {
        "user_id": user_id,
        "sessions": user_sessions[user_id]["sessions"],
        "total_sessions": len(user_sessions[user_id]["sessions"])
    }


@app.delete("/sessions/{user_id}/{session_id}")
async def delete_session(user_id: str, session_id: str):
    """Delete a specific session"""
    if user_id in user_sessions and session_id in user_sessions[user_id]["sessions"]:
        del user_sessions[user_id]["sessions"][session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


def main():
    """Main entry point for the Inno-Researcher service"""
    agent_logger.info("Starting Inno-Researcher service...")
    
    # Run the FastAPI server
    uvicorn.run(
        "agents.ResearchAgent.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
