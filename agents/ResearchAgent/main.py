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

from agent_engine.agent_logger import agent_logger
from .deep_research.deep_research_engine import DeepResearchEngine
from .config import PDF_STROAGE_DIR, ARXIV_DATABASE_DIR, QIJI_LIBRARY_DIR, QIJI_ARTICLES_DIR


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


class ResearchRequest(BaseModel):
    """Request model for research tasks"""
    query: str
    user_id: str
    research_type: str = "general"  # general, deep, literature_review
    parameters: Optional[Dict[str, Any]] = None


class ResearchResponse(BaseModel):
    """Response model for research tasks"""
    results: Dict[str, Any]
    session_id: str
    timestamp: str
    status: str = "success"
    message: str


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

# Global variables for service state
research_engine: Optional[DeepResearchEngine] = None
user_sessions: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the research engine on startup"""
    global research_engine
    try:
        agent_logger.info("Initializing Inno-Researcher service...")
        
        # Initialize the deep research engine
        research_engine = DeepResearchEngine()
        await research_engine.initialize()
        
        agent_logger.info("Inno-Researcher service initialized successfully")
        
    except Exception as e:
        agent_logger.error(f"Failed to initialize Inno-Researcher service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global research_engine
    if research_engine:
        await research_engine.cleanup()
    agent_logger.info("Inno-Researcher service shutdown")


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
        "engine_ready": research_engine is not None,
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
        
        # Generate response using the research engine
        if research_engine:
            response_text = await research_engine.process_query(
                query=request.message,
                user_id=request.user_id,
                session_id=session_id,
                context=request.context or {}
            )
        else:
            response_text = "Research engine is not available. Please try again later."
        
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


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Research endpoint for conducting research tasks
    """
    try:
        agent_logger.info(f"Research request from user {request.user_id}: {request.query[:100]}...")
        
        session_id = f"research_{request.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Conduct research using the engine
        if research_engine:
            results = await research_engine.conduct_research(
                query=request.query,
                research_type=request.research_type,
                user_id=request.user_id,
                session_id=session_id,
                parameters=request.parameters or {}
            )
            message = "Research completed successfully"
        else:
            results = {"error": "Research engine is not available"}
            message = "Research engine is not available"
        
        agent_logger.info(f"Research completed for user {request.user_id}")
        
        return ResearchResponse(
            results=results,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            message=message
        )
        
    except Exception as e:
        agent_logger.error(f"Error in research endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
