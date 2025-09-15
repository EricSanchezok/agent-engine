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


Record Memory Server - FastAPI server for RecordMemory operations

This server provides REST API endpoints for managing capabilities, agents, and task results
in the RecordMemory system.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from record_memory import RecordMemory

# Initialize FastAPI app
app = FastAPI(
    title="Record Memory Server",
    description="REST API server for RecordMemory operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RecordMemory instance
record_memory = RecordMemory()

# Pydantic models for request/response validation
class CapabilityRequest(BaseModel):
    name: str
    definition: str

class SearchCapabilitiesRequest(BaseModel):
    name: str
    definition: str
    top_k: int = 5
    threshold: float = 0.55

class TaskResultRequest(BaseModel):
    agent_name: str
    agent_url: str
    capability_name: str
    capability_definition: str
    success: bool
    task_content: str
    task_result: str

class DeleteTaskResultRequest(BaseModel):
    agent_name: str
    agent_url: str
    capability_name: str
    capability_definition: str
    task_content: str
    task_result: str
    timestamp: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Record Memory Server is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "record_memory_server"}

@app.get("/capabilities", response_model=List[Dict[str, Any]])
async def get_all_capabilities():
    """
    Get all capabilities from the memory
    
    Returns:
        List of all capabilities with their metadata
    """
    try:
        capabilities = await record_memory.get_all_capabilities()
        return capabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")

@app.post("/capabilities/search", response_model=List[Dict[str, Any]])
async def search_similar_capabilities(request: SearchCapabilitiesRequest):
    """
    Search for similar capabilities based on name and definition
    
    Args:
        request: SearchCapabilitiesRequest containing search parameters
        
    Returns:
        List of similar capabilities with similarity scores
    """
    try:
        similar_capabilities = await record_memory.search_similar_capabilities(
            capability_name=request.name,
            capability_definition=request.definition,
            top_k=request.top_k,
            threshold=request.threshold
        )
        return similar_capabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search capabilities: {str(e)}")

@app.post("/capabilities/agents", response_model=List[Dict[str, Any]])
async def get_agents_for_capability(request: CapabilityRequest):
    """
    Get all agents that can perform a specific capability
    
    Args:
        request: CapabilityRequest containing capability name and definition
        
    Returns:
        List of agents that can perform the capability
    """
    try:
        agents = await record_memory.get_agents_for_capability(
            capability_name=request.name,
            capability_definition=request.definition
        )
        return agents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agents for capability: {str(e)}")

@app.get("/agents/{agent_name}/capabilities", response_model=List[Dict[str, Any]])
async def get_agent_capabilities(agent_name: str, agent_url: str):
    """
    Get all capabilities that a specific agent can perform
    
    Args:
        agent_name: Name of the agent
        agent_url: URL of the agent
        
    Returns:
        List of capabilities that the agent can perform
    """
    try:
        capabilities = await record_memory.get_agent_capabilities(
            agent_name=agent_name,
            agent_url=agent_url
        )
        return capabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent capabilities: {str(e)}")

@app.post("/task-result")
async def add_task_result(request: TaskResultRequest):
    """
    Add a task execution result for an agent
    
    Args:
        request: TaskResultRequest containing task execution details
        
    Returns:
        Success message with operation status
    """
    try:
        success = await record_memory.add_task_result(
            agent_name=request.agent_name,
            agent_url=request.agent_url,
            capability_name=request.capability_name,
            capability_definition=request.capability_definition,
            success=request.success,
            task_content=request.task_content,
            task_result=request.task_result
        )
        if success:
            return {"message": "Task result added successfully", "success": True}
        else:
            return {"message": "Failed to add task result. Please verify that the capability and agent exist in the system.", "success": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add task result: {str(e)}")

@app.delete("/task-result")
async def delete_task_result(request: DeleteTaskResultRequest):
    """
    Delete a specific task execution result for an agent
    
    Args:
        request: DeleteTaskResultRequest containing task details to delete
        
    Returns:
        Success message with operation status
    """
    try:
        success = await record_memory.delete_task_result(
            agent_name=request.agent_name,
            agent_url=request.agent_url,
            capability_name=request.capability_name,
            capability_definition=request.capability_definition,
            task_content=request.task_content,
            task_result=request.task_result,
            timestamp=request.timestamp
        )
        if success:
            return {"message": "Task result deleted successfully", "success": True}
        else:
            return {"message": "Failed to delete task result. Please verify that the capability and agent exist in the system.", "success": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete task result: {str(e)}")

@app.post("/capabilities/performance", response_model=List[Dict[str, Any]])
async def get_capability_performance(request: CapabilityRequest):
    """
    Get performance information for all agents that can perform a capability
    
    Args:
        request: CapabilityRequest containing capability name and definition
        
    Returns:
        List of agent performance information
    """
    try:
        performance = await record_memory.get_capability_performance(
            capability_name=request.name,
            capability_definition=request.definition
        )
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get capability performance: {str(e)}")

@app.post("/capabilities/history", response_model=Dict[str, Any])
async def get_capability_history(request: CapabilityRequest):
    """
    Get history information for a specific capability
    
    Args:
        request: CapabilityRequest containing capability name and definition
        
    Returns:
        Task history for all agents performing the capability
    """
    try:
        history = await record_memory.get_capability_history(
            capability_name=request.name,
            capability_definition=request.definition
        )
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get capability history: {str(e)}")

@app.get("/agents", response_model=List[Dict[str, str]])
async def get_all_agents():
    """
    Get all unique agents from all capabilities
    
    Returns:
        List of unique agent dictionaries with name and url
    """
    try:
        agents = await record_memory.get_all_agents()
        return agents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get all agents: {str(e)}")

@app.delete("/agents/{agent_name}/task-history")
async def delete_agent_task_history(agent_name: str, agent_url: str):
    """
    Delete task history for a specific agent from all capabilities
    
    Args:
        agent_name: Name of the agent
        agent_url: URL of the agent
        
    Returns:
        Success message with operation status
    """
    try:
        success = await record_memory.delete_agent_task_history(
            agent_name=agent_name,
            agent_url=agent_url
        )
        if success:
            return {"message": f"Task history deleted successfully for agent {agent_name}", "success": True}
        else:
            return {"message": f"No task history found for agent {agent_name}", "success": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete agent task history: {str(e)}")

@app.delete("/task-history")
async def delete_all_task_history():
    """
    Delete all task history from all capabilities
    
    Returns:
        Success message with operation status
    """
    try:
        success = await record_memory.delete_all_task_history()
        if success:
            return {"message": "All task history deleted successfully", "success": True}
        else:
            return {"message": "No task history found to delete", "success": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete all task history: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
