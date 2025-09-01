"""
JudgeMemory class for multi-armed bandit algorithm in agent capability selection.

This class extends the base Memory class to provide specialized methods for
capability-agent matching and performance tracking.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

from agent_engine.memory import Memory
from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.utils import get_current_file_dir

logger = AgentLogger(__name__)

load_dotenv()


class JudgeMemory(Memory):
    """Extended memory class for judge agent with capability-agent matching capabilities"""
    
    def __init__(self, name: str = 'judge_memory', db_path: Optional[str] = None):
        """
        Initialize JudgeMemory
        
        Args:
            name: Name of the memory database
            db_path: Path to the database file
        """
        if db_path is None:
            db_path = get_current_file_dir() / 'database' / 'judge_memory.db'
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        super().__init__(name=name, db_path=str(db_path))
    
    async def add_capability(self, name: str, definition: str, alias: List[str] = None, agents: List[Dict] = None):
        """
        Add a capability with proper structure for judge memory
        
        Args:
            name: Capability name
            definition: Capability definition
            alias: List of aliases for the capability
            agents: List of agents that can perform this capability
        """
        # Create capability content with only name and definition
        capability_content = {
            'name': name,
            'definition': definition
        }
        
        # Create metadata with alias and agents
        metadata = {}
        if alias:
            metadata['alias'] = alias
        if agents:
            metadata['agents'] = agents
        
        vector = await self.llm_client.embedding(json.dumps(capability_content, ensure_ascii=False, indent=4), model_name='text-embedding-3-small')
        # Add to memory
        self.add(json.dumps(capability_content, ensure_ascii=False, indent=4), vector=vector, metadata=metadata)

    async def delete_capability(self, name: str, definition: str):
        """
        Delete a capability from memory
        
        Args:
            name: Capability name
            definition: Capability definition
        """
        content = json.dumps({'name': name, 'definition': definition}, ensure_ascii=False, indent=4)
        self.delete_by_content(content)
    
    async def search_similar_capabilities(self, name: str, definition: str, top_k: int = 5, threshold: float = 0.55) -> List[Dict[str, Any]]:
        """
        Search for similar capabilities based on text similarity
        
        Args:
            name: Capability name
            definition: Capability definition
            top_k: Number of top similar capabilities to return
            
        Returns:
            List of capability content dictionaries
        """
        vector = await self.llm_client.embedding(json.dumps({'name': name, 'definition': definition}, ensure_ascii=False, indent=4), model_name='text-embedding-3-small')
        results = self.search(vector, top_k=top_k)
        similar_capabilities = []
        
        for content, similarity_score, metadata in results:
            if similarity_score > threshold:  # Lower similarity threshold for better matching
                capability_content = json.loads(content)
                capability_content['similarity_score'] = similarity_score
                capability_content['metadata'] = metadata
                similar_capabilities.append(capability_content)
        
        return similar_capabilities
    
    def get_agents_for_capability(self, capability_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get all agents that can perform a specific capability
        
        Args:
            capability_content: Capability content dictionary with name and definition
            
        Returns:
            List of agent dictionaries
        """
        content_str = json.dumps(capability_content, ensure_ascii=False, indent=4)
        vector, metadata = self.get_by_content(content_str)
        
        if metadata and 'agents' in metadata:
            return metadata['agents']
        return []
    
    def get_agent_capabilities(self, agent_name: str, agent_url: str) -> List[Dict[str, Any]]:
        """
        Get all capabilities that a specific agent can perform
        
        Args:
            agent_name: Name of the agent
            agent_url: URL of the agent
            
        Returns:
            List of capability content dictionaries
        """
        agent_info = {'name': agent_name, 'url': agent_url}
        all_contents = self.get_all_contents()
        agent_capabilities = []
        
        for content_str in all_contents:
            content = json.loads(content_str)
            vector, metadata = self.get_by_content(content_str)
            
            if metadata and 'agents' in metadata:
                for agent in metadata['agents']:
                    if (agent.get('name') == agent_name and 
                        agent.get('url') == agent_url):
                        agent_capabilities.append(content)
                        break
        
        return agent_capabilities
    
    async def record_task_result(self, agent_name: str, agent_url: str, 
                          success: bool, capability_content: Dict[str, Any], 
                          task_name: str):
        """
        Record a task execution result for an agent
        
        Args:
            agent_name: Name of the agent
            agent_url: URL of the agent
            success: Whether the task was successful
            capability_content: Capability content dictionary
            task_name: Name of the task
        """
        content_str = json.dumps(capability_content, ensure_ascii=False, indent=4)
        vector, metadata = self.get_by_content(content_str)
        
        if not metadata:
            metadata = {}
        
        # Initialize task_history if not exists
        if 'task_history' not in metadata:
            metadata['task_history'] = {}
        
        agent_key = f"{agent_name}_{agent_url}"
        if agent_key not in metadata['task_history']:
            metadata['task_history'][agent_key] = {
                'agent_name': agent_name,
                'agent_url': agent_url,
                'success_count': 0,
                'total_count': 0,
                'tasks': []
            }
        
        # Update task history
        task_record = {
            'task_name': task_name,
            'success': success,
            'timestamp': self._get_current_timestamp()
        }
        
        metadata['task_history'][agent_key]['tasks'].append(task_record)
        metadata['task_history'][agent_key]['total_count'] += 1
        if success:
            metadata['task_history'][agent_key]['success_count'] += 1
        
        # Update the content in memory with new metadata
        self.delete_by_content(content_str)
        self.add(content_str, vector=vector, metadata=metadata)
    
    def get_agent_performance_info(self, capability_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get performance information for all agents that can perform a capability
        
        Args:
            capability_content: Capability content dictionary
            
        Returns:
            List of agent performance information dictionaries
        """
        content_str = json.dumps(capability_content, ensure_ascii=False, indent=4)
        vector, metadata = self.get_by_content(content_str)
        
        agent_performance = []
        
        if metadata and 'task_history' in metadata:
            for agent_key, history in metadata['task_history'].items():
                performance_info = {
                    'name': history['agent_name'],
                    'url': history['agent_url'],
                    'success_count': history['success_count'],
                    'total_count': history['total_count'],
                    'success_rate': history['success_count'] / history['total_count'] if history['total_count'] > 0 else 0
                }
                agent_performance.append(performance_info)
        
        return agent_performance
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_capability_with_metadata(self, capability_content: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Get capability content with its metadata
        
        Args:
            capability_content: Capability content dictionary
            
        Returns:
            Tuple of (capability_content, metadata) or None if not found
        """
        content_str = json.dumps(capability_content, ensure_ascii=False, indent=4)
        vector, metadata = self.get_by_content(content_str)
        
        if vector is not None:
            return capability_content, metadata
        return None
    
    async def update_capability_agents(self, capability_content: Dict[str, Any], agents: List[Dict[str, Any]]):
        """
        Update the agents list for a capability
        
        Args:
            capability_content: Capability content dictionary
            agents: New list of agents
        """
        content_str = json.dumps(capability_content, ensure_ascii=False, indent=4)
        vector, metadata = self.get_by_content(content_str)
        
        if vector is not None:
            if not metadata:
                metadata = {}
            metadata['agents'] = agents
            
            # Update the content in memory
            self.delete_by_content(content_str)
            self.add(content_str, vector=vector, metadata=metadata)
    
    def delete_agent_task_history(self, agent_name: str, agent_url: str):
        """
        Delete task history for a specific agent from all capabilities
        
        Args:
            agent_name: Name of the agent
            agent_url: URL of the agent
        """
        agent_key = f"{agent_name}_{agent_url}"
        all_items = self.get_all()
        
        for content_str, vector, metadata in all_items:
            if metadata and 'task_history' in metadata:
                if agent_key in metadata['task_history']:
                    # Remove the agent's task history
                    del metadata['task_history'][agent_key]
                    
                    # Update the content in memory
                    self.delete_by_content(content_str)
                    self.add(content_str, vector=vector, metadata=metadata)
                    
                    logger.info(f"Deleted task history for agent {agent_name} from capability {json.loads(content_str)['name']}")
    
    def delete_all_task_history(self):
        """
        Delete all task history from all capabilities
        """
        all_items = self.get_all()
        
        for content_str, vector, metadata in all_items:
            if metadata and 'task_history' in metadata:
                # Remove all task history
                del metadata['task_history']
                
                # Update the content in memory
                self.delete_by_content(content_str)
                self.add(content_str, vector=vector, metadata=metadata)
                
                logger.info(f"Deleted all task history from capability {json.loads(content_str)['name']}")
    
    def get_all_agents(self) -> List[Dict[str, str]]:
        """
        Get all unique agents from all capabilities
        
        Returns:
            List of unique agent dictionaries with name and url
        """
        all_agents = set()
        all_items = self.get_all()
        
        for content_str, vector, metadata in all_items:
            if metadata and 'agents' in metadata:
                for agent in metadata['agents']:
                    agent_key = (agent['name'], agent['url'])
                    all_agents.add(agent_key)
        
        return [{'name': name, 'url': url} for name, url in all_agents]
