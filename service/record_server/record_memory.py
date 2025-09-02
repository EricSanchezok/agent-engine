"""
RecordMemory class for multi-armed bandit algorithm in agent capability selection.

This class extends the base Memory class to provide specialized methods for
capability-agent matching and performance tracking.
"""

import asyncio
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


class RecordMemory(Memory):
    """Extended memory class for record server with capability-agent matching capabilities"""
    
    def __init__(self, name: str = 'record_memory', db_path: Optional[str] = None):
        """
        Initialize RecordMemory
        
        Args:
            name: Name of the memory database
            db_path: Path to the database file
        """
        if db_path is None:
            db_path = 'database/record_memory.sqlite'
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        super().__init__(name=name, db_path=str(db_path))
    
    async def add_capability(self, name: str, definition: str, alias: List[str] = None, agents: List[Dict] = None):
        """
        Add a capability with proper structure for record memory
        
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

    async def delete_capability(self, capability_name: str, capability_definition: str):
        """
        Delete a capability from memory
        
        Args:
            capability_name: Capability name
            capability_definition: Capability definition
        """
        capability_content_str = json.dumps({'name': capability_name, 'definition': capability_definition}, ensure_ascii=False, indent=4)
        logger.info(f"Attempting to delete capability: {capability_name}")
        logger.info(f"Content to delete: {capability_content_str}")
        
        # Check if content exists before deletion
        vector, metadata = self.get_by_content(capability_content_str)
        if vector is None:
            logger.warning(f"Capability not found for deletion: {capability_name}")
            return
        
        logger.info(f"Found capability for deletion: {capability_name}")
        result = self.delete_by_content(capability_content_str)
        if result:
            logger.info(f"Successfully deleted capability: {capability_name}")
        else:
            logger.error(f"Failed to delete capability: {capability_name}")
    
    async def search_similar_capabilities(self, capability_name: str, capability_definition: str, top_k: int = 5, threshold: float = 0.55) -> List[Dict[str, Any]]:
        """
        Search for similar capabilities based on text similarity
        
        Args:
            capability_name: Capability name
            capability_definition: Capability definition
            top_k: Number of top similar capabilities to return
            threshold: Threshold for similarity score
            
        Returns:
            List of capability content dictionaries
        """
        capability_content_str = json.dumps({'name': capability_name, 'definition': capability_definition}, ensure_ascii=False, indent=4)
        vector = await self.llm_client.embedding(capability_content_str, model_name='text-embedding-3-small')
        results = self.search(vector, top_k=top_k)
        similar_capabilities = []
        
        for _capability_content_str, _similarity_score, _metadata in results:
            if _similarity_score > threshold:
                capability_content = json.loads(_capability_content_str)
                capability_content['similarity_score'] = _similarity_score
                capability_content['metadata'] = _metadata
                similar_capabilities.append(capability_content)
        
        return similar_capabilities
    
    async def get_agents_for_capability(self, capability_name: str, capability_definition: str) -> List[Dict[str, Any]]:
        """
        Get all agents that can perform a specific capability
        
        Args:
            capability_name: Capability name
            capability_definition: Capability definition
            
        Returns:
            List of agent dictionaries
        """
        capability_content_str = json.dumps({'name': capability_name, 'definition': capability_definition}, ensure_ascii=False, indent=4)
        vector, metadata = self.get_by_content(capability_content_str)
        
        if metadata and 'agents' in metadata:
            return metadata['agents']
        return []
    
    async def get_agent_capabilities(self, agent_name: str, agent_url: str) -> List[Dict[str, Any]]:
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
        
        for capability_content_str in all_contents:
            capability_content = json.loads(capability_content_str)
            vector, metadata = self.get_by_content(capability_content_str)
            
            if metadata and 'agents' in metadata:
                for agent in metadata['agents']:
                    if (agent.get('name') == agent_name and 
                        agent.get('url') == agent_url):
                        agent_capabilities.append(capability_content)
                        break
        
        return agent_capabilities
    
    async def record_task_result(self, agent_name: str, agent_url: str, 
                          capability_name: str, capability_definition: str, 
                          success: bool, task_content: str, task_result: str):
        """
        Record a task execution result for an agent
        
        Args:
            agent_name: Name of the agent
            agent_url: URL of the agent
            capability_name: Capability name
            capability_definition: Capability definition
            success: Whether the task was successful
            task_content: Content of the task
            task_result: Result of the task
        """
        capability_content_str = json.dumps({'name': capability_name, 'definition': capability_definition}, ensure_ascii=False, indent=4)
        vector, metadata = self.get_by_content(capability_content_str)
        
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
            'task_content': task_content,
            'task_result': task_result,
            'success': success,
            'timestamp': self._get_current_timestamp()
        }
        
        metadata['task_history'][agent_key]['tasks'].append(task_record)
        metadata['task_history'][agent_key]['total_count'] += 1
        if success:
            metadata['task_history'][agent_key]['success_count'] += 1
        
        # Update the content in memory with new metadata
        self.delete_by_content(capability_content_str)
        self.add(capability_content_str, vector=vector, metadata=metadata)

    async def get_capability_history(self, capability_name: str, capability_definition: str) -> List[Dict[str, Any]]:
        """
        Get history information for a specific capability
        """
        capability_content_str = json.dumps({'name': capability_name, 'definition': capability_definition}, ensure_ascii=False, indent=4)
        vector, metadata = self.get_by_content(capability_content_str)
        return metadata['task_history']

    async def get_agent_performance(self, agent_name: str, agent_url: str) -> List[Dict[str, Any]]:
        """
        Get performance information for a specific agent
        """
        agent_key = f"{agent_name}_{agent_url}"
        all_items = self.get_all()
        return [metadata['task_history'][agent_key] for metadata in all_items if 'task_history' in metadata]
    
    async def get_capability_performance(self, capability_name: str, capability_definition: str) -> List[Dict[str, Any]]:
        """
        Get performance information for all agents that can perform a capability
        
        Args:
            capability_name: Capability name
            capability_definition: Capability definition
            
        Returns:
            List of agent performance information dictionaries
        """
        capability_content_str = json.dumps({'name': capability_name, 'definition': capability_definition}, ensure_ascii=False, indent=4)
        vector, metadata = self.get_by_content(capability_content_str)
        
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
    
    async def get_capability_with_metadata(self, capability_name: str, capability_definition: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Get capability content with its metadata
        
        Args:
            capability_name: Capability name
            capability_definition: Capability definition
            
        Returns:
            Tuple of (capability_content, metadata) or None if not found
        """
        capability_content_str = json.dumps({'name': capability_name, 'definition': capability_definition}, ensure_ascii=False, indent=4)
        vector, metadata = self.get_by_content(capability_content_str)
        
        if vector is not None:
            return {'name': capability_name, 'definition': capability_definition}, metadata
        return None
    
    async def delete_agent_task_history(self, agent_name: str, agent_url: str):
        """
        Delete task history for a specific agent from all capabilities
        
        Args:
            agent_name: Name of the agent
            agent_url: URL of the agent
        """
        agent_key = f"{agent_name}_{agent_url}"
        all_items = self.get_all()
        
        for capability_content_str, vector, metadata in all_items:
            if metadata and 'task_history' in metadata:
                if agent_key in metadata['task_history']:
                    # Remove the agent's task history
                    del metadata['task_history'][agent_key]
                    
                    # Update the content in memory
                    self.delete_by_content(capability_content_str)
                    self.add(capability_content_str, vector=vector, metadata=metadata)
                    
                    logger.info(f"Deleted task history for agent {agent_name} from capability {json.loads(capability_content_str)['name']}")
    
    async def delete_all_task_history(self):
        """
        Delete all task history from all capabilities
        """
        all_items = self.get_all()
        
        for capability_content_str, vector, metadata in all_items:
            if metadata and 'task_history' in metadata:
                # Remove all task history
                del metadata['task_history']
                
                # Update the content in memory
                self.delete_by_content(capability_content_str)
                self.add(capability_content_str, vector=vector, metadata=metadata)
                
                logger.info(f"Deleted all task history from capability {json.loads(capability_content_str)['name']}")
    
    async def get_all_agents(self) -> List[Dict[str, str]]:
        """
        Get all unique agents from all capabilities
        
        Returns:
            List of unique agent dictionaries with name and url
        """
        all_agents = set()
        all_items = self.get_all()
        
        for capability_content_str, vector, metadata in all_items:
            if metadata and 'agents' in metadata:
                for agent in metadata['agents']:
                    agent_key = (agent['name'], agent['url'])
                    all_agents.add(agent_key)
        
        return [{'name': name, 'url': url} for name, url in all_agents]

    async def get_all_capabilities(self) -> List[Dict[str, Any]]:
        capabilities = []
        all_items = self.get_all()
        
        for content_str, vector, metadata in all_items:
            content = json.loads(content_str)
            
            # Combine content with metadata
            capability = {
                'name': content['name'],
                'definition': content['definition'],
                'alias': metadata.get('alias', []),
                'agents': metadata.get('agents', [])
            }
            capabilities.append(capability)
        return capabilities


if __name__ == "__main__":
    from pprint import pprint
    memory = RecordMemory()

    pprint(asyncio.run(memory.get_capability_history("Chat with Conversational AI Assistant", "This service enables users to engage in a natural language conversation with a conversational AI assistant. It accepts text queries and provides concise, clear, and helpful responses that may include answers, explanations, summaries, or clarifications. It supports multi-turn dialogues and handling of complex topics.")))