"""
Test script for RecordMemory task history deletion functionality
"""

import asyncio
import json
import sys
import os
import random
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.JudgeAgent.judge_memory import RecordMemory
from agent_engine.agent_logger import AgentLogger

logger = AgentLogger(__name__)


class TaskHistoryTester:
    """Test class for task history deletion functionality"""
    
    def __init__(self):
        self.judge_memory = RecordMemory(name='test_task_history_memory')
        self.capabilities_with_tasks = []
    
    def load_capabilities_with_tasks(self):
        """Load capabilities with tasks from JSON file"""
        try:
            json_file_path = Path(__file__).parent.parent / 'agents' / 'JudgeAgent' / 'capabilities_with_tasks.json'
            if json_file_path.exists():
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    self.capabilities_with_tasks = json.load(f)
                print(f"Loaded {len(self.capabilities_with_tasks)} capabilities with tasks")
            else:
                print(f"File not found: {json_file_path}")
                return False
        except Exception as e:
            print(f"Error loading capabilities: {e}")
            return False
        return True
    
    async def simulate_agent_execution(self, capability: Dict[str, Any], task: str, agent: Dict[str, Any]) -> bool:
        """
        Simulate agent execution with 50% success rate
        
        Args:
            capability: Capability dictionary
            task: Task to execute
            agent: Agent dictionary
            
        Returns:
            True if successful, False otherwise
        """
        # Simulate execution time
        await asyncio.sleep(0.1)
        
        # 50% success rate
        success = random.random() > 0.5
        
        # Record the result
        await self.judge_memory.record_task_result(
            agent_name=agent['name'],
            agent_url=agent['url'],
            success=success,
            capability_content={
                'name': capability['name'],
                'definition': capability['definition']
            },
            task_name=task
        )
        
        return success
    
    async def setup_test_data(self):
        """Setup test data by running some simulations"""
        print("Setting up test data...")
        
        # Run a few rounds of simulation to create task history
        for round_num in range(3):
            print(f"Round {round_num + 1}")
            
            for capability in self.capabilities_with_tasks:
                if 'tasks' not in capability or not capability['tasks']:
                    continue
                
                # Select a random task
                task = random.choice(capability['tasks'])
                agents = capability.get('agents', [])
                
                if agents:
                    # Simulate execution for each agent
                    for agent in agents:
                        await self.simulate_agent_execution(capability, task, agent)
        
        print("Test data setup completed")
    
    def test_get_all_agents(self):
        """Test get_all_agents method"""
        print(f"\n{'='*60}")
        print("Testing Get All Agents")
        print(f"{'='*60}")
        
        agents = self.judge_memory.get_all_agents()
        print(f"Found {len(agents)} unique agents:")
        
        for agent in agents:
            print(f"  - {agent['name']} ({agent['url']})")
        
        return agents
    
    def test_task_history_before_deletion(self):
        """Test task history before deletion"""
        print(f"\n{'='*60}")
        print("Testing Task History Before Deletion")
        print(f"{'='*60}")
        
        all_items = self.judge_memory.get_all()
        capabilities_with_history = 0
        
        for content_str, vector, metadata in all_items:
            content = json.loads(content_str)
            capability_name = content['name']
            
            if 'task_history' in metadata and metadata['task_history']:
                capabilities_with_history += 1
                print(f"\nCapability: {capability_name}")
                print(f"Task history entries: {len(metadata['task_history'])}")
                
                for agent_key, history in metadata['task_history'].items():
                    print(f"  Agent: {history['agent_name']}")
                    print(f"    Total tasks: {history['total_count']}")
                    print(f"    Successful: {history['success_count']}")
        
        print(f"\nTotal capabilities with task history: {capabilities_with_history}")
        return capabilities_with_history
    
    def test_delete_specific_agent_task_history(self, agent_name: str, agent_url: str):
        """Test deleting task history for a specific agent"""
        print(f"\n{'='*60}")
        print(f"Testing Delete Specific Agent Task History: {agent_name}")
        print(f"{'='*60}")
        
        # Delete task history for the specific agent
        self.judge_memory.delete_agent_task_history(agent_name, agent_url)
        
        # Verify deletion
        all_items = self.judge_memory.get_all()
        agent_key = f"{agent_name}_{agent_url}"
        
        for content_str, vector, metadata in all_items:
            content = json.loads(content_str)
            capability_name = content['name']
            
            if 'task_history' in metadata:
                if agent_key in metadata['task_history']:
                    print(f"ERROR: Agent {agent_name} still has task history in capability {capability_name}")
                else:
                    print(f"✓ Agent {agent_name} task history successfully deleted from capability {capability_name}")
            else:
                print(f"✓ Capability {capability_name} has no task history")
    
    def test_delete_all_task_history(self):
        """Test deleting all task history"""
        print(f"\n{'='*60}")
        print("Testing Delete All Task History")
        print(f"{'='*60}")
        
        # Delete all task history
        self.judge_memory.delete_all_task_history()
        
        # Verify deletion
        all_items = self.judge_memory.get_all()
        capabilities_with_history = 0
        
        for content_str, vector, metadata in all_items:
            content = json.loads(content_str)
            capability_name = content['name']
            
            if 'task_history' in metadata and metadata['task_history']:
                capabilities_with_history += 1
                print(f"ERROR: Capability {capability_name} still has task history")
            else:
                print(f"✓ Capability {capability_name} task history successfully deleted")
        
        print(f"\nTotal capabilities with remaining task history: {capabilities_with_history}")
        return capabilities_with_history
    
    async def test_delete_all_task_history_by_agent(self):
        """Test deleting all task history by iterating through all agents"""
        print(f"\n{'='*60}")
        print("Testing Delete All Task History By Agent")
        print(f"{'='*60}")
        
        # First, setup some test data again
        await self.setup_test_data()
        
        # Get all agents
        agents = self.judge_memory.get_all_agents()
        print(f"Found {len(agents)} agents to delete task history for")
        
        # Delete task history for each agent
        for agent in agents:
            print(f"Deleting task history for agent: {agent['name']}")
            self.judge_memory.delete_agent_task_history(agent['name'], agent['url'])
        
        # Verify all task history is deleted
        all_items = self.judge_memory.get_all()
        capabilities_with_history = 0
        
        for content_str, vector, metadata in all_items:
            content = json.loads(content_str)
            capability_name = content['name']
            
            if 'task_history' in metadata and metadata['task_history']:
                capabilities_with_history += 1
                print(f"ERROR: Capability {capability_name} still has task history")
            else:
                print(f"✓ Capability {capability_name} task history successfully deleted")
        
        print(f"\nTotal capabilities with remaining task history: {capabilities_with_history}")
        return capabilities_with_history
    
    async def run_all_tests(self):
        """Run all tests"""
        print("Starting Task History Deletion Tests for RecordMemory")
        print("=" * 80)
        
        # Load capabilities with tasks
        if not self.load_capabilities_with_tasks():
            print("Failed to load capabilities with tasks")
            return False
        
        try:
            # Setup test data
            await self.setup_test_data()
            
            # Test get all agents
            agents = self.test_get_all_agents()
            
            # Test task history before deletion
            initial_history_count = self.test_task_history_before_deletion()
            
            if initial_history_count == 0:
                print("No task history found, skipping deletion tests")
                return True
            
            # Test deleting specific agent task history
            if agents:
                test_agent = agents[0]
                self.test_delete_specific_agent_task_history(test_agent['name'], test_agent['url'])
            
            # Test deleting all task history
            remaining_history = self.test_delete_all_task_history()
            
            # Test deleting all task history by agent
            remaining_history_by_agent = await self.test_delete_all_task_history_by_agent()
            
            print(f"\n{'='*80}")
            print("All tests completed successfully!")
            print(f"Initial task history count: {initial_history_count}")
            print(f"Remaining history after bulk delete: {remaining_history}")
            print(f"Remaining history after agent-by-agent delete: {remaining_history_by_agent}")
            print(f"{'='*80}")
            
            return True
            
        except Exception as e:
            print(f"\nTest failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Main test function"""
    tester = TaskHistoryTester()
    success = await tester.run_all_tests()
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
