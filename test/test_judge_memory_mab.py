"""
Test script for JudgeMemory multi-armed bandit algorithm functionality
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

from agents.JudgeAgent.judge_memory import JudgeMemory
from agent_engine.agent_logger import AgentLogger

logger = AgentLogger(__name__)


class MultiArmedBanditTester:
    """Test class for multi-armed bandit algorithm functionality"""
    
    def __init__(self):
        self.judge_memory = JudgeMemory(name='test_mab_memory')
        self.capabilities_with_tasks = []
        self.simulation_results = []
    
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
    
    async def run_simulation(self, num_rounds: int = 10):
        """
        Run simulation for multiple rounds
        
        Args:
            num_rounds: Number of simulation rounds
        """
        print(f"\n{'='*60}")
        print(f"Starting Multi-Armed Bandit Simulation ({num_rounds} rounds)")
        print(f"{'='*60}")
        
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1} ---")
            
            # For each capability, simulate task execution
            for capability in self.capabilities_with_tasks:
                if 'tasks' not in capability or not capability['tasks']:
                    continue
                
                # Select a random task
                task = random.choice(capability['tasks'])
                
                # Get all agents that can perform this capability
                agents = capability.get('agents', [])
                if not agents:
                    continue
                
                print(f"\nCapability: {capability['name']}")
                print(f"Task: {task[:50]}...")
                print(f"Available agents: {len(agents)}")
                
                # Simulate execution for each agent
                for agent in agents:
                    success = await self.simulate_agent_execution(capability, task, agent)
                    status = "SUCCESS" if success else "FAILED"
                    print(f"  {agent['name']}: {status}")
    
    def test_agent_performance_analysis(self):
        """Test agent performance analysis"""
        print(f"\n{'='*60}")
        print("Testing Agent Performance Analysis")
        print(f"{'='*60}")
        
        # Get all capabilities from memory
        all_items = self.judge_memory.get_all()
        
        for content_str, vector, metadata in all_items:
            content = json.loads(content_str)
            capability_name = content['name']
            
            print(f"\nCapability: {capability_name}")
            
            # Get performance info for this capability
            performance_info = self.judge_memory.get_agent_performance_info(content)
            
            if performance_info:
                print(f"Agent Performance:")
                for info in performance_info:
                    print(f"  {info['name']}: {info['success_count']}/{info['total_count']} "
                          f"(success rate: {info['success_rate']:.2f})")
            else:
                print("  No performance data available")
    
    def test_agent_capability_mapping(self):
        """Test agent-capability mapping functionality"""
        print(f"\n{'='*60}")
        print("Testing Agent-Capability Mapping")
        print(f"{'='*60}")
        
        # Get all agents from the loaded data
        all_agents = set()
        for capability in self.capabilities_with_tasks:
            for agent in capability.get('agents', []):
                agent_key = (agent['name'], agent['url'])
                all_agents.add(agent_key)
        
        print(f"Found {len(all_agents)} unique agents")
        
        # Test get_agent_capabilities for each agent
        for agent_name, agent_url in list(all_agents)[:5]:  # Test first 5 agents
            print(f"\nAgent: {agent_name} ({agent_url})")
            try:
                capabilities = self.judge_memory.get_agent_capabilities(agent_name, agent_url)
                print(f"Capabilities: {len(capabilities)}")
                for cap in capabilities:
                    print(f"  - {cap['name']}")
            except Exception as e:
                print(f"Error getting capabilities: {e}")
    
    def test_capability_agent_mapping(self):
        """Test capability-agent mapping functionality"""
        print(f"\n{'='*60}")
        print("Testing Capability-Agent Mapping")
        print(f"{'='*60}")
        
        # Test for each capability in the loaded data
        for capability in self.capabilities_with_tasks:
            capability_content = {
                'name': capability['name'],
                'definition': capability['definition']
            }
            
            print(f"\nCapability: {capability['name']}")
            try:
                agents = self.judge_memory.get_agents_for_capability(capability_content)
                print(f"Agents that can perform this capability: {len(agents)}")
                for agent in agents:
                    print(f"  - {agent['name']} ({agent['url']})")
            except Exception as e:
                print(f"Error getting agents: {e}")
    
    def test_multi_armed_bandit_selection(self):
        """Test multi-armed bandit selection logic"""
        print(f"\n{'='*60}")
        print("Testing Multi-Armed Bandit Selection Logic")
        print(f"{'='*60}")
        
        # Simulate MAB selection for a specific capability
        test_capability = {
            'name': 'Chat with Task Assistant',
            'definition': 'This capability provides a conversational interface that accepts user text queries and delivers clear, concise, and helpful text responses through a dialogue-based interaction.'
        }
        
        print(f"Testing MAB selection for: {test_capability['name']}")
        
        # Get all agents that can perform this capability
        agents = self.judge_memory.get_agents_for_capability(test_capability)
        
        if not agents:
            print("No agents found for this capability")
            return
        
        print(f"Available agents: {len(agents)}")
        
        # Get performance data for each agent
        performance_info = self.judge_memory.get_agent_performance_info(test_capability)
        
        if performance_info:
            print("\nAgent Performance Data:")
            for info in performance_info:
                print(f"  {info['name']}: {info['success_count']}/{info['total_count']} "
                      f"(success rate: {info['success_rate']:.2f})")
            
            # Simple UCB-like selection (Upper Confidence Bound)
            print("\nUCB-based Agent Selection:")
            for info in performance_info:
                # UCB formula: success_rate + sqrt(2 * log(total_rounds) / total_count)
                total_rounds = sum(p['total_count'] for p in performance_info)
                if info['total_count'] > 0:
                    ucb_score = info['success_rate'] + (2 * total_rounds / info['total_count']) ** 0.5
                else:
                    ucb_score = float('inf')  # Prioritize unexplored agents
                
                print(f"  {info['name']}: UCB score = {ucb_score:.3f}")
            
            # Select best agent based on UCB
            best_agent = max(performance_info, key=lambda x: 
                x['success_rate'] + (2 * sum(p['total_count'] for p in performance_info) / x['total_count']) ** 0.5 
                if x['total_count'] > 0 else float('inf'))
            
            print(f"\nSelected agent: {best_agent['name']} (UCB-based selection)")
        else:
            print("No performance data available for MAB selection")
    
    def test_task_execution_history(self):
        """Test task execution history tracking"""
        print(f"\n{'='*60}")
        print("Testing Task Execution History")
        print(f"{'='*60}")
        
        # Get all capabilities and their task history
        all_items = self.judge_memory.get_all()
        
        for content_str, vector, metadata in all_items:
            content = json.loads(content_str)
            capability_name = content['name']
            
            if 'task_history' in metadata:
                print(f"\nCapability: {capability_name}")
                print(f"Task history entries: {len(metadata['task_history'])}")
                
                for agent_key, history in metadata['task_history'].items():
                    print(f"  Agent: {history['agent_name']}")
                    print(f"    Total tasks: {history['total_count']}")
                    print(f"    Successful: {history['success_count']}")
                    print(f"    Success rate: {history['success_count']/history['total_count']:.2f}")
                    
                    # Show recent tasks
                    recent_tasks = history['tasks'][-3:]  # Last 3 tasks
                    print(f"    Recent tasks:")
                    for task in recent_tasks:
                        status = "✓" if task['success'] else "✗"
                        print(f"      {status} {task['task_name']} ({task['timestamp']})")
            else:
                print(f"\nCapability: {capability_name} - No task history")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("Starting Multi-Armed Bandit Tests for JudgeMemory")
        print("=" * 80)
        
        # Load capabilities with tasks
        if not self.load_capabilities_with_tasks():
            print("Failed to load capabilities with tasks")
            return False
        
        try:
            # Run simulation
            await self.run_simulation(num_rounds=5)
            
            # Run various tests
            self.test_agent_performance_analysis()
            self.test_agent_capability_mapping()
            self.test_capability_agent_mapping()
            self.test_multi_armed_bandit_selection()
            self.test_task_execution_history()
            
            print(f"\n{'='*80}")
            print("All tests completed successfully!")
            print(f"{'='*80}")
            
            return True
            
        except Exception as e:
            print(f"\nTest failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Main test function"""
    tester = MultiArmedBanditTester()
    success = await tester.run_all_tests()
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
