#!/usr/bin/env python3
"""
Demo script showing multi-agent logging separation
"""

import os
import sys
import tempfile
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.agent_logger import AgentLogger, set_agent_log_directory, get_agent_log_directory


def simulate_agent_process(agent_name: str, log_dir: Path):
    """Simulate an agent process with its own logging"""
    print(f"\n🚀 Starting {agent_name} process...")
    
    # Set the log directory for this thread (simulating a process)
    set_agent_log_directory(str(log_dir))
    print(f"📁 {agent_name} log directory set to: {get_agent_log_directory()}")
    
    # Create various loggers that would be used in different modules
    agent_logger = AgentLogger(f"{agent_name}.Main")
    skill_logger = AgentLogger(f"{agent_name}.SkillIdentifier")
    llm_logger = AgentLogger(f"{agent_name}.LLMClient")
    prompt_logger = AgentLogger(f"{agent_name}.PromptLoader")
    
    # Simulate some logging activity
    agent_logger.info(f"{agent_name} initialized successfully")
    skill_logger.info(f"{agent_name} skill identification started")
    llm_logger.info(f"{agent_name} LLM client connected")
    prompt_logger.info(f"{agent_name} prompts loaded")
    
    # Simulate some error logging
    agent_logger.warning(f"{agent_name} encountered a minor issue")
    skill_logger.error(f"{agent_name} skill identification failed")
    
    print(f"✅ {agent_name} process completed. Logs saved to: {log_dir}")


def main():
    """Main demo function"""
    print("🎯 Multi-Agent Logging Separation Demo")
    print("=" * 50)
    
    # Create temporary directories for each agent
    with tempfile.TemporaryDirectory() as temp_base:
        temp_base_path = Path(temp_base)
        
        # Create log directories for each agent
        arxiv_logs = temp_base_path / "ArxivSearchAgent" / "logs"
        filter_logs = temp_base_path / "PaperFilterAgent" / "logs"
        fetch_logs = temp_base_path / "PaperFetchAgent" / "logs"
        analysis_logs = temp_base_path / "PaperAnalysisAgent" / "logs"
        
        # Create directories
        for log_dir in [arxiv_logs, filter_logs, fetch_logs, analysis_logs]:
            log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Created temporary log directories in: {temp_base_path}")
        
        # Simulate multiple agent processes running simultaneously
        threads = []
        
        # ArxivSearchAgent
        arxiv_thread = threading.Thread(
            target=simulate_agent_process,
            args=("ArxivSearchAgent", arxiv_logs)
        )
        threads.append(arxiv_thread)
        
        # PaperFilterAgent
        filter_thread = threading.Thread(
            target=simulate_agent_process,
            args=("PaperFilterAgent", filter_logs)
        )
        threads.append(filter_thread)
        
        # PaperFetchAgent
        fetch_thread = threading.Thread(
            target=simulate_agent_process,
            args=("PaperFetchAgent", fetch_logs)
        )
        threads.append(fetch_thread)
        
        # PaperAnalysisAgent
        analysis_thread = threading.Thread(
            target=simulate_agent_process,
            args=("PaperAnalysisAgent", analysis_logs)
        )
        threads.append(analysis_thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Show results
        print("\n" + "=" * 50)
        print("📊 Logging Results Summary:")
        print("=" * 50)
        
        for agent_name, log_dir in [
            ("ArxivSearchAgent", arxiv_logs),
            ("PaperFilterAgent", filter_logs),
            ("PaperFetchAgent", fetch_logs),
            ("PaperAnalysisAgent", analysis_logs)
        ]:
            log_files = list(log_dir.glob("*.log"))
            print(f"📁 {agent_name}: {len(log_files)} log files")
            for log_file in log_files:
                print(f"   └─ {log_file.name}")
        
        print(f"\n🎉 Demo completed! All agent logs are separated in their own directories.")
        print(f"💡 In a real scenario, each agent would have its logs in agents/{agent_name}/logs/")


if __name__ == "__main__":
    main()
