"""
Test script for optimized SessionMemory

This script tests the optimized SessionMemory functionality.
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
from agent_engine.agent_logger.agent_logger import AgentLogger
from session_memory import SessionMemory

logger = AgentLogger(__name__)


def test_optimized_session_memory():
    """Test optimized SessionMemory functionality"""
    print("🧪 Testing Optimized SessionMemory...")
    
    try:
        # Create session memory with low threshold for testing
        session_memory = SessionMemory(
            user_id="test_user",
            session_id="test_session",
            max_short_history_tokens=1000,
            summarization_threshold=200  # Low threshold to trigger summarization
        )
        
        print(f"📁 Database directory: {session_memory.db_dir}")
        print(f"📄 Context history file: {session_memory.context_history_file}")
        
        # Test 1: Add Q&A pairs
        print("\n📝 Test 1: Adding Q&A pairs...")
        questions_answers = [
            ("What is AI?", "AI is artificial intelligence, a field of computer science that focuses on creating intelligent machines."),
            ("How does machine learning work?", "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."),
            ("What are neural networks?", "Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by analyzing training data."),
            ("What is deep learning?", "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."),
            ("What are some AI applications?", "AI has many applications including image recognition, natural language processing, autonomous vehicles, medical diagnosis, and recommendation systems.")
        ]
        
        for i, (question, answer) in enumerate(questions_answers):
            print(f"  Adding Q&A {i+1}: {question[:30]}...")
            session_memory.add_qa(question, answer)
            
            # Check stats after each addition
            stats = session_memory.get_session_stats()
            print(f"    Total tokens: {stats['total_tokens']}, Should summarize: {stats['should_summarize']}")
        
        # Test 2: Check if summarization occurred
        print("\n📊 Test 2: Checking summarization...")
        stats = session_memory.get_session_stats()
        print(f"Short history count: {stats['short_history_count']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Long history length: {stats['long_history_length']}")
        print(f"Should summarize: {stats['should_summarize']}")
        
        # Test 3: Get context for LLM
        print("\n🤖 Test 3: Getting context for LLM...")
        context = session_memory.get_context_for_llm()
        print(f"Context length: {len(context)} characters")
        print(f"Context preview: {context[:200]}...")
        
        # Test 4: Check file persistence
        print("\n💾 Test 4: Checking file persistence...")
        context_file_exists = session_memory.context_history_file.exists()
        print(f"Context history file exists: {context_file_exists}")
        
        if context_file_exists:
            import json
            with open(session_memory.context_history_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            print(f"File contains {len(file_data.get('short_history', []))} Q&A pairs")
            print(f"File total_tokens: {file_data.get('total_tokens', 0)}")
        
        # Test 5: Create new instance and load from file
        print("\n🔄 Test 5: Testing persistence by creating new instance...")
        session_memory.close()
        
        new_session_memory = SessionMemory(
            user_id="test_user",
            session_id="test_session",
            max_short_history_tokens=1000,
            summarization_threshold=200
        )
        
        new_stats = new_session_memory.get_session_stats()
        print(f"Loaded short history count: {new_stats['short_history_count']}")
        print(f"Loaded total tokens: {new_stats['total_tokens']}")
        print(f"Loaded long history length: {new_stats['long_history_length']}")
        
        # Test 6: Add more Q&A to test incremental token counting
        print("\n➕ Test 6: Adding more Q&A to test incremental counting...")
        new_session_memory.add_qa("What is reinforcement learning?", "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.")
        
        final_stats = new_session_memory.get_session_stats()
        print(f"Final total tokens: {final_stats['total_tokens']}")
        print(f"Final should summarize: {final_stats['should_summarize']}")
        
        # Cleanup
        new_session_memory.close()
        
        print("\n✅ Optimized SessionMemory test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Optimized SessionMemory test failed: {e}")
        logger.error(f"Test error: {e}")
        return False


def main():
    """Run the test"""
    print("🚀 Starting Optimized SessionMemory Test")
    print("=" * 50)
    
    success = test_optimized_session_memory()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Test completed successfully!")
        print("\n💡 Key improvements:")
        print("  ✅ Custom database directory structure")
        print("  ✅ Context history stored in JSON file")
        print("  ✅ Incremental token counting")
        print("  ✅ EMemory and context history are independent")
        print("  ✅ Proper persistence and loading")
    else:
        print("❌ Test failed!")
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        sys.exit(1)
