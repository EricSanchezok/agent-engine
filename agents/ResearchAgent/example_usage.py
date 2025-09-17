"""
Example usage of the Research Agent system

This script demonstrates how to use the Research Agent system programmatically.
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
from researcher import Researcher

logger = AgentLogger(__name__)


async def example_usage():
    """Example of using the Research Agent system"""
    
    print("🤖 Research Agent Example Usage")
    print("=" * 50)
    
    # Check if Azure API key is set
    azure_api_key = os.getenv("AZURE_API_KEY")
    if not azure_api_key:
        print("❌ Error: AZURE_API_KEY environment variable is required")
        print("Please set your Azure OpenAI API key:")
        print("  export AZURE_API_KEY=your_api_key_here")
        return
    
    try:
        # Initialize researcher
        print("🚀 Initializing Researcher...")
        researcher = Researcher(
            azure_api_key=azure_api_key,
            model_name="gpt-4.1"
        )
        
        # Example conversation
        user_id = "example_user"
        session_id = "example_session"
        
        print(f"👤 User ID: {user_id}")
        print(f"💬 Session ID: {session_id}")
        print()
        
        # First question
        print("🤔 Question 1: What is artificial intelligence?")
        response1 = await researcher.chat(
            user_id=user_id,
            session_id=session_id,
            user_message="What is artificial intelligence?",
            max_tokens=500,
            temperature=0.7
        )
        print(f"🤖 Response: {response1}")
        print()
        
        # Second question (should have context from first)
        print("🤔 Question 2: How does it relate to machine learning?")
        response2 = await researcher.chat(
            user_id=user_id,
            session_id=session_id,
            user_message="How does it relate to machine learning?",
            max_tokens=500,
            temperature=0.7
        )
        print(f"🤖 Response: {response2}")
        print()
        
        # Third question (should have context from both previous)
        print("🤔 Question 3: What are some practical applications?")
        response3 = await researcher.chat(
            user_id=user_id,
            session_id=session_id,
            user_message="What are some practical applications?",
            max_tokens=500,
            temperature=0.7
        )
        print(f"🤖 Response: {response3}")
        print()
        
        # Get session statistics
        print("📊 Session Statistics:")
        stats = researcher.get_session_stats(user_id, session_id)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()
        
        # Example of streaming response
        print("🌊 Streaming Response Example:")
        print("🤔 Question: Can you explain neural networks in simple terms?")
        print("🤖 Response: ", end="", flush=True)
        
        async for chunk in researcher.chat_stream(
            user_id=user_id,
            session_id=session_id,
            user_message="Can you explain neural networks in simple terms?",
            max_tokens=300,
            temperature=0.7
        ):
            print(chunk, end="", flush=True)
        print("\n")
        
        # Cleanup
        await researcher.close()
        
        print("✅ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Example usage error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example error: {e}")
