from agents.PaperFilterAgent.agent import PaperFilterAgent
import asyncio

agent = PaperFilterAgent()

async def init():
    await agent._initialize_qiji_memory()
    
if __name__ == "__main__":
    asyncio.run(init())
    agent.run_server()