from agents.PaperFilterAgent.agent import PaperFilterAgent
import asyncio

async def main():
    agent = PaperFilterAgent()
    await agent._initialize_qiji_memory()
    agent.run_server()

if __name__ == "__main__":
    asyncio.run(main())