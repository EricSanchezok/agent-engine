from agent_engine.llm_client.azure_client import AzureClient
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()


async def test_chat_stream():
    client = AzureClient(api_key=os.getenv("AZURE_API_KEY"))
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the weather like today?"
    async for chunk in client.chat_stream(system_prompt, user_prompt):
        print(chunk)

if __name__ == "__main__":
    asyncio.run(test_chat_stream())