import asyncio
from dotenv import load_dotenv
import os
load_dotenv()
from browser_use import Agent, ChatOpenAI, ChatAzureOpenAI

async def main():
    agent = Agent(
        task="Find the number of stars of the browser-use repo",
        llm=ChatAzureOpenAI(
            model="gpt-4o",
            api_version = "2025-04-01-preview",
            base_url = 'https://gpt.yunstorm.com/',
            api_key = os.getenv('AZURE_API_KEY'),
        ),
    )
    await agent.run()

asyncio.run(main())