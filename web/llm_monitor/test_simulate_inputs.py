import asyncio
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from agent_engine.llm_client import AzureClient
from agent_engine.agent_logger import AgentLogger
from agent_engine.utils.project_root import get_project_root


logger = AgentLogger("LLMMonitorInputsTest")


load_dotenv()


async def simulate_inputs(user_inputs: List[str]) -> None:
    api_key = os.getenv("AZURE_API_KEY")
    if not api_key:
        logger.error("AZURE_API_KEY is not set. Please set it in .env or environment.")
        return

    base_url = os.getenv("AZURE_BASE_URL", "https://gpt.yunstorm.com/")
    api_version = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
    model_name = os.getenv("AZURE_CHAT_MODEL", "gpt-4.1-nano")

    client = AzureClient(api_key=api_key, base_url=base_url, api_version=api_version)

    system_prompt = "You are a helpful assistant."

    for idx, user_prompt in enumerate(user_inputs, start=1):
        logger.info(f"Submitting input #{idx}: {user_prompt}")
        try:
            response = await client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=model_name,
                max_tokens=512,
                temperature=0.7,
            )
            preview = (response or "")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            logger.info(f"Response #{idx} preview: {preview}")
        except Exception as e:
            logger.error(f"Chat failed for input #{idx}: {e}")
        # Let the frontend poll and update
        await asyncio.sleep(2.0)

    await client.close()


def main():
    # Start the web server
    web_dir = Path(__file__).parent
    server = subprocess.Popen([sys.executable, str(web_dir / "run.py")])

    try:
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8765/index.html")

        # Prepare test inputs
        user_inputs = [
            "Say hello and include the word 'monitoring' (#1)",
            "Tell me a quick fun fact about space. Keep it short. (#2)",
            "Summarize in one sentence why logs should use IDs. (#3)",
            "Give a short tip for debugging long LLM prompts. (#4)",
            "Say goodbye with a friendly tone. (#5)",
            "How are you? (#6)",
            "What is the capital of France? (#7)",
            "What is the capital of China? (#8)",
            "What is the capital of Japan? (#9)",
            "What is the capital of Korea? (#10)",
            "What is the capital of Germany? (#11)",
            "What is the capital of Italy? (#12)",
            "What is the capital of Spain? (#13)"
        ]

        asyncio.run(simulate_inputs(user_inputs))

        print("Simulation completed. Server running. Press Ctrl+C to stop.")
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.terminate()


if __name__ == "__main__":
    main()


