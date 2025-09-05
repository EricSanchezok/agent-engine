import asyncio
import os
import json
from typing import Any, Dict, List

from dotenv import load_dotenv

from agent_engine.llm_client import AzureClient
from agent_engine.llm_client import LLMChatMonitor
from agent_engine.agent_logger import AgentLogger

logger = AgentLogger("LLMMonitorTest")


load_dotenv()


async def _run_chat_test() -> None:
    """Run a simple chat and verify it is recorded by LLMChatMonitor."""
    api_key = os.getenv("AZURE_API_KEY")
    if not api_key:
        logger.error("AZURE_API_KEY is not set. Please set it in .env or environment.")
        return

    base_url = os.getenv("AZURE_BASE_URL", "https://gpt.yunstorm.com/")
    api_version = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
    model_name = os.getenv("AZURE_CHAT_MODEL", "o3-mini")

    client = AzureClient(api_key=api_key, base_url=base_url, api_version=api_version)

    system_prompt = "You are a helpful assistant."
    user_prompt = "Say hello and include the word 'monitoring'."

    logger.info("Starting chat test...")
    response = await client.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        max_tokens=512,
        temperature=0.7,
    )

    if response is None:
        logger.error("Chat returned None. Check network or API key configuration.")
    else:
        preview = response if len(response) < 200 else response[:200] + "..."
        logger.info(f"Chat response preview: {preview}")

    # Try to read the most recent monitor entry
    try:
        monitor = getattr(client, "monitor", None) or LLMChatMonitor(name="llm_chats", enable_vectors=False)
        metadatas: List[Dict[str, Any]] = monitor.memory.get_all_metadata()
        if not metadatas:
            logger.warning("No monitoring records found.")
        else:
            # Sort by started_at desc if available
            def _started_at(md: Dict[str, Any]) -> str:
                return md.get("started_at") or ""

            metadatas_sorted = sorted(metadatas, key=_started_at, reverse=True)
            latest = metadatas_sorted[0]
            latest_id = latest.get("id")
            logger.info(f"Latest monitoring record id: {latest_id}, status: {latest.get('status')}")
            if latest_id:
                record = monitor.get_chat(latest_id)
                # Print full stored content and metadata as JSON, even if empty
                content_json = json.dumps(record.get('content', {}), ensure_ascii=False, indent=2, sort_keys=True)
                logger.info("Full monitoring record (content):\n" + content_json)
    except Exception as e:
        logger.warning(f"Failed to read monitoring record: {e}")

    await client.close()


async def main() -> None:
    await _run_chat_test()


if __name__ == "__main__":
    asyncio.run(main())


