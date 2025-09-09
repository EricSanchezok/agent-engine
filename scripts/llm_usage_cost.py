from agent_engine.llm_client import LLMChatMonitor
from pprint import pprint

if __name__ == "__main__":
    monitor = LLMChatMonitor()
    summary = monitor.summarize_usage_cost()
    pprint(summary)