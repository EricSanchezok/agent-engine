from pathlib import Path
from agent_engine.agent_logger import set_agent_log_directory

current_file_dir = Path(__file__).parent
log_dir = current_file_dir / 'logs'
set_agent_log_directory(str(log_dir))

from agents.PaperFilterAgent.agent import PaperFilterAgent


if __name__ == "__main__":
    agent = PaperFilterAgent()
    agent.run_server()