from pathlib import Path
from agent_engine.agent_logger import set_agent_log_directory

current_file_dir = Path(__file__).parent
log_dir = current_file_dir / 'logs'
set_agent_log_directory(str(log_dir))

from agents.PaperAnalysisAgent.agent import PaperAnalysisAgent


if __name__ == "__main__":
    agent = PaperAnalysisAgent()
    agent.run_server()
