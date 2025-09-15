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


from typing import Any, Dict, List

from agent_engine.agent_logger import AgentLogger

from core.icu_web.frontend_bridge import FrontendBridge


logger = AgentLogger(__name__)


class CollectorToFrontendAdapter:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.bridge = FrontendBridge.get_instance()

    async def on_plan_start(self, iteration: int, max_iterations: int) -> str:
        header = f"Building plan iteration {iteration}/{max_iterations}"
        return await self.bridge.create_message_block(self.session_id, header, block_type="agent")

    async def on_plan_llm(self, block_id: str, llm_raw: Any) -> None:
        await self.bridge.update_message_block(self.session_id, block_id, llm_raw)

    async def on_plan_result(self, block_id: str, plan: List[Dict[str, Any]]) -> None:
        await self.bridge.complete_message_block(self.session_id, block_id, {"plan": plan})

    async def on_assess_start(self) -> str:
        header = "Assessing coverage sufficiency"
        return await self.bridge.create_message_block(self.session_id, header, block_type="agent")

    async def on_assess_llm(self, block_id: str, llm_raw: Any) -> None:
        await self.bridge.update_message_block(self.session_id, block_id, llm_raw)

    async def on_assess_result(self, block_id: str, assessor: Dict[str, Any]) -> None:
        await self.bridge.complete_message_block(self.session_id, block_id, assessor)

    async def on_result_start(self) -> str:
        header = "Final aggregated result"
        return await self.bridge.create_message_block(self.session_id, header, block_type="agent")

    async def on_result_complete(self, block_id: str, events: List[Dict[str, Any]]) -> None:
        await self.bridge.complete_message_block(self.session_id, block_id, {"events": events})


