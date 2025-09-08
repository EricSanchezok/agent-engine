import json
import asyncio
import os
from pprint import pprint
from dotenv import load_dotenv
import re
from typing import Optional

# Agent Engine imports
from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_current_file_dir


# Local imports
from agents.ICURiskPredictionAgent.risks_table import RisksTable

load_dotenv()

logger = AgentLogger(__name__)

class RiskDiagnosis:
    def __init__(self):
        self.logger = AgentLogger(self.__class__.__name__)

        # LLM client
        api_key = os.getenv("AZURE_API_KEY", "").strip()
        base_url = os.getenv("AZURE_BASE_URL", "https://gpt.yunstorm.com/")
        api_version = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
        self._translate_model = os.getenv("AGENT_ENGINE_TRANSLATE_MODEL", "gpt-4o")

        if not api_key:
            self.logger.warning("AZURE_API_KEY not set; translation is disabled")
            self._llm: Optional[AzureClient] = None
        else:
            self._llm = AzureClient(api_key=api_key, base_url=base_url, api_version=api_version)

        # Prompts
        try:
            prompts_path = get_current_file_dir() / "prompts.yaml"
            self._prompt_loader = PromptLoader(prompts_path)
        except Exception as e:
            self.logger.warning(f"PromptLoader init failed: {e}. Using fallback prompts.")
            self._prompt_loader = None
        self.risks_table = RisksTable()

    async def invoke(self, event: dict) -> list:
        event_content = event.get("event_content", "")
        if not event_content:
            return []

        system_prompt = self._prompt_loader.get_prompt(
            section="risk_diagnosis",
            prompt_type="system",
            risks_table=self.risks_table.table
        )
        user_prompt = self._prompt_loader.get_prompt(
            section="risk_diagnosis",
            prompt_type="user",
            event=event_content
        )

        initial_diagnosis = {}
        
        try:
            response = await self._llm.chat(system_prompt, user_prompt, model_name='o3-mini', max_tokens=8192)
            response = re.sub(r'\$\s*\{', '{', response)
            initial_diagnosis = json.loads(response)
        except Exception as e:
            logger.error(f"Error predicting risks: {e}")
            initial_diagnosis = {}

        system_prompt = self._prompt_loader.get_prompt(
            section="risk_audit",
            prompt_type="system",
            risks_table=self.risks_table.table
        )
        user_prompt = self._prompt_loader.get_prompt(
            section="risk_audit",
            prompt_type="user",
            initial_diagnosis=initial_diagnosis,
            event=event
        )

        try:
            response = await self._llm.chat(system_prompt, user_prompt, model_name='o3-mini', max_tokens=8192)
            response = re.sub(r'\$\s*\{', '{', response)
            final_diagnosis = json.loads(response)
        except Exception as e:
            logger.error(f"Error auditing risks: {e}")
            return []

        return final_diagnosis.get("risks", [])



async def main():
    event = {
        "id": "deae1a08-9e0c-4c6b-b8f8-647487285ba7",
        "timestamp": "2024-05-22T09:23:25",
        "event_type": "exam",
        "sub_type": "exam",
        "event_content": {
            "exams_name": "胸部(胸腔)(CT平扫64层)",
            "exams_findings": "脊柱侧弯两肺见散在少许条索影右肺中叶钙化灶各级支气管通畅无扩张与狭窄双侧肺门不大纵隔居中其内未见肿大淋巴结心脏饱满主动脉及部分冠脉见多发钙化未见胸腔积液和心包积液胸壁软组织未见异常附见双侧肾盂扩张",
            "exams_result": "1.两肺散在少许纤维灶右肺中叶钙化灶 \n2.心脏饱满主动脉及部分冠脉钙化 \n3.脊柱侧弯 \n附见双侧肾盂扩张"
        },
        "risks": [],
        "flag": 0,
        "metadata": {}
    }

    risk_diagnosis = RiskDiagnosis()
    risks = await risk_diagnosis.invoke(event)
    pprint(risks)

if __name__ == "__main__":
    asyncio.run(main())