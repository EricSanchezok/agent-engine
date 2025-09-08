import json
import asyncio
import os
from pprint import pprint
from dotenv import load_dotenv


# Agent Engine imports
from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_relative_path_from_current_file


# Local imports
from agents.ICURiskPredictionAgent.risks_table import RisksTable

load_dotenv()

logger = AgentLogger(__name__)

class RiskDiagnosis:
    def __init__(self):
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        self.risks_table = RisksTable()

    async def invoke(self, event: dict) -> dict:
        system_prompt = self.prompt_loader.get_prompt(
            section="risk_diagnosis",
            prompt_type="system",
            risks_table=self.risks_table.table
        )
        user_prompt = self.prompt_loader.get_prompt(
            section="risk_diagnosis",
            prompt_type="user",
            event=event
        )

        initial_diagnosis = {}
        
        try:
            response = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini', max_tokens=32000)
            if "$" in response:
                response = response.replace("$", "")
            initial_diagnosis = json.loads(response)
        except Exception as e:
            logger.error(f"Error predicting risks: {e}")
            initial_diagnosis = {}

        system_prompt = self.prompt_loader.get_prompt(
            section="risk_audit",
            prompt_type="system",
            risks_table=self.risks_table.table
        )
        user_prompt = self.prompt_loader.get_prompt(
            section="risk_audit",
            prompt_type="user",
            initial_diagnosis=initial_diagnosis,
            event=event
        )

        try:
            response = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini', max_tokens=32000)
            if "$" in response:
                response = response.replace("$", "")
            final_diagnosis = json.loads(response)
        except Exception as e:
            logger.error(f"Error auditing risks: {e}")
            return []

        return final_diagnosis.get("risks", [])



async def main():
    event = {
        "timestamp": "2024-06-27T07:53:56",
        "event_type": "exam",
        "event_content": {
            "exams_name": "胸部正位(床边)",
            "exams_findings": "床旁胸片大致所示胸部术后胸骨见缝线影两侧胸廓对称气管居中两肺纹理增多两肺絮状渗出影两肺门大小位置如常纵膈未见增宽主动脉未见迂曲增宽钙化卧位心影增大两侧膈面及肋膈角模糊右侧深静脉导管置管",
            "exams_result": "胸部术后改变心影增大两肺渗出两侧胸腔积液"
        },
        "flag": 0,
        "risks": []
    }

    risk_diagnosis = RiskDiagnosis()
    risks = await risk_diagnosis.invoke(event)
    pprint(risks)

if __name__ == "__main__":
    asyncio.run(main())