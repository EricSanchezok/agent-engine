import json
from typing import List, Tuple


# A2A framework imports
from a2a.types import AgentSkill

# Internal imports
from ..agent_logger.agent_logger import AgentLogger
from ..prompt.prompt_loader import PromptLoader
from ..utils.file_utils import get_relative_path_from_current_file
from ..llm_client import LLMClient

logger = AgentLogger(__name__)


class SkillsDescGenerator:
    def prompt(self, skills: List[AgentSkill]) -> str:
        if not skills:
            return ""
        
        prompt = "## 可用技能列表\n\n"
        
        for skill in skills:
            prompt += f"### 技能ID: {skill.id}\n"
            
            if hasattr(skill, 'name') and skill.name:
                prompt += f"- 中文名称: {skill.name}\n"
            if hasattr(skill, 'name_en') and skill.name_en:
                prompt += f"- 英文名称: {skill.name_en}\n"
            
            if hasattr(skill, 'description') and skill.description:
                prompt += f"- 中文描述: {skill.description}\n"
            if hasattr(skill, 'description_en') and skill.description_en:
                prompt += f"- 英文描述: {skill.description_en}\n"
            
            if hasattr(skill, 'tags') and skill.tags:
                prompt += f"- 标签: {', '.join(skill.tags)}\n"
            
            if hasattr(skill, 'examples') and skill.examples:
                prompt += "- 使用示例:\n"
                for i, example in enumerate(skill.examples, 1):
                    prompt += f"  {i}. {example}\n"
            
            prompt += "\n"
        
        return prompt

class SkillIdentifier:
    def __init__(self, llm_client: LLMClient, model_name: str = "o3-mini"):
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))
        self.llm_client = llm_client
        self.model_name = model_name
        self.skills_desc_generator = SkillsDescGenerator()
        
    async def invoke(self, user_input: str, skills: List[AgentSkill]) -> Tuple[str, str]:
        skills_desc = self.skills_desc_generator.prompt(skills)

        if not skills_desc:
            logger.error("No skills description found")
            return "", "No skills description found"

        # system_prompt = self.prompt_loader.get_prompt(
        #     section='task_purifier',
        #     prompt_type='system'
        # )
        # user_prompt = self.prompt_loader.get_prompt(
        #     section='task_purifier',
        #     prompt_type='user',
        #     user_input=user_input
        # )
        # try:
        #     result = await self.llm_client.chat(system_prompt, user_prompt, model_name=self.model_name)
        #     result = json.loads(result)
        #     core_task = result.get("core_task", "")
        # except Exception as e:
        #     logger.error(f"Task purifier error: {e}")
        #     return "", "Task purifier error"

        # if not core_task:
        #     logger.error("No core task found")
        #     return "", "No core task found"
        
        system_prompt = self.prompt_loader.get_prompt(
            section='skill_identifier',
            prompt_type='system',
            skills_desc=skills_desc
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='skill_identifier',
            prompt_type='user',
            user_input=user_input
        )
        try:
            result = await self.llm_client.chat(system_prompt, user_prompt, model_name=self.model_name)
            result = json.loads(result)
            return result.get("skill_id", ""), result.get("reason", "")
        except Exception as e:
            logger.error(f"Skill identifier error: {e}")
            return "", "Skill identifier error"