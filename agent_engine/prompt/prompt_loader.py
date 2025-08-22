import yaml
from string import Template
from typing import Dict, Any

# Internal imports
from ..agent_logger.agent_logger import AgentLogger

logger = AgentLogger('PromptLoader')

class PromptLoader:
    def __init__(self, file_path: str = None):
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            self.prompt_data = yaml.safe_load(file)
    
    def get_prompt(self, section: str, prompt_type: str, **variables: Dict[str, Any]) -> str:
        section_data = self.prompt_data.get(section)
        if not section_data:
            raise ValueError(f"Section '{section}' not found in prompt data")
        
        prompt_key = f"{prompt_type}_prompt"
        prompt_template = section_data.get(prompt_key)
        if not prompt_template:
            raise ValueError(f"Prompt type '{prompt_key}' not found in section '{section}'")
        
        input_vars = prompt_template.get('input_variables', [])
        missing_vars = [var for var in input_vars if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        template_content = prompt_template['template']
        safe_template = Template(template_content.replace('{', '${'))
        
        return safe_template.safe_substitute(variables)