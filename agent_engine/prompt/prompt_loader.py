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