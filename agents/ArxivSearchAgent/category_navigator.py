import asyncio
import re
import json
from typing import List, Dict, Optional
from difflib import get_close_matches
import os
from dotenv import load_dotenv

# AgentEngine imports
from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_relative_path_from_current_file

logger = AgentLogger(__name__)

load_dotenv()

with open(get_relative_path_from_current_file('arxiv_categories.json'), 'r', encoding='utf-8') as f:
    ARXIV_CATEGORIES = json.load(f)

class ArXivCategoryNavigator:
    def __init__(self, category_data: List[Dict] = ARXIV_CATEGORIES):
        self.category_data = category_data
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))

    async def invoke(self, user_input: str) -> Optional[List[str]]:
        system_prompt = self.prompt_loader.get_prompt(
            section='cat_majors',
            prompt_type='system',
            majors=self.majors(),
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='cat_majors',
            prompt_type='user',
            user_input=user_input,
        )
        majors = []
        try:
            result = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
            result = json.loads(result)
            for _major_name in result.get("result", []):
                major_name = self._format_major_name(_major_name)
                if major_name is not None:
                    majors.append(major_name)
        except Exception as e:
            logger.error(f"Category navigator select majors error: {e}")
        
        logger.info(f"Majors: {majors}")
        system_prompt = self.prompt_loader.get_prompt(
            section='cat_categories',
            prompt_type='system',
            categories=self.categories(majors),
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='cat_categories',
            prompt_type='user',
            user_input=user_input,
        )
        categories = []
        try:
            result = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
            result = json.loads(result)
            for _category_abbr in result.get("result", []):
                category_abbr = self._format_category_abbr(_category_abbr, self.categories(majors))
                if category_abbr is not None:
                    categories.append(category_abbr)
        except Exception as e:
            logger.error(f"Category navigator select categories error: {e}")

        logger.info(f"Categories: {categories}")
        return categories

    def groups(self) -> List[str]:
        return [group["group_name"] for group in self.category_data]
    
    def majors(self, groups: List[str] = None) -> List[Dict]:
        if groups is None:
            groups = self.groups()
        majors = []
        for group in self.category_data:
            if group["group_name"] in groups:
                for major in group["majors"]:
                    majors.append({
                        "major_name": major.get("major_name", ""),
                        "major_abbr": major.get("major_abbr", "")
                    })
        return majors
    
    def categories(self, majors: List[str] = None) -> List[Dict]:
        if majors is None:
            majors = self.majors()
        categories = []
        for group in self.category_data:
            for major in group["majors"]:
                if major["major_name"] in majors:
                    for category in major["categories"]:
                        categories.append({
                            "category_name": category.get("category_name", ""),
                            "category_abbr": category.get("category_abbr", ""),
                            "explanation": category.get("explanation", "")
                        })
        return categories

    def _format_major_name(self, major_name: str) -> Optional[str]:
        major_names = [major["major_name"] for major in self.majors()]
        matched_name = None
        match_type = "exact"

        if major_name in major_names:
            return major_name

        if not matched_name:
            matches = get_close_matches(major_name, major_names, n=1, cutoff=0.7)
            if matches:
                matched_name = matches[0]
                match_type = "fuzzy"

        if not matched_name:
            for valid_name in major_names:
                if major_name.lower() in valid_name.lower().split():
                    matched_name = valid_name
                    match_type = "abbreviation"
                    break

        if not matched_name:
            for valid_name in major_names:
                if any(word in valid_name.lower() for word in major_name.lower().split()):
                    matched_name = valid_name
                    match_type = "keyword"
                    break

        if not matched_name and major_name:
            first_letter = major_name[0].upper()
            for valid_name in major_names:
                if valid_name[0].upper() == first_letter:
                    matched_name = valid_name
                    match_type = "first_letter"
                    break

        if matched_name:
            logger.warning(f"Major name matched: '{major_name}' -> '{matched_name}' ({match_type} match)")
            return matched_name
        else:
            logger.warning(f"Major name match failed: '{major_name}'")
            return None

    def _format_category_abbr(self, category_abbr: str, categories: List[Dict]) -> Optional[str]:
        category_abbrs = [category["category_abbr"] for category in categories]
        original_abbr = category_abbr
        matched_abbr = None
        match_type = "exact"
        
        if category_abbr in category_abbrs:
            return category_abbr

        lower_abbr = category_abbr.lower()
        for abbr in category_abbrs:
            if abbr.lower() == lower_abbr:
                matched_abbr = abbr
                match_type = "lowercase"
                break
        
        if not matched_abbr and '.' not in category_abbr:
            parts = re.split(r'[\s\-_]+', category_abbr)
            if len(parts) == 2:
                dotted_abbr = f"{parts[0]}.{parts[1]}"
                if dotted_abbr in category_abbrs:
                    matched_abbr = dotted_abbr
                    match_type = "dotted"
                else:
                    for abbr in category_abbrs:
                        if abbr.lower() == dotted_abbr.lower():
                            matched_abbr = abbr
                            match_type = "dotted_lowercase"
                            break
        
        if not matched_abbr:
            suffix = category_abbr.split('.')[-1] if '.' in category_abbr else category_abbr
            for abbr in category_abbrs:
                if abbr.split('.')[-1].lower() == suffix.lower():
                    matched_abbr = abbr
                    match_type = "suffix"
                    break
        
        if not matched_abbr:
            prefix = category_abbr.split('.')[0] if '.' in category_abbr else category_abbr
            prefix_matches = []
            for abbr in category_abbrs:
                if abbr.split('.')[0].lower() == prefix.lower():
                    prefix_matches.append(abbr)
            
            if prefix_matches:
                matched_abbr = prefix_matches[0]
                match_type = "prefix"
        
        if not matched_abbr:
            matches = get_close_matches(
                category_abbr, 
                category_abbrs, 
                n=1, 
                cutoff=0.8
            )
            if matches:
                matched_abbr = matches[0]
                match_type = "fuzzy"
        
        if not matched_abbr:
            keywords = re.split(r'[.\s\-_]+', category_abbr)
            for abbr in category_abbrs:
                if any(kw.lower() in abbr.lower() for kw in keywords):
                    matched_abbr = abbr
                    match_type = "keyword"
                    break
        
        if matched_abbr:
            if matched_abbr != original_abbr:
                logger.warning(f"Category abbr matched: '{original_abbr}' -> '{matched_abbr}' ({match_type} match)")
            return matched_abbr
        else:
            logger.warning(f"Category abbr match failed: '{original_abbr}'")
            return None


async def main():
    navigator = ArXivCategoryNavigator()
    # print(navigator.categories(['Economics']))
    
    result = await navigator.invoke("我对经济学和人工智能的领域比较感兴趣，想查一下相关的文献")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())