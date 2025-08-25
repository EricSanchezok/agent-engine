import json
import re
from pathlib import Path
from typing import List, Optional
from agent_engine.llm_client import LLMClient
from agent_engine.prompt import PromptLoader
from agent_engine.agent_logger import AgentLogger

logger = AgentLogger(__name__)

class ArxivIdParser:
    """Parse arXiv IDs from user input using LLM when JSON parsing fails"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        # Get the current file directory and construct path to prompts
        current_dir = Path(__file__).parent
        prompts_path = current_dir / 'prompts.yaml'
        self.prompt_loader = PromptLoader(prompts_path)
    
    async def extract_arxiv_ids(self, user_input: str) -> List[str]:
        """
        Extract arXiv IDs from user input.
        First try JSON parsing, then fallback to LLM parsing.
        
        Args:
            user_input: User input string that may contain arXiv IDs
            
        Returns:
            List of extracted arXiv IDs
        """
        # First try to parse as JSON
        try:
            parsed_data = json.loads(user_input)
            arxiv_ids = parsed_data.get('arxiv_ids', [])
            if not arxiv_ids:
                return await self._parse_with_llm(user_input)
            else:
                return arxiv_ids
        except Exception as e:
            logger.warning(f"Error during JSON parsing: {e}")
            return await self._parse_with_llm(user_input)
    
    async def _parse_with_llm(self, user_input: str) -> List[str]:
        """
        Use LLM to parse arXiv IDs from user input
        
        Args:
            user_input: User input string
            
        Returns:
            List of extracted arXiv IDs
        """
        try:
            # Get prompts
            system_prompt = self.prompt_loader.get_prompt('arxiv_id_parser', 'system')
            user_prompt = self.prompt_loader.get_prompt('arxiv_id_parser', 'user', user_input=user_input)

            # Call LLM with o3-mini model
            response = await self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name='o3-mini'
            )
            
            if response:
                # Parse LLM response
                parsed_ids = self._parse_llm_response(response)
                if parsed_ids:
                    logger.info(f"LLM successfully parsed {len(parsed_ids)} arXiv IDs")
                    return parsed_ids
                else:
                    logger.warning("LLM response could not be parsed for arXiv IDs")
            else:
                logger.error("LLM call failed")
                
        except Exception as e:
            logger.error(f"Error during LLM parsing: {e}")
        
        # Final fallback: try regex extraction
        logger.info("LLM parsing failed, trying regex extraction")
        return self._extract_with_regex(user_input)
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """
        Parse LLM response to extract arXiv IDs
        
        Args:
            response: LLM response string
            
        Returns:
            List of extracted arXiv IDs
        """
        try:
            parsed_data = json.loads(response)
            arxiv_ids = parsed_data.get('arxiv_ids', [])
            return arxiv_ids
        except Exception as e:
            logger.warning(f"Error during LLM response parsing: {e}")
            return []
    
    def _extract_with_regex(self, text: str) -> List[str]:
        """
        Extract arXiv IDs using regex as final fallback
        
        Args:
            text: Input text
            
        Returns:
            List of extracted arXiv IDs
        """
        # Regex pattern for arXiv IDs: YYYY.MMMMM or YYYY.MMMMMvN
        pattern = r'\b(20\d{2}\.\d{5}(?:v\d+)?)\b'
        matches = re.findall(pattern, text)
        
        # Clean and validate matches
        valid_ids = []
        for match in matches:
            if self._is_valid_arxiv_id(match):
                valid_ids.append(self._clean_arxiv_id(match))
        
        if valid_ids:
            logger.info(f"Regex extracted {len(valid_ids)} arXiv IDs")
        else:
            logger.warning("No valid arXiv IDs found with regex")
        
        return valid_ids
    
    def _is_valid_arxiv_id(self, arxiv_id: str) -> bool:
        """
        Validate if a string is a valid arXiv ID format
        
        Args:
            arxiv_id: String to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(arxiv_id, str):
            return False
        
        # Pattern: YYYY.MMMMM or YYYY.MMMMMvN
        pattern = r'^20\d{2}\.\d{5}(?:v\d+)?$'
        return bool(re.match(pattern, arxiv_id))
    
    def _clean_arxiv_id(self, arxiv_id: str) -> str:
        """
        Clean and normalize arXiv ID
        
        Args:
            arxiv_id: Raw arXiv ID string
            
        Returns:
            Cleaned arXiv ID string
        """
        if not isinstance(arxiv_id, str):
            return ""
        
        # Remove extra whitespace and convert to lowercase
        cleaned = arxiv_id.strip().lower()
        
        # Ensure proper format
        if re.match(r'^20\d{2}\.\d{5}(?:v\d+)?$', cleaned):
            return cleaned
        
        return ""
