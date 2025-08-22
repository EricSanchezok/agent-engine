import asyncio
import re
import datetime
import json
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# AgentEngine imports
from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.agent_logger import AgentLogger
from agent_engine.utils import get_relative_path_from_current_file

# Core imports
from core.utils import get_weekday, get_last_week_range

logger = AgentLogger(__name__)

load_dotenv()

class ArXivQueryParser:
    def __init__(self):
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))

    async def invoke(self, user_input: str) -> Optional[Dict]:
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        yesterday = yesterday.strftime("%Y%m%d")

        system_prompt = self.prompt_loader.get_prompt(
            section='query_parser',
            prompt_type='system'
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='query_parser',
            prompt_type='user',
            user_input=user_input,
            yesterday=yesterday,
            weekday=get_weekday(yesterday),
            last_week_range=get_last_week_range()
        )

        try:
            result = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
            result = json.loads(result)
        except Exception as e:
            logger.error(f"Query builder error: {e}")
            return {}

        if not result:
            logger.error("No query string provided")
            return {}

        date_range = self._format_date(result.get("date_range", {}))
        result["date_range"] = date_range
        return result

    def _format_date(self, date_range: Dict) -> Dict:
        if date_range.get("start") is None or date_range.get("end") is None:
            logger.warning("No date range provided, using last year as default")
            return self._last_year_range()

        start_dt = self._parse_date(date_range.get("start"))
        end_dt = self._parse_date(date_range.get("end"))
        if start_dt is None or end_dt is None:
            logger.warning("Invalid date format, using last year as default")
            return self._last_year_range()
        
        if start_dt == end_dt:
            logger.warning("Same date range, using yesterday as default")
            return self._yesterday_range()
        
        if start_dt > end_dt:
            logger.warning("Start date is after end date, using last year as default")
            return self._last_year_range()
        
        now = datetime.datetime.now()
        if start_dt > now:
            logger.warning("Start date is in the future, using last year as default")
            return self._last_year_range()
        
        return {"start": start_dt.strftime("%Y%m%d"), "end": end_dt.strftime("%Y%m%d")}

    @staticmethod
    def _yesterday_range() -> Dict[str, str]:
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        start_dt = yesterday - datetime.timedelta(days=1)
        end_dt = yesterday
        return {"start": start_dt.strftime("%Y%m%d"), "end": end_dt.strftime("%Y%m%d")}

    @staticmethod
    def _last_year_range() -> Dict[str, str]:
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        start_dt = yesterday.replace(year=yesterday.year - 1)
        end_dt = yesterday
        return {"start": start_dt.strftime("%Y%m%d"), "end": end_dt.strftime("%Y%m%d")}

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime.datetime]:
        if re.match(r'^\d{14}$', date_str):
            try:
                return datetime.datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                pass
        
        formats_to_try = [
            '%Y%m%d',
            '%Y%m%d %H:%M:%S',
            '%Y%m%d %H:%M',
            '%Y%m%d %H:%M:%S.%f',
            '%Y%m%d %H:%M:%S.%f%z',
            '%Y%m%d %H:%M:%S.%f%z',
        ]
        
        for fmt in formats_to_try:
            try:
                return datetime.datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Cannot parse date format: {date_str}")
        return None


async def main():
    parser = ArXivQueryParser()
    result = await parser.invoke("我对经济学和人工智能的领域比较感兴趣，想查一下相关的文献")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())