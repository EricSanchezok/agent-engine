from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv

from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.memory import ScalableMemory
from agent_engine.prompt import PromptLoader
from agent_engine.utils import get_current_file_dir

load_dotenv()

class UMLSClinicalTranslator:
    """UMLS-oriented professional translator with persistent cache.

    - Translates Chinese ICU event content to professional English aligned with biomedical terminology
    - Caches translations in a local ScalableMemory under this agent's directory
    - Supports overwrite (re-translate even if a cached entry exists)
    - Provides a method to clear the entire translation cache
    """

    def __init__(self) -> None:
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

        # Persistent cache
        persist_dir = get_current_file_dir() / "database"
        self._mem = ScalableMemory(
            name="icu_translation_cache",
            persist_dir=str(persist_dir),
            enable_vectors=False,
            db_backend="duckdb",
        )

        # No runner; use async methods directly

    def clear_cache(self) -> None:
        """Delete all cached translations."""
        try:
            self._mem.clear()
            self.logger.info("Translation cache cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear translation cache: {e}")

    async def translate_event_content(self, event_id: Optional[str], text_cn: str, real_translate: bool = False, overwrite: bool = False) -> Optional[str]:
        """Get or create an English translation for the given Chinese text.

        - If event_id is provided and overwrite=False, try cache first
        - If real_translate is False, return the original text
        - If not in cache or overwrite=True, call LLM and upsert into cache when event_id is available
        - If no LLM configured, return None
        """

        if not real_translate:
            return text_cn

        text_cn = (text_cn or "").strip()
        if not text_cn:
            return None

        if event_id and not overwrite:
            content, _, _ = self._mem.get_by_id(event_id)
            if content:
                return content

        if not self._llm:
            return None

        try:
            sys_p = self._prompt_loader.get_prompt(
                section="translate_event_to_english",
                prompt_type="system",
            )
            usr_p = self._prompt_loader.get_prompt(
                section="translate_event_to_english",
                prompt_type="user",
                event_content_cn=text_cn,
            )
            translated = await self._llm.chat(
                system_prompt=sys_p,
                user_prompt=usr_p,
                model_name=self._translate_model,
                max_tokens=8192,
                temperature=0.2,
            )
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return None

        content_en = (translated or "").strip()
        if not content_en:
            return None

        if event_id:
            try:
                # Upsert translation into cache
                await self._mem.add(
                    content=content_en,
                    item_id=event_id,
                )
            except Exception as e:
                self.logger.warning(f"Failed to cache translation for {event_id}: {e}")

        return content_en

    # Alias
    async def get_translation(self, event_id: Optional[str], text_cn: str, real_translate: bool = False, overwrite: bool = False) -> Optional[str]:
        return await self.translate_event_content(event_id, text_cn, real_translate=real_translate, overwrite=overwrite)


if __name__ == '__main__':
    translator = UMLSClinicalTranslator()
    print(translator._mem.count())