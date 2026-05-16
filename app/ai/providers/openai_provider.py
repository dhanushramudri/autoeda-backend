"""OpenAI provider using the openai SDK."""
import logging
import os
from typing import Optional

from .base import LLMProvider

logger = logging.getLogger("autoeda.ai.providers.openai")

_MODEL = "gpt-4o-mini"


class OpenAIProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("OpenAI generate failed: %s", e)
            return None
