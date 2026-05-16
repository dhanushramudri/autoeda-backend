"""Gemini provider using the google-genai SDK."""
import logging
import os
from typing import Optional

from .base import LLMProvider

logger = logging.getLogger("autoeda.ai.providers.gemini")

_MODEL = "gemini-2.5-flash"


class GeminiProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "gemini"

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return None
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    # Disable thinking budget — Gemini 2.5 Flash uses thinking tokens
                    # by default which eats into max_output_tokens, truncating replies.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.warning("Gemini generate failed: %s", e)
            return None
