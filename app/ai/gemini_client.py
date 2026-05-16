"""Thin wrapper around the Gemini generative AI SDK."""
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger("autoeda.ai.gemini")

_MODEL_NAME = "gemini-1.5-flash"


@lru_cache(maxsize=1)
def _get_model():
    try:
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(_MODEL_NAME)
    except Exception as e:
        logger.warning("Gemini client init failed: %s", e)
        return None


def generate(prompt: str, temperature: float = 0.3) -> Optional[str]:
    """Call Gemini and return the text response, or None on failure."""
    model = _get_model()
    if model is None:
        return None
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 1024},
        )
        return response.text.strip()
    except Exception as e:
        logger.warning("Gemini generate failed: %s", e)
        return None
