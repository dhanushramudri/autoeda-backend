"""Thin wrapper around the Gemini generative AI SDK (google-genai)."""
import logging
import os
from typing import Optional

logger = logging.getLogger("autoeda.ai.gemini")

_MODEL_NAME = "gemini-2.5-flash"


def generate(prompt: str, temperature: float = 0.3) -> Optional[str]:
    """Call Gemini and return the text response, or None on failure."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set")
        return None
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=1024,
                # Disable thinking budget — otherwise Gemini 2.5 Flash consumes
                # output tokens on internal reasoning, truncating the actual reply.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning("Gemini generate failed: %s", e)
        return None
