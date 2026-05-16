"""
LLM router — selects provider based on available API keys.

Priority: OPENAI_API_KEY → GEMINI_API_KEY → None

Adding a new provider later:
  1. Create app/ai/providers/<name>.py implementing LLMProvider
  2. Add it to _build_provider() below
  3. Set the corresponding env var
"""
import logging
import os
from typing import Optional

from .providers.base import LLMProvider

logger = logging.getLogger("autoeda.ai.llm")

_cached_provider: Optional[LLMProvider] = None
_provider_checked = False


def _build_provider() -> Optional[LLMProvider]:
    if os.environ.get("OPENAI_API_KEY"):
        from .providers.openai_provider import OpenAIProvider
        logger.info("AI provider: OpenAI")
        return OpenAIProvider()

    if os.environ.get("GEMINI_API_KEY"):
        from .providers.gemini import GeminiProvider
        logger.info("AI provider: Gemini")
        return GeminiProvider()

    logger.warning("No AI provider configured — set OPENAI_API_KEY or GEMINI_API_KEY")
    return None


def get_provider() -> Optional[LLMProvider]:
    """Return the active provider (cached per process)."""
    global _cached_provider, _provider_checked
    if not _provider_checked:
        _cached_provider = _build_provider()
        _provider_checked = True
    return _cached_provider


def generate(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Optional[str]:
    """Generate text using the active provider. Returns None if no provider."""
    provider = get_provider()
    if provider is None:
        return None
    return provider.generate(prompt, temperature=temperature, max_tokens=max_tokens)


def provider_name() -> str:
    """Return active provider name or 'none'."""
    p = get_provider()
    return p.provider_name if p else "none"
