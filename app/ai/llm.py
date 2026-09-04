"""
LLM router — selects provider based on available API keys.

Priority: OPENAI_API_KEY / Azure OpenAI → ANTHROPIC_API_KEY → GEMINI_API_KEY → None
(OpenAI — Azure-hosted in production — is the primary provider; Claude is
a secondary fallback; Gemini is the last-resort fallback while no paid
key is configured.)

Adding a new provider later:
  1. Create app/ai/providers/<name>.py implementing LLMProvider
  2. Add it to _build_provider() below
  3. Set the corresponding env var

Note: keys must be read via `settings` (pydantic-settings), not raw
os.environ — nothing in this app calls load_dotenv(), so values that
only live in .env never reach os.environ directly. Only declared
Settings fields actually pick up .env values.
"""
import logging
from typing import Optional

from .providers.base import LLMProvider

logger = logging.getLogger("autoeda.ai.llm")

_cached_provider: Optional[LLMProvider] = None
_provider_checked = False


def _build_provider() -> Optional[LLMProvider]:
    from ..config import settings

    openai_key = settings.OPENAI_API_KEY
    azure_key = settings.AZURE_OPENAI_API_KEY or settings.TENALI_AI_API
    azure_configured = bool(azure_key and settings.AZURE_OPENAI_ENDPOINT and settings.AZURE_OPENAI_DEPLOYMENT)
    anthropic_key = settings.ANTHROPIC_API_KEY
    gemini_key = settings.GEMINI_API_KEY

    if openai_key or azure_configured:
        from .providers.openai_provider import OpenAIProvider
        logger.info("AI provider: %s", "Azure OpenAI" if azure_configured else "OpenAI")
        return OpenAIProvider()

    if anthropic_key:
        from .providers.claude import ClaudeProvider
        logger.info("AI provider: Claude")
        return ClaudeProvider()

    if gemini_key:
        from .providers.gemini import GeminiProvider
        logger.info("AI provider: Gemini")
        return GeminiProvider()

    logger.warning("No AI provider configured — set AZURE_OPENAI_* (or OPENAI_API_KEY), ANTHROPIC_API_KEY, or GEMINI_API_KEY")
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
