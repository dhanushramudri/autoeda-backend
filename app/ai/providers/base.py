"""Abstract interface every LLM provider must implement."""
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """Return generated text or None on failure."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider label (e.g. 'gemini', 'openai')."""
        ...
