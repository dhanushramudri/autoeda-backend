"""Abstract interface every LLM provider must implement."""
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, TypedDict


class QuotaExceededError(Exception):
    """Raised by a provider when the failure is specifically a rate-limit/quota/
    billing error, as opposed to a generic outage — callers use this distinction
    to show users an accurate "AI quota reached" message instead of a generic
    "couldn't find an answer"."""


def is_quota_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in (
        "quota", "rate limit", "rate_limit", "resource_exhausted",
        "credit balance", "insufficient_quota",
    ))


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict


class ToolTurn(TypedDict):
    """Result of one generate_with_tools() round: either a final answer
    (content set, tool_calls empty) or a request to run tools (content may
    be None, tool_calls non-empty)."""
    content: Optional[str]
    tool_calls: list[ToolCall]


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

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Optional[ToolTurn]:
        """Tool-calling turn for agentic features (e.g. Scout).

        `messages` follows OpenAI chat-message shape: role in
        user/assistant/tool/system, plus `tool_calls`/`tool_call_id`/`name`
        where applicable. `tools` is a list of
        {"name", "description", "parameters": <JSON schema>} specs.

        Not implemented by default — only providers backing agentic
        features need it; the existing single-shot generate() callers are
        unaffected.
        """
        raise NotImplementedError(f"{self.provider_name} does not support tool calling")

    def stream_text(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1536,
    ) -> Iterator[str]:
        """Stream a plain-text completion (no tool calling) token-by-token,
        for the final answer once an agent loop has no more tool calls to
        make. `messages` uses the same shape as generate_with_tools.

        Not implemented by default; falls back to a single non-streamed
        chunk via generate_with_tools in the orchestrator if a provider
        doesn't override this.
        """
        raise NotImplementedError(f"{self.provider_name} does not support streaming")

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider label (e.g. 'gemini', 'openai')."""
        ...
