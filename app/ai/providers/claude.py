"""Claude provider using the Anthropic SDK — the intended production provider.

Claude's Messages API differs from OpenAI's in three structural ways this file
has to bridge to fit the shared (OpenAI-shaped) internal message format used
by the orchestrator:
  1. The system prompt is a top-level `system` param, not a message.
  2. Tool results must be grouped into a single "user" message containing
     one `tool_result` content block per tool call — Claude's API requires
     strict user/assistant alternation, so consecutive tool-result messages
     can't be sent as separate turns the way OpenAI/Gemini allow.
  3. Images ride on an internal message's optional `image` key (set only on
     the current turn — see orchestrator._resolve_image) and get expanded into
     a `[{"type": "image", ...}, {"type": "text", ...}]` content list. This is
     currently the only provider that looks for that key — OpenAI/Gemini stay
     text-only until/unless they get the same treatment.
"""
import logging
import os
from typing import Any, Iterator, Optional

from .base import LLMProvider, ToolTurn

logger = logging.getLogger("autoeda.ai.providers.claude")

_MODEL = "claude-sonnet-4-6"


def _to_claude_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Optional[str]]:
    system_parts = [m["content"] for m in messages if m["role"] == "system" and m.get("content")]

    claude_messages: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results():
        if pending_tool_results:
            claude_messages.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "tool":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m.get("content") or "",
            })
            continue

        flush_tool_results()
        if role == "user":
            image = m.get("image")
            if image:
                content: list[dict[str, Any]] = [{
                    "type": "image",
                    "source": {"type": "base64", "media_type": image["media_type"], "data": image["data"]},
                }]
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                claude_messages.append({"role": "user", "content": content})
            else:
                claude_messages.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            if m.get("tool_calls"):
                content: list[dict[str, Any]] = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
                claude_messages.append({"role": "assistant", "content": content})
            else:
                claude_messages.append({"role": "assistant", "content": m.get("content") or ""})
    flush_tool_results()
    return claude_messages, ("\n\n".join(system_parts) or None)


class ClaudeProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "claude"

    def _client(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None
        import anthropic  # type: ignore
        return anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        client = self._client()
        if client is None:
            return None
        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in response.content if b.type == "text")
            return text.strip() or None
        except Exception as e:
            logger.warning("Claude generate failed: %s", e)
            return None

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Optional[ToolTurn]:
        client = self._client()
        if client is None:
            return None
        try:
            claude_messages, system = _to_claude_messages(messages)

            claude_tools = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ]

            response = client.messages.create(
                model=_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=claude_messages,
                **({"tools": claude_tools} if claude_tools else {}),
            )

            tool_calls = []
            text_parts = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({"id": block.id, "name": block.name, "arguments": dict(block.input or {})})
                elif block.type == "text":
                    text_parts.append(block.text)

            return {"content": ("\n".join(text_parts).strip() or None) if not tool_calls else None, "tool_calls": tool_calls}
        except Exception as e:
            logger.warning("Claude generate_with_tools failed: %s", e)
            return None

    def stream_text(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1536,
    ) -> Iterator[str]:
        client = self._client()
        if client is None:
            return
        try:
            claude_messages, system = _to_claude_messages(messages)
            with client.messages.stream(
                model=_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=claude_messages,
            ) as stream:
                yield from stream.text_stream
        except Exception as e:
            logger.warning("Claude stream_text failed: %s", e)
            return
