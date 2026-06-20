"""OpenAI provider using the openai SDK."""
import json
import logging
import os
from typing import Any, Iterator, Optional

from .base import LLMProvider, ToolTurn

logger = logging.getLogger("autoeda.ai.providers.openai")

_MODEL = "gpt-4o-mini"


def _to_oa_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    oa_messages = []
    for m in messages:
        if m["role"] == "tool":
            oa_messages.append({
                "role": "tool",
                "tool_call_id": m["tool_call_id"],
                "content": m.get("content") or "",
            })
        elif m["role"] == "assistant" and m.get("tool_calls"):
            oa_messages.append({
                "role": "assistant",
                "content": m.get("content"),
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                    }
                    for tc in m["tool_calls"]
                ],
            })
        else:
            oa_messages.append({"role": m["role"], "content": m.get("content") or ""})
    return oa_messages


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

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Optional[ToolTurn]:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            oa_messages = _to_oa_messages(messages)

            oa_tools = [
                {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
                for t in tools
            ]

            response = client.chat.completions.create(
                model=_MODEL,
                messages=oa_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **({"tools": oa_tools, "tool_choice": "auto"} if oa_tools else {}),
            )
            choice = response.choices[0].message

            if choice.tool_calls:
                tool_calls = [
                    {"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments or "{}")}
                    for tc in choice.tool_calls
                ]
                return {"content": None, "tool_calls": tool_calls}

            return {"content": (choice.content or "").strip() or None, "tool_calls": []}
        except Exception as e:
            logger.warning("OpenAI generate_with_tools failed: %s", e)
            return None

    def stream_text(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1536,
    ) -> Iterator[str]:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            stream = client.chat.completions.create(
                model=_MODEL,
                messages=_to_oa_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            logger.warning("OpenAI stream_text failed: %s", e)
            return
