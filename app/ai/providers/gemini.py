"""Gemini provider using the google-genai SDK."""
import logging
import os
from typing import Any, Iterator, Optional

from .base import LLMProvider, QuotaExceededError, ToolTurn, is_quota_error

logger = logging.getLogger("autoeda.ai.providers.gemini")

_MODEL = "gemini-2.5-flash"


def _json_schema_to_gemini(schema: dict) -> dict:
    """Translate a plain JSON-schema dict (lowercase types) into the shape
    google-genai's Schema expects (uppercase Type enum values)."""
    type_map = {
        "object": "OBJECT", "string": "STRING", "number": "NUMBER",
        "integer": "INTEGER", "boolean": "BOOLEAN", "array": "ARRAY",
    }
    out: dict[str, Any] = {}
    t = schema.get("type")
    if t:
        out["type"] = type_map.get(t, t.upper())
    if "description" in schema:
        out["description"] = schema["description"]
    if "enum" in schema:
        out["enum"] = schema["enum"]
    if "properties" in schema:
        out["properties"] = {k: _json_schema_to_gemini(v) for k, v in schema["properties"].items()}
    if "items" in schema:
        out["items"] = _json_schema_to_gemini(schema["items"])
    if "required" in schema:
        out["required"] = schema["required"]
    return out


class GeminiProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "gemini"

    def _client(self):
        from ...config import settings
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return None
        from google import genai  # type: ignore
        return genai.Client(api_key=api_key)

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
            from google.genai import types  # type: ignore

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
            if is_quota_error(e):
                raise QuotaExceededError(str(e)) from e
            logger.warning("Gemini generate failed: %s", e)
            return None

    def _to_contents(self, messages: list[dict[str, Any]]) -> tuple[list[Any], Optional[str]]:
        from google.genai import types  # type: ignore

        system_parts = [m["content"] for m in messages if m["role"] == "system" and m.get("content")]
        contents: list[Any] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                continue
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m.get("content") or "")]))
            elif role == "assistant":
                if m.get("tool_calls"):
                    parts = [
                        types.Part(function_call=types.FunctionCall(name=tc["name"], args=tc["arguments"]))
                        for tc in m["tool_calls"]
                    ]
                    contents.append(types.Content(role="model", parts=parts))
                else:
                    contents.append(types.Content(role="model", parts=[types.Part(text=m.get("content") or "")]))
            elif role == "tool":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=m["name"],
                        response={"result": m.get("content")},
                    )],
                ))
        return contents, ("\n\n".join(system_parts) or None)

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
            from google.genai import types  # type: ignore

            contents, system_instruction = self._to_contents(messages)

            gemini_tools = [types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=_json_schema_to_gemini(t["parameters"]),
                )
                for t in tools
            ])] if tools else None

            response = client.models.generate_content(
                model=_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    tools=gemini_tools,
                    system_instruction=system_instruction,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            candidate = response.candidates[0] if response.candidates else None
            if candidate is None or candidate.content is None:
                return {"content": None, "tool_calls": []}

            tool_calls = []
            text_parts = []
            for i, part in enumerate(candidate.content.parts or []):
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_calls.append({"id": f"call_{i}", "name": fc.name, "arguments": dict(fc.args or {})})
                elif getattr(part, "text", None):
                    text_parts.append(part.text)

            return {"content": ("\n".join(text_parts).strip() or None) if not tool_calls else None, "tool_calls": tool_calls}
        except Exception as e:
            if is_quota_error(e):
                raise QuotaExceededError(str(e)) from e
            logger.warning("Gemini generate_with_tools failed: %s", e)
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
            from google.genai import types  # type: ignore

            contents, system_instruction = self._to_contents(messages)
            stream = client.models.generate_content_stream(
                model=_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system_instruction,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            if is_quota_error(e):
                raise QuotaExceededError(str(e)) from e
            logger.warning("Gemini stream_text failed: %s", e)
            return
