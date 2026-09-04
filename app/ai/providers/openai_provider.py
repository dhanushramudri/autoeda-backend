"""OpenAI provider using the openai SDK — Azure OpenAI when Azure config is
present (the primary production setup), plain OpenAI otherwise."""
import json
import logging
from typing import Any, Iterator, Optional

from .base import LLMProvider, QuotaExceededError, ToolTurn, is_quota_error

logger = logging.getLogger("autoeda.ai.providers.openai")

_DEFAULT_MODEL = "gpt-4o-mini"


def _get_client_and_model():
    """Return (client, model) using Azure OpenAI if configured, else plain
    OpenAI. Returns (None, None) if neither is configured."""
    from ...config import settings

    azure_key = settings.AZURE_OPENAI_API_KEY or settings.TENALI_AI_API
    if azure_key and settings.AZURE_OPENAI_ENDPOINT and settings.AZURE_OPENAI_DEPLOYMENT:
        from openai import OpenAI  # type: ignore

        # This endpoint is Azure AI Foundry's unified "v1" API surface
        # (already includes /openai/v1) — it's OpenAI-API-compatible and
        # wants the plain client pointed at it via base_url, NOT the
        # AzureOpenAI class (which appends its own /openai/deployments/...
        # path and produces a 404 against an already-v1 endpoint).
        client = OpenAI(api_key=azure_key, base_url=settings.AZURE_OPENAI_ENDPOINT)
        return client, settings.AZURE_OPENAI_DEPLOYMENT

    if settings.OPENAI_API_KEY:
        from openai import OpenAI  # type: ignore

        return OpenAI(api_key=settings.OPENAI_API_KEY), _DEFAULT_MODEL

    return None, None


def _is_unsupported_temperature(exc: Exception) -> bool:
    """True for the specific 400 some reasoning-model deployments (e.g. the
    gpt-5.x family) raise when a non-default temperature is passed — those
    models only support the implicit default (1)."""
    msg = str(exc).lower()
    return "temperature" in msg and ("unsupported_value" in msg or "does not support" in msg)


def _create(client, **kwargs):
    """chat.completions.create(), with one retry that drops `temperature`
    if the model rejects a non-default value outright."""
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        if "temperature" in kwargs and _is_unsupported_temperature(e):
            kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
            return client.chat.completions.create(**kwargs)
        raise


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
        elif m["role"] == "user" and m.get("image"):
            image = m["image"]
            content: list[dict[str, Any]] = [{
                "type": "image_url",
                "image_url": {"url": f"data:{image['media_type']};base64,{image['data']}"},
            }]
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            oa_messages.append({"role": "user", "content": content})
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
        client, model = _get_client_and_model()
        if client is None:
            return None
        try:
            response = _create(
                client,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if is_quota_error(e):
                raise QuotaExceededError(str(e)) from e
            logger.warning("OpenAI generate failed: %s", e)
            return None

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Optional[ToolTurn]:
        client, model = _get_client_and_model()
        if client is None:
            return None
        try:
            oa_messages = _to_oa_messages(messages)

            oa_tools = [
                {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
                for t in tools
            ]

            response = _create(
                client,
                model=model,
                messages=oa_messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
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
            if is_quota_error(e):
                raise QuotaExceededError(str(e)) from e
            logger.warning("OpenAI generate_with_tools failed: %s", e)
            return None

    def stream_text(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1536,
    ) -> Iterator[str]:
        client, model = _get_client_and_model()
        if client is None:
            return
        try:
            stream = _create(
                client,
                model=model,
                messages=_to_oa_messages(messages),
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            if is_quota_error(e):
                raise QuotaExceededError(str(e)) from e
            logger.warning("OpenAI stream_text failed: %s", e)
            return
