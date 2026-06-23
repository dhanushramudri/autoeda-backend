"""Scout's agent loop: feed the LLM a question + tool list, execute whatever
tools it asks for, feed results back, repeat until it gives a final answer
or the iteration cap is hit.

Reuses the existing single-shot LLM provider selection in app/ai/llm.py —
this is purely additive, the narrative/chat/hypothesis features are
untouched and keep calling provider.generate() directly.

Two entry points share the same tool-calling loop (_run_tool_loop):
  - run_agent_turn: synchronous, returns the complete answer at once.
  - run_agent_turn_stream: yields progress events as they happen, and
    streams the final answer token-by-token via the provider's stream_text.
"""
import json
import logging
from typing import Any, Iterator

from sqlalchemy.orm import Session

from ..llm import get_provider
from ..providers.base import QuotaExceededError
from ...models.user import User
from .tools import TOOL_SPECS, execute_tool

logger = logging.getLogger("autoeda.ai.agent")

_MAX_ITERATIONS_AGENT = 10
# 2, not 1: chat mode should allow a single tool lookup *and* a chance to
# answer using it — capping at 1 means even one tool call burns the only
# iteration, forcing an extra fallback call just to produce an answer.
_MAX_ITERATIONS_CHAT = 2

# Tool results can be large (full profile/correlation matrices) — fine for
# the frontend to render, but wasteful and costly to replay into the LLM's
# own context on every loop iteration. Trim before handing back to the LLM.
_MAX_CHARS_FOR_LLM = 6000

_NO_PROVIDER_MSG = "Scout needs an AI provider configured (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY) to answer questions."
_UNREACHABLE_MSG = "Scout couldn't reach the AI provider. Please try again."
_NO_ANSWER_MSG = "I wasn't able to find an answer."
_RAN_OUT_MSG = "I gathered some data but ran out of steps to fully answer — try narrowing the question."
_QUOTA_MSG = "AI quota reached — the AI provider's free-tier credits are exhausted for now. Please try again later."


def _system_prompt(workspace_id: int) -> str:
    return (
        "You are Scout, a data analyst agent embedded in AutoEDA, answering "
        f"questions scoped to one workspace (internal id {workspace_id} — "
        "this is NOT a dataset_id and must never be passed as one). You have "
        "tools covering the full EDA surface: profiling, missing values, "
        "correlations, outliers (univariate and multivariate), feature "
        "importance, distributions, text analysis, time series, and quality "
        "scoring — plus read-only SQL both within a single dataset (run_sql) "
        "and across every dataset in the workspace at once (run_workspace_sql, "
        "using slugs from list_datasets). Use search_columns when the user "
        "mentions a column but you don't know which dataset it lives in. "
        "Call get_known_relationships early when a question spans multiple "
        "datasets — it recalls join keys discovered in past conversations, "
        "so you may not need to rediscover them via trial-and-error SQL. "
        "After you successfully join two datasets with run_workspace_sql, "
        "call remember_relationship so future conversations benefit too.\n\n"
        "You can also go beyond analysis: get_shap_explanations for per-feature "
        "impact direction (not just ranking); run_statistical_test for a rigorous "
        "t-test/ANOVA/chi-square/normality check instead of eyeballing numbers; "
        "evaluate_quality_rules and add_quality_rule to check and extend this "
        "dataset's data-quality rules; save_chart and create_segment to persist "
        "a finding the user can revisit later; preview_transform to show what a "
        "cleaning step would do without saving it. For analysis none of these "
        "cover, run_python executes real pandas/numpy/scipy code against the "
        "dataset (assign your answer to `result`) — prefer a dedicated tool "
        "when one already fits, but don't avoid run_python out of caution; "
        "it's there precisely for the gaps the other tools don't cover.\n\n"
        "You do not know any dataset_id or slug until you call list_datasets "
        "— call it first, every conversation, before any other tool, even if "
        "a dataset was discussed earlier in the chat history. Never guess or "
        "infer a dataset_id from context. Always ground answers in real tool "
        "output — never guess numbers. Plan multi-step investigations when "
        "the question calls for it (e.g. find outliers, then check whether "
        "they cluster in a particular time range or category) rather than "
        "stopping after one lookup. Cite actual column names and values you "
        "retrieved. If a tool errors, explain what went wrong rather than "
        "making up an answer. Keep the final answer concise — the underlying "
        "data is rendered separately, so don't restate large tables in prose."
    )


def _trim_for_llm(result: dict) -> str:
    text = json.dumps(result, default=str)
    if len(text) > _MAX_CHARS_FOR_LLM:
        text = text[:_MAX_CHARS_FOR_LLM] + "... [truncated]"
    return text


def _resolve_image(image: dict[str, Any] | None) -> dict[str, str] | None:
    """Fetches an attached image's bytes from S3 and base64-encodes them for the
    provider. Called once, for the current turn's message only — history replay
    never re-resolves past images, so one attachment doesn't silently inflate the
    token cost of every later turn in the same conversation."""
    if not image or not image.get("key"):
        return None
    import base64
    from ...s3_attachments import get_object_bytes

    data = get_object_bytes(image["key"])
    if not data:
        return None
    return {"media_type": image.get("media_type") or "image/png", "data": base64.b64encode(data).decode()}


def _build_messages(
    workspace_id: int, history: list[dict[str, str]], message: str, image: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": _system_prompt(workspace_id)}]
    for h in history[-12:]:
        messages.append({"role": h["role"], "content": h["content"]})
    user_msg: dict[str, Any] = {"role": "user", "content": message}
    resolved_image = _resolve_image(image)
    if resolved_image:
        user_msg["image"] = resolved_image
    messages.append(user_msg)
    return messages


def _run_tool_loop(
    provider,
    messages: list[dict[str, Any]],
    *,
    workspace_id: int,
    db: Session,
    user: User,
    max_iterations: int,
    tool_trace: list[dict[str, Any]],
    tool_specs: list[dict[str, Any]] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1536,
) -> Iterator[dict[str, Any]]:
    """Mutates `messages`/`tool_trace` in place; yields progress events.
    `tool_specs`/`temperature`/`max_tokens` default to Scout's own settings;
    callers needing different limits (e.g. hypothesis_orchestrator's
    read-only allowlist and larger structured-JSON final answers) can
    override them without affecting Scout's own behavior.
    Terminal event is always either {"type": "error", ...} or
    {"type": "ready", "content": str | None} — "ready" means no more tool
    calls are pending (content is the already-generated answer if the model
    stopped naturally, or None if the iteration cap was hit instead)."""
    specs = tool_specs if tool_specs is not None else TOOL_SPECS
    for _ in range(max_iterations):
        try:
            turn = provider.generate_with_tools(messages, specs, temperature=temperature, max_tokens=max_tokens)
        except QuotaExceededError:
            yield {"type": "error", "code": "quota_exceeded", "message": _QUOTA_MSG}
            return
        if turn is None:
            yield {"type": "error", "message": _UNREACHABLE_MSG}
            return

        if not turn["tool_calls"]:
            yield {"type": "ready", "content": turn["content"]}
            return

        messages.append({"role": "assistant", "content": turn["content"], "tool_calls": turn["tool_calls"]})

        for tc in turn["tool_calls"]:
            yield {"type": "tool_call", "tool": tc["name"], "arguments": tc["arguments"]}
            result = execute_tool(tc["name"], tc["arguments"], workspace_id=workspace_id, db=db, user=user)
            tool_trace.append({"tool": tc["name"], "arguments": tc["arguments"], "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["name"],
                "content": _trim_for_llm(result),
            })
            yield {"type": "tool_result", "tool": tc["name"], "result": result}

    yield {"type": "ready", "content": None}


def run_agent_turn(
    *,
    message: str,
    history: list[dict[str, str]],
    workspace_id: int,
    db: Session,
    user: User,
    mode: str = "agent",
    image: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one user turn synchronously. Returns {"answer": str, "tool_trace": [...]}.
    `image`, if given, is {"key": str, "media_type": str} — an S3 reference, not raw
    bytes; resolved (fetched + base64-encoded) once inside _build_messages."""
    provider = get_provider()
    if provider is None:
        return {"answer": _NO_PROVIDER_MSG, "tool_trace": []}

    max_iterations = _MAX_ITERATIONS_AGENT if mode == "agent" else _MAX_ITERATIONS_CHAT
    messages = _build_messages(workspace_id, history, message, image=image)
    tool_trace: list[dict[str, Any]] = []
    final_content: str | None = None

    for event in _run_tool_loop(provider, messages, workspace_id=workspace_id, db=db, user=user, max_iterations=max_iterations, tool_trace=tool_trace):
        if event["type"] == "error":
            return {"answer": event["message"], "tool_trace": tool_trace}
        if event["type"] == "ready":
            final_content = event["content"]

    if final_content:
        return {"answer": final_content, "tool_trace": tool_trace}

    # Iteration cap hit without a natural stop — ask once more for a final answer.
    try:
        final = provider.generate_with_tools(messages, [], temperature=0.2, max_tokens=1536)
    except QuotaExceededError:
        return {"answer": _QUOTA_MSG, "tool_trace": tool_trace}
    answer = (final["content"] if final else None) or _RAN_OUT_MSG
    return {"answer": answer, "tool_trace": tool_trace}


def run_agent_turn_stream(
    *,
    message: str,
    history: list[dict[str, str]],
    workspace_id: int,
    db: Session,
    user: User,
    mode: str = "agent",
    image: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Streaming variant. Yields progress events as they happen:
      {"type": "tool_call", "tool": str, "arguments": dict}
      {"type": "tool_result", "tool": str, "result": dict}
      {"type": "answer_chunk", "text": str}
      {"type": "done", "answer": str, "tool_trace": [...]}
      {"type": "error", "message": str}

    The tool-calling phase itself isn't streamed (each step needs a complete,
    parseable response to decide what to call next) — only the final answer
    streams token-by-token via the provider's stream_text. Note this means
    one extra LLM call versus run_agent_turn: the call that detects "no more
    tools needed" produces a throwaway answer, which stream_text then
    regenerates for real, streamed output. That's the accepted cost of true
    token streaming without fragile incremental tool-call-argument parsing.

    `image`, if given, is {"key": str, "media_type": str} — see run_agent_turn.
    """
    provider = get_provider()
    if provider is None:
        yield {"type": "error", "message": _NO_PROVIDER_MSG}
        return

    max_iterations = _MAX_ITERATIONS_AGENT if mode == "agent" else _MAX_ITERATIONS_CHAT
    messages = _build_messages(workspace_id, history, message, image=image)
    tool_trace: list[dict[str, Any]] = []
    ready_content: str | None = None

    for event in _run_tool_loop(provider, messages, workspace_id=workspace_id, db=db, user=user, max_iterations=max_iterations, tool_trace=tool_trace):
        if event["type"] == "error":
            yield event
            return
        if event["type"] == "ready":
            ready_content = event["content"]
            break
        yield event  # tool_call / tool_result

    full_answer = ""
    try:
        for chunk in provider.stream_text(messages, temperature=0.2, max_tokens=1536):
            full_answer += chunk
            yield {"type": "answer_chunk", "text": chunk}
    except QuotaExceededError:
        yield {"type": "error", "code": "quota_exceeded", "message": _QUOTA_MSG}
        return
    except NotImplementedError:
        full_answer = ""

    if not full_answer:
        # Provider doesn't support streaming, or streaming produced nothing —
        # fall back to whatever non-streamed content is available, making one
        # more forced call only if the iteration cap was hit with no answer yet.
        if ready_content is None:
            try:
                final = provider.generate_with_tools(messages, [], temperature=0.2, max_tokens=1536)
            except QuotaExceededError:
                yield {"type": "error", "code": "quota_exceeded", "message": _QUOTA_MSG}
                return
            ready_content = (final["content"] if final else None) or _RAN_OUT_MSG
        full_answer = ready_content or _NO_ANSWER_MSG
        yield {"type": "answer_chunk", "text": full_answer}

    yield {"type": "done", "answer": full_answer, "tool_trace": tool_trace}
