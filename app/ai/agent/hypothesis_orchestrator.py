"""Hypothesis generation/validation as a specialized Scout agent mode.

Reuses Scout's tool-calling loop (_run_tool_loop), provider selection
(get_provider), and tool layer (TOOL_SPECS/execute_tool) directly — this is
NOT a parallel AI system. The differences from Scout's own chat/agent modes:

  - A read-only tool allowlist (no add_quality_rule/save_chart/
    create_segment/remember_relationship) — a verification agent
    investigates, it doesn't mutate the product's data. Enforced by
    filtering TOOL_SPECS before the model ever sees the options, via
    _run_tool_loop's tool_specs override (added for this purpose; Scout's
    own call sites don't pass it, so its behavior is unchanged).
  - The final answer must be structured JSON (a verdict object for
    validation, a list of pre-verified hypotheses for generation), not free
    prose — parsed server-side, the same fence-stripping pattern the old
    hypothesis_generator.py/assumption_validator.py already used.
  - Explicit terminal-state handling: hitting the iteration cap forces
    status="inconclusive" deterministically rather than asking for a
    best-effort prose answer (Scout's run_agent_turn behavior, which is
    fine for chat but not for a status enum that drives UI color-coding).

This file is purely additive — scout.py's own orchestrator, prompt, and
behavior are untouched.
"""
import json
import logging
from typing import Any, Iterator

from sqlalchemy.orm import Session

from ..llm import get_provider
from ...models.user import User
from .orchestrator import _run_tool_loop, _NO_PROVIDER_MSG
from .tools import TOOL_SPECS

logger = logging.getLogger("autoeda.ai.agent.hypothesis")

_MUTATING_TOOLS = {"add_quality_rule", "save_chart", "create_segment", "remember_relationship"}
_READONLY_TOOL_SPECS = [t for t in TOOL_SPECS if t["name"] not in _MUTATING_TOOLS]

_MAX_ITERATIONS_VALIDATE = 6
_MAX_ITERATIONS_GENERATE = 14
_GENERATE_MAX_TOKENS = 2560
_VALIDATE_MAX_TOKENS = 1536
_TEMPERATURE = 0.15  # lower than Scout chat's 0.2 — verification should be conservative

_INCONCLUSIVE_CAP_HIT = "Could not reach a verdict within the investigation budget — try narrowing the hypothesis to one specific, testable claim."
_PARSE_FAILURE_MSG = "The investigation ran but its findings couldn't be parsed into a structured verdict. Try rephrasing the hypothesis."

_SCOPE_RULES = (
    "You do not know any dataset_id or slug until you call list_datasets — "
    "call it first, before any other tool. Never guess or infer a dataset_id. "
    "Tools operate on real data only — never fabricate a statistic, p-value, "
    "or column value that didn't come from an actual tool result."
)

_VAGUE_CLAIM_RULE = (
    "If the hypothesis is too vague or subjective to map to a concrete test "
    "or query (e.g. \"this data looks weird\", \"something seems off\"), do "
    "not force a tool call or invent a test just to have something to "
    "report — immediately return status=\"inconclusive\" with a verdict "
    "explaining what would make the claim testable (e.g. \"specify which "
    "column or what kind of anomaly you suspect\")."
)


def _validate_system_prompt(workspace_id: int, dataset_id: int | None) -> str:
    scope = (
        f"Focus primarily on dataset id {dataset_id} in this workspace, but you "
        "may reference other datasets if directly relevant to the claim."
        if dataset_id is not None
        else "This claim may span any dataset in the workspace — investigate cross-dataset relationships when relevant (run_workspace_sql/run_workspace_python)."
    )
    return (
        f"You are Scout's hypothesis-validation mode, scoped to workspace #{workspace_id} "
        f"(internal id, not a dataset_id). {scope}\n\n"
        "The user has stated a specific claim about this data. Your only job is to "
        "verify it using real tools until you have actual computed evidence — never "
        "narrate a guess from memory or from a cached summary. Prefer run_statistical_test "
        "for comparative/distributional claims (it returns a real p-value), get_correlations "
        "for relationship claims (it already includes p-values), run_sql/run_workspace_sql "
        "for precise aggregations, and run_python/run_workspace_python for anything those "
        "don't cover. " + _SCOPE_RULES + "\n\n" + _VAGUE_CLAIM_RULE + "\n\n"
        "When you're done investigating, respond with ONLY a JSON object (no markdown "
        "fences, no text outside the JSON):\n"
        '{"status": "supported"|"refuted"|"inconclusive", "verdict": "<2-3 sentence plain-English conclusion>", '
        '"evidence_summary": "<short, specific, e.g. \'p=0.003, r=-0.82, n=46011\'>", '
        '"confidence": "high"|"medium"|"low", "columns": ["<relevant column names>"]}'
    )


def _generate_system_prompt(workspace_id: int, dataset_id: int | None, count: int) -> str:
    scope = (
        f"Focus on dataset id {dataset_id} in this workspace, but reference other "
        "datasets too if you find a cross-dataset pattern worth surfacing."
        if dataset_id is not None
        else "Investigate across the whole workspace, including cross-dataset relationships (run_workspace_sql/run_workspace_python) where they reveal something interesting."
    )
    return (
        f"You are Scout's hypothesis-generation mode, scoped to workspace #{workspace_id} "
        f"(internal id, not a dataset_id). {scope}\n\n"
        f"Investigate the data and propose up to {count} hypotheses — but only after "
        "using tools to verify each one actually holds. Every hypothesis you report must "
        "already carry the real evidence that supports it (a correlation with its p-value, "
        "an outlier percentage, a statistical test result, a skewness value, a missing-data "
        "rate, etc.) — do not propose a hypothesis you haven't already checked with a tool. "
        "Investigate broadly first (profile, correlations, outliers, feature importance, "
        "quality score, distributions) before committing to your final list, so the "
        f"{count} you report are the most interesting, evidence-backed findings, not just "
        "the first things you noticed. " + _SCOPE_RULES + "\n\n"
        "When you're done investigating, respond with ONLY a JSON array (no markdown "
        "fences, no text outside the JSON), each item shaped:\n"
        '{"title": "<short headline>", "statement": "<the hypothesis as a claim>", '
        '"category": "correlation"|"distribution"|"missing"|"outlier"|"quality"|"feature"|"pattern", '
        '"status": "supported"|"refuted", "verdict": "<2-3 sentence explanation>", '
        '"evidence_summary": "<short, specific>", "confidence": "high"|"medium"|"low", '
        '"severity": "info"|"warning"|"danger", "columns": ["<relevant column names>"]}'
    )


def _build_messages(system_prompt: str, user_message: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def _parse_json_block(raw: str) -> Any | None:
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("hypothesis JSON parse error: %s | raw=%s", e, raw[:400])
        return None


def _force_final_json(provider, messages: list[dict[str, Any]], max_tokens: int) -> Any | None:
    """One extra call asking explicitly for JSON-only, used when the loop's
    natural final turn didn't parse — gives the model one chance to comply
    before we give up and report a parse failure."""
    messages = messages + [{"role": "user", "content": "Respond with ONLY the JSON now, no other text."}]
    final = provider.generate_with_tools(messages, [], temperature=_TEMPERATURE, max_tokens=max_tokens)
    if final is None or not final.get("content"):
        return None
    return _parse_json_block(final["content"])


def run_hypothesis_validation(
    *, statement: str, workspace_id: int, dataset_id: int | None, db: Session, user: User,
) -> dict[str, Any]:
    """Returns {"status", "verdict", "evidence_summary", "confidence", "columns", "tool_trace"}."""
    provider = get_provider()
    if provider is None:
        return {"status": "error", "verdict": _NO_PROVIDER_MSG, "evidence_summary": None, "confidence": None, "columns": [], "tool_trace": []}

    messages = _build_messages(_validate_system_prompt(workspace_id, dataset_id), statement)
    tool_trace: list[dict[str, Any]] = []
    final_content: str | None = None

    for event in _run_tool_loop(
        provider, messages, workspace_id=workspace_id, db=db, user=user,
        max_iterations=_MAX_ITERATIONS_VALIDATE, tool_trace=tool_trace,
        tool_specs=_READONLY_TOOL_SPECS, temperature=_TEMPERATURE, max_tokens=_VALIDATE_MAX_TOKENS,
    ):
        if event["type"] == "error":
            return {"status": "error", "verdict": event["message"], "evidence_summary": None, "confidence": None, "columns": [], "tool_trace": tool_trace}
        if event["type"] == "ready":
            final_content = event["content"]

    parsed = _parse_json_block(final_content) if final_content else None
    if parsed is None and final_content is not None:
        parsed = _force_final_json(provider, messages, _VALIDATE_MAX_TOKENS)

    if parsed is None:
        verdict = _INCONCLUSIVE_CAP_HIT if final_content is None else _PARSE_FAILURE_MSG
        return {"status": "inconclusive", "verdict": verdict, "evidence_summary": None, "confidence": "low", "columns": [], "tool_trace": tool_trace}

    return {
        "status": parsed.get("status", "inconclusive"),
        "verdict": parsed.get("verdict"),
        "evidence_summary": parsed.get("evidence_summary"),
        "confidence": parsed.get("confidence"),
        "columns": parsed.get("columns", []),
        "tool_trace": tool_trace,
    }


def run_hypothesis_validation_stream(
    *, statement: str, workspace_id: int, dataset_id: int | None, db: Session, user: User,
) -> Iterator[dict[str, Any]]:
    """Streaming variant. Yields tool_call/tool_result events live, then a
    single terminal {"type": "result", "hypothesis": {...}} or {"type": "error", ...}."""
    provider = get_provider()
    if provider is None:
        yield {"type": "error", "message": _NO_PROVIDER_MSG}
        return

    messages = _build_messages(_validate_system_prompt(workspace_id, dataset_id), statement)
    tool_trace: list[dict[str, Any]] = []
    final_content: str | None = None

    for event in _run_tool_loop(
        provider, messages, workspace_id=workspace_id, db=db, user=user,
        max_iterations=_MAX_ITERATIONS_VALIDATE, tool_trace=tool_trace,
        tool_specs=_READONLY_TOOL_SPECS, temperature=_TEMPERATURE, max_tokens=_VALIDATE_MAX_TOKENS,
    ):
        if event["type"] == "error":
            yield event
            return
        if event["type"] == "ready":
            final_content = event["content"]
            break
        yield event

    parsed = _parse_json_block(final_content) if final_content else None
    if parsed is None and final_content is not None:
        parsed = _force_final_json(provider, messages, _VALIDATE_MAX_TOKENS)

    if parsed is None:
        verdict = _INCONCLUSIVE_CAP_HIT if final_content is None else _PARSE_FAILURE_MSG
        yield {"type": "result", "hypothesis": {"status": "inconclusive", "verdict": verdict, "evidence_summary": None, "confidence": "low", "columns": []}, "tool_trace": tool_trace}
        return

    yield {
        "type": "result",
        "hypothesis": {
            "status": parsed.get("status", "inconclusive"),
            "verdict": parsed.get("verdict"),
            "evidence_summary": parsed.get("evidence_summary"),
            "confidence": parsed.get("confidence"),
            "columns": parsed.get("columns", []),
        },
        "tool_trace": tool_trace,
    }


def run_hypothesis_generation(
    *, workspace_id: int, dataset_id: int | None, count: int, db: Session, user: User,
) -> dict[str, Any]:
    """Returns {"hypotheses": list[dict], "tool_trace": [...]}."""
    provider = get_provider()
    if provider is None:
        return {"hypotheses": [], "tool_trace": [], "error": _NO_PROVIDER_MSG}

    scope_msg = f"Generate up to {count} hypotheses" + (f" about dataset id {dataset_id}." if dataset_id is not None else " about this workspace.")
    messages = _build_messages(_generate_system_prompt(workspace_id, dataset_id, count), scope_msg)
    tool_trace: list[dict[str, Any]] = []
    final_content: str | None = None

    for event in _run_tool_loop(
        provider, messages, workspace_id=workspace_id, db=db, user=user,
        max_iterations=_MAX_ITERATIONS_GENERATE, tool_trace=tool_trace,
        tool_specs=_READONLY_TOOL_SPECS, temperature=_TEMPERATURE, max_tokens=_GENERATE_MAX_TOKENS,
    ):
        if event["type"] == "error":
            return {"hypotheses": [], "tool_trace": tool_trace, "error": event["message"]}
        if event["type"] == "ready":
            final_content = event["content"]

    parsed = _parse_json_block(final_content) if final_content else None
    if parsed is None and final_content is not None:
        parsed = _force_final_json(provider, messages, _GENERATE_MAX_TOKENS)
    if not isinstance(parsed, list):
        return {"hypotheses": [], "tool_trace": tool_trace, "error": _PARSE_FAILURE_MSG if final_content else _INCONCLUSIVE_CAP_HIT}

    return {"hypotheses": parsed, "tool_trace": tool_trace}


def run_hypothesis_generation_stream(
    *, workspace_id: int, dataset_id: int | None, count: int, db: Session, user: User,
) -> Iterator[dict[str, Any]]:
    provider = get_provider()
    if provider is None:
        yield {"type": "error", "message": _NO_PROVIDER_MSG}
        return

    scope_msg = f"Generate up to {count} hypotheses" + (f" about dataset id {dataset_id}." if dataset_id is not None else " about this workspace.")
    messages = _build_messages(_generate_system_prompt(workspace_id, dataset_id, count), scope_msg)
    tool_trace: list[dict[str, Any]] = []
    final_content: str | None = None

    for event in _run_tool_loop(
        provider, messages, workspace_id=workspace_id, db=db, user=user,
        max_iterations=_MAX_ITERATIONS_GENERATE, tool_trace=tool_trace,
        tool_specs=_READONLY_TOOL_SPECS, temperature=_TEMPERATURE, max_tokens=_GENERATE_MAX_TOKENS,
    ):
        if event["type"] == "error":
            yield event
            return
        if event["type"] == "ready":
            final_content = event["content"]
            break
        yield event

    parsed = _parse_json_block(final_content) if final_content else None
    if parsed is None and final_content is not None:
        parsed = _force_final_json(provider, messages, _GENERATE_MAX_TOKENS)
    if not isinstance(parsed, list):
        yield {"type": "error", "message": _PARSE_FAILURE_MSG if final_content else _INCONCLUSIVE_CAP_HIT}
        return

    yield {"type": "result", "hypotheses": parsed, "tool_trace": tool_trace}
