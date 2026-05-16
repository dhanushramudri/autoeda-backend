"""Gemini-powered NL query parser with regex fallback."""
import json
import logging

from .gemini_client import generate
from ..nl_parser import parse_nl_query as _regex_parse

logger = logging.getLogger("autoeda.ai.nl_query")

_SYSTEM_PROMPT = """You are a natural language query router for a data analysis platform called AutoEDA.
Given a user's plain-English query and a list of dataset column names, return a JSON object with exactly these fields:
  - action: one of "navigate", "transform", "filter", "unknown"
  - params: dict of parameters relevant to the action
  - message: a short human-readable confirmation (under 15 words)

Available pages for action="navigate":
  overview, profile, distributions, correlations, missing, outliers, feature-importance,
  timeseries, text, graph, pivot, charts

For action="navigate" params should include "page" and optionally "column", "col1", "col2", "target".

For action="transform" params should include "op" (one of: drop, fill_missing, drop_high_missing)
  and optionally "column", "method" (mean/median/mode/custom), "threshold".

For action="filter" params should include "column", "operator" (equals/gt/lt/gte/lte/not_equals), "value".

Respond with ONLY a valid JSON object, no markdown, no explanation.

Column names available: {columns}

User query: {query}"""


def parse_nl_query_ai(query: str, columns: list[str]) -> dict:
    """Try Gemini first; fall back to regex parser on failure."""
    cols_repr = ", ".join(columns[:60]) if columns else "(none)"
    prompt = _SYSTEM_PROMPT.format(columns=cols_repr, query=query)

    raw = generate(prompt, temperature=0.1)
    if raw:
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            result = json.loads(cleaned.strip())
            if isinstance(result, dict) and "action" in result and "params" in result:
                result.setdefault("message", "")
                logger.info("Gemini parsed: action=%s params=%s", result["action"], result["params"])
                return result
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Gemini response not valid JSON: %s | raw=%s", e, raw[:200])

    # Fallback to regex
    logger.info("Falling back to regex parser for query: %s", query[:80])
    return _regex_parse(query, columns)
