"""LLM-powered NL query parser with regex fallback."""
import json
import logging

from .llm import generate
from ..nl_parser import parse_nl_query as _regex_parse

logger = logging.getLogger("autoeda.ai.nl_query")

_PROMPT = """You are a query router for AutoEDA, a data analysis platform.
Parse the user's plain-English query into a JSON action object.

Return ONLY valid JSON with exactly these fields:
  action: "navigate" | "transform" | "filter" | "unknown"
  params: dict (see rules below)
  message: short confirmation under 15 words

Navigate pages: overview, profile, distributions, correlations, missing, outliers,
  feature-importance, timeseries, text, graph, pivot, charts
  params: {page, column?, col1?, col2?, target?}

Transform ops: drop, fill_missing, drop_high_missing
  params: {op, column?, method?(mean/median/mode/custom), threshold?}

Filter:
  params: {column, operator(equals/gt/lt/gte/lte/not_equals), value}

Available columns: {columns}
Query: {query}"""


def parse_nl_query_ai(query: str, columns: list[str]) -> dict:
    """Try LLM first; fall back to regex on failure or missing provider."""
    cols_repr = ", ".join(columns[:60]) if columns else "(none)"
    prompt = _PROMPT.format(columns=cols_repr, query=query)

    raw = generate(prompt, temperature=0.1, max_tokens=256)
    if raw:
        try:
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            result = json.loads(cleaned)
            if isinstance(result, dict) and "action" in result and "params" in result:
                result.setdefault("message", "")
                return result
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("LLM NL response not valid JSON: %s | raw=%s", e, raw[:200])

    return _regex_parse(query, columns)
