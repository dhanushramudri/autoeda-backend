"""Convert natural language instructions into structured transform operations."""
import json
import logging

from .llm import generate

logger = logging.getLogger("autoeda.ai.nl_transform")

_PROMPT = """You are a data wrangling assistant. Convert the user's instruction into exactly one transform operation.

Available columns: {columns}
Column semantic types: {column_types}

User instruction: "{prompt}"

Return a JSON object with exactly two fields:
  "op": the transform operation object (see valid types below)
  "explanation": one clear sentence describing what this does and the expected result

Valid op types:
  {{"type": "drop_columns", "columns": [col, ...]}}
  {{"type": "fill_missing", "column": col, "strategy": "mean"|"median"|"mode"|"constant"|"ffill"|"bfill", "value": str}}
  {{"type": "rename_column", "old_name": col, "new_name": str}}
  {{"type": "cast_type", "column": col, "to_type": "int"|"float"|"str"|"datetime"|"bool"}}
  {{"type": "create_column", "name": str, "expression": "pandas expression using column names"}}
  {{"type": "filter_rows", "column": col, "operator": "eq"|"neq"|"gt"|"gte"|"lt"|"lte"|"contains"|"not_contains", "value": str}}
  {{"type": "drop_duplicates"}}
  {{"type": "drop_rows_where_null", "columns": [col, ...]}}
  {{"type": "sort_rows", "by": [col, ...], "ascending": [true|false, ...]}}
  {{"type": "scale", "column": col, "method": "standard"|"minmax"|"robust"}}
  {{"type": "encode", "column": col, "method": "label"|"onehot"}}
  {{"type": "log_transform", "column": col, "variant": "log"|"log1p"}}
  {{"type": "sqrt_transform", "column": col}}
  {{"type": "bin", "column": col, "bins": int, "strategy": "cut"|"qcut"}}
  {{"type": "extract_datetime", "column": col, "parts": ["year","month","day","hour","minute","weekday","quarter"]}}
  {{"type": "text_clean", "column": col, "strip": bool, "lowercase": bool, "remove_special": bool}}
  {{"type": "cap_outliers", "column": col, "method": "iqr"|"percentile"}}
  {{"type": "drop_outliers", "column": col, "method": "iqr"|"zscore"}}
  {{"type": "select_columns", "columns": [col, ...]}}
  {{"type": "sample_rows", "n": int}} or {{"type": "sample_rows", "frac": float}}
  {{"type": "clip", "column": col, "lower": number, "upper": number}}

Rules:
- Use only column names that exist in the available columns list
- For create_column expressions, use bare column names (e.g. revenue / users, not df["revenue"])
- For filter_rows values, use plain strings without quotes

Return ONLY a valid JSON object. No markdown fences. No extra explanation outside the JSON."""


def generate_transform_step(
    prompt: str,
    columns: list[str],
    column_types: dict[str, str],
) -> dict:
    """Return {op, explanation} dict or raise ValueError."""
    filled = _PROMPT.format(
        prompt=prompt,
        columns=", ".join(columns[:60]),
        column_types=", ".join(f"{k}:{v}" for k, v in list(column_types.items())[:60]),
    )
    raw = generate(filled, temperature=0.1, max_tokens=512)
    if not raw:
        raise ValueError("AI provider not configured")

    cleaned = raw.strip()
    # Strip markdown fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip().rstrip("```").strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("nl_transform parse error: %s | raw=%s", e, raw[:300])
        raise ValueError(f"Could not parse AI response: {e}")

    if not isinstance(result, dict) or "op" not in result:
        raise ValueError("AI returned unexpected structure")

    op = result["op"]
    if not isinstance(op, dict) or "type" not in op:
        raise ValueError("Op missing 'type' field")

    return {
        "op": op,
        "explanation": result.get("explanation", "Transform step generated from your description."),
    }
