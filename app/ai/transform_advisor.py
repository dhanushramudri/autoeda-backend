"""AI-powered transform suggestions based on dataset profile + quality data."""
import json
import logging
from typing import Optional

from .llm import generate

logger = logging.getLogger("autoeda.ai.transform_advisor")

_PROMPT = """You are a data cleaning expert analyzing a dataset profile.
Based on the statistics below, return a JSON array of transform suggestions.

Each suggestion must have:
  priority: integer 1-10 (1 = most important)
  reason: one sentence explaining why
  label: short human-readable action label (under 8 words)
  op: a valid transform operation object

Valid op types and their required fields:
  drop_duplicates: {{}}
  drop_columns: {{columns: [...]}}
  fill_missing: {{column: str, strategy: "mean"|"median"|"mode"|"constant"|"ffill"|"bfill"}}
  drop_rows_where_null: {{columns: [...]}}
  cap_outliers: {{column: str, method: "iqr"}}
  drop_outliers: {{column: str, method: "iqr"|"zscore"}}
  encode: {{column: str, method: "label"|"onehot"}}
  scale: {{column: str, method: "standard"|"minmax"|"robust"}}
  cast_type: {{column: str, to_type: "int"|"float"|"str"|"datetime"}}
  log_transform: {{column: str, variant: "log1p"}}
  text_clean: {{column: str, strip: true, lowercase: true}}

Dataset profile:
{profile}

Return ONLY a valid JSON array. No markdown. No explanation outside the array. Maximum 8 suggestions."""


def _build_profile_summary(profile: dict, quality: dict) -> str:
    """Build a compact profile string to fit in the prompt."""
    lines = []

    rows = profile.get("total_rows", 0)
    cols_count = profile.get("total_columns", 0)
    dup_pct = profile.get("duplicate_pct", 0)
    lines.append(f"Rows: {rows}, Columns: {cols_count}, Duplicates: {dup_pct:.1f}%")

    col_summaries = []
    for c in profile.get("columns", [])[:40]:
        name = c.get("name", "")
        dtype = c.get("dtype", "")
        sem = c.get("semantic_type", "")
        miss = c.get("missing_pct", 0)
        unique = c.get("unique_count", 0)
        skew = c.get("skewness")
        parts = [f"{name} ({dtype}/{sem})"]
        if miss > 0:
            parts.append(f"missing={miss:.1f}%")
        if unique == rows and rows > 0:
            parts.append("id-like/100%unique")
        if skew is not None and abs(skew) > 2:
            parts.append(f"skew={skew:.1f}")
        col_summaries.append(", ".join(parts))

    lines.append("Columns:\n" + "\n".join(f"  - {s}" for s in col_summaries))

    issues = [i.get("description", "") for i in quality.get("issues", [])[:5]]
    if issues:
        lines.append("Quality issues: " + "; ".join(issues))

    return "\n".join(lines)


def get_transform_suggestions(profile: dict, quality: dict) -> list[dict]:
    """Return AI-generated transform suggestions as a list of dicts."""
    summary = _build_profile_summary(profile, quality)
    prompt = _PROMPT.format(profile=summary)

    raw = generate(prompt, temperature=0.1, max_tokens=1024)
    if not raw:
        return _rule_based_suggestions(profile, quality)

    try:
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        suggestions = json.loads(cleaned)
        if isinstance(suggestions, list):
            # Sort by priority, validate each has required fields
            valid = [
                s for s in suggestions
                if isinstance(s, dict) and "op" in s and "label" in s
            ]
            return sorted(valid, key=lambda x: x.get("priority", 99))[:8]
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("AI transform suggestions parse failed: %s | raw=%s", e, raw[:300])

    return _rule_based_suggestions(profile, quality)


def _rule_based_suggestions(profile: dict, quality: dict) -> list[dict]:
    """Fallback: deterministic suggestions when AI is unavailable."""
    suggestions = []
    priority = 1
    rows = profile.get("total_rows", 0)

    if profile.get("duplicate_pct", 0) > 0:
        dup_count = int(profile.get("duplicate_pct", 0) / 100 * rows)
        suggestions.append({
            "priority": priority,
            "label": f"Remove {dup_count} duplicate rows",
            "reason": f"{profile.get('duplicate_pct', 0):.1f}% of rows are duplicates.",
            "op": {"type": "drop_duplicates"},
        })
        priority += 1

    for col in profile.get("columns", []):
        name = col.get("name", "")
        miss = col.get("missing_pct", 0)
        sem = col.get("semantic_type", "")
        skew = col.get("skewness") or 0
        unique = col.get("unique_count", 0)

        if miss > 50:
            suggestions.append({
                "priority": priority,
                "label": f"Drop '{name}' ({miss:.0f}% missing)",
                "reason": f"Column '{name}' has {miss:.1f}% missing values — too sparse to impute.",
                "op": {"type": "drop_columns", "columns": [name]},
            })
            priority += 1
        elif miss > 5:
            strategy = "median" if sem == "numeric" else "mode"
            suggestions.append({
                "priority": priority,
                "label": f"Fill missing '{name}' with {strategy}",
                "reason": f"Column '{name}' has {miss:.1f}% missing values.",
                "op": {"type": "fill_missing", "column": name, "strategy": strategy},
            })
            priority += 1

        if unique == rows and rows > 0 and sem not in ("datetime",):
            suggestions.append({
                "priority": priority,
                "label": f"Drop ID column '{name}'",
                "reason": f"'{name}' has 100% unique values — likely an ID, not useful for analysis.",
                "op": {"type": "drop_columns", "columns": [name]},
            })
            priority += 1

        if sem == "numeric" and abs(skew) > 2:
            suggestions.append({
                "priority": priority,
                "label": f"Log-transform '{name}' (skew={skew:.1f})",
                "reason": f"'{name}' is highly skewed ({skew:.1f}). Log transform reduces skewness.",
                "op": {"type": "log_transform", "column": name, "variant": "log1p"},
            })
            priority += 1

        if priority > 8:
            break

    return suggestions[:8]
