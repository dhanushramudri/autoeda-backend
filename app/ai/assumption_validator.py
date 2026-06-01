"""
Validate user-provided assumptions/hypotheses against actual EDA results.

Takes a business assumption (e.g. "High-value customers churn less") and validates it
against profile, correlations, outliers, feature importance data.

Returns: verdict, evidence, confidence, status (supported|refuted|inconclusive), columns.
"""
import json
import logging
from typing import Any

from .llm import generate

logger = logging.getLogger("autoeda.ai.assumption_validator")

# ── Prompt template ────────────────────────────────────────────────────────────

_VALIDATOR_PROMPT = """You are a senior data analyst validating a business hypothesis against actual data.

User Assumption: "{assumption}"

Dataset Profile ({rows} rows × {cols} columns):
{col_profiles}

Top Correlations (r values):
{correlations}

Outlier Summary:
{outliers}

Feature Importance (if available):
{feature_importance}

Your task: Validate this assumption strictly against the data provided.

Return ONLY a JSON object with these exact fields:
{{
  "verdict": "1-2 sentence verdict citing SPECIFIC column names and NUMERIC values from the data. Be precise.",
  "evidence": "The key statistic that supports or refutes (e.g., r=0.87, 23% missing, skew=4.2, outlier_pct=8.5)",
  "confidence": "high|medium|low",
  "status": "supported|refuted|inconclusive",
  "columns": ["relevant", "column", "names", "from", "the", "data"]
}}

Rules:
- ONLY cite columns that actually exist in the dataset
- ONLY use numbers from the data provided
- If the assumption cannot be validated with available data, mark as "inconclusive"
- If data clearly contradicts the assumption, mark as "refuted"
- Only "supported" if data shows evidence aligning with the assumption
- Be conservative with "high" confidence — use "medium" or "low" if uncertain

No markdown fences. No text outside the JSON."""


# ── EDA summary builder ───────────────────────────────────────────────────────

def _safe(val: Any, default: Any = "N/A") -> str:
    if val is None:
        return str(default)
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _build_context(
    profile: dict,
    correlations: dict,
    outliers: dict,
    feature_importance: dict,
) -> dict:
    """Compress EDA results into strings for the validator prompt."""

    # Column profiles
    col_lines = []
    for c in profile.get("columns", [])[:40]:
        name = c.get("name", "")
        dtype = c.get("dtype", "")
        miss = c.get("missing_pct", 0)
        skew = c.get("skewness")
        unique = c.get("unique_count", 0)
        total = profile.get("total_rows", 1) or 1
        unique_pct = round(unique / total * 100, 1)
        skew_str = f"{skew:.2f}" if skew is not None else "-"
        col_lines.append(
            f"  {name} | {dtype} | miss={miss:.1f}% | skew={skew_str} | unique={unique_pct}%"
        )
    col_profiles = "\n".join(col_lines) if col_lines else "  (not profiled)"

    # Correlations — top pairs
    corr_lines = []
    top_pairs = correlations.get("top_pairs", [])
    if top_pairs:
        for p in sorted(top_pairs, key=lambda x: abs(x.get("correlation", 0)), reverse=True)[:10]:
            r = p.get("correlation", 0)
            corr_lines.append(f"  {p.get('col1')} ↔ {p.get('col2')}: r={r:.3f}")
    elif correlations.get("matrix"):
        # Derive from matrix
        matrix = correlations.get("matrix", {})
        seen = set()
        pairs = []
        for c1, row in matrix.items():
            for c2, val in (row or {}).items():
                if c1 != c2 and (c2, c1) not in seen and val is not None:
                    pairs.append((c1, c2, float(val)))
                    seen.add((c1, c2))
        for c1, c2, r in sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
            corr_lines.append(f"  {c1} ↔ {c2}: r={r:.3f}")

    corr_str = "\n".join(corr_lines) if corr_lines else "  (not computed)"

    # Outliers
    out_lines = []
    for col in outliers.get("columns", []):
        name = col.get("name", "")
        pct = col.get("outlier_pct", 0)
        cnt = col.get("outlier_count", 0)
        if pct > 0:
            out_lines.append(f"  {name}: {pct:.1f}% ({cnt} rows)")
    out_str = "\n".join(out_lines) if out_lines else "  (no outliers detected)"

    # Feature importance
    fi_lines = []
    for item in feature_importance.get("importances", [])[:8]:
        feat = item.get("feature", "")
        imp = item.get("importance", 0)
        fi_lines.append(f"  {feat}: importance={imp:.4f}")
    target = feature_importance.get("target", "")
    fi_str = (
        f"  Target: {target}\n" + "\n".join(fi_lines)
        if fi_lines
        else "  (not computed)"
    )

    return {
        "col_profiles": col_profiles,
        "correlations": corr_str,
        "outliers": out_str,
        "feature_importance": fi_str,
    }


# ── AI validation ──────────────────────────────────────────────────────────────

def _call_ai(
    assumption: str,
    rows: int,
    cols: int,
    context: dict,
) -> dict | None:
    """Call LLM to validate assumption. Returns None if AI unavailable."""

    prompt = _VALIDATOR_PROMPT.format(
        assumption=assumption,
        rows=rows,
        cols=cols,
        col_profiles=context["col_profiles"],
        correlations=context["correlations"],
        outliers=context["outliers"],
        feature_importance=context["feature_importance"],
    )

    raw = generate(prompt, temperature=0.1, max_tokens=800)
    if not raw:
        return None

    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip().rstrip("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("assumption validator AI parse error: %s | raw=%s", e, raw[:300])
        return None

    # Validate required fields
    required = {"verdict", "evidence", "confidence", "status", "columns"}
    if not required.issubset(parsed.keys()):
        return None

    # Coerce to valid enums
    valid_conf = {"high", "medium", "low"}
    valid_status = {"supported", "refuted", "inconclusive"}

    if parsed.get("confidence") not in valid_conf:
        parsed["confidence"] = "medium"
    if parsed.get("status") not in valid_status:
        parsed["status"] = "inconclusive"
    if not isinstance(parsed.get("columns"), list):
        parsed["columns"] = []

    return parsed


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _rule_based(
    assumption: str,
    profile: dict,
    correlations: dict,
    outliers: dict,
    feature_importance: dict,
) -> dict:
    """
    Fallback: attempt simple pattern matching on assumption against data.
    Returns inconclusive if no clear pattern matches.
    """
    cols = [c.get("name", "") for c in profile.get("columns", [])]
    col_lower = [c.lower() for c in cols]
    assumption_lower = assumption.lower()

    # Very basic heuristics — just return inconclusive by default
    # since rule-based validation of free-form text assumptions is unreliable

    return {
        "verdict": "Could not validate with available data patterns. Please check EDA results manually.",
        "evidence": "—",
        "confidence": "low",
        "status": "inconclusive",
        "columns": [],
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def validate_assumption(
    assumption: str,
    profile: dict,
    correlations: dict,
    outliers: dict,
    feature_importance: dict,
) -> dict:
    """
    Validate a user assumption against EDA results.

    Returns:
    {
      "verdict": str,
      "evidence": str,
      "confidence": "high|medium|low",
      "status": "supported|refuted|inconclusive",
      "columns": list[str]
    }
    """

    if not assumption or not assumption.strip():
        return {
            "verdict": "Empty assumption.",
            "evidence": "—",
            "confidence": "low",
            "status": "inconclusive",
            "columns": [],
        }

    rows = profile.get("total_rows", 0)
    cols = profile.get("total_columns", 0)

    if not rows or not cols:
        return {
            "verdict": "Cannot validate — dataset profile incomplete.",
            "evidence": "—",
            "confidence": "low",
            "status": "inconclusive",
            "columns": [],
        }

    context = _build_context(profile, correlations, outliers, feature_importance)

    # 1. Try AI
    ai_result = _call_ai(assumption, rows, cols, context)
    if ai_result:
        logger.info("Assumption validation: AI returned result for '%s'", assumption[:60])
        return ai_result

    # 2. Fallback to rule-based
    logger.info("Assumption validation: using rule-based fallback for '%s'", assumption[:60])
    return _rule_based(assumption, profile, correlations, outliers, feature_importance)
