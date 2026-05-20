"""
Proactive hypothesis generation from EDA results.

Uses AI when configured, falls back to a data-driven rule engine that
derives real findings from actual column stats — never generic text.
"""
import json
import logging
import re
from typing import Any

from .llm import generate

logger = logging.getLogger("autoeda.ai.hypothesis_generator")

# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT = """You are a senior data analyst reviewing an automated EDA report.
Your job: generate 6-8 proactive, analyst-quality hypotheses about this dataset.

Rules:
- Every hypothesis MUST reference actual column names and numeric values from the data below
- State the finding WITH magnitude (e.g. "4× higher", "r=0.87", "23% of outliers")
- Write as if briefing a business stakeholder — actionable, not just statistical
- Do NOT repeat what the profile already says — interpret it
- Categories: correlation | distribution | missing | outlier | quality | feature | pattern

Dataset: {name}
Shape: {rows} rows × {cols} columns
Quality score: {quality_score}/100
Missing overall: {missing_pct}%
Duplicates: {dup_pct}%

Column profiles (name | dtype | semantic | missing% | skew | unique):
{col_profiles}

Top correlation pairs:
{correlations}

Outlier summary (column | outlier% | method):
{outliers}

Feature importance (if available):
{feature_importance}

Return ONLY a JSON array. Each object must have ALL of these fields:
  "title"      : string  — max 8 words, punchy headline
  "hypothesis" : string  — 1-2 sentences with evidence and magnitude
  "evidence"   : string  — the key stat (e.g. "r=0.91", "skew=4.8", "14.2% outliers")
  "category"   : one of correlation|distribution|missing|outlier|quality|feature|pattern
  "confidence" : one of high|medium|low
  "severity"   : one of info|warning|danger
  "columns"    : array of relevant column name strings from the dataset

No markdown fences. No text outside the JSON array."""


# ── EDA summary builder ───────────────────────────────────────────────────────

def _safe(val: Any, default: Any = "N/A") -> str:
    if val is None:
        return str(default)
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _build_summary(
    profile: dict,
    correlations: dict,
    outliers: dict,
    feature_importance: dict,
) -> dict:
    """Compress EDA results into compact strings for the prompt."""

    # Column profiles
    col_lines = []
    for c in profile.get("columns", [])[:50]:
        name = c.get("name", "")
        dtype = c.get("dtype", "")
        sem = c.get("semantic_type", "")
        miss = c.get("missing_pct", 0)
        skew = c.get("skewness")
        unique = c.get("unique_count", 0)
        total = profile.get("total_rows", 1) or 1
        unique_pct = round(unique / total * 100, 1)
        skew_str = f"{skew:.2f}" if skew is not None else "-"
        col_lines.append(
            f"  {name} | {dtype} | {sem} | miss={miss:.1f}% | skew={skew_str} | unique={unique_pct}%"
        )
    col_profiles = "\n".join(col_lines) if col_lines else "  (not yet profiled)"

    # Correlations — top 10 pairs by abs value
    corr_lines = []
    matrix = correlations.get("matrix", {})
    top_pairs = correlations.get("top_pairs", [])
    if top_pairs:
        for p in sorted(top_pairs, key=lambda x: abs(x.get("correlation", 0)), reverse=True)[:10]:
            r = p.get("correlation", 0)
            corr_lines.append(f"  {p.get('col1')} ↔ {p.get('col2')}: r={r:.3f}")
    elif matrix:
        # Derive from matrix if top_pairs absent
        seen = set()
        pairs = []
        for c1, row in matrix.items():
            for c2, val in row.items():
                if c1 != c2 and (c2, c1) not in seen and val is not None:
                    pairs.append((c1, c2, float(val)))
                    seen.add((c1, c2))
        for c1, c2, r in sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
            corr_lines.append(f"  {c1} ↔ {c2}: r={r:.3f}")
    corr_str = "\n".join(corr_lines) if corr_lines else "  (not yet computed)"

    # Outliers
    out_lines = []
    for col in outliers.get("columns", []):
        name = col.get("name", "")
        pct = col.get("outlier_pct", 0)
        cnt = col.get("outlier_count", 0)
        if pct > 0:
            out_lines.append(f"  {name}: {pct:.1f}% ({cnt} rows) | method={outliers.get('method','IQR')}")
    out_str = "\n".join(out_lines) if out_lines else "  (not yet computed or no outliers)"

    # Feature importance
    fi_lines = []
    for item in feature_importance.get("importances", [])[:10]:
        feat = item.get("feature", "")
        imp = item.get("importance", 0)
        fi_lines.append(f"  {feat}: importance={imp:.4f}")
    target = feature_importance.get("target", "")
    fi_str = (
        f"  Target: {target}\n" + "\n".join(fi_lines)
        if fi_lines
        else "  (not yet computed — no target selected)"
    )

    return {
        "col_profiles": col_profiles,
        "correlations": corr_str,
        "outliers": out_str,
        "feature_importance": fi_str,
    }


# ── AI path ───────────────────────────────────────────────────────────────────

def _call_ai(
    name: str,
    rows: int,
    cols: int,
    quality_score: Any,
    missing_pct: float,
    dup_pct: float,
    summary: dict,
) -> list[dict] | None:
    prompt = _PROMPT.format(
        name=name,
        rows=rows,
        cols=cols,
        quality_score=quality_score,
        missing_pct=f"{missing_pct:.1f}",
        dup_pct=f"{dup_pct:.1f}",
        col_profiles=summary["col_profiles"],
        correlations=summary["correlations"],
        outliers=summary["outliers"],
        feature_importance=summary["feature_importance"],
    )

    raw = generate(prompt, temperature=0.2, max_tokens=2048)
    if not raw:
        return None

    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip().rstrip("```").strip()

    # Sometimes the model wraps in an outer object
    if cleaned.startswith("{"):
        try:
            obj = json.loads(cleaned)
            cleaned = json.dumps(obj.get("hypotheses", obj.get("items", [])))
        except json.JSONDecodeError:
            pass

    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("hypothesis AI parse error: %s | raw=%s", e, raw[:400])
        return None

    if not isinstance(items, list):
        return None

    required = {"title", "hypothesis", "evidence", "category", "confidence", "severity", "columns"}
    valid_cats = {"correlation", "distribution", "missing", "outlier", "quality", "feature", "pattern"}
    valid_conf = {"high", "medium", "low"}
    valid_sev = {"info", "warning", "danger"}

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if not required.issubset(item.keys()):
            # Fill in defaults for missing fields rather than dropping the card
            item.setdefault("title", item.get("hypothesis", "Finding")[:50])
            item.setdefault("evidence", "")
            item.setdefault("category", "pattern")
            item.setdefault("confidence", "medium")
            item.setdefault("severity", "info")
            item.setdefault("columns", [])
        # Coerce to valid enum values
        if item["category"] not in valid_cats:
            item["category"] = "pattern"
        if item["confidence"] not in valid_conf:
            item["confidence"] = "medium"
        if item["severity"] not in valid_sev:
            item["severity"] = "info"
        if not isinstance(item["columns"], list):
            item["columns"] = []
        results.append(item)

    return results if results else None


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based(
    profile: dict,
    correlations: dict,
    outliers: dict,
    feature_importance: dict,
) -> list[dict]:
    """
    Derive genuine, data-driven hypotheses from actual EDA numbers.
    Every card references real column names and values — no generic text.
    """
    cards: list[dict] = []
    cols = profile.get("columns", [])
    total_rows = profile.get("total_rows", 1) or 1

    # ── Correlations ────────────────────────────────────────────────────────
    top_pairs = correlations.get("top_pairs", [])
    if not top_pairs:
        # Derive from matrix
        matrix = correlations.get("matrix", {})
        seen: set = set()
        for c1, row in matrix.items():
            for c2, val in (row or {}).items():
                if c1 != c2 and (c2, c1) not in seen and val is not None:
                    top_pairs.append({"col1": c1, "col2": c2, "correlation": float(val)})
                    seen.add((c1, c2))
        top_pairs.sort(key=lambda x: abs(x.get("correlation", 0)), reverse=True)

    for pair in top_pairs[:3]:
        r = pair.get("correlation", 0)
        c1, c2 = pair.get("col1", ""), pair.get("col2", "")
        if not c1 or not c2:
            continue
        abs_r = abs(r)
        direction = "positive" if r > 0 else "negative"
        if abs_r >= 0.9:
            cards.append({
                "title": f"{c1} & {c2} almost perfectly correlated",
                "hypothesis": (
                    f"'{c1}' and '{c2}' have near-perfect {direction} correlation (r={r:.2f}). "
                    "One of these columns is likely redundant — consider dropping it before modelling to avoid multicollinearity."
                ),
                "evidence": f"r={r:.3f}",
                "category": "correlation",
                "confidence": "high",
                "severity": "warning",
                "columns": [c1, c2],
            })
        elif abs_r >= 0.7:
            cards.append({
                "title": f"Strong link between {c1} and {c2}",
                "hypothesis": (
                    f"'{c1}' and '{c2}' show strong {direction} correlation (r={r:.2f}). "
                    "This relationship is worth investigating — it may encode useful signal or cause data leakage."
                ),
                "evidence": f"r={r:.3f}",
                "category": "correlation",
                "confidence": "high",
                "severity": "info",
                "columns": [c1, c2],
            })

    # ── VIF / Multicollinearity ──────────────────────────────────────────────
    for vif in (correlations.get("vif") or [])[:2]:
        val = vif.get("vif", 0)
        col_name = vif.get("column", "")
        if val > 10 and col_name:
            cards.append({
                "title": f"Severe multicollinearity in {col_name}",
                "hypothesis": (
                    f"'{col_name}' has a VIF of {val:.1f}, indicating severe multicollinearity with other features. "
                    "Linear models trained on this data will have unstable, unreliable coefficients."
                ),
                "evidence": f"VIF={val:.1f}",
                "category": "correlation",
                "confidence": "high",
                "severity": "danger",
                "columns": [col_name],
            })

    # ── Skewed distributions ─────────────────────────────────────────────────
    skewed = [
        c for c in cols
        if c.get("skewness") is not None and abs(c["skewness"]) > 2
        and c.get("semantic_type") == "numeric"
    ]
    skewed.sort(key=lambda c: abs(c.get("skewness", 0)), reverse=True)
    for c in skewed[:2]:
        skew = c["skewness"]
        direction = "right" if skew > 0 else "left"
        cards.append({
            "title": f"{c['name']} is heavily {direction}-skewed",
            "hypothesis": (
                f"'{c['name']}' has a skewness of {skew:.2f}, indicating a heavily {direction}-skewed distribution. "
                "A log or Box-Cox transform is strongly recommended before using this feature in any ML model."
            ),
            "evidence": f"skew={skew:.2f}",
            "category": "distribution",
            "confidence": "high",
            "severity": "warning",
            "columns": [c["name"]],
        })

    # ── High missing columns ─────────────────────────────────────────────────
    high_miss = [
        c for c in cols if c.get("missing_pct", 0) > 15
    ]
    high_miss.sort(key=lambda c: c.get("missing_pct", 0), reverse=True)
    for c in high_miss[:2]:
        miss = c["missing_pct"]
        sev = "danger" if miss > 50 else "warning"
        action = "dropping this column before modelling" if miss > 50 else "investigating whether the missingness is systematic (MNAR)"
        cards.append({
            "title": f"{c['name']} has critical missing data",
            "hypothesis": (
                f"'{c['name']}' is missing {miss:.1f}% of its values ({int(miss/100 * total_rows):,} rows). "
                f"This level of missingness suggests {action}."
            ),
            "evidence": f"{miss:.1f}% missing",
            "category": "missing",
            "confidence": "high",
            "severity": sev,
            "columns": [c["name"]],
        })

    # ── Outlier-heavy columns ────────────────────────────────────────────────
    out_cols = outliers.get("columns", [])
    heavy_outliers = [
        c for c in out_cols if c.get("outlier_pct", 0) > 5
    ]
    heavy_outliers.sort(key=lambda c: c.get("outlier_pct", 0), reverse=True)
    for c in heavy_outliers[:2]:
        pct = c["outlier_pct"]
        cnt = c.get("outlier_count", int(pct / 100 * total_rows))
        cards.append({
            "title": f"High outlier rate in {c['name']}",
            "hypothesis": (
                f"'{c['name']}' contains {pct:.1f}% outliers ({cnt:,} rows) by {outliers.get('method','IQR')} method. "
                "This could indicate data entry errors, sensor faults, or genuine extreme events worth investigating."
            ),
            "evidence": f"{pct:.1f}% outliers ({cnt:,} rows)",
            "category": "outlier",
            "confidence": "high",
            "severity": "warning",
            "columns": [c["name"]],
        })

    # ── Feature importance ───────────────────────────────────────────────────
    importances = feature_importance.get("importances", [])
    target = feature_importance.get("target", "")
    if importances and target:
        top = importances[0]
        top_feat = top.get("feature", "")
        top_imp = top.get("importance", 0)
        if top_feat and top_imp > 0.1:
            cards.append({
                "title": f"{top_feat} dominates prediction of {target}",
                "hypothesis": (
                    f"'{top_feat}' is the single most important predictor of '{target}' with importance={top_imp:.3f}. "
                    "Investigate whether this reflects genuine causation or data leakage before deploying a model."
                ),
                "evidence": f"importance={top_imp:.3f}",
                "category": "feature",
                "confidence": "high",
                "severity": "info",
                "columns": [top_feat, target] if target else [top_feat],
            })

        # Near-zero importance features
        low_imp = [f for f in importances if f.get("importance", 1) < 0.005]
        if len(low_imp) >= 3:
            names = [f["feature"] for f in low_imp[:5]]
            cards.append({
                "title": f"{len(low_imp)} features add near-zero value",
                "hypothesis": (
                    f"{len(low_imp)} features have near-zero importance for predicting '{target}' "
                    f"(e.g. {', '.join(names[:3])}). Removing these will reduce model complexity with no accuracy loss."
                ),
                "evidence": f"{len(low_imp)} features with importance < 0.005",
                "category": "feature",
                "confidence": "medium",
                "severity": "info",
                "columns": names,
            })

    # ── ID-like / constant columns ───────────────────────────────────────────
    id_cols = [c for c in cols if c.get("semantic_type") == "id_like"]
    if id_cols:
        names = [c["name"] for c in id_cols[:3]]
        cards.append({
            "title": f"{len(id_cols)} identifier column(s) detected",
            "hypothesis": (
                f"{', '.join(names)} appear{'s' if len(names)==1 else ''} to be identifier column(s) "
                f"(near-100% unique values). These must be dropped before any ML training."
            ),
            "evidence": f"{len(id_cols)} columns with ~100% unique values",
            "category": "quality",
            "confidence": "high",
            "severity": "warning",
            "columns": names,
        })

    const_cols = [c for c in cols if c.get("semantic_type") == "constant"]
    if const_cols:
        names = [c["name"] for c in const_cols[:3]]
        cards.append({
            "title": "Constant columns carry no signal",
            "hypothesis": (
                f"{', '.join(names)} ha{'s' if len(names)==1 else 've'} zero variance (constant value). "
                "These columns carry no information and must be removed before modelling."
            ),
            "evidence": f"{len(const_cols)} constant column(s)",
            "category": "quality",
            "confidence": "high",
            "severity": "danger",
            "columns": names,
        })

    # ── Duplicates ───────────────────────────────────────────────────────────
    dup_pct = profile.get("duplicate_pct", 0)
    dup_count = profile.get("duplicate_count", int(dup_pct / 100 * total_rows))
    if dup_pct > 3:
        cards.append({
            "title": f"{dup_pct:.1f}% duplicate rows detected",
            "hypothesis": (
                f"The dataset contains {dup_count:,} duplicate rows ({dup_pct:.1f}%). "
                "These will bias any aggregate statistics and must be removed before analysis or modelling."
            ),
            "evidence": f"{dup_count:,} duplicate rows ({dup_pct:.1f}%)",
            "category": "quality",
            "confidence": "high",
            "severity": "warning",
            "columns": [],
        })

    return cards[:8]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_hypotheses(
    name: str,
    profile: dict,
    correlations: dict,
    outliers: dict,
    feature_importance: dict,
) -> list[dict]:
    """
    Return a list of hypothesis cards.
    Tries AI first; falls back to rule-based engine if AI is unavailable or fails.
    """
    rows = profile.get("total_rows", 0)
    cols = profile.get("total_columns", 0)
    quality_score = "?"
    missing_pct = 0.0
    dup_pct = profile.get("duplicate_pct", 0.0)

    if rows and cols:
        total_missing = sum(c.get("missing_count", 0) for c in profile.get("columns", []))
        total_cells = rows * cols
        missing_pct = round(total_missing / total_cells * 100, 1) if total_cells else 0.0

    summary = _build_summary(profile, correlations, outliers, feature_importance)

    # 1. Try AI
    ai_result = _call_ai(
        name=name,
        rows=rows,
        cols=cols,
        quality_score=quality_score,
        missing_pct=missing_pct,
        dup_pct=dup_pct,
        summary=summary,
    )
    if ai_result:
        logger.info("Hypotheses: AI returned %d cards for '%s'", len(ai_result), name)
        return ai_result

    # 2. Rule-based fallback
    logger.info("Hypotheses: using rule-based fallback for '%s'", name)
    return _rule_based(profile, correlations, outliers, feature_importance)
