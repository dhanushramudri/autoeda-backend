"""Fully-automated AutoML planning — no manual dataset/target/feature
selection. One-shot LLM call (not an agentic tool loop, since everything it
needs — dataset profiles and this workspace's own validated Hypotheses — is
cheap to gather directly here) that grounds the plan in what's already been
discovered in this workspace, so the model that gets queued isn't picked
blind off a raw column list.
"""
import json
import logging

from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..llm import get_provider
from ...models.auto_eda import AutoEdaRun
from ...models.dataset import Dataset, EDAResult
from ...models.hypothesis import Hypothesis

logger = logging.getLogger("autoeda.ai.automl_planner")

_TEMPERATURE = 0.2
_MAX_TOKENS = 4000
_MAX_SAMPLE_ROWS = 500  # enough to profile columns fast — not a training sample
_MAX_DATASETS_CONSIDERED = 3
_MAX_COLUMNS_PER_DATASET = 50  # a wide real-world dataset otherwise bloats the prompt enough to truncate the response
_MAX_HYPOTHESES_CONSIDERED = 20
_MAX_AUTO_EDA_EXCERPT_CHARS = 2000

# The only feature-engineering operations the plan may reference — kept as a
# fixed, safe toolbox (see agent/feature_tools.py) rather than letting the
# model emit arbitrary pandas/Python, which would be both a code-execution
# risk and much more likely to just be wrong.
FEATURE_ENGINEERING_TOOLS = {
    "difference": "difference(a, b) — a minus b; works for two datetime-ish columns (gives days) or two numeric columns",
    "ratio": "ratio(a, b) — a divided by b, safely handling zero/missing",
    "datetime_part": "datetime_part(column, part) — part is one of: year, month, day, dayofweek, is_weekend",
    "log1p": "log1p(column) — log(1+x); use for a right-skewed numeric column",
    "frequency_encoding": "frequency_encoding(column) — replaces a high-cardinality categorical with how often each value occurs",
    "interaction": "interaction(a, b) — a multiplied by b",
}


class PlanningError(Exception):
    """Raised with a message that's safe to show the engineer directly."""


def _cached_profile_columns(db: Session, dataset_id: int) -> list[dict] | None:
    """The dataset's own full Profile tab result, if it's already been run
    — richer than a from-scratch sample (semantic type, skew/kurtosis, top
    values) and reuses work already done instead of redoing a shallow
    version of it."""
    row = (
        db.query(EDAResult)
        .filter(EDAResult.dataset_id == dataset_id, EDAResult.analysis_type == "profile")
        .order_by(desc(EDAResult.computed_at))
        .first()
    )
    if not row:
        return None
    try:
        data = json.loads(row.result_data)
        columns = data.get("columns")
        return columns if isinstance(columns, list) else None
    except Exception:
        return None


def _cached_top_correlations(db: Session, dataset_id: int, limit: int = 10) -> list[dict]:
    """Strongest already-computed pairwise correlations, if the
    Correlations tab has been run on this dataset — a numeric column
    strongly correlated with a plausible outcome column is itself a signal
    worth the plan seeing."""
    row = (
        db.query(EDAResult)
        .filter(EDAResult.dataset_id == dataset_id, EDAResult.analysis_type == "correlations")
        .order_by(desc(EDAResult.computed_at))
        .first()
    )
    if not row:
        return []
    try:
        pairs = json.loads(row.result_data).get("top_pairs")
        if not isinstance(pairs, list):
            return []
        ranked = sorted(pairs, key=lambda p: abs(p.get("correlation") or 0), reverse=True)
        return ranked[:limit]
    except Exception:
        return []


def _recent_auto_eda_excerpts(db: Session, workspace_id: int, limit: int = 2) -> list[dict]:
    """The most recent completed Auto EDA report(s) for this workspace —
    already-synthesized findings across a dataset, not just raw stats."""
    runs = (
        db.query(AutoEdaRun)
        .filter(AutoEdaRun.workspace_id == workspace_id, AutoEdaRun.status == "completed", AutoEdaRun.markdown.isnot(None))
        .order_by(desc(AutoEdaRun.updated_at))
        .limit(limit)
        .all()
    )
    out = []
    for run in runs:
        excerpt = (run.markdown or "")[:_MAX_AUTO_EDA_EXCERPT_CHARS]
        out.append({
            "title": run.title, "dataset_ids": json.loads(run.dataset_ids_json) if run.dataset_ids_json else [],
            "report_excerpt": excerpt,
        })
    return out


def _profile_columns(df) -> list[dict]:
    import pandas as pd

    sample = df.head(_MAX_SAMPLE_ROWS)
    out = []
    for col in df.columns:
        s = sample[col]
        info: dict = {
            "name": col,
            "dtype": str(df[col].dtype),
            "unique_pct": round(100 * df[col].nunique() / max(len(df), 1), 1),
            "missing_pct": round(100 * float(df[col].isna().mean()), 1),
        }
        if pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()
            info["min"] = float(non_null.min()) if len(non_null) else None
            info["max"] = float(non_null.max()) if len(non_null) else None
        else:
            info["sample_values"] = [str(v) for v in s.dropna().unique()[:5]]
        out.append(info)
    return out


def _build_prompt(
    datasets_context: list[dict], hypotheses_context: list[dict], auto_eda_context: list[dict],
) -> str:
    tools_desc = "\n".join(f"- {name}: {desc}" for name, desc in FEATURE_ENGINEERING_TOOLS.items())
    hyp_block = json.dumps(hypotheses_context, indent=2) if hypotheses_context else "(none validated yet)"
    auto_eda_block = json.dumps(auto_eda_context, indent=2) if auto_eda_context else "(no Auto EDA run has been completed yet)"
    return f"""You are planning a machine learning experiment for a data science team, with no human choosing the dataset, target, or features. You must pick ONE dataset, ONE target column to predict, the problem type, which columns are legitimate predictors vs which to exclude, and a handful of engineered features.

Everything below is analysis that has ALREADY been run in this workspace — use it as evidence rather than reasoning from column names alone. Each dataset's "columns" block is its own real Profile result (or a fast equivalent) and "top_correlations" is its own real Correlations result, when those have been run; "auto_eda_reports" are full narrative EDA reports already written for this workspace.

DATASETS AVAILABLE (pick exactly one, by its dataset_id):
{json.dumps(datasets_context, indent=2)}

RECENT AUTO EDA REPORTS FOR THIS WORKSPACE (already-synthesized findings — prefer a target these reports actually discuss):
{auto_eda_block}

VALIDATED FINDINGS FROM THIS WORKSPACE'S HYPOTHESES (a supported hypothesis that "X relates to Y" is real evidence Y is worth targeting and X is worth including; a refuted one is evidence to leave that relationship out):
{hyp_block}

FEATURE ENGINEERING TOOLS YOU MAY REFERENCE (only these — do not invent others):
{tools_desc}

Rules:
- Pick a target column that is a genuine business outcome (e.g. a churn/status/outcome flag, a renewal or revenue number) — never an ID, a free-text column, or a column that is effectively the target already.
- List in excluded_columns ONLY the columns that are clearly ID-like (near-unique per row), free-text, or leakage — do not enumerate every column you are keeping, and do not restate obvious inclusions.
- Propose 0-5 engineered features using ONLY the tools listed above, only where they would plausibly help, referencing only real column names from the chosen dataset.
- Keep the JSON compact — short strings, no repeated explanations per column.
- Respond with ONLY this JSON, no other text, no markdown fences, and make sure it is complete and valid (all brackets closed):

{{
  "dataset_id": <int>,
  "target_column": "<string>",
  "problem_type": "classification" | "regression",
  "rationale": "<2-3 sentences on why this dataset and target, grounded in the context above>",
  "excluded_columns": ["<column names to exclude, with no explanation needed here>"],
  "engineered_features": [
    {{"tool": "<tool name>", "args": {{...}}, "output": "<new column name>", "reason": "<short reason>"}}
  ]
}}"""


def _parse_json_block(raw: str):
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("automl planner JSON parse error: %s | raw=%s", e, raw[:500])
        return None


def plan_experiment(workspace_id: int, db: Session) -> dict:
    """Returns a validated plan dict, or raises PlanningError with a
    message that's safe to surface directly to the engineer."""
    from ...routers.eda import _load_df

    provider = get_provider()
    if provider is None:
        raise PlanningError("No AI provider is configured — can't auto-plan an experiment without one.")

    datasets = (
        db.query(Dataset)
        .filter(Dataset.workspace_id == workspace_id, Dataset.status == "ready")
        .order_by(desc(Dataset.updated_at))
        .limit(_MAX_DATASETS_CONSIDERED)
        .all()
    )
    if not datasets:
        raise PlanningError("No processed datasets in this workspace yet — upload and process one first.")

    datasets_context = []
    profiles_by_id: dict[int, list[dict]] = {}
    for ds in datasets:
        columns = _cached_profile_columns(db, ds.id)
        used_cache = columns is not None
        if columns is None:
            try:
                df = _load_df(ds, row_limit=_MAX_SAMPLE_ROWS)
            except Exception:
                logger.warning("automl planner: couldn't load dataset %s, skipping", ds.id)
                continue
            columns = _profile_columns(df)

        # Validation only needs to recognize what the model picks, so keep
        # the FULL list here even if we trim what's actually shown to the
        # model below — otherwise a column past the cap could never be
        # excluded or targeted, silently narrowing what the plan can do.
        profiles_by_id[ds.id] = columns
        truncated_note = None
        shown_columns = columns
        if len(columns) > _MAX_COLUMNS_PER_DATASET:
            shown_columns = columns[:_MAX_COLUMNS_PER_DATASET]
            truncated_note = f"...and {len(columns) - _MAX_COLUMNS_PER_DATASET} more columns not shown"

        top_correlations = _cached_top_correlations(db, ds.id)
        datasets_context.append({
            "dataset_id": ds.id, "name": ds.name,
            "row_count": ds.row_count, "column_count": ds.column_count,
            "profile_source": "already-computed Profile result" if used_cache else "quick sample (Profile hasn't been run on this dataset yet)",
            "columns": shown_columns,
            **({"note": truncated_note} if truncated_note else {}),
            **({"top_correlations": top_correlations} if top_correlations else {}),
        })

    if not datasets_context:
        raise PlanningError("Couldn't load any dataset in this workspace to plan against.")

    auto_eda_context = _recent_auto_eda_excerpts(db, workspace_id)

    hyps = (
        db.query(Hypothesis)
        .filter(Hypothesis.workspace_id == workspace_id, Hypothesis.status.in_(["supported", "refuted"]))
        .order_by(desc(Hypothesis.validated_at))
        .limit(_MAX_HYPOTHESES_CONSIDERED)
        .all()
    )
    hypotheses_context = [
        {
            "statement": h.statement, "verdict": h.status, "dataset_id": h.dataset_id,
            "columns": json.loads(h.columns_json) if h.columns_json else [],
            "evidence_summary": h.evidence_summary,
        }
        for h in hyps
    ]

    prompt = _build_prompt(datasets_context, hypotheses_context, auto_eda_context)
    raw = provider.generate(prompt, temperature=_TEMPERATURE, max_tokens=_MAX_TOKENS)
    plan = _parse_json_block(raw) if raw else None
    if not isinstance(plan, dict):
        # Most likely cause: the response got cut off mid-JSON (a wide
        # dataset means a lot to enumerate) rather than the model failing
        # outright — one retry with an explicit "keep it complete and
        # compact" nudge recovers far more often than not.
        logger.info("automl planner: first attempt didn't parse, retrying once")
        retry_prompt = prompt + "\n\nYour previous response did not parse as valid, complete JSON. Respond again with ONLY the JSON object described above — complete, compact, and with every bracket closed."
        raw = provider.generate(retry_prompt, temperature=_TEMPERATURE, max_tokens=_MAX_TOKENS)
        plan = _parse_json_block(raw) if raw else None
    if not isinstance(plan, dict):
        raise PlanningError("The AI couldn't produce a usable plan — try again in a moment.")

    try:
        dataset_id = int(plan.get("dataset_id"))
    except (TypeError, ValueError):
        raise PlanningError("The AI's plan didn't identify a valid dataset — try again.")
    if dataset_id not in profiles_by_id:
        raise PlanningError("The AI picked a dataset that wasn't actually offered — try again.")
    column_names = {c["name"] for c in profiles_by_id[dataset_id]}
    target = plan.get("target_column")
    if target not in column_names:
        raise PlanningError(f"The AI picked a target column ('{target}') that doesn't exist in that dataset — try again.")

    excluded = [c for c in plan.get("excluded_columns", []) if isinstance(c, str) and c in column_names and c != target]
    engineered = [
        feat for feat in plan.get("engineered_features", [])[:5]
        if isinstance(feat, dict) and feat.get("tool") in FEATURE_ENGINEERING_TOOLS and isinstance(feat.get("output"), str)
    ]
    problem_type = plan.get("problem_type") if plan.get("problem_type") in ("classification", "regression") else None

    return {
        "dataset_id": dataset_id,
        "dataset_name": next(d.name for d in datasets if d.id == dataset_id),
        "target_column": target,
        "problem_type": problem_type,
        "rationale": (plan.get("rationale") or "").strip(),
        "excluded_columns": excluded,
        "engineered_features": engineered,
    }
