"""Auto EDA: an autonomous, agentic exploratory-data-analysis run.

Given a dataset, this seeds a worklist of investigations, executes them one
by one against the REAL data using the same compute functions the rest of
the product already uses (profiler/correlations/distributions/outliers/
missing/quality_score/timeseries/text_analysis — no LLM guessing for the
math itself), renders each result as a chart image (app/eda/chart_render.py)
or a Markdown table, asks the LLM for a short grounded caption, and asks it
whether this specific finding warrants adding follow-up items to the
worklist — which can genuinely grow the queue mid-run, capped for safety.

Two analyses exist specifically to answer "what actually drives the outcome"
rather than profile each column in isolation, and only run when a business
context is given (that's what defines "the outcome" at all):
  - "feature_importance" (per dataset, if a target/outcome column is
    detected in it): RF importance + correlation/ANOVA + leakage/redundancy
    checks against that target, via eda/feature_importance.py.
  - "hypothesis_investigation" (once per run, workspace-wide): reuses
    Scout's hypothesis-generation agent (hypothesis_orchestrator.py) to
    propose and statistically test named claims — including cross-dataset
    ones via SQL/python — and persists them as real Hypothesis rows too, so
    they also surface in the Hypotheses tab.
Both are tier0 (guaranteed), not optional — they're the point of the run
when a business context is given, not competing line items.

Design choice: the INITIAL worklist is seeded deterministically from the
dataset's own profile (not invented freehand by the LLM) — this guarantees
every seeded item is actually executable and never references a column
that doesn't exist. The LLM's job is captioning and proposing bounded,
validated follow-ups, not inventing the whole plan from scratch.

The growing report is persisted as one Markdown string with every chart
embedded as a base64 data URI — fully self-contained, no external image
URLs that could expire or 404 later.
"""
import json
import logging
from typing import Any, Iterator

import pandas as pd
from sqlalchemy.orm import Session

from ..llm import get_provider
from ..providers.base import QuotaExceededError
from .hypothesis_orchestrator import run_hypothesis_generation
from .sandbox import exec_sandboxed
from ...eda import chart_render as cr
from ...eda.profiler import run_profile
from ...eda.correlations import compute_num_matrix
from ...eda.distributions import run_distribution
from ...eda.outliers import run_outlier_detection
from ...eda.missing import run_missing_analysis
from ...eda.quality_score import run_quality_score
from ...eda.timeseries import run_timeseries
from ...eda.text_analysis import run_text_analysis
from ...eda.feature_importance import run_feature_importance
from ...eda.stats_tests import run_statistical_test
from ...process_pool import AnalysisCrashed, AnalysisTimeout, run_isolated
from ...models.auto_eda import AutoEdaChatMessage
from ...models.hypothesis import Hypothesis
from ...models.user import User

logger = logging.getLogger("autoeda.ai.agent.auto_eda")

MAX_ROWS = 200_000  # keeps a full autonomous run fast — not the per-analysis max a human picks manually
MAX_TOTAL_ITEMS = 150  # per dataset, before the overall ceiling below applies
# Guardrail: a run — single- or multi-dataset — never plans more than this
# many items total. This exists purely to bound pathological cases (a
# dataset with hundreds of columns) — it is NOT meant to be the number of
# items a normal run actually produces; _plan_worklist is explicitly told
# not to pad up to it. Fallback default only — settings.AUTO_EDA_MAX_ITEMS
# (env var) is the actual value used at runtime, and should be set high
# enough that a real, business-context-driven candidate set (now including
# one target_relationship test per feature column, where a target was
# found — see _enumerate_candidates) is rarely if ever truncated.
MAX_TOTAL_ITEMS_CEILING = 150
MAX_FOLLOWUPS_PER_ITEM = 2
# These are a SAFETY ceiling on how many columns become candidates at all —
# not the selection itself. A 150-column table shouldn't get 150 distribution
# items, but it also shouldn't get silently pre-filtered down to the first 8
# by column order before the planner ever sees the rest. Selection down to
# the actual item budget happens in _plan_worklist, grounded in real stats
# and business context — this ceiling only bounds prompt size for pathological
# cases (hundreds of columns), so it's deliberately generous.
MAX_NUMERIC_COLS = 40
MAX_CATEGORICAL_COLS = 30
MAX_DATETIME_PAIRS = 6
MAX_TEXT_COLS = 8

_CAPTION_MAX_TOKENS = 500
_FOLLOWUP_MAX_TOKENS = 700
# The active deployment is a reasoning-model family (gpt-5.x) where
# max_tokens caps hidden reasoning tokens AND the visible answer combined —
# too small a budget here silently returns an EMPTY completion (not an
# error), which _plan_worklist would otherwise mistake for "AI declined"
# and fall back to naive ordering. Generous on purpose.
_PLANNING_MAX_TOKENS = 1200
_TARGET_DETECT_MAX_TOKENS = 300
_TEMPERATURE = 0.2

_VALID_KINDS = {
    "profile", "missing", "quality_score", "correlations",
    "distribution", "outliers", "categorical_breakdown",
    "timeseries", "text_analysis", "custom_python", "feature_importance",
    "hypothesis_investigation", "target_relationship",
}

# Kind -> which arg keys name a column that must actually exist on the dataset.
_COLUMN_ARG_KEYS = ("column", "time_col", "value_col", "target", "feature")

_FEATURE_IMPORTANCE_METHODS = ["rf", "metadata", "correlation", "anova", "mi"]
_FEATURE_IMPORTANCE_TIMEOUT_S = 240  # matches the existing manual feature-importance endpoint's own timeout
# A categorical column past this many distinct values isn't a usable model
# feature anyway (it's an identifier or a near-unique free-form field) and
# label-encoding it is what makes the RF fit / pairwise redundancy check
# blow up — see the exclude-list comment in _plan_all_datasets.
_FEATURE_IMPORTANCE_MAX_CATEGORIES = 200
_HYPOTHESIS_COUNT = 6
# A target needs few enough distinct values to be a meaningful grouping
# variable for chi2/ANOVA (Membership_Renewal_Decision: 4, Prospect_Outcome:
# 3 — this is the common case; a near-continuous "target" isn't one these
# tests can use as groups, so target_relationship items simply aren't
# generated for it — feature_importance still runs regardless, since it
# handles a continuous target as a regression problem on its own).
_MAX_TARGET_GROUPS_FOR_STATS = 12
# Periodic "reconsider the remaining plan" checkpoint (see _self_review) —
# runs automatically, not just in response to a user chat message. Interval
# is a count of completed items, not wall-clock time; capped separately so a
# very long run doesn't rack up an unbounded number of extra planning calls.
_SELF_REVIEW_INTERVAL = 15
_SELF_REVIEW_MAX_CALLS = 8


def _new_item(kind: str, title: str, args: dict | None = None) -> dict:
    return {"kind": kind, "title": title, "args": args or {}, "status": "pending"}


def _dedup_key(item: dict) -> tuple:
    """Identity of an item for duplicate detection — deliberately keyed on
    (kind, dataset, identifying args), NOT title. Two follow-ups worded
    differently ("Drivers of prospect outcome" vs "Rank drivers of
    Prospect_Outcome") but naming the same kind+target are the exact same
    analysis, and titles are free text the LLM can phrase however it likes —
    only the args actually determine what gets computed. Without this, every
    item that sees the same finding (e.g. a failed feature_importance run)
    can independently propose "let's retry that" as its own follow-up, with
    zero visibility into what every OTHER item already proposed — observed
    in production as a single failed billings feature_importance spawning a
    dozen near-duplicate "drivers of prospect outcome" items."""
    args = item.get("args", {})
    identifying = tuple(sorted(
        (k, tuple(v) if isinstance(v, list) else v)
        for k, v in args.items()
        if k in ("column", "time_col", "value_col", "target", "feature", "columns")
    ))
    return (item["kind"], item.get("dataset_id"), identifying)


def _looks_like_datetime(series: pd.Series) -> bool:
    """profiler.classify_column only checks pandas dtype, so a date column
    loaded from CSV (always a plain string dtype — file_connector doesn't
    parse dates) gets misclassified as categorical/text and would otherwise
    never produce a timeseries item. Sniff the actual string values instead."""
    sample = series.dropna().astype(str)
    if len(sample) == 0:
        return False
    sample = sample.sample(n=min(50, len(sample)), random_state=0)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9


def _enumerate_candidates(
    df: pd.DataFrame, profile: dict, target_col: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Every analysis this dataset actually supports — deterministic, no
    LLM involved, so nothing here can reference a column that doesn't
    exist. Split into:
      - tier0: profile/missing only — always run, non-negotiable. Cheap,
        structural, not worth spending the LLM's judgment on.
      - optional: everything else, INCLUDING correlations/outliers (they're
        high-value but not free, so they compete for budget like anything
        else rather than being silently force-included even when the
        report is capped tight). Tagged neither by dataset nor trimmed to
        the final budget here — _plan_worklist does that, across ALL of a
        run's datasets at once, so business context can favor one dataset
        over another instead of every dataset getting an equal slice.

    When `target_col` is given and usable as a grouping variable (few enough
    distinct values — see _MAX_TARGET_GROUPS_FOR_STATS), every OTHER
    numeric/categorical column gets a "target_relationship" candidate
    instead of a plain distribution/categorical_breakdown — this is the
    difference between "here's what column X looks like" (profiling) and
    "here's whether column X actually relates to the outcome" (the point of
    doing EDA for a modeling project at all). The target column itself still
    gets its own plain breakdown/distribution (its base rate is useful
    context on its own).
    """
    cols_by_type: dict[str, list[str]] = {}
    for c in profile["columns"]:
        cols_by_type.setdefault(c["semantic_type"], []).append(c["name"])

    numeric = list(cols_by_type.get("numeric", []))
    categorical = cols_by_type.get("categorical", [])[:]
    datetime_cols = list(cols_by_type.get("datetime", []))
    text_cols = cols_by_type.get("text", [])[:MAX_TEXT_COLS]

    # profiler.classify_column tags any high-cardinality column as "id_like"
    # (unique_ratio > 0.95), which is right for order_id/customer_id but
    # wrongly swallows genuine continuous float measures (e.g. a computed
    # "revenue" column where almost every row is a distinct amount) — those
    # would otherwise get skipped from distribution/correlation entirely.
    # Real ID columns are virtually never floats, so rescue only float dtype.
    for col in cols_by_type.get("id_like", []):
        if pd.api.types.is_float_dtype(df[col]):
            numeric.append(col)
    numeric = numeric[:MAX_NUMERIC_COLS]

    # Rescue string-encoded dates out of "categorical" so they feed timeseries
    # analysis instead of a meaningless top-15-values breakdown.
    still_categorical = []
    for col in categorical:
        if len(datetime_cols) < 5 and _looks_like_datetime(df[col]):
            datetime_cols.append(col)
        else:
            still_categorical.append(col)
    categorical = still_categorical[:MAX_CATEGORICAL_COLS]

    col_stats = profile.get("_col_stats", {})
    target_usable = (
        target_col is not None
        and target_col in col_stats
        and 2 <= col_stats[target_col].get("unique_count", 0) <= _MAX_TARGET_GROUPS_FOR_STATS
    )

    # quality_score's completeness/consistency/uniqueness framing is a generic
    # data-quality rubric, not a business-relevance signal — it competes for
    # budget with everything else instead of being forced, so a run with a
    # real target/business context spends its guaranteed slots on profile +
    # missing (both genuinely foundational for feature-engineering decisions)
    # rather than a fixed score most users don't need repeated per dataset.
    tier0 = [
        _new_item("profile", "Dataset overview & column profile"),
        _new_item("missing", "Missing-value analysis"),
    ]

    optional = [_new_item("quality_score", "Overall data quality score")]
    if len(numeric) >= 2:
        optional.append(_new_item("correlations", "Correlation between numeric columns", {"columns": numeric}))
        optional.append(_new_item("outliers", "Outlier detection across numeric columns", {"columns": numeric}))
    elif len(numeric) == 1:
        optional.append(_new_item("outliers", "Outlier detection", {"columns": numeric}))
    for col in numeric:
        if target_usable and col != target_col:
            optional.append(_new_item(
                "target_relationship", f"`{col}` vs `{target_col}`", {"feature": col, "target": target_col},
            ))
        else:
            optional.append(_new_item("distribution", f"Distribution of `{col}`", {"column": col}))
    for col in categorical:
        if target_usable and col != target_col:
            optional.append(_new_item(
                "target_relationship", f"`{col}` vs `{target_col}`", {"feature": col, "target": target_col},
            ))
        else:
            optional.append(_new_item("categorical_breakdown", f"Breakdown of `{col}`", {"column": col}))
    if datetime_cols and numeric:
        pairs = 0
        for tcol in datetime_cols[:2]:
            for vcol in numeric:
                if pairs >= MAX_DATETIME_PAIRS:
                    break
                optional.append(_new_item("timeseries", f"`{vcol}` over `{tcol}`", {"time_col": tcol, "value_col": vcol}))
                pairs += 1
    for col in text_cols:
        optional.append(_new_item("text_analysis", f"Text analysis of `{col}`", {"column": col}))

    return tier0, optional


def _classify_dataset(
    profile: dict, dataset_name: str, business_context: str | None, provider,
) -> dict:
    """Asks the LLM two things about THIS dataset, in one call:
      - "target": does it contain a column representing the outcome/target
        the business context cares about (e.g. churn, cancellation, renewal
        decision)? This was previously entirely missing: every candidate was
        profiled or broken down in isolation, never tested against an actual
        outcome. Deliberately conservative — most datasets in a multi-dataset
        run won't have one, and this returns None for those rather than
        forcing a guess.
      - "is_reference_table": is this a data dictionary / lookup / metadata
        table describing OTHER datasets' columns (e.g. a "column name ->
        description" table), rather than actual observations to analyze?
        Those don't benefit from per-column breakdowns/distributions — their
        content is context, not a subject of statistical analysis.
    Returns {"target": str|None, "is_reference_table": bool}. Only ever
    detects a target when a business context is given — without one there's
    no basis to know what the "outcome" even is (is_reference_table doesn't
    need one, and still runs without it)."""
    empty = {"target": None, "is_reference_table": False}
    if provider is None:
        return empty

    col_stats = profile.get("_col_stats", {})
    usable = [c for c in profile["columns"] if c["semantic_type"] != "text" and c["unique_count"] >= 2][:80]
    all_cols = [c["name"] for c in profile["columns"]][:80]
    if not usable and not all_cols:
        return empty
    col_desc = "\n".join(
        f"- {c['name']} ({c['semantic_type']}, {c['unique_count']} unique, {c['missing_pct']}% missing)"
        for c in usable
    ) or "\n".join(f"- {c}" for c in all_cols)

    context_block = (
        f"Business context: \"{business_context.strip()[:1200]}\"\n\n" if business_context and business_context.strip() else ""
    )
    target_question = (
        "1. Does this dataset contain ONE column that best represents the outcome/target the business context "
        "cares about (e.g. churn, cancellation, renewal decision, disengagement, retention outcome)? Only pick a "
        "column that is clearly an outcome/label — never an identifier, free-text note, or purely administrative "
        "field. If no business context is given, or none fits, answer null.\n"
        if context_block else
        "1. Leave this null — no business context was given, so there's no basis to identify a target.\n"
    )
    prompt = (
        "You are scoping an automated EDA run.\n\n"
        f"{context_block}"
        f"Dataset \"{dataset_name}\" columns:\n{col_desc}\n\n"
        f"{target_question}"
        "2. Is this dataset itself a data dictionary / lookup / metadata table — e.g. rows describing what OTHER "
        "columns/datasets mean (\"column name\", \"description\") — rather than real observations/records to "
        "analyze statistically?\n\n"
        "Respond with ONLY a JSON object, no markdown fences: "
        '{"target": "<exact column name>" or null, "is_reference_table": true|false}'
    )
    try:
        text = provider.generate(prompt, temperature=_TEMPERATURE, max_tokens=_TARGET_DETECT_MAX_TOKENS)
    except QuotaExceededError:
        return empty
    except Exception as e:
        logger.warning("auto_eda dataset classification failed: %s", e)
        return empty
    if not text:
        return empty
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        return empty
    if not isinstance(parsed, dict):
        return empty

    target = parsed.get("target")
    if not isinstance(target, str) or target not in col_stats or col_stats[target].get("semantic_type") == "constant":
        target = None
    is_reference = bool(parsed.get("is_reference_table"))
    return {"target": target, "is_reference_table": is_reference}


def _describe_candidate(item: dict, col_stats: dict[str, dict], dataset_name: str | None) -> str:
    args = item["args"]
    cols = [args[k] for k in ("column", "time_col", "value_col") if k in args]
    cols += args.get("columns", []) if isinstance(args.get("columns"), list) else []
    bits = []
    for c in cols[:2]:
        s = col_stats.get(c)
        if not s:
            continue
        desc = f"{s['semantic_type']}, {s['missing_pct']}% missing, {s['unique_count']} unique"
        if s.get("std") is not None:
            desc += f", std {s['std']}"
        bits.append(f"{c} ({desc})")
    prefix = f"[{dataset_name}] " if dataset_name else ""
    return prefix + item["title"] + (f" — {'; '.join(bits)}" if bits else "")


def _plan_worklist(
    candidates: list[dict], profiles: dict[int, dict], dataset_names: dict[int, str],
    business_context: str | None, max_items: int, provider,
) -> list[dict]:
    """Selects and prioritizes which optional candidates are actually worth
    running, within budget — across every dataset in the run at once, so a
    business context that clearly favors one dataset over another can
    concentrate the budget there instead of every dataset getting an equal,
    isolated slice. Grounded in each column's real stats; falls back to
    natural order if AI is unavailable or its response can't be trusted —
    this is a prioritization layer on top of a safe, deterministic base,
    never the only thing standing between a hallucination and the report.

    Ordering matters even when nothing needs to be CUT: the report reads in
    execution order, so with a business context given, still ask for a
    priority ranking of the full candidate list — otherwise the highest-
    signal target_relationship items (e.g. the one column that actually
    turns out to matter) end up buried in raw column order rather than
    read first, purely because the budget happened to be generous enough
    that nothing needed trimming. Only skip the LLM call outright when
    there's truly nothing to prioritize by (no context) or nothing to
    reorder (a single candidate)."""
    if not candidates:
        return []
    has_context = bool(business_context and business_context.strip())
    if len(candidates) <= 1 or (len(candidates) <= max_items and not has_context):
        return candidates
    if provider is None:
        return candidates[:max_items]

    multi = len(dataset_names) > 1
    candidate_desc = "\n".join(
        f"{i}. {_describe_candidate(it, profiles[it['dataset_id']].get('_col_stats', {}), dataset_names.get(it['dataset_id']) if multi else None)}"
        for i, it in enumerate(candidates)
    )

    context_block = ""
    priority_hint = (
        "Prioritize high-signal columns: low missingness, meaningful variance/cardinality, and anything likely to "
        "matter for typical business questions (revenue, cost, dates, status/category fields, likely outcome/target "
        "columns) over administrative or near-constant columns. Think like a data scientist scoping feature "
        "engineering: favor columns/relationships that would make good model features or reveal multicollinearity, "
        "over redundant or low-signal ones."
    )
    if business_context and business_context.strip():
        context_block = (
            "The user provided this business context — this EDA exists to support that goal (including, if it "
            "implies building a predictive model later, identifying candidate features, likely target/outcome "
            "columns, and feature-engineering opportunities). Prioritize whatever is most relevant to it, and only "
            f"fall back to general data-quality/variance signals for anything it doesn't cover:\n\"{business_context.strip()[:1500]}\"\n\n"
        )
        priority_hint = (
            "Prioritize whatever is most relevant to the business context above — including relationships between "
            "candidate features and any outcome/target the context implies."
        )
    if multi:
        context_block += (
            "This run spans multiple datasets (shown in [brackets] below) — it's fine, and often correct, to spend "
            "most of the budget on whichever dataset(s) actually matter for the business context rather than "
            "splitting evenly.\n\n"
        )

    prompt = (
        f"You are planning an automated EDA run. There are {len(candidates)} candidate analyses and a hard ceiling "
        f"of {max_items} (a safety cap, not a quota to fill).\n\n"
        f"{context_block}"
        f"Candidates (index: description):\n{candidate_desc}\n\n"
        f"{priority_hint}\n\n"
        f"Respond with ONLY a JSON array of the chosen indices, in priority order (most valuable first), no "
        f"markdown fences, e.g. [3, 0, 7]. Choose up to {max_items} — include every candidate that's genuinely "
        f"worth running, but do not pad the list with low-value or redundant items just to reach the ceiling; "
        f"returning fewer than {max_items} is normal and expected whenever that's all that's actually worthwhile."
    )
    # The output is a JSON array of up to max_items indices (each a few
    # tokens), on top of whatever hidden reasoning the active deployment
    # spends before it — see _PLANNING_MAX_TOKENS's own comment. A fixed
    # budget sized for a ~30-candidate pool silently starves once the pool
    # (and therefore the expected output array) grows into the hundreds —
    # this previously fell back to candidates[:max_items], a raw positional
    # truncation of construction order (profile/missing/quality first, then
    # dataset-by-dataset) rather than a merit-based choice, which is exactly
    # why a later dataset's categorical candidates could vanish entirely
    # while an earlier one's numeric candidates all survived.
    planning_max_tokens = max(_PLANNING_MAX_TOKENS, max_items * 40)
    try:
        text = provider.generate(prompt, temperature=_TEMPERATURE, max_tokens=planning_max_tokens)
    except QuotaExceededError:
        return candidates[:max_items]
    except Exception as e:
        logger.warning("auto_eda planning failed: %s", e)
        return candidates[:max_items]
    if not text:
        logger.warning("auto_eda planning returned empty text (%d candidates, max_items=%d) — falling back to positional truncation", len(candidates), max_items)
        return candidates[:max_items]
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        logger.warning("auto_eda planning returned unparseable JSON (%d candidates) — falling back to positional truncation: %s", len(candidates), text[:500])
        return candidates[:max_items]
    if not isinstance(parsed, list):
        logger.warning("auto_eda planning returned non-list JSON (%d candidates) — falling back to positional truncation", len(candidates))
        return candidates[:max_items]

    chosen: list[dict] = []
    seen: set[int] = set()
    for idx in parsed:
        if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            chosen.append(candidates[idx])
        if len(chosen) >= max_items:
            break
    return chosen if chosen else candidates[:max_items]


def _plan_all_datasets(
    loaded: dict[int, tuple], max_total: int, business_context: str | None, provider, workspace_id: int,
) -> list[dict]:
    """Builds the full worklist across every dataset in the run: tier0
    (profile/missing/quality) is a non-negotiable floor per dataset, and
    everything else is planned globally against whatever budget remains —
    see _plan_worklist. If there are so many datasets that even the tier0
    floor doesn't fit the overall cap, falls back to round-robin (one
    dataset's profile before any dataset's second item) so a huge workspace
    gets broad coverage rather than exhausting the budget on the first few
    datasets alone."""
    multi = len(loaded) > 1
    dataset_names = {did: ds.name for did, (ds, _) in loaded.items()}

    all_tier0: list[dict] = []
    all_optional: list[dict] = []
    profiles: dict[int, dict] = {}
    for did, (ds, df) in loaded.items():
        profile = run_profile(df)
        profile["_col_stats"] = {c["name"]: c for c in profile["columns"]}
        profiles[did] = profile

        classification = _classify_dataset(profile, ds.name, business_context, provider)
        target_col = classification["target"]

        tier0, optional = _enumerate_candidates(df, profile, target_col)
        if classification["is_reference_table"]:
            # A data dictionary / lookup table (e.g. "column name" ->
            # "description") isn't itself a subject of statistical analysis —
            # per-column breakdowns/distributions/correlations of ITS rows
            # are meaningless. Keep only profile+missing so it's still
            # visible in the report, but don't burn optional budget on it.
            optional = []

        # The one guaranteed, business-grounded analysis: if this dataset has
        # a plausible outcome column, test every other feature's relationship
        # to it in one shot (RF importance + correlation/ANOVA + leakage/
        # redundancy checks — see eda/feature_importance.py, already used
        # elsewhere in the product but never previously wired into Auto EDA).
        # Tier0, not optional — this is exactly what turns generic per-column
        # profiling into "what actually predicts/relates to the thing the
        # business cares about", so it should never lose to budget pressure.
        if target_col:
            # Excluding only what the profiler itself tagged "id_like" isn't
            # enough — that tag needs a very high unique-ratio to trigger, so
            # a near-unique column the profiler still calls "categorical"
            # (e.g. a 5,000-category date string, a customer ref with tens
            # of thousands of values) slips through, gets label-encoded, and
            # fed into the RF fit AND the O(k^2) pairwise redundancy check —
            # that combination is what timed out billings' feature_importance
            # (Co_Ref: 47,826 unique; Registration_Date: 5,789; etc.). Cap
            # raw categorical cardinality directly instead of trusting the
            # profiler's own (differently-tuned) id_like threshold.
            exclude = [
                c["name"] for c in profile["columns"]
                if c["semantic_type"] == "text"
                or (c["semantic_type"] == "id_like" and not pd.api.types.is_float_dtype(df[c["name"]]))
                or (c["semantic_type"] == "categorical" and c["unique_count"] > _FEATURE_IMPORTANCE_MAX_CATEGORIES)
            ]
            tier0.append(_new_item(
                "feature_importance",
                f"What drives `{target_col}`? — feature importance & relationships",
                {"target": target_col, "exclude": exclude},
            ))

        for it in tier0 + optional:
            it["dataset_id"] = did
            if multi:
                it["title"] = f"{ds.name}: {it['title']}"
        all_tier0.extend(tier0)
        all_optional.extend(optional)

    # A second, complementary lens on the same "what drives the outcome"
    # question: feature_importance (above) is a fast, deterministic,
    # single-dataset ranking; this is a real agentic investigation (Scout's
    # hypothesis-generation mode, reused as-is — see hypothesis_orchestrator.py)
    # that can run SQL/python across ALL of this workspace's datasets at once
    # (e.g. joining a churn outcome in one dataset against signals recorded in
    # another) and reports each finding as a named, tested claim with a real
    # p-value — the narrative "hypothesis" style this was missing entirely
    # before. Workspace-scoped (dataset_id=None), so it can only run once per
    # run, not once per dataset; tier0 because it's the single most direct
    # answer to "does this actually relate to the target" and shouldn't be at
    # the mercy of budget competition. Only runs when there's a business
    # context to ground it — without one, "what's a good hypothesis" has no
    # answer. Results are also persisted as real Hypothesis rows (see
    # _execute_item), so they show up in the Hypotheses tab too, not just
    # embedded in this report.
    if business_context and business_context.strip() and provider is not None and loaded:
        placeholder_did = next(iter(loaded))
        hyp_item = _new_item(
            "hypothesis_investigation",
            "Tested hypotheses about what drives the business outcome",
            {"count": _HYPOTHESIS_COUNT},
        )
        hyp_item["dataset_id"] = placeholder_did
        all_tier0.append(hyp_item)

    if len(all_tier0) > max_total:
        # Round-robin so every dataset gets at least a shot at basic coverage
        # instead of the first few datasets eating the whole budget.
        by_dataset: dict[int, list[dict]] = {}
        for it in all_tier0:
            by_dataset.setdefault(it["dataset_id"], []).append(it)
        result: list[dict] = []
        idx = 0
        while len(result) < max_total and any(by_dataset.values()):
            for did in list(by_dataset.keys()):
                if by_dataset[did]:
                    result.append(by_dataset[did].pop(0))
                    if len(result) >= max_total:
                        break
            idx += 1
            if idx > max_total:  # safety valve, should never actually trigger
                break
        return result[:max_total]

    remaining_budget = max_total - len(all_tier0)
    selected = _plan_worklist(all_optional, profiles, dataset_names, business_context, remaining_budget, provider)
    return (all_tier0 + selected)[:max_total]


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


_CUSTOM_TABLE_MAX_ROWS = 20


def _fmt_cell(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v) if v is not None else ""


def _render_custom_result(result: Any) -> str | None:
    """A custom_python check's `result` can be any JSON-serializable shape.
    A raw JSON dump reads badly in a report (and gets truncated for long
    ones) — render the common shapes as real tables instead, and drop
    anything else entirely (the AI caption already explains the finding
    in prose, so there's nothing worth showing here)."""
    if isinstance(result, list) and result and all(isinstance(r, dict) for r in result):
        headers = list(result[0].keys())
        rows = [[_fmt_cell(r.get(h)) for h in headers] for r in result[:_CUSTOM_TABLE_MAX_ROWS]]
        table = _md_table(headers, rows)
        if len(result) > _CUSTOM_TABLE_MAX_ROWS:
            table += f"\n\n_(showing {_CUSTOM_TABLE_MAX_ROWS} of {len(result)} rows)_"
        return table

    if isinstance(result, dict) and result and not any(isinstance(v, (list, dict)) for v in result.values()):
        rows = [[k, _fmt_cell(v)] for k, v in result.items()]
        return _md_table(["Field", "Value"], rows)

    return None


def _execute_item(
    item: dict, df: pd.DataFrame, *, workspace_id: int | None = None, db: Session | None = None,
    user: User | None = None, business_context: str | None = None,
) -> tuple[dict, str | None]:
    """Returns (result_dict, markdown_body). markdown_body already includes
    any embedded chart image — the caller appends the AI caption after it.
    workspace_id/db/user/business_context are only used by kinds that need
    more than the one dataset's own dataframe (currently just
    "hypothesis_investigation", which runs cross-dataset SQL/python)."""
    kind, args = item["kind"], item["args"]

    if kind == "profile":
        result = run_profile(df)
        rows = [[c["name"], c["semantic_type"], c["dtype"], f"{c['missing_pct']}%", c["unique_count"]] for c in result["columns"]]
        body = f"**{result['total_rows']:,} rows × {result['total_columns']} columns** · {result['duplicate_count']} duplicate rows ({result['duplicate_pct']}%)\n\n"
        body += _md_table(["Column", "Type", "Dtype", "Missing %", "Unique"], rows)
        return result, body

    if kind == "missing":
        result = run_missing_analysis(df)
        cols = result.get("columns", [])
        if cols:
            img = cr.render_bar([c["name"] for c in cols], [c["pct"] for c in cols], "Missing % by column", "% missing")
            return result, f"![Missing values]({img})"
        return result, "No missing values found."

    if kind == "quality_score":
        result = run_quality_score(df)
        img = cr.render_bar(
            ["Completeness", "Consistency", "Uniqueness"],
            [result.get("completeness", 0), result.get("consistency", 0), result.get("uniqueness", 0)],
            f"Quality score: {result.get('overall', 0)}/100", "score",
        )
        return result, f"![Quality score]({img})"

    if kind == "correlations":
        cols = [c for c in args["columns"] if c in df.columns]
        result = compute_num_matrix(df, cols, "pearson")
        matrix = [[result["matrix"][a].get(b) for b in cols] for a in cols]
        img = cr.render_heatmap(cols, matrix, "Correlation matrix (Pearson)")
        return result, f"![Correlation matrix]({img})"

    if kind == "outliers":
        result = run_outlier_detection(df, method="iqr")
        summary = result.get("columns", [])
        if summary:
            names = [c["name"] for c in summary]
            pcts = [c["outlier_pct"] for c in summary]
            img = cr.render_bar(names, pcts, "Outlier % by column (IQR)", "% outliers")
            return result, f"![Outliers]({img})"
        return result, "No numeric columns with detectable outliers."

    if kind == "distribution":
        col = args["column"]
        result = run_distribution(df, col)
        values = df[col].dropna().tolist()
        img = cr.render_histogram(values, f"Distribution of {col}", col)
        return result, f"![Distribution of {col}]({img})"

    if kind == "categorical_breakdown":
        col = args["column"]
        vc = df[col].value_counts().head(15)
        result = {"column": col, "top_values": [{"value": str(k), "count": int(v)} for k, v in vc.items()]}
        img = cr.render_bar([str(k) for k in vc.index], [int(v) for v in vc.values], f"Top values of {col}", "count")
        return result, f"![{col} breakdown]({img})"

    if kind == "timeseries":
        time_col, value_col = args["time_col"], args["value_col"]
        result = run_timeseries(df, time_col, value_col)
        line = result.get("line_data") or {}
        dates, values = line.get("dates"), line.get("values")
        if dates and values:
            img = cr.render_line(dates, values, f"{value_col} over {time_col}", value_col)
            return result, f"![{value_col} over time]({img})"
        return result, "Could not build a plottable time series (check for a valid date column)."

    if kind == "text_analysis":
        col = args["column"]
        result = run_text_analysis(df, col)
        words = result.get("word_freq", [])[:15]
        if words:
            img = cr.render_bar([w["word"] for w in words], [w["count"] for w in words], f"Top words in {col}", "count")
            return result, f"![{col} word frequency]({img})"
        return result, "No usable text found in this column."

    if kind == "target_relationship":
        feature, target = args["feature"], args["target"]
        if feature not in df.columns or target not in df.columns:
            return {"error": "column not found"}, None

        feature_is_numeric = pd.api.types.is_numeric_dtype(df[feature])
        test = "anova" if feature_is_numeric else "chi2"
        stat = run_statistical_test(df, test=test, column=feature, group_column=target)
        if "error" in stat:
            return stat, f"_Could not test `{feature}` against `{target}`: {stat['error']}_"

        if feature_is_numeric:
            means = df.groupby(target)[feature].mean().round(3).sort_index()
            table = _md_table([target, f"mean {feature}"], [[str(k), v] for k, v in means.items()])
            img = cr.render_bar([str(k) for k in means.index], [float(v) for v in means.values], f"Mean {feature} by {target}", feature)
        else:
            top_cats = df[feature].value_counts().head(10).index
            ct = (pd.crosstab(df[feature], df[target], normalize="index") * 100).round(1)
            ct = ct.loc[ct.index.intersection(top_cats)]
            table = _md_table([feature] + [str(c) for c in ct.columns], [[str(idx)] + [f"{v}%" for v in row] for idx, row in ct.iterrows()])
            img = None

        body = (
            f"**`{feature}` vs `{target}`** — {stat['test'].upper()}: statistic={stat['statistic']:.3f}, "
            f"p={stat['p_value']:.4f} → {stat['interpretation']}\n\n{table}"
        )
        if img:
            body += f"\n\n![{feature} by {target}]({img})"
        return stat, body

    if kind == "feature_importance":
        target = args["target"]
        if target not in df.columns:
            return {"error": f"target column not found: {target}"}, None
        exclude = [c for c in args.get("exclude", []) if c in df.columns and c != target]
        feature_df = df.drop(columns=exclude) if exclude else df
        try:
            result = run_isolated(
                run_feature_importance, feature_df, target,
                methods=_FEATURE_IMPORTANCE_METHODS, timeout=_FEATURE_IMPORTANCE_TIMEOUT_S,
            )
        except (AnalysisTimeout, AnalysisCrashed) as e:
            return {"error": str(e)}, f"_Feature importance analysis for `{target}` timed out or crashed: {e}_"
        if result.get("error"):
            return result, f"_Could not compute feature importance for `{target}`: {result['error']}_"

        meta = result.get("feature_meta", [])[:15]
        rows = [
            [m["feature"], m.get("rf_importance"), m.get("correlation"), m.get("anova_f"),
             f"{m.get('missing_pct', 0)}%", m["recommendation"]]
            for m in meta
        ]
        body = (
            f"**Target: `{target}`** · problem type: {result.get('problem_type')} · "
            f"model OOB score: {result.get('model_score')}\n\n"
        )
        body += _md_table(
            ["Feature", "RF importance", "Correlation/Eta", "ANOVA F", "Missing %", "Recommendation"], rows,
        )
        if result.get("leakage_suspects"):
            names = "; ".join(f"`{s['feature']}`" for s in result["leakage_suspects"])
            body += f"\n\n**Possible target leakage — verify these would exist before the outcome does:** {names}"
        top = [m["feature"] for m in meta[:10]]
        imps = [m.get("rf_importance") or 0 for m in meta[:10]]
        img = cr.render_bar(top, imps, f"Top features for {target}", "RF importance") if top else None
        body += f"\n\n![Feature importance]({img})" if img else ""

        # Compact result for the caption/follow-up LLM calls — the full
        # feature_meta/correlations/anova lists are already rendered as a
        # table above, so captioning from the full dict would waste most of
        # its truncated budget re-describing rows already visible in `body`.
        compact = {
            "target": target,
            "problem_type": result.get("problem_type"),
            "n_samples": result.get("n_samples"),
            "model_score": result.get("model_score"),
            "cv_score_mean": result.get("cv_score_mean"),
            "top_features": result.get("top_features"),
            "drop_candidates": result.get("drop_candidates"),
            "leakage_suspects": result.get("leakage_suspects"),
            "warnings": result.get("warnings"),
            "top_feature_detail": meta,
        }
        return compact, body

    if kind == "hypothesis_investigation":
        count = args.get("count", _HYPOTHESIS_COUNT)
        result = run_hypothesis_generation(
            workspace_id=workspace_id, dataset_id=None, count=count, db=db, user=user,
            business_context=business_context,
        )
        if result.get("error"):
            return result, f"_Hypothesis investigation failed: {result['error']}_"
        hyps = result.get("hypotheses", [])
        if not hyps:
            return result, "No testable hypotheses were found."

        # Persist as real Hypothesis rows too — same table the Hypotheses tab
        # reads from, so these tested findings show up there, not only here.
        for h in hyps:
            if not isinstance(h, dict) or not h.get("statement"):
                continue
            db.add(Hypothesis(
                workspace_id=workspace_id, dataset_id=None, created_by=None, origin="ai",
                title=h.get("title"), statement=h.get("statement", ""), category=h.get("category"),
                status=h.get("status", "supported"), verdict=h.get("verdict"),
                evidence_summary=h.get("evidence_summary"), confidence=h.get("confidence"),
                severity=h.get("severity"), columns_json=json.dumps(h.get("columns", [])),
                tool_trace_json=json.dumps(result.get("tool_trace", []), default=str),
            ))
        db.commit()

        lines = []
        for h in hyps:
            if not isinstance(h, dict):
                continue
            status = h.get("status")
            icon = "✅" if status == "supported" else "❌" if status == "refuted" else "❓"
            lines.append(
                f"**{icon} {h.get('title') or 'Untitled'}** — {h.get('statement', '')}\n\n"
                f"{h.get('verdict', '')} _{h.get('evidence_summary', '')}_"
            )
        body = "\n\n".join(lines)
        summary = {
            "count": len(hyps),
            "supported": sum(1 for h in hyps if isinstance(h, dict) and h.get("status") == "supported"),
            "refuted": sum(1 for h in hyps if isinstance(h, dict) and h.get("status") == "refuted"),
            "titles": [h.get("title") for h in hyps if isinstance(h, dict)],
        }
        return summary, body

    if kind == "custom_python":
        code = args.get("code", "")
        out = exec_sandboxed(df, code)
        if "error" in out:
            return out, f"_Custom check failed: {out['error']}_"
        return out, _render_custom_result(out.get("result"))

    return {"error": f"Unknown kind {kind}"}, None


def _caption_prompt(item: dict, result: dict, business_context: str | None) -> str:
    trimmed = json.dumps(result, default=str)[:4000]
    context_line = f"\nBusiness context to keep in mind: \"{business_context.strip()[:1000]}\"\n" if business_context and business_context.strip() else ""
    return (
        f"You are writing one short paragraph for an automated EDA report. "
        f"The section is: \"{item['title']}\" (analysis type: {item['kind']}).\n"
        f"{context_line}\n"
        f"Here is the ACTUAL computed result — ground your caption in these exact numbers, "
        f"never invent a number that isn't here:\n{trimmed}\n\n"
        "Write 2-4 plain-English sentences interpreting this for a business reader"
        + (", tying it back to the business context where relevant" if context_line else "") + ". "
        + (
            "If genuinely relevant, you may briefly note whether this looks like a useful model feature or flags "
            "a modeling concern (e.g. needs encoding, a transform for skew, or risks multicollinearity) — but only "
            "when it's actually worth saying, not as a rote add-on. "
            if context_line else ""
        )
        + "No markdown headers, no restating the raw JSON — just the interpretation. "
        "If the result shows nothing noteworthy, say so briefly rather than padding."
    )


def _caption_for(item: dict, result: dict, provider, business_context: str | None = None) -> str:
    if provider is None:
        return ""
    try:
        text = provider.generate(_caption_prompt(item, result, business_context), temperature=_TEMPERATURE, max_tokens=_CAPTION_MAX_TOKENS)
        return (text or "").strip()
    except QuotaExceededError:
        return "_(AI caption unavailable — quota exceeded)_"
    except Exception as e:
        logger.warning("auto_eda caption failed: %s", e)
        return ""


_FOLLOWUP_SCHEMA_HINT = (
    'Respond with ONLY a JSON array (no markdown fences), 0 to 2 items, each shaped:\n'
    '{"kind": "correlations"|"distribution"|"outliers"|"categorical_breakdown"|"timeseries"|"text_analysis"|'
    '"feature_importance"|"custom_python", "title": "<short title>", "args": {...}} — for '
    '"distribution"/"categorical_breakdown"/"text_analysis" args is {"column": "<existing column name>"}; for '
    '"timeseries" args is {"time_col": "...", "value_col": "..."}; for "correlations"/"outliers" args is '
    '{"columns": ["...", "..."]}; for "feature_importance" args is {"target": "<existing column name>"} — use this '
    'when a finding suggests a specific column is (or might be) the outcome/target worth explaining, to rank every '
    'other feature\'s relationship to it; for "custom_python" args is {"code": "<python using df, pd, np — assign '
    'to result>"}. '
    "Only propose a follow-up if this specific finding genuinely warrants deeper investigation — "
    "an empty array is a completely valid answer, and is expected most of the time. When a business context is "
    "given, a good follow-up often examines a feature's relationship to whatever outcome/target it implies, not "
    "just the column in isolation."
)


def _suggest_followups(item: dict, result: dict, df_columns: list[str], provider, business_context: str | None = None) -> list[dict]:
    if provider is None:
        return []
    trimmed = json.dumps(result, default=str)[:3000]
    context_line = f"Business context to keep in mind: \"{business_context.strip()[:1000]}\"\n\n" if business_context and business_context.strip() else ""
    prompt = (
        f"You just completed this EDA step: \"{item['title']}\" ({item['kind']}).\n"
        f"Result: {trimmed}\n\n"
        f"{context_line}"
        f"Available columns in this dataset: {df_columns}\n\n"
        f"{_FOLLOWUP_SCHEMA_HINT}"
    )
    try:
        text = provider.generate(prompt, temperature=_TEMPERATURE, max_tokens=_FOLLOWUP_MAX_TOKENS)
    except QuotaExceededError:
        return []
    except Exception as e:
        logger.warning("auto_eda followup suggestion failed: %s", e)
        return []
    if not text:
        return []
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    valid: list[dict] = []
    for p in parsed[:MAX_FOLLOWUPS_PER_ITEM]:
        if not isinstance(p, dict) or p.get("kind") not in _VALID_KINDS:
            continue
        args = p.get("args") or {}
        cols_named = [v for k, v in args.items() if k in _COLUMN_ARG_KEYS]
        cols_named += args.get("columns", []) if isinstance(args.get("columns"), list) else []
        if any(c not in df_columns for c in cols_named):
            continue  # reject anything naming a column that doesn't actually exist
        valid.append(_new_item(p["kind"], p.get("title", p["kind"]), args))
    return valid


_STEERING_SCHEMA_HINT = (
    'Respond with ONLY a JSON object (no markdown fences), shaped:\n'
    '{"remove": [<worklist index>, ...], "add": [{"kind": "...", "title": "...", "args": {...}, "dataset_id": <id>}, ...], '
    '"reply": "<one short sentence telling the user what you changed, or why you didn\'t>"}\n'
    'kind must be one of: profile, missing, quality_score, correlations, distribution, outliers, '
    'categorical_breakdown, timeseries, text_analysis, feature_importance, custom_python. args follows the same '
    'shape as a follow-up: distribution/categorical_breakdown/text_analysis -> {"column": "..."}; '
    'timeseries -> {"time_col": "...", "value_col": "..."}; correlations/outliers -> {"columns": [...]}; '
    'feature_importance -> {"target": "<existing column name>"} (ranks every other feature\'s relationship to that '
    'target); custom_python -> {"code": "<python using df, pd, np — assign to result>"}. '
    'dataset_id must be one of the dataset ids listed below. '
    '"remove" and "add" may both be empty — e.g. if the message is just a question rather than an instruction — '
    "but always include a short \"reply\"."
)


def _apply_steering(
    worklist: list[dict], instruction: str, loaded: dict, provider, *,
    max_total: int | None = None, findings_context: str | None = None,
) -> tuple[list[dict], str]:
    """Lets a user redirect a still-running pipeline mid-flight — e.g. "skip
    the categorical breakdowns" or "focus more on revenue". Removes/adds are
    validated the same way follow-ups are (real kind, real columns, real
    dataset) so a bad AI response can't corrupt the worklist; "remove" only
    ever touches items still "pending" (never done/running/already-skipped).

    Also reused (not just for user chat instructions) by _self_review as the
    periodic "reconsider the whole remaining plan" checkpoint — that's why
    `findings_context` (a digest of what's been found so far, so redundancy/
    new-value judgments are grounded in actual results, not just titles) and
    `max_total` (so additions don't silently blow past the run's own budget
    ceiling) are both optional: a plain user instruction doesn't need either.
    """
    if provider is None:
        return worklist, "AI isn't configured, so I can't act on that right now."

    pending_items = [(i, it) for i, it in enumerate(worklist) if it["status"] == "pending"]
    datasets_info = {did: (ds.name, list(df.columns)) for did, (ds, df) in loaded.items()}
    pending_desc = "\n".join(f"- index {i}: [dataset {it['dataset_id']}] {it['title']} (kind: {it['kind']})" for i, it in pending_items)
    datasets_desc = "\n".join(f"- dataset {did} \"{name}\": columns {cols}" for did, (name, cols) in datasets_info.items())
    findings_block = f"Findings so far (recent excerpt of the growing report):\n{findings_context[-4000:]}\n\n" if findings_context else ""
    prompt = (
        f"You are steering an in-progress automated EDA run based on a user instruction.\n\n"
        f"User instruction: \"{instruction}\"\n\n"
        f"{findings_block}"
        f"Datasets in this run:\n{datasets_desc}\n\n"
        f"Remaining (not-yet-executed) worklist items:\n{pending_desc or '(none remaining)'}\n\n"
        f"{_STEERING_SCHEMA_HINT}"
    )

    try:
        text = provider.generate(prompt, temperature=_TEMPERATURE, max_tokens=_FOLLOWUP_MAX_TOKENS)
    except QuotaExceededError:
        return worklist, "AI quota exceeded — couldn't act on that."
    except Exception as e:
        logger.warning("auto_eda steering failed: %s", e)
        return worklist, "Something went wrong applying that — no changes made."

    if not text:
        return worklist, "No response from the AI — no changes made."
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        return worklist, "Couldn't understand that response — no changes made."
    if not isinstance(parsed, dict):
        return worklist, "Unexpected response — no changes made."

    reply = str(parsed.get("reply") or "Done.")

    remove_idx = {
        idx for idx in (parsed.get("remove") or [])
        if isinstance(idx, int) and 0 <= idx < len(worklist) and worklist[idx]["status"] == "pending"
    }
    for idx in remove_idx:
        worklist[idx]["status"] = "skipped"

    room = (max_total - len(worklist)) if max_total is not None else None
    existing_keys = {_dedup_key(it) for it in worklist}
    for p in (parsed.get("add") or [])[:MAX_FOLLOWUPS_PER_ITEM * 3]:
        if room is not None and room <= 0:
            break
        if not isinstance(p, dict) or p.get("kind") not in _VALID_KINDS:
            continue
        did = p.get("dataset_id")
        if did not in datasets_info:
            continue
        _, cols = datasets_info[did]
        args = p.get("args") or {}
        cols_named = [v for k, v in args.items() if k in _COLUMN_ARG_KEYS]
        cols_named += args.get("columns", []) if isinstance(args.get("columns"), list) else []
        if any(c not in cols for c in cols_named):
            continue
        item = _new_item(p["kind"], p.get("title", p["kind"]), args)
        item["dataset_id"] = did
        key = _dedup_key(item)
        if key in existing_keys:
            continue  # same kind+target/column already planned (done, pending, or even errored) — pure duplicate
        existing_keys.add(key)
        worklist.append(item)
        if room is not None:
            room -= 1

    return worklist, reply


_SELF_REVIEW_INSTRUCTION = (
    "This is an automatic checkpoint, not a message from the user — periodically reconsider the remaining plan "
    "in light of what's actually been found so far. Remove any pending item that now looks redundant, low-value, "
    "or unlikely to add anything new (e.g. several similar low-signal categorical fields already tested with no "
    "significant relationship to the target, a repeat of an analysis type whose earlier instances all came back "
    "unremarkable, or a comparison that's close to tautological — like a derived column plotted against the very "
    "column it was derived from). Add any new item a SPECIFIC finding below genuinely warrants investigating "
    "further. If nothing needs to change, that is a completely valid answer — leave both remove and add empty, "
    "and say so briefly in \"reply\". Be conservative: only remove items you're genuinely confident add little, "
    "and only add items clearly grounded in an actual finding, not a hunch."
)


def _self_review(
    worklist: list[dict], loaded: dict, provider, markdown: str, max_total: int,
) -> tuple[list[dict], str | None]:
    """Periodic "does this plan still make sense" checkpoint, run automatically
    every _SELF_REVIEW_INTERVAL completed items (see run_auto_eda_stream) —
    not just reactive to a user chat message. Reuses _apply_steering's exact
    remove/add validation (same safety guarantees: only touches still-pending
    items, only adds real kinds/columns/datasets, never exceeds the run's own
    budget) with the growing report itself as grounding context, so
    "redundant" and "worth adding" are judged against actual results instead
    of guesswork. Returns (worklist, reply_or_None) — None means nothing
    changed, so the caller can skip posting a no-op chat message."""
    if provider is None:
        return worklist, None
    # _apply_steering mutates `worklist` in place and returns that same
    # object, so "did anything change" has to be judged against a snapshot
    # taken before the call — comparing the return value against `worklist`
    # afterward would just compare the mutated list against itself.
    before_len = len(worklist)
    before_statuses = [w["status"] for w in worklist]
    new_worklist, reply = _apply_steering(
        worklist, _SELF_REVIEW_INSTRUCTION, loaded, provider,
        max_total=max_total, findings_context=markdown,
    )
    changed = len(new_worklist) != before_len or [w["status"] for w in new_worklist[:before_len]] != before_statuses
    return new_worklist, (reply if changed else None)


def _persist(db: Session, run_row, worklist: list[dict], markdown: str, status: str, error: str | None = None, title: str | None = None):
    run_row.worklist_json = json.dumps(worklist)
    run_row.markdown = markdown
    run_row.status = status
    if title:
        run_row.title = title
    if error:
        run_row.error = error
    db.add(run_row)
    db.commit()


def run_auto_eda_stream(
    *, workspace_id: int, dataset_ids: list[int], db: Session, user: User, run_row, resume: bool = False,
    business_context: str | None = None, require_approval: bool = True, report_title: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yields progress events:
      {"type": "worklist", "worklist": [...]}
      {"type": "planned", "worklist": [...]}
      {"type": "item_start", "index": int, "item": {...}}
      {"type": "item_done", "index": int, "item": {...}, "markdown_chunk": str}
      {"type": "paused"}
      {"type": "error", "message": str}
      {"type": "done", "markdown": str}
    Persists `run_row` (an AutoEdaRun) after every item so a reload mid-run
    shows real progress, not just the final state.

    With require_approval=True (the default), a first-time (non-resume) call
    stops right after planning: it persists status="planned" and returns
    WITHOUT executing anything — a human reviews (and can steer via chat,
    see routers/auto_eda.py's chat endpoint) the proposed worklist, then a
    separate call to routers/auto_eda.py's /approve endpoint flips status to
    "running" and calls this function again with resume=True, which picks
    up execution from the first still-"pending" item — i.e. everything,
    since nothing ran yet. Approving and resuming-from-pause are the exact
    same continuation mechanism; only how status got to "running" differs.

    A run can cover several datasets at once (e.g. "run on the whole
    workspace") — each is loaded and profiled independently, seeded into
    one combined worklist tagged with which dataset each item belongs to,
    with each dataset getting a fair, even share of the overall item budget
    (MAX_TOTAL_ITEMS_CEILING — a hard guardrail regardless of dataset count).

    Pass resume=True to continue an existing paused run: `run_row`'s
    already-persisted worklist/markdown are picked up where they left off
    instead of seeding fresh. Before each item, this re-reads run_row's own
    status from the DB — a separate request can flip it to "pausing" (see
    routers/auto_eda.py's pause endpoint) to have this loop stop cleanly
    after whichever item is currently in flight, rather than mid-computation.
    """
    from ...routers.eda import _load_df, _get_authorized_dataset

    provider = get_provider()

    loaded: dict[int, tuple] = {}
    try:
        for did in dataset_ids:
            ds = _get_authorized_dataset(did, user, db)
            df = _load_df(ds, row_limit=MAX_ROWS)
            loaded[did] = (ds, df)
    except Exception as e:
        yield {"type": "error", "message": f"Could not load dataset: {e}"}
        _persist(db, run_row, [], "", "error", error=str(e))
        return

    from ...config import settings

    multi = len(loaded) > 1
    items_ceiling = getattr(settings, "AUTO_EDA_MAX_ITEMS", MAX_TOTAL_ITEMS_CEILING)
    max_total = min(MAX_TOTAL_ITEMS * len(loaded), items_ceiling)

    if resume and run_row.worklist_json:
        worklist = json.loads(run_row.worklist_json)
        markdown = run_row.markdown or ""
        for it in worklist:
            if it["status"] == "running":  # was mid-flight when the pause landed
                it["status"] = "pending"
        yield {"type": "worklist", "worklist": worklist}
    else:
        worklist = _plan_all_datasets(loaded, max_total, business_context, provider, workspace_id)

        # Title: an explicit report_title wins; otherwise the workspace's own
        # name (stable, always available) rather than the dataset name(s) —
        # a run covering several datasets shouldn't be titled after whichever
        # one happened to load first, and the workspace is what the user
        # actually thinks of this project as.
        from ...models.workspace import Workspace
        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        base_name = (report_title or "").strip() or (ws.name if ws else None) or "Workspace"
        title = f"Automated EDA — {base_name}"
        scope_note = f" across {len(loaded)} datasets" if multi else ""
        context_note = f"\n\n> **Business context:** {business_context.strip()}" if business_context and business_context.strip() else ""
        markdown = f"# {title}\n\n_Generated automatically{scope_note} — {len(worklist)} planned investigations, more may be added as findings emerge._{context_note}\n"

        if require_approval:
            _persist(db, run_row, worklist, markdown, "planned", title=title)
            yield {"type": "planned", "worklist": worklist}
            return

        _persist(db, run_row, worklist, markdown, "running", title=title)
        yield {"type": "worklist", "worklist": worklist}

    i = next((idx for idx, it in enumerate(worklist) if it["status"] == "pending"), len(worklist))
    completed_count = 0
    self_review_count = 0
    while i < len(worklist) and i < max_total:
        db.refresh(run_row)
        if run_row.status == "pausing":
            _persist(db, run_row, worklist, markdown, "paused")
            yield {"type": "paused"}
            return

        # Steering: a user can post a chat message ("skip the categorical
        # breakdowns", "focus more on revenue") while this run is live —
        # apply any not-yet-handled ones before touching the next item.
        new_messages = (
            db.query(AutoEdaChatMessage)
            .filter(AutoEdaChatMessage.run_id == run_row.id, AutoEdaChatMessage.role == "user", AutoEdaChatMessage.applied.is_(False))
            .order_by(AutoEdaChatMessage.created_at)
            .all()
        )
        for msg in new_messages:
            worklist, reply = _apply_steering(worklist, msg.content, loaded, provider)
            msg.applied = True
            db.add(msg)
            db.add(AutoEdaChatMessage(run_id=run_row.id, role="assistant", content=reply, applied=True))
            db.commit()
            _persist(db, run_row, worklist, markdown, "running")
            yield {"type": "worklist", "worklist": worklist}
            yield {"type": "chat", "role": "assistant", "content": reply}

        item = worklist[i]
        if item["status"] != "pending":  # skipped by steering, or already handled
            i += 1
            continue
        ds, df = loaded[item["dataset_id"]]
        df_columns = list(df.columns)

        item["status"] = "running"
        yield {"type": "item_start", "index": i, "item": item}

        try:
            result, body = _execute_item(
                item, df, workspace_id=workspace_id, db=db, user=user, business_context=business_context,
            )
        except Exception as e:
            item["status"] = "error"
            markdown += f"\n\n## {item['title']}\n\n_This check failed: {e}_\n"
            _persist(db, run_row, worklist, markdown, "running")
            yield {"type": "item_done", "index": i, "item": item, "markdown_chunk": ""}
            i += 1
            continue

        caption = _caption_for(item, result, provider, business_context)
        item["status"] = "done"

        chunk = f"\n\n## {item['title']}\n\n"
        if body:
            chunk += body + "\n\n"
        if caption:
            chunk += caption + "\n"
        markdown += chunk

        followups = []
        if len(worklist) < max_total:
            followups = _suggest_followups(item, result, df_columns, provider, business_context)
            for f in followups:
                f["dataset_id"] = item["dataset_id"]
                if multi:
                    f["title"] = f"{ds.name}: {f['title']}"
            # Same analysis, freshly worded, proposed independently by many
            # different items (e.g. every "X vs target" result separately
            # concluding "let's dig into the target's drivers") is a pure
            # duplicate, not a new investigation — see _dedup_key.
            existing_keys = {_dedup_key(it) for it in worklist}
            deduped = []
            for f in followups:
                key = _dedup_key(f)
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                deduped.append(f)
            followups = deduped
            room = max_total - len(worklist)
            worklist.extend(followups[:room])

        _persist(db, run_row, worklist, markdown, "running")
        yield {"type": "item_done", "index": i, "item": item, "markdown_chunk": chunk}
        if followups:
            yield {"type": "worklist", "worklist": worklist}

        completed_count += 1
        # Periodic self-review: every _SELF_REVIEW_INTERVAL completed items,
        # step back and reconsider the WHOLE remaining plan (not just
        # append-only follow-ups off the one item that just finished) —
        # prune anything now looking redundant given accumulated findings,
        # add anything a real finding genuinely warrants. Capped separately
        # from the interval so a very long run can't rack up unbounded extra
        # planning calls.
        if (
            completed_count % _SELF_REVIEW_INTERVAL == 0
            and self_review_count < _SELF_REVIEW_MAX_CALLS
            and provider is not None
            and any(it["status"] == "pending" for it in worklist)
        ):
            self_review_count += 1
            worklist, review_reply = _self_review(worklist, loaded, provider, markdown, max_total)
            _persist(db, run_row, worklist, markdown, "running")
            yield {"type": "worklist", "worklist": worklist}
            if review_reply:
                db.add(AutoEdaChatMessage(run_id=run_row.id, role="assistant", content=f"Self-review: {review_reply}", applied=True))
                db.commit()
                yield {"type": "chat", "role": "assistant", "content": f"Self-review: {review_reply}"}

        i += 1

    _persist(db, run_row, worklist, markdown, "completed")
    yield {"type": "done", "markdown": markdown}
