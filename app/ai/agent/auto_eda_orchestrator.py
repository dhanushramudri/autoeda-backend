"""Auto EDA: an autonomous, agentic exploratory-data-analysis run.

Given a dataset, this seeds a worklist of investigations, executes them one
by one against the REAL data using the same compute functions the rest of
the product already uses (profiler/correlations/distributions/outliers/
missing/quality_score/timeseries/text_analysis — no LLM guessing for the
math itself), renders each result as a chart image (app/eda/chart_render.py)
or a Markdown table, asks the LLM for a short grounded caption, and asks it
whether this specific finding warrants adding follow-up items to the
worklist — which can genuinely grow the queue mid-run, capped for safety.

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
from ...models.auto_eda import AutoEdaChatMessage
from ...models.user import User

logger = logging.getLogger("autoeda.ai.agent.auto_eda")

MAX_ROWS = 200_000  # keeps a full autonomous run fast — not the per-analysis max a human picks manually
MAX_TOTAL_ITEMS = 30  # per dataset, before the overall ceiling below applies
# Guardrail: a run — single- or multi-dataset — never plans more than this
# many items total. Fallback default only — settings.AUTO_EDA_MAX_ITEMS
# (env var, raised for production) is the actual value used at runtime.
MAX_TOTAL_ITEMS_CEILING = 30
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
_TEMPERATURE = 0.2

_VALID_KINDS = {
    "profile", "missing", "quality_score", "correlations",
    "distribution", "outliers", "categorical_breakdown",
    "timeseries", "text_analysis", "custom_python",
}


def _new_item(kind: str, title: str, args: dict | None = None) -> dict:
    return {"kind": kind, "title": title, "args": args or {}, "status": "pending"}


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


def _enumerate_candidates(df: pd.DataFrame, profile: dict) -> tuple[list[dict], list[dict]]:
    """Every analysis this dataset actually supports — deterministic, no
    LLM involved, so nothing here can reference a column that doesn't
    exist. Split into:
      - tier0: profile/missing/quality only — always run, non-negotiable.
        Cheap, structural, not worth spending the LLM's judgment on.
      - optional: everything else, INCLUDING correlations/outliers (they're
        high-value but not free, so they compete for budget like anything
        else rather than being silently force-included even when the
        report is capped tight). Tagged neither by dataset nor trimmed to
        the final budget here — _plan_worklist does that, across ALL of a
        run's datasets at once, so business context can favor one dataset
        over another instead of every dataset getting an equal slice.
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

    tier0 = [
        _new_item("profile", "Dataset overview & column profile"),
        _new_item("missing", "Missing-value analysis"),
        _new_item("quality_score", "Overall data quality score"),
    ]

    optional = []
    if len(numeric) >= 2:
        optional.append(_new_item("correlations", "Correlation between numeric columns", {"columns": numeric}))
        optional.append(_new_item("outliers", "Outlier detection across numeric columns", {"columns": numeric}))
    elif len(numeric) == 1:
        optional.append(_new_item("outliers", "Outlier detection", {"columns": numeric}))
    for col in numeric:
        optional.append(_new_item("distribution", f"Distribution of `{col}`", {"column": col}))
    for col in categorical:
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
    never the only thing standing between a hallucination and the report."""
    if not candidates:
        return []
    if len(candidates) <= max_items:
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
        f"You are planning an automated EDA run. There are {len(candidates)} candidate analyses but only budget "
        f"for {max_items} of them.\n\n"
        f"{context_block}"
        f"Candidates (index: description):\n{candidate_desc}\n\n"
        f"{priority_hint}\n\n"
        f"Respond with ONLY a JSON array of the chosen indices, in priority order (most valuable first), no "
        f"markdown fences, e.g. [3, 0, 7]. Choose exactly {max_items} (fewer only if genuinely fewer are worthwhile)."
    )
    try:
        text = provider.generate(prompt, temperature=_TEMPERATURE, max_tokens=_PLANNING_MAX_TOKENS)
    except QuotaExceededError:
        return candidates[:max_items]
    except Exception as e:
        logger.warning("auto_eda planning failed: %s", e)
        return candidates[:max_items]
    if not text:
        return candidates[:max_items]
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        return candidates[:max_items]
    if not isinstance(parsed, list):
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
    loaded: dict[int, tuple], max_total: int, business_context: str | None, provider,
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
        tier0, optional = _enumerate_candidates(df, profile)
        for it in tier0 + optional:
            it["dataset_id"] = did
            if multi:
                it["title"] = f"{ds.name}: {it['title']}"
        all_tier0.extend(tier0)
        all_optional.extend(optional)

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


def _execute_item(item: dict, df: pd.DataFrame) -> tuple[dict, str | None]:
    """Returns (result_dict, markdown_body). markdown_body already includes
    any embedded chart image — the caller appends the AI caption after it."""
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
    '{"kind": "correlations"|"distribution"|"outliers"|"categorical_breakdown"|"timeseries"|"text_analysis"|"custom_python", '
    '"title": "<short title>", "args": {...}} — for "distribution"/"categorical_breakdown"/"text_analysis" args is '
    '{"column": "<existing column name>"}; for "timeseries" args is {"time_col": "...", "value_col": "..."}; '
    'for "correlations"/"outliers" args is {"columns": ["...", "..."]}; for "custom_python" args is '
    '{"code": "<python using df, pd, np — assign to result>"}. '
    "Only propose a follow-up if this specific finding genuinely warrants deeper investigation — "
    "an empty array is a completely valid answer, and is expected most of the time. When a business context is "
    "given, a good follow-up often examines a feature's relationship to whatever outcome/target it implies (e.g. "
    "a correlation or breakdown segmented by that outcome), not just the column in isolation."
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
        cols_named = [v for k, v in args.items() if k in ("column", "time_col", "value_col")]
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
    'categorical_breakdown, timeseries, text_analysis, custom_python. args follows the same shape as a follow-up: '
    'distribution/categorical_breakdown/text_analysis -> {"column": "..."}; timeseries -> {"time_col": "...", "value_col": "..."}; '
    'correlations/outliers -> {"columns": [...]}; custom_python -> {"code": "<python using df, pd, np — assign to result>"}. '
    'dataset_id must be one of the dataset ids listed below. '
    '"remove" and "add" may both be empty — e.g. if the message is just a question rather than an instruction — '
    "but always include a short \"reply\"."
)


def _apply_steering(worklist: list[dict], instruction: str, loaded: dict, provider) -> tuple[list[dict], str]:
    """Lets a user redirect a still-running pipeline mid-flight — e.g. "skip
    the categorical breakdowns" or "focus more on revenue". Removes/adds are
    validated the same way follow-ups are (real kind, real columns, real
    dataset) so a bad AI response can't corrupt the worklist; "remove" only
    ever touches items still "pending" (never done/running/already-skipped).
    """
    if provider is None:
        return worklist, "AI isn't configured, so I can't act on that right now."

    pending_items = [(i, it) for i, it in enumerate(worklist) if it["status"] == "pending"]
    datasets_info = {did: (ds.name, list(df.columns)) for did, (ds, df) in loaded.items()}
    pending_desc = "\n".join(f"- index {i}: [dataset {it['dataset_id']}] {it['title']} (kind: {it['kind']})" for i, it in pending_items)
    datasets_desc = "\n".join(f"- dataset {did} \"{name}\": columns {cols}" for did, (name, cols) in datasets_info.items())
    prompt = (
        f"You are steering an in-progress automated EDA run based on a user instruction.\n\n"
        f"User instruction: \"{instruction}\"\n\n"
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

    for p in (parsed.get("add") or [])[:MAX_FOLLOWUPS_PER_ITEM * 3]:
        if not isinstance(p, dict) or p.get("kind") not in _VALID_KINDS:
            continue
        did = p.get("dataset_id")
        if did not in datasets_info:
            continue
        _, cols = datasets_info[did]
        args = p.get("args") or {}
        cols_named = [v for k, v in args.items() if k in ("column", "time_col", "value_col")]
        cols_named += args.get("columns", []) if isinstance(args.get("columns"), list) else []
        if any(c not in cols for c in cols_named):
            continue
        item = _new_item(p["kind"], p.get("title", p["kind"]), args)
        item["dataset_id"] = did
        worklist.append(item)

    return worklist, reply


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
        worklist = _plan_all_datasets(loaded, max_total, business_context, provider)

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
            result, body = _execute_item(item, df)
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
            room = max_total - len(worklist)
            worklist.extend(followups[:room])

        _persist(db, run_row, worklist, markdown, "running")
        yield {"type": "item_done", "index": i, "item": item, "markdown_chunk": chunk}
        if followups:
            yield {"type": "worklist", "worklist": worklist}

        i += 1

    _persist(db, run_row, worklist, markdown, "completed")
    yield {"type": "done", "markdown": markdown}
