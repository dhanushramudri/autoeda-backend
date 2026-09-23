"""The fixed feature-engineering toolbox the auto-planner is allowed to
reference (see app/ai/agent/automl_planner.py on the backend). Kept as a
small set of named, deterministic operations rather than letting the LLM
emit arbitrary pandas/Python — safer (this still runs on the engineer's own
machine, but there's no reason to accept arbitrary code from a model
output) and far more likely to actually work than free-form generated code.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("autoeda-agent.feature_tools")


def _difference(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    sa, sb = df[a], df[b]
    if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
        return sa - sb
    # Try as dates — the common case is "days between two date-ish columns"
    da, db_ = pd.to_datetime(sa, errors="coerce"), pd.to_datetime(sb, errors="coerce")
    return (da - db_).dt.days


def _ratio(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    denom = pd.to_numeric(df[b], errors="coerce").replace(0, np.nan)
    return pd.to_numeric(df[a], errors="coerce") / denom


def _datetime_part(df: pd.DataFrame, column: str, part: str) -> pd.Series:
    dt = pd.to_datetime(df[column], errors="coerce")
    if part == "year":
        return dt.dt.year
    if part == "month":
        return dt.dt.month
    if part == "day":
        return dt.dt.day
    if part == "dayofweek":
        return dt.dt.dayofweek
    if part == "is_weekend":
        return (dt.dt.dayofweek >= 5).astype(int)
    raise ValueError(f"Unknown datetime part: {part}")


def _log1p(df: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(df[column], errors="coerce")
    return np.log1p(numeric.clip(lower=0))


def _frequency_encoding(df: pd.DataFrame, column: str) -> pd.Series:
    counts = df[column].value_counts()
    return df[column].map(counts)


def _interaction(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    return pd.to_numeric(df[a], errors="coerce") * pd.to_numeric(df[b], errors="coerce")


_TOOLS = {
    "difference": _difference,
    "ratio": _ratio,
    "datetime_part": _datetime_part,
    "log1p": _log1p,
    "frequency_encoding": _frequency_encoding,
    "interaction": _interaction,
}


def apply_engineered_features(df: pd.DataFrame, recipes: list[dict], log=print) -> pd.DataFrame:
    """Applies each {tool, args, output, reason} recipe in order, skipping
    (not raising on) any recipe that fails — one bad column reference from
    the plan shouldn't take down the whole experiment."""
    df = df.copy()
    for recipe in recipes:
        tool_name = recipe.get("tool")
        output = recipe.get("output")
        args = recipe.get("args") or {}
        fn = _TOOLS.get(tool_name)
        if not fn or not output:
            continue
        missing = [v for v in args.values() if isinstance(v, str) and v not in df.columns]
        if missing:
            log(f"  Skipping engineered feature '{output}' — column(s) not found: {missing}")
            continue
        try:
            df[output] = fn(df, **args)
            log(f"  Derived feature '{output}' via {tool_name}({args}) — {recipe.get('reason', '')}")
        except Exception as e:
            log(f"  Skipping engineered feature '{output}' — {tool_name} failed: {e}")
    return df
