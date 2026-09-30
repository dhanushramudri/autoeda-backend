"""
Time-column parsing and time/value column detection.

Shared by the Time Series page (column pickers), the time-series analysis
itself (`timeseries._prepare_series`) and the Data Science Flows engine, so
every consumer agrees on what counts as a usable time axis / value series.

The profiler's `semantic_type` is deliberately NOT used here: it labels
continuous, high-cardinality numerics (revenue, amounts) as "id_like" and
leaves string dates as "categorical", both of which are exactly the columns a
time series needs.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

_TIME_NAME_HINT = re.compile(
    r"(date|time|timestamp|datetime|period|month|year|yr|fy|quarter|week|day|dt)", re.I
)
_ID_NAME = re.compile(r"(^|_)(id|uuid|guid|key|index|idx|row|rownum|serial)($|_)|id$", re.I)
_MIN_PARSE_RATE = 0.9
_SAMPLE = 500


def _numeric_to_datetime(s: pd.Series) -> pd.Series | None:
    """Year (2019), year-month (201906) or yyyymmdd (20190630) integers.
    Epoch numbers are intentionally not guessed at."""
    v = pd.to_numeric(s, errors="coerce")
    nn = v.dropna()
    if nn.empty or not np.all(np.mod(nn, 1) == 0):
        return None
    lo, hi = nn.min(), nn.max()
    fmt = None
    if 1800 <= lo and hi <= 2200:
        fmt = "%Y"
    elif 180001 <= lo and hi <= 220012:
        fmt = "%Y%m"
    elif 18000101 <= lo and hi <= 22001231:
        fmt = "%Y%m%d"
    if fmt is None:
        return None
    txt = v.map(lambda x: None if pd.isna(x) else str(int(x)))
    return pd.to_datetime(txt, format=fmt, errors="coerce")


def _infer_dayfirst(txt: pd.Series) -> bool:
    """dd-mm-yyyy vs mm-dd-yyyy: a first part > 12 proves day-first, a second part > 12 proves
    month-first. Ambiguous columns default to month-first (pandas' own default)."""
    parts = txt.dropna().head(2000).str.extract(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}")
    if parts.dropna().empty:
        return False
    a = pd.to_numeric(parts[0], errors="coerce")
    b = pd.to_numeric(parts[1], errors="coerce")
    if (a > 12).any() and not (b > 12).any():
        return True
    return False


def parse_time_column(s: pd.Series) -> pd.Series:
    """Best-effort conversion to datetime64. Unparseable cells become NaT."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce").dt.tz_localize(None) if getattr(s.dt, "tz", None) else s
    if pd.api.types.is_bool_dtype(s):
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if pd.api.types.is_numeric_dtype(s):
        out = _numeric_to_datetime(s)
        return out if out is not None else pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    txt = s.astype("string").str.strip()
    # numeric-looking strings ("2019", "201906") -> year / yyyymm handling
    if txt.dropna().str.fullmatch(r"\d{4,8}").mean() > 0.9:
        out = _numeric_to_datetime(pd.to_numeric(txt, errors="coerce"))
        if out is not None:
            return out
    dayfirst = _infer_dayfirst(txt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pd.to_datetime(txt, errors="coerce", dayfirst=dayfirst)
        if out.notna().mean() < _MIN_PARSE_RATE:
            try:  # "2023-Q1" / "2023Q1"
                q = pd.PeriodIndex(txt.str.replace(r"[\s-]", "", regex=True), freq="Q")
                return pd.Series(q.to_timestamp(), index=s.index)
            except Exception:
                pass
            try:  # mixed formats
                alt = pd.to_datetime(txt, errors="coerce", format="mixed")
                if alt.notna().sum() > out.notna().sum():
                    out = alt
            except Exception:
                pass
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


def _time_score(df: pd.DataFrame, col: str) -> float:
    s = df[col]
    nn = s.dropna()
    if len(nn) < 10:
        return 0.0
    hint = bool(_TIME_NAME_HINT.search(str(col)))
    if pd.api.types.is_datetime64_any_dtype(s):
        return 1.0
    if pd.api.types.is_bool_dtype(s):
        return 0.0
    if pd.api.types.is_numeric_dtype(s):
        if not hint:  # plain ints in the year range are far more often counts than years
            return 0.0
    elif not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return 0.0
    sample = nn.sample(min(len(nn), _SAMPLE), random_state=0) if len(nn) > _SAMPLE else nn
    parsed = parse_time_column(sample)
    rate = float(parsed.notna().mean())
    if rate < _MIN_PARSE_RATE or parsed.dropna().nunique() < 3:
        return 0.0
    return round(0.6 + 0.3 * rate + (0.1 if hint else 0.0), 3)


def detect_time_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for col in df.columns:
        try:
            sc = _time_score(df, col)
        except Exception:
            sc = 0.0
        if sc > 0:
            out.append({"name": str(col), "score": sc, "dtype": str(df[col].dtype)})
    return sorted(out, key=lambda d: -d["score"])


def detect_value_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    """Numeric columns usable as a series. Keeps continuous high-cardinality
    columns (revenue) that the profiler calls id_like; drops constants,
    booleans and integer surrogate keys."""
    exclude = exclude or set()
    out = []
    n = max(len(df), 1)
    for col in df.columns:
        if str(col) in exclude:
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            continue
        nn = s.dropna()
        if len(nn) < 10 or nn.nunique() <= 1:
            continue
        is_int = pd.api.types.is_integer_dtype(s) or bool(np.all(np.mod(nn, 1) == 0))
        if is_int and _ID_NAME.search(str(col)) and nn.nunique() / n > 0.9:
            continue
        out.append({"name": str(col), "dtype": str(s.dtype), "missing_pct": round(float(s.isna().mean() * 100), 2)})
    return out


def detect_time_series_columns(df: pd.DataFrame) -> dict[str, Any]:
    times = detect_time_columns(df)
    values = detect_value_columns(df, exclude={t["name"] for t in times[:1]})
    # Prefer money-like value columns as the default series
    money = re.compile(r"(revenue|sales|amount|turnover|income|arr|mrr|gmv|value|price|profit|cost|qty|quantity|volume)", re.I)
    values.sort(key=lambda v: (0 if money.search(v["name"]) else 1))
    return {
        "time_columns": times,
        "value_columns": values,
        "recommended": {
            "time_col": times[0]["name"] if times else None,
            "value_col": values[0]["name"] if values else None,
        },
    }
