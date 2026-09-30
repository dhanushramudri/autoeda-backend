"""Shared helpers for Data Science Flows: JSON sanitising and column-role detection."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from ..eda.ts_columns import detect_time_columns, parse_time_column


def clean(obj: Any) -> Any:
    """Recursively convert numpy/pandas values into plain JSON types (NaN/inf -> None)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        try:
            return pd.Timestamp(obj).strftime("%Y-%m-%d")
        except Exception:
            return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(obj, pd.DataFrame):
        return clean(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return clean(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if obj is pd.NaT:
        return None
    return obj


# ---------------------------------------------------------------------------
# Role detection
# ---------------------------------------------------------------------------

_TARGET_NAME = re.compile(r"(churn|target|label|attrit|cancel|lost|left|exit|defect|flag|outcome)", re.I)
_TARGET_STRONG = re.compile(r"(churn|target|attrit|cancel|defect|lapse|lapsed)", re.I)
_ENTITY_NAME = re.compile(
    r"(account|customer|client|user|member|company|org|subscriber|acct|cust|co_ref|tenant|contract)", re.I
)
_ID_SUFFIX = re.compile(r"(_id|id|_ref|ref|_key|key|_no|number|_code)$", re.I)
_VALUE_NAME = re.compile(r"(arr|mrr|revenue|amount|spend|sales|value|gmv|billing|contract_value|acv|tcv)", re.I)
_VALUE_BAD = re.compile(r"(future|change|diff|delta|log_|percent|pct|ratio|growth|discount|next|forecast)", re.I)
_POS_WORDS = {"1", "true", "yes", "y", "t", "churn", "churned", "lost", "left", "cancelled", "canceled", "attrited", "exited"}


def _binary_info(s: pd.Series) -> dict | None:
    nn = s.dropna()
    if len(nn) < 20:
        return None
    vals = nn.unique()
    if len(vals) != 2:
        return None
    if pd.api.types.is_bool_dtype(s):
        pos = True
    elif pd.api.types.is_numeric_dtype(s):
        pos = max(vals)
    else:
        low = {str(v).strip().lower(): v for v in vals}
        hit = [orig for k, orig in low.items() if k in _POS_WORDS]
        pos = hit[0] if len(hit) == 1 else None
        if pos is None:
            return None
    rate = float((nn == pos).mean())
    return {"positive": pos.item() if hasattr(pos, "item") else pos, "rate": rate}


def encode_target(s: pd.Series, positive: Any = None) -> pd.Series:
    """1.0 = churned, 0.0 = retained, NaN = unknown."""
    if positive is None:
        info = _binary_info(s)
        positive = info["positive"] if info else None
    if positive is None:
        raise ValueError(f"Column '{s.name}' is not a binary churn label")
    y = pd.Series(np.nan, index=s.index, dtype="float64")
    known = s.notna()
    y[known] = (s[known] == positive).astype(float)
    return y


def _top(scores: list[tuple[str, float]], k: int = 5) -> list[dict]:
    return [{"name": n, "score": round(sc, 3)} for n, sc in sorted(scores, key=lambda t: -t[1])[:k] if sc > 0]


def detect_churn_roles(df: pd.DataFrame) -> dict[str, Any]:
    """Finds: target (binary churn label), entity (customer/account id), date (snapshot/period), value (ARR/revenue)."""
    n = max(len(df), 1)
    cols = [str(c) for c in df.columns]

    # --- target ---
    tscores = []
    for c in cols:
        info = _binary_info(df[c])
        if not info or not (0.005 <= info["rate"] <= 0.7):
            continue
        # a binary column is only a churn-label candidate if its NAME says so; "flag"/"outcome" alone is too
        # weak (e.g. Auto_Renewal_Flag is a feature, not an outcome), so those are never auto-selected
        if not _TARGET_STRONG.search(c):
            continue
        sc = 1.0
        if re.search(r"(engaged|increase|decrease|adopted|enabled|above|activated)", c, re.I):
            sc -= 1.0
        if _TARGET_STRONG.search(c):
            sc += 3
        elif _TARGET_NAME.search(c):
            sc += 1.5
        if 0.02 <= info["rate"] <= 0.4:
            sc += 0.5
        tscores.append((c, sc))
    targets = _top(tscores)
    target = targets[0]["name"] if targets else None

    # --- date ---
    time_cols = [t for t in detect_time_columns(df) if t["name"] != target]
    dscores = []
    for t in time_cols:
        sc = t["score"]
        if re.search(r"(snapshot|calculated|as_of|period|month|date)", t["name"], re.I):
            sc += 0.3
        dscores.append((t["name"], sc))
    dates = _top(dscores)
    date = dates[0]["name"] if dates else None

    # --- entity ---
    escores = []
    for c in cols:
        if c == target or c == date:
            continue
        s = df[c]
        if pd.api.types.is_float_dtype(s) and not (_ENTITY_NAME.search(c) or _ID_SUFFIX.search(c)):
            continue
        nu = s.nunique(dropna=True)
        if nu < 2 or nu > n:
            continue
        name_hit = bool(_ENTITY_NAME.search(c))
        id_hit = bool(_ID_SUFFIX.search(c))
        if not id_hit:  # an entity key must look like a key, not merely mention "customer"
            continue
        sc = (2.0 if name_hit else 0) + (1.5 if id_hit else 0)
        if nu < n * 0.98:  # repeats across rows -> a panel of snapshots
            sc += 1.0
        if nu < 2 or nu == n and not (name_hit and id_hit):
            sc -= 0.5
        escores.append((c, sc))
    entities = _top(escores)
    entity = entities[0]["name"] if entities else None

    # --- value (ARR / revenue) ---
    vscores = []
    for c in cols:
        if c in (target, date, entity):
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or s.nunique() < 5:
            continue
        if not _VALUE_NAME.search(c) or _VALUE_BAD.search(c):
            continue
        sc = 1.0 + (1.0 if re.search(r"(current|total|^arr|^mrr)", c, re.I) else 0) - (0.5 if s.isna().mean() > 0.3 else 0)
        vscores.append((c, sc))
    values = _top(vscores)
    value = values[0]["name"] if values else None

    return {
        "target": target, "entity": entity, "date": date, "value": value,
        "candidates": {"target": targets, "entity": entities, "date": dates, "value": values},
    }


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "numeric_columns": int(df.select_dtypes("number").shape[1]),
        "missing_pct": round(float(df.isna().mean().mean() * 100), 2) if len(df) else 0.0,
    }


def prepare_dates(df: pd.DataFrame, date_col: str | None) -> pd.Series | None:
    if not date_col or date_col not in df.columns:
        return None
    d = parse_time_column(df[date_col])
    return d if d.notna().mean() >= 0.5 else None


_SUFFIXES = [("__missing", " (missing)"), ("__delta_prev", " (change vs previous)"), ("__days_before_period", " (days before period)"),
             ("__days_since_last", " (days since last)"), ("__events_last_90d", " (events, last 90 days)"), ("__n_events", " (number of events)")]


def _titled(s: str) -> str:
    txt = ": ".join(p.replace("_", " ") for p in s.split("__") if p)
    return (txt[:1].upper() + txt[1:]) if txt else s


def pretty_feature(f: str) -> str:
    """Readable name for an engineered / raw feature (used in the UI, narrative and report)."""
    s = str(f)
    if "=" in s and not s.endswith("__missing"):
        a, b = s.split("=", 1)
        return f"{pretty_feature(a)}: {b.strip() or '(blank)'}"
    for suf, txt in _SUFFIXES:
        if s.endswith(suf):
            return pretty_feature(s[: -len(suf)]) + txt
    if s.endswith("__yes__avg"):
        return _titled(s[:-10]) + " (share yes)"
    if s.endswith("__avg"):
        return _titled(s[:-5]) + " (average)"
    return _titled(s)
