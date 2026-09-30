"""
Zero-configuration data discovery for the churn flow.

Given every dataset in a workspace, figure out — with no user input —:
  * which table holds the churn outcome, and how to turn it into 0 / 1 / unknown
    (binary flag, or a multi-value outcome such as Won / Churned / Open),
  * which column links the tables together (the account key),
  * which tables are event tables (calls, emails, tickets...) to be attached as features,
    and which one is a data dictionary (used only as a hint for picking the label),
  * the period date, and a revenue column that does not itself leak the outcome.

Event tables are attached to each base row using ONLY events that happened before the period start
(lead-time safe), so the model never sees the decision it is trying to predict.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..eda.ts_columns import detect_time_columns, parse_time_column
from .common import _binary_info, clean, detect_churn_roles

warnings.filterwarnings("ignore")

CHURN_WORDS = re.compile(r"(churn|lost|cancel|attrit|terminat|non[- ]?renew|lapse|defect|left)", re.I)
NOT_CHURN = re.compile(r"(not|non|no)[ _-]*churn", re.I)
WON_WORDS = re.compile(r"(^won$|renew|retain|active|stay|success|kept)", re.I)
OPEN_WORDS = re.compile(r"(open|pending|in progress|tbd|unknown|n/?a|not yet|undecided|current)", re.I)
LABEL_NAME = re.compile(r"(outcome|status|result|churn|label|target|decision)", re.I)
KEY_NAME = re.compile(r"(_id|id|_ref|ref|_key|key|_no|number|code)$", re.I)
EVENT_DATE_NAME = re.compile(r"(date|time|created|sent|call|contact)", re.I)
PERIOD_NAME = re.compile(r"(renewal[_ ]?(month|period)|snapshot|as_of|calculated|billing[_ ]?month|period|^month)", re.I)
VALUE_NAME = re.compile(r"(total_amount|revenue|arr|mrr|amount|net|gross|price|fee|value|spend)", re.I)
VALUE_BAD = re.compile(r"(paid|future|change|diff|delta|log_|percent|pct|ratio|growth|discount|next|forecast|last_|starting_|vat)", re.I)
MAX_EVENT_FEATURES = 60


def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


# ---------------------------------------------------------------------------
# table classification
# ---------------------------------------------------------------------------

def _dictionary_hints(tables: dict[str, pd.DataFrame]) -> tuple[list[str], dict[str, str]]:
    """A dictionary table is small and has a 'column/field' column beside a 'description' column."""
    names, hints = [], {}
    for name, df in tables.items():
        if len(df) > 3000 or df.shape[1] > 6:
            continue
        cols = {str(c): str(c).lower() for c in df.columns}
        col_c = next((c for c, l in cols.items() if re.search(r"(column|field|variable|attribute)", l)), None)
        desc_c = next((c for c, l in cols.items() if re.search(r"(descr|definition|meaning|comment)", l)), None)
        if col_c and desc_c and col_c != desc_c:
            names.append(name)
            for a, b in zip(df[col_c].fillna("").astype(str), df[desc_c].fillna("").astype(str)):
                hints.setdefault(a.strip().lower(), b)
    return names, hints


def _key_series(s: pd.Series) -> pd.Series:
    """Normalised account keys; blanks/NaN become NaN so they can never match each other in a join
    (pandas 2 turns NaN into the string 'nan' under astype(str), pandas 3 keeps NaN — handle both)."""
    k = s.astype("string").str.strip().str.lower()
    return k.where(~k.isin(["", "nan", "none", "null", "<na>"]))


def _norm_key_values(s: pd.Series) -> pd.Series:
    return _key_series(s).dropna()


def find_link_key(tables: dict[str, pd.DataFrame]) -> dict | None:
    """The column name shared by the most tables whose values actually overlap."""
    by_name: dict[str, dict[str, str]] = {}
    for t, df in tables.items():
        for c in df.columns:
            if KEY_NAME.search(str(c)) and df[c].nunique(dropna=True) >= 10 and not pd.api.types.is_float_dtype(df[c]):
                by_name.setdefault(str(c).strip().lower(), {})[t] = str(c)
    best = None
    for lname, cols in by_name.items():
        if len(cols) < 2:
            continue
        sets = {t: set(_norm_key_values(tables[t][c]).head(60_000)) for t, c in cols.items()}
        biggest = max(sets, key=lambda t: len(sets[t]))
        linked = [t for t in sets if t == biggest or len(sets[t] & sets[biggest]) / max(min(len(sets[t]), len(sets[biggest])), 1) >= 0.3]
        if len(linked) >= 2 and (best is None or len(linked) > len(best["tables"])):
            best = {"name": lname, "columns": {t: cols[t] for t in linked}, "tables": linked}
    return best


# ---------------------------------------------------------------------------
# label detection
# ---------------------------------------------------------------------------

def detect_label(df: pd.DataFrame, hints: dict[str, str] | None = None) -> dict | None:
    hints = hints or {}
    best = None

    def consider(spec, score):
        nonlocal best
        if best is None or score > best[0]:
            best = (score, spec)

    for c in df.columns:
        c = str(c)
        s = df[c]
        desc = hints.get(c.lower(), "")
        hint_score = 1.5 if re.search(r"(churn|outcome|cancel|renew)", desc, re.I) else 0.0
        if _is_text(s) and 2 <= s.nunique(dropna=True) <= 12:
            vals = [v for v in s.dropna().astype(str).unique()]
            pos = [v for v in vals if CHURN_WORDS.search(v) and not NOT_CHURN.search(v)]
            if not pos:
                continue
            neg = [v for v in vals if v not in pos and WON_WORDS.search(v)]
            unk = [v for v in vals if v not in pos and v not in neg]
            lab = s.astype(str).isin(pos + neg)
            rate = s.astype(str).isin(pos)[lab].mean() if lab.any() else 0
            if lab.mean() < 0.5 or not (0.005 <= rate <= 0.7) or not neg:
                continue
            score = 3.0 + (1.5 if LABEL_NAME.search(c) else 0) + hint_score
            consider({"column": c, "kind": "categorical", "positive": pos, "negative": neg, "unknown": unk}, score)
        else:
            info = _binary_info(s)
            if (info and 0.005 <= info["rate"] <= 0.7
                    and re.search(r"(^|[_\s-])(churn\w*|target|target_flag|attrition|cancel\w*|defect\w*|lapse\w*)([_\s-]|$)", c, re.I)
                    and not re.search(r"(engaged|increase|decrease|adopted|enabled|above|activated)", c, re.I)):
                vals = list(s.dropna().unique())
                pos = [info["positive"]]
                neg = [v for v in vals if v != info["positive"]]
                consider({"column": c, "kind": "binary", "positive": pos, "negative": neg, "unknown": []}, 4.0 + hint_score)
    return best[1] if best else None


def apply_label(df: pd.DataFrame, spec: dict) -> pd.Series:
    col = df[spec["column"]]
    if spec["kind"] == "categorical":
        v = col.astype(str)
        y = pd.Series(np.nan, index=df.index, dtype="float64")
        y[v.isin(spec["negative"])] = 0.0
        y[v.isin(spec["positive"])] = 1.0
        return y
    y = pd.Series(np.nan, index=df.index, dtype="float64")
    known = col.notna()
    y[known] = col[known].isin(spec["positive"]).astype(float)
    return y


# ---------------------------------------------------------------------------
# base-table preparation
# ---------------------------------------------------------------------------

def _period_date(df: pd.DataFrame, exclude: set[str]) -> str | None:
    cands = [t for t in detect_time_columns(df) if t["name"] not in exclude]
    if not cands:
        return None
    scored = []
    for t in cands:
        sc = t["score"] + (1.0 if PERIOD_NAME.search(t["name"]) else 0.0)
        scored.append((sc, t["name"]))
    return max(scored)[1]


def _pick_value(df: pd.DataFrame, y: pd.Series, exclude: set[str]) -> str | None:
    """Revenue column by name, rejecting ones that reveal the outcome (e.g. 'amount paid' is empty when churned)."""
    from scipy import stats

    lab = y.notna().to_numpy()
    cands = []
    for c in df.columns:
        c = str(c)
        if c in exclude or not VALUE_NAME.search(c) or VALUE_BAD.search(c):
            continue
        s = pd.to_numeric(df[c].replace("", np.nan), errors="coerce") if _is_text(df[c]) else df[c]
        if not pd.api.types.is_numeric_dtype(s) or s.nunique() < 5 or s.isna().mean() > 0.3:
            continue
        x = s.to_numpy(dtype=float)[lab]
        yy = y.to_numpy()[lab]
        m = ~np.isnan(x)
        if m.sum() < 50 or yy[m].sum() < 10:
            continue
        r = stats.rankdata(x[m])
        n1 = yy[m].sum()
        n0 = m.sum() - n1
        auc = (r[yy[m] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        if max(auc, 1 - auc) > 0.85:
            continue
        pri = (2 if re.search(r"^total[_ ]?amount$|revenue|arr|mrr", c, re.I) else 1) - 0.1 * len(cands)
        cands.append((pri, c))
    return max(cands)[1] if cands else None


def _derive_date_features(base: pd.DataFrame, period: pd.Series, skip: set[str]) -> tuple[pd.DataFrame, list[str]]:
    """Text columns that are dates -> 'days before the period start'. Only dates that precede the period
    for ~all rows qualify (a date that falls after it, like a close date, describes the outcome)."""
    out, used = {}, []
    for c in base.columns:
        c = str(c)
        if c in skip or not _is_text(base[c]):
            continue
        s = base[c].replace("", np.nan)
        if s.notna().mean() < 0.3:
            continue
        sample = s.dropna().head(400)
        if parse_time_column(sample).notna().mean() < 0.9:
            continue
        d = parse_time_column(s)
        days = (period - d).dt.days
        valid = days.dropna()
        if len(valid) and (valid >= 0).mean() >= 0.95:
            out[f"{c}__days_before_period"] = days.clip(lower=0)
            used.append(c)
    return pd.DataFrame(out, index=base.index), used


# ---------------------------------------------------------------------------
# event-table attachment
# ---------------------------------------------------------------------------

def _event_indicators(ev: pd.DataFrame, skip: set[str]) -> pd.DataFrame:
    """Turn an event table's columns into numeric indicators that can be averaged per account."""
    out: dict[str, pd.Series] = {}
    scored: list[tuple[float, str]] = []
    n = max(len(ev), 1)
    for c in ev.columns:
        c = str(c)
        if c in skip:
            continue
        s = ev[c]
        nu = s.nunique(dropna=True)
        if nu < 2 or nu / n > 0.5:
            continue
        if pd.api.types.is_bool_dtype(s):
            out[f"{c}"] = s.astype(float)
            scored.append((s.notna().mean(), c))
        elif pd.api.types.is_numeric_dtype(s):
            out[f"{c}"] = s.astype(float)
            scored.append((s.notna().mean() * 0.8, c))
        elif _is_text(s):
            t = s.astype(str).str.strip()
            blank = t.eq("") | t.str.lower().isin(["nan", "none", "null"])
            num = pd.to_numeric(t.where(~blank), errors="coerce")
            if num.notna().sum() >= 0.9 * max((~blank).sum(), 1) and num.nunique() > 2:
                out[f"{c}"] = num
                scored.append(((~blank).mean() * 0.8, c))
                continue
            if nu > 30 or t.str.len().mean() > 40:
                continue
            low = t.str.lower()
            if low.isin(["yes", "no"]).sum() >= 0.3 * max((~blank).sum(), 1):
                out[f"{c}__yes"] = (low == "yes").astype(float).where(~blank)
                scored.append(((~blank).mean(), f"{c}__yes"))
            else:
                top = low[~blank & ~low.isin(["not mentioned", "not discussed", "not applicable", "unknown", "xxxx"])].value_counts().head(3)
                for cat in top.index:
                    if top[cat] >= 0.02 * n:
                        name = f"{c}__{re.sub(r'[^a-z0-9]+', '_', cat)[:30].strip('_')}"
                        out[name] = (low == cat).astype(float).where(~blank)
                        scored.append(((~blank).mean() * 0.9, name))
    keep = [name for _sc, name in sorted(scored, reverse=True)[:MAX_EVENT_FEATURES]]
    return pd.DataFrame({k: out[k] for k in keep}, index=ev.index)


def _attach_events(base: pd.DataFrame, key: str, period: pd.Series | None, ev: pd.DataFrame, ev_key: str, name: str,
                   lead_days: int = 0, window_days: int = 365) -> tuple[pd.DataFrame, dict]:
    rid = pd.RangeIndex(len(base), name="rid")
    bk = pd.DataFrame({"rid": rid, "k": _key_series(base[key]).to_numpy(dtype=object)})
    bk = bk[bk["k"].notna()].reset_index(drop=True)
    if period is not None:
        bk["R"] = period.to_numpy()[bk["rid"].to_numpy()]

    ev = ev.reset_index(drop=True)
    evk = _key_series(ev[ev_key])
    ev_date_col = next((t["name"] for t in detect_time_columns(ev) if EVENT_DATE_NAME.search(t["name"]) and t["name"] != ev_key), None)
    year_col = next((c for c in ev.columns if re.fullmatch(r"(?i)(renewal_)?year", str(c)) and pd.api.types.is_integer_dtype(ev[c])), None)
    skip = {ev_key, ev_date_col, year_col} - {None}
    # non-feature identifiers: high-cardinality id columns
    for c in ev.columns:
        if KEY_NAME.search(str(c)) and ev[c].nunique() > 0.5 * len(ev):
            skip.add(str(c))
    # a timing column that marks events AFTER the decision must not be used
    timing = next((c for c in ev.columns if re.search(r"(time_to_renewal|timing|stage)", str(c), re.I) and _is_text(ev[c])), None)
    post_mask = ev[timing].astype(str).str.contains(r"post|after|lapsed|expired", case=False, regex=True) if timing else pd.Series(False, index=ev.index)

    ind = _event_indicators(ev[~post_mask], skip)
    body = pd.concat([pd.DataFrame({"k": evk[~post_mask].to_numpy(dtype=object)}), ind.reset_index(drop=True)], axis=1)
    mode = "account-level"
    if ev_date_col and period is not None:
        body["D"] = parse_time_column(ev.loc[~post_mask, ev_date_col]).to_numpy()
        body = body.dropna(subset=["D", "k"])
        pairs = bk.dropna(subset=["R"]).merge(body, on="k", how="inner")
        pairs = pairs[(pairs["D"] < pairs["R"] - pd.Timedelta(days=lead_days)) & (pairs["D"] >= pairs["R"] - pd.Timedelta(days=window_days))]
        mode = f"events in the {window_days} days before the period start"
    elif year_col is not None and period is not None:
        body["Y"] = ev.loc[~post_mask, year_col].to_numpy()
        body = body.dropna(subset=["Y", "k"])
        bk["Y"] = period.dt.year.to_numpy()
        pairs = bk.merge(body, left_on=["k", "Y"], right_on=["k", "Y"], how="inner")
        mode = "events of the same renewal year (all pre-decision by design)"
    else:
        pairs = bk.merge(body.dropna(subset=["k"]), on="k", how="inner")

    feats = list(ind.columns)
    g = pairs.groupby("rid")
    agg = g[feats].mean().add_prefix(f"{name}__").add_suffix("__avg") if feats else pd.DataFrame(index=pd.Index([], name="rid"))
    agg[f"{name}__n_events"] = g.size()
    if "D" in pairs.columns and "R" in pairs.columns:
        agg[f"{name}__days_since_last"] = (pairs.groupby("rid")["R"].first() - g["D"].max()).dt.days
        recent = pairs[pairs["D"] >= pairs["R"] - pd.Timedelta(days=90)]
        agg[f"{name}__events_last_90d"] = recent.groupby("rid").size()
    agg = agg.reindex(rid)
    agg[f"{name}__n_events"] = agg[f"{name}__n_events"].fillna(0)
    if f"{name}__events_last_90d" in agg:
        agg[f"{name}__events_last_90d"] = agg[f"{name}__events_last_90d"].fillna(0)
    agg.index = base.index
    info = {"table": name, "mode": mode, "features": int(agg.shape[1]), "accounts_covered_pct": float((agg[f"{name}__n_events"] > 0).mean() * 100)}
    return agg, info


# ---------------------------------------------------------------------------
# public entry points (top-level so they can run in the process pool)
# ---------------------------------------------------------------------------

def _analyse(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    dict_tables, hints = _dictionary_hints(tables)
    data_tables = {n: d for n, d in tables.items() if n not in dict_tables}
    link = find_link_key(data_tables) if len(data_tables) > 1 else None

    base_name, spec = None, None
    best_score = -1.0
    for n, d in data_tables.items():
        s = detect_label(d, hints)
        if s:
            score = {"binary": 4.0, "categorical": 3.0}[s["kind"]] + min(len(d) / 1e6, 1)
            if score > best_score:
                base_name, spec, best_score = n, s, score
    return {"dict_tables": dict_tables, "hints": hints, "data_tables": data_tables, "link": link, "base": base_name, "label": spec}


def plan_workspace(tables: dict[str, pd.DataFrame], meta: dict[str, dict] | None = None) -> dict[str, Any]:
    """Fast, detection-only view used by the page before anything is run."""
    from .registry import FLOWS, scan_dataset

    a = _analyse(tables)
    base_name, spec, link = a["base"], a["label"], a["link"]
    rows = []
    for n, d in tables.items():
        role = "dictionary" if n in a["dict_tables"] else "base" if n == base_name else "events" if (link and n in link["tables"]) else "other"
        rows.append({"name": n, "rows": int(len(d)), "columns": int(d.shape[1]), "role": role, **((meta or {}).get(n, {}))})

    # per-flow feasibility: best score across tables (churn is judged on the discovered label)
    flows: dict[str, dict] = {}
    for n, d in tables.items():
        if n in a["dict_tables"]:
            continue
        sc = scan_dataset(d)
        for f in sc["flows"]:
            cur = flows.get(f["key"])
            if cur is None or f["feasibility"]["score"] > cur["feasibility"]["score"]:
                flows[f["key"]] = {**f, "table": n}
    if spec:
        y = apply_label(tables[base_name], spec)
        pos, neg, unk = int((y == 1).sum()), int((y == 0).sum()), int(y.isna().sum())
        sig = [f"Outcome '{spec['column']}' in {base_name}: {pos:,} churned, {neg:,} retained" + (f", {unk:,} still open" if unk else "")]
        if link:
            sig.append(f"{len(link['tables']) - 1} linked table(s) via {link['columns'][base_name] if base_name in link['columns'] else link['name']}")
        flows["churn"]["feasibility"] = {"score": 100 if pos >= 100 else 60, "verdict": "strong" if pos >= 100 else "possible", "signals": sig, "missing": []}
    else:
        flows["churn"]["feasibility"] = {"score": 10, "verdict": "not_detected", "signals": [],
                                         "missing": ["No churn outcome found in any dataset (a column such as Outcome = Churned / Won, or a 0/1 churn flag)"]}
    flow_list = [flows[f["key"]] for f in FLOWS if f["key"] in flows]
    for f in flow_list:
        f.pop("table", None)
    return clean({
        "tables": rows, "link_key": link["name"] if link else None, "base_table": base_name,
        "label": ({"column": spec["column"], "kind": spec["kind"], "churned": spec["positive"], "retained": spec["negative"], "open": spec["unknown"], "counts": {"churned": pos, "retained": neg, "open": unk}} if spec else None),
        "flows": flow_list, "runnable": bool(spec),
        "reason": None if spec else "No churn outcome was found in this workspace's datasets.",
    })


def stage_discover(ctx):
    """Stage 0: build the modelling table (one row per base row, in the original order)."""
    tables: dict[str, pd.DataFrame] = ctx["tables"]
    a = _analyse(tables)
    if not a["label"]:
        raise ValueError("No churn outcome found. Looked for an outcome column with values such as Churned / Won / Lost, "
                         "or a 0/1 churn flag, in: " + ", ".join(tables))
    base_name, spec, link = a["base"], a["label"], a["link"]
    base = tables[base_name].reset_index(drop=True)
    y = apply_label(base, spec)

    # account key
    key = link["columns"][base_name] if link and base_name in link["columns"] else None
    roles = detect_churn_roles(base.assign(**{"__tmp_label__": y}))
    if key is None:
        key = roles["entity"]
    label_cols = {spec["column"]}
    period_name = _period_date(base, exclude={key} if key else set())
    period = parse_time_column(base[period_name]) if period_name else None
    if period is not None and period.notna().mean() < 0.5:
        period_name, period = None, None
    value = _pick_value(base, y, label_cols | {c for c in (key, period_name) if c})

    merged = base.copy()
    merged["__label__"] = y.to_numpy()
    log: list[str] = []
    attached: list[dict] = []
    if period is not None:
        derived, used = _derive_date_features(base, period, label_cols | {c for c in (key, period_name) if c})
        if not derived.empty:
            merged = pd.concat([merged, derived], axis=1)
            log.append(f"Turned {len(used)} date column(s) into 'days before the period' features (only dates known before the period start)")
    if link and key:
        for n, d in a["data_tables"].items():
            if n == base_name or n not in link["tables"]:
                continue
            agg, info = _attach_events(base, key, period, d, link["columns"][n], re.sub(r"[^A-Za-z0-9]+", "_", n).strip("_"))
            merged = pd.concat([merged, agg], axis=1)
            attached.append(info)
            log.append(f"Linked '{n}' to '{base_name}' on {link['columns'][n]}: {info['features']} features from {info['mode']}; "
                       f"{info['accounts_covered_pct']:.0f}% of rows have history")
    roles_out = {"target": "__label__", "entity": key, "date": period_name, "value": value, "positive": 1.0}
    exclude = [spec["column"]]
    result = {
        "base_table": base_name, "link_key": key,
        "label": {"column": spec["column"], "kind": spec["kind"], "churned": spec["positive"], "retained": spec["negative"], "open": spec["unknown"]},
        "label_counts": {"churned": int((y == 1).sum()), "retained": int((y == 0).sum()), "unlabeled": int(y.isna().sum())},
        "period_col": period_name, "value_col": value,
        "tables": [{"name": n, "rows": int(len(d)), "role": "dictionary" if n in a["dict_tables"] else "base" if n == base_name else ("events" if link and n in link["tables"] else "unused")} for n, d in tables.items()],
        "attached": attached, "merged_columns": int(merged.shape[1]), "merged_rows": int(len(merged)), "log": log,
        "roles": roles_out, "exclude_columns": exclude,
    }
    return clean(result), {"merged": merged, "base": base}
