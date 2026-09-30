"""
Catalogue of JMAN's data-science offerings and the *opportunity scan*: a fast, content-based check of
which offerings a dataset can support. The scan is what lets the tool say "your data has a churn
opportunity" before anyone runs a model.

Only flows with status "available" can be run; the rest are still scanned so the feasibility picture is
complete (and so adding a flow later is just: implement its stages + flip the status).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ..eda.ts_columns import detect_time_columns, parse_time_column
from .common import clean, dataset_summary, detect_churn_roles, encode_target

FLOWS: list[dict[str, Any]] = [
    {
        "key": "churn", "category": "Churn", "name": "Churn prediction & retention", "status": "available", "priority": 3,
        "tagline": "Identify at-risk customers early and design retention interventions that protect revenue and reduce leakage.",
        "outputs": ["Churn probability & risk tier per row", "Top risk drivers per account", "Revenue at risk", "Retention playbook"],
        "stages": ["Understand the data", "Audit for data leakage", "Explore churn patterns", "Test hypotheses", "Engineer features",
                   "Select features & split", "Train & compare models", "Explain drivers", "Risk tiers & revenue at risk", "Validate", "Build deliverables"],
    },
    {
        "key": "revenue_growth", "category": "Revenue Growth", "name": "Cross-sell, upsell & lead scoring", "status": "coming_soon", "priority": 1,
        "tagline": "Cross-sell, upsell and lead-scoring playbooks that drive top-line growth.",
        "outputs": ["Next-best-product per customer", "Propensity scores", "Revenue opportunity sizing"], "stages": [],
    },
    {
        "key": "forecasting", "category": "Forecasting", "name": "Revenue, demand & cash-flow forecasting", "status": "coming_soon", "priority": 2,
        "tagline": "Revenue, demand and cash-flow forecasting models to support planning, budgeting and diligence.",
        "outputs": ["Forecast with intervals", "Backtested model comparison", "Seasonality & trend findings"], "stages": [],
    },
    {
        "key": "pricing", "category": "Pricing", "name": "Pricing & discount optimisation", "status": "coming_soon", "priority": 4,
        "tagline": "Re-pricing, discounting and monetization strategies to defend or grow margin.",
        "outputs": ["Price elasticity", "Discount leakage", "Margin uplift scenarios"], "stages": [],
    },
    {
        "key": "efficiency_cost", "category": "Efficiency & Cost", "name": "Efficiency & cost automation", "status": "coming_soon", "priority": 5,
        "tagline": "Automation and productivity playbooks that improve EBITDA and operating efficiency.",
        "outputs": ["Cost drivers", "Automation candidates", "EBITDA impact"], "stages": [],
    },
]

_FLOW_BY_KEY = {f["key"]: f for f in FLOWS}


def get_flow(key: str) -> dict[str, Any] | None:
    return _FLOW_BY_KEY.get(key)


def _hits(cols: list[str], pattern: str) -> list[str]:
    """Whole-token match ("price" matches unit_price, not "surprise"; "rate" never matches by accident)."""
    rx = re.compile(rf"(^|[_\s-])({pattern})([_\s-]|$)", re.I)
    return [c for c in cols if rx.search(c)]


def _verdict(score: int) -> str:
    return "strong" if score >= 70 else "possible" if score >= 40 else "weak" if score >= 15 else "not_detected"


def scan_dataset(df: pd.DataFrame) -> dict[str, Any]:
    cols = [str(c) for c in df.columns]
    roles = detect_churn_roles(df)
    times = detect_time_columns(df)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    n = len(df)
    feas: dict[str, dict] = {}

    # --- churn ---
    sig, miss, score = [], [], 0
    if roles["target"]:
        pos = int(encode_target(df[roles["target"]]).sum())
        sig.append(f"Binary outcome column '{roles['target']}' ({pos:,} positive rows)")
        score += 40 + (15 if pos >= 100 else 5 if pos >= 50 else 0)
    else:
        miss.append("No 0/1 churn label found (a label such as churned / target / attrition is needed)")
    if roles["entity"]:
        sig.append(f"Account / customer key '{roles['entity']}'")
        score += 20
    else:
        miss.append("No account or customer identifier")
    if roles["date"]:
        sig.append(f"Snapshot / period date '{roles['date']}'")
        score += 15
    if roles["value"]:
        sig.append(f"Revenue column '{roles['value']}' (enables revenue-at-risk)")
        score += 10
    feas["churn"] = {"score": min(score, 100 if roles["target"] else 25), "signals": sig, "missing": miss}

    # --- forecasting ---
    sig, miss, score = [], [], 0
    if times:
        npts = int(parse_time_column(df[times[0]["name"]]).nunique()) if n else 0
        sig.append(f"Time axis '{times[0]['name']}' with {npts:,} distinct periods")
        score += 40 if npts >= 24 else 25 if npts >= 10 else 10
        money = _hits(numeric, r"revenue|sales|amount|arr|mrr|value|demand|volume|qty|quantity|cost|spend|total_amount")
        if money:
            sig.append(f"Numeric series to forecast: {', '.join(money[:3])}")
            score += 25
        else:
            miss.append("No obvious revenue / demand measure to forecast")
    else:
        miss.append("No date or time column")
    feas["forecasting"] = {"score": min(score, 100), "signals": sig, "missing": miss}

    # --- revenue growth (cross-sell / upsell / lead scoring) ---
    sig, miss, score = [], [], 0
    prod = [c for c in _hits(cols, r"product|item|sku|plan|service|module|offering|package")
            if not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique() <= 500]
    if prod and roles["entity"]:
        sig.append(f"Customer key + product dimension '{prod[0]}' (cross-sell basket analysis)")
        score += 55
    elif prod:
        miss.append("Product column found but no customer key")
    lead = [c for c in _hits(cols, r"convert|converted|lead|won|opportunity|upsell|expansion|propensity")
            if df[c].nunique(dropna=True) == 2]
    if lead:
        sig.append(f"Outcome-like column(s) for lead / upsell scoring: {', '.join(lead[:3])}")
        score += 30
    if roles["value"]:
        score += 10
    if not sig:
        miss.append("No customer × product structure or conversion outcome detected")
    feas["revenue_growth"] = {"score": min(score, 100), "signals": sig, "missing": miss}

    # --- pricing ---
    sig, miss, score = [], [], 0
    price = _hits(numeric, r"price|unit_price|fee|tariff")
    qty = _hits(numeric, r"qty|quantity|units|volume")
    disc = _hits(numeric, r"discount|rebate|promo")
    if price:
        sig.append(f"Price column(s): {', '.join(price[:3])}")
        score += 40
    if qty:
        sig.append(f"Volume column(s): {', '.join(qty[:3])}")
        score += 25
    if disc:
        sig.append(f"Discount column(s): {', '.join(disc[:3])}")
        score += 25
    if not price:
        miss.append("No price column")
    feas["pricing"] = {"score": min(score, 100), "signals": sig, "missing": miss}

    # --- efficiency & cost ---
    sig, miss, score = [], [], 0
    cost = _hits(numeric, r"cost|expense|opex|hours|handle_time|resolution|effort|headcount|overtime|utili[sz]ation")
    if cost:
        sig.append(f"Cost / effort measures: {', '.join(cost[:4])}")
        score += 50 + (15 if len(cost) >= 3 else 0)
    else:
        miss.append("No cost, effort or utilisation measures")
    if times:
        score += 10
    feas["efficiency_cost"] = {"score": min(score, 100), "signals": sig, "missing": miss}

    flows = []
    best, best_score = None, -1
    for f in FLOWS:
        fe = feas[f["key"]]
        fe["verdict"] = _verdict(fe["score"])
        entry = {k: f[k] for k in ("key", "category", "name", "status", "tagline", "outputs", "priority")}
        entry["feasibility"] = fe
        flows.append(entry)
        if f["status"] == "available" and fe["score"] > best_score:
            best, best_score = f["key"], fe["score"]
    return clean({
        "summary": dataset_summary(df),
        "roles": {k: roles[k] for k in ("target", "entity", "date", "value")},
        "role_candidates": roles["candidates"],
        "columns": cols,
        "flows": flows,
        "recommended_flow": best if best_score >= 40 else None,
    })
