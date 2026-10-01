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
        "key": "revenue_growth", "category": "Revenue Growth", "name": "Cross-sell, upsell & lead scoring", "status": "available", "priority": 1,
        "tagline": "Cross-sell, upsell and lead-scoring playbooks that drive top-line growth.",
        "outputs": ["Next-best-product per customer", "Propensity scores", "Revenue opportunity sizing"], "stages": [],
    },
    {
        "key": "forecasting", "category": "Forecasting", "name": "Revenue, demand & cash-flow forecasting", "status": "available", "priority": 2,
        "tagline": "Revenue, demand and cash-flow forecasting models to support planning, budgeting and diligence.",
        "outputs": ["Forecast with intervals", "Backtested model comparison", "Seasonality & trend findings"], "stages": [],
    },
    {
        "key": "pricing", "category": "Pricing", "name": "Pricing & discount optimisation", "status": "available", "priority": 4,
        "tagline": "Re-pricing, discounting and monetization strategies to defend or grow margin.",
        "outputs": ["Price elasticity", "Discount leakage", "Margin uplift scenarios"], "stages": [],
    },
    {
        "key": "efficiency_cost", "category": "Efficiency & Cost", "name": "Efficiency & cost automation", "status": "available", "priority": 5,
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


def _col_values_match(df: pd.DataFrame, col: str, pattern: str) -> bool:
    """Check whether any sampled VALUES of a categorical column match a regex."""
    try:
        s = df[col].dropna().astype(str).head(500)
        return bool(s.str.contains(pattern, case=False, regex=True, na=False).any())
    except Exception:
        return False


def scan_dataset(df: pd.DataFrame) -> dict[str, Any]:
    cols = [str(c) for c in df.columns]
    roles = detect_churn_roles(df)
    times = detect_time_columns(df)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    categorical = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique(dropna=True) <= 500]
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
        # Broader revenue / amount pattern — catches invoice_amount, sub_total_gbp, ltm_revenue etc.
        money = _hits(numeric, r"revenue|sales|amount|arr|mrr|value|demand|volume|qty|quantity|cost|spend|total|ltm|billing|invoice|gbp|usd|eur")
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

    # Product-like dimension: classic names OR broader equivalents (solution, channel, vertical, tier…)
    PROD_NAME = r"product|item|sku|plan|service|module|offering|package|solution|channel|segment|vertical|tier|category|bundle|service_type"
    prod = [c for c in categorical if re.search(rf"(^|[_\s-])({PROD_NAME})([_\s-]|$)", c, re.I)]
    if prod and roles["entity"]:
        sig.append(f"Customer key + product/solution dimension '{prod[0]}' (cross-sell basket analysis)")
        score += 55
    elif prod:
        sig.append(f"Product/solution dimension '{prod[0]}' found")
        score += 25

    # Explicit outcome column names
    LEAD_NAME = r"convert|converted|lead|won|opportunity|upsell|expansion|propensity|cross.?sell|netsell|new.?client|new.?business"
    lead_by_name = [c for c in cols if re.search(rf"(^|[_\s-])({LEAD_NAME})([_\s-]|$)", c, re.I)]
    if lead_by_name:
        sig.append(f"Growth-outcome column(s): {', '.join(lead_by_name[:3])}")
        score += 35

    # Value-based: a categorical column whose VALUES contain cross-sell / growth keywords (e.g. bucket = "Solution Cross-Sell")
    LEAD_VALUES = r"cross.?sell|upsell|netsell|new.?client|new.?business|expansion|propensity|lead|won|converted"
    value_cols = [c for c in categorical if not lead_by_name and _col_values_match(df, c, LEAD_VALUES)]
    if value_cols:
        sig.append(f"Cross-sell / growth labels found in column values: {', '.join(value_cols[:3])}")
        score += 40

    # Customer segmentation enriches revenue growth modelling
    SEG_NAME = r"customer_type|business_model|vertical|fund_potential|depth|relationship|segment|tier|band|type|region"
    seg = [c for c in categorical if re.search(rf"(^|[_\s-])({SEG_NAME})([_\s-]|$)", c, re.I)]
    if seg:
        sig.append(f"Customer segmentation attribute(s): {', '.join(seg[:3])}")
        score += 15

    if roles["value"]:
        score += 10

    if not sig:
        miss.append("No customer × product structure or conversion outcome detected")
    feas["revenue_growth"] = {"score": min(score, 100), "signals": sig, "missing": miss}

    # --- pricing ---
    sig, miss, score = [], [], 0
    # Broader price detection: catches invoice_amount, sub_total_gbp_fixed, invoice_amount_gbp etc.
    PRICE_PAT = r"price|unit_price|fee|tariff|amount|invoice|billing|sub_total|charge|rate|gbp|usd|eur|total"
    price = [c for c in numeric if re.search(rf"(^|[_\s-])({PRICE_PAT})([_\s-]|$)", c, re.I)
             and df[c].nunique() > 5 and df[c].median() != 0]
    qty = _hits(numeric, r"qty|quantity|units|volume|count|invoices|num_invoices")
    disc = _hits(numeric, r"discount|rebate|promo|concession|reduction")
    # Dimension columns that enable price-by-segment analysis (solution, channel, customer_type…)
    DIM_PAT = r"solution|channel|segment|customer_type|region|vertical|tier|band|category|product|service"
    dims = [c for c in categorical if re.search(rf"(^|[_\s-])({DIM_PAT})([_\s-]|$)", c, re.I)]
    if price:
        sig.append(f"Invoice / revenue column(s): {', '.join(price[:3])}")
        score += 40
    if dims:
        sig.append(f"Pricing dimension(s) — {', '.join(dims[:3])} — enable price-by-segment analytics")
        score += 20
    if qty:
        sig.append(f"Volume column(s): {', '.join(qty[:3])}")
        score += 15
    if disc:
        sig.append(f"Discount column(s): {', '.join(disc[:3])}")
        score += 20
    if not price:
        miss.append("No price or invoice amount column detected")
    if not disc:
        miss.append("No discount / margin columns (limits full pricing optimisation)")
    feas["pricing"] = {"score": min(score, 100), "signals": sig, "missing": miss}

    # --- efficiency & cost ---
    sig, miss, score = [], [], 0
    # Broader cost / effort pattern — also picks up duration, project age, SLA etc.
    COST_PAT = r"cost|expense|opex|hours|handle_time|resolution|effort|headcount|overtime|utili[sz]ation|duration|days|age|sla|lead_time|cycle_time|turnaround|completion"
    cost = [c for c in numeric if re.search(rf"(^|[_\s-])({COST_PAT})([_\s-]|$)", c, re.I)]
    # Project / operational tables: status + date columns imply delivery analytics
    STATUS_PAT = r"status|state|stage|phase|milestone|flag|billable"
    status_cols = [c for c in categorical if re.search(rf"(^|[_\s-])({STATUS_PAT})([_\s-]|$)", c, re.I)]
    if cost:
        sig.append(f"Cost / effort measures: {', '.join(cost[:4])}")
        score += 50 + (15 if len(cost) >= 3 else 0)
    if status_cols and times:
        sig.append(f"Project / operational status column(s) + time axis (delivery analytics)")
        score += 30
    elif status_cols:
        sig.append(f"Project / operational status column(s): {', '.join(status_cols[:3])}")
        score += 15
    if times:
        score += 10
    if not cost and not status_cols:
        miss.append("No cost, effort, utilisation, or project delivery columns")
    elif not cost:
        miss.append("No cost / hours data — limits true cost-efficiency modelling")
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
