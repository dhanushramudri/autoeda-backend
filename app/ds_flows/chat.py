"""Chat about a finished churn run, for the executive dashboard.

The assistant is grounded: it only sees a fact sheet built from the run's own results and customer list (plus the rows of
any customer ID mentioned in the question), and is told to answer from that sheet alone. No model vocabulary, no invented
numbers.
"""

from __future__ import annotations

import io
import json
import re

import pandas as pd

from .common import pretty_feature

SYSTEM = (
    "You are the assistant on a customer churn dashboard used by business executives. Answer ONLY from the FACTS below. "
    "Use plain business language: no statistics or machine-learning terms (no AUC, SHAP, model, feature, probability score). "
    "Say 'chance of churning' instead of probability. Be concise: a short answer first, then bullets if useful. "
    "Quote numbers exactly as given. Refer to customers by their ID. If the facts do not contain the answer, say so plainly and "
    "say what you can answer instead. Do not invent customers, numbers or causes; drivers are associations, not proof of cause. "
    "When asked how to retain customers, base the advice on the warning signs and reasons in the facts."
)


def _num(x) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    a = abs(x)
    return f"{x / 1e6:.2f}M" if a >= 1e6 else f"{x / 1e3:.1f}K" if a >= 1e4 else f"{x:,.0f}"


def _driver(d: str) -> str:
    feat, _, val = str(d).partition(" = ")
    name = pretty_feature(feat)
    return f"{name} = {val}" if val and val not in ("0", "1") else name


def load_customers(run) -> pd.DataFrame | None:
    if not run.accounts_csv:
        return None
    return pd.read_csv(io.BytesIO(run.accounts_csv), low_memory=False)


def build_facts(run, df: pd.DataFrame | None, message: str) -> str:
    R = json.loads(run.results_json or "{}")
    v, eda, ex, hy = R.get("value") or {}, R.get("eda") or {}, R.get("explain") or {}, R.get("hypotheses") or {}
    val = v.get("value_column")
    out: list[str] = []
    out.append(f"{v.get('accounts_scored')} customers are due for renewal and have been scored (data as of {str(v.get('as_of'))[:10]}). "
               f"Risk levels: High = the 10% of customers with the highest chance of churning, Medium = the next 20%, Low = the rest.")
    for t in v.get("tiers", []):
        out.append(f"- {t['tier']} risk: {t['accounts']} customers, average chance of churning {t['avg_probability'] * 100:.1f}%"
                   + (f", revenue {_num(t.get('value'))}, expected revenue loss {_num(t.get('expected_loss'))}" if val else ""))
    if v.get("expected_loss_total") is not None:
        out.append(f"Expected revenue loss across all customers due for renewal: {_num(v['expected_loss_total'])} (total revenue {_num(v.get('total_value'))}).")
    if eda.get("base_rate") is not None:
        out.append(f"Usual churn rate in past renewals: {eda['base_rate'] * 100:.1f}%.")
    signs = [f"{pretty_feature(t['feature'])} ({'lower values' if str(t.get('direction', '')).startswith('lower') else 'higher values'} go with more churn)"
             for t in (ex.get("top_features") or [])[:8]]
    if signs:
        out.append("Biggest warning signs: " + "; ".join(signs) + ".")
    pairs = [h for h in (hy.get("hypotheses") or []) if h.get("verdict") == "supported" and h.get("churn_rate_high") is not None and h.get("test") == "Mann-Whitney U"][:6]
    if pairs:
        out.append("Churn rate when a factor is below / above its average: " + "; ".join(
            f"{pretty_feature(h['feature'])}: {h['churn_rate_low'] * 100:.0f}% / {h['churn_rate_high'] * 100:.0f}%" for h in pairs) + ".")
    segs = [s for s in (eda.get("segments") or []) if (s.get("lift") or 0) > 1][:6]
    if segs:
        out.append("Groups that churn more than average: " + "; ".join(
            f"{s['dimension'].replace('_', ' ')} = {s['group']}: {s['churn_rate'] * 100:.0f}% ({s['lift']:.1f}x average, {s['n']} rows)" for s in segs) + ".")

    if df is not None and len(df):
        d = df.copy()
        loss_col = "expected_value_at_risk" if "expected_value_at_risk" in d.columns else "churn_probability"
        top = d.sort_values(loss_col, ascending=False).head(25)
        if "as_of_snapshot" in d.columns:
            dt = pd.to_datetime(d["as_of_snapshot"], errors="coerce")
            if dt.notna().any():
                d["_m"] = dt.dt.strftime("%Y-%m")
                lines = []
                for m, g in d.dropna(subset=["_m"]).groupby("_m"):
                    lines.append(f"{m}: {len(g)} customers due, about {g['churn_probability'].sum():.0f} expected to churn, {int((g['risk_tier'] == 'High').sum())} high-risk"
                                 + (f", expected loss {_num(g['expected_value_at_risk'].sum())}" if "expected_value_at_risk" in g.columns else ""))
                out.append("By renewal month (customers whose renewal falls due in that month): " + "; ".join(lines[:18]) + ".")
        out.append("Top customers by expected loss (ID | renewal due | chance of churning | level | revenue | expected loss | reasons):")
        for r in top.itertuples(index=False):
            rr = r._asdict()
            why = ", ".join(_driver(rr[k]) for k in ("driver_1", "driver_2", "driver_3") if k in rr and isinstance(rr[k], str) and rr[k])
            out.append(f"  {rr['account']} | {str(rr.get('as_of_snapshot', ''))[:10]} | {rr['churn_probability'] * 100:.0f}% | {rr['risk_tier']} | {_num(rr.get(val, ''))} | {_num(rr.get('expected_value_at_risk', ''))} | {why or 'several factors'}")
        fixed = {"account", "customer_name", "as_of_snapshot", "_m", "churn_probability", "risk_tier", "expected_value_at_risk", "driver_1", "driver_2", "driver_3", val}
        for c in [c for c in d.columns if c not in fixed][:3]:
            g = d.groupby([c, "risk_tier"]).size().unstack(fill_value=0)
            out.append(f"Customers by {c} and risk level: " + "; ".join(f"{k}: " + ", ".join(f"{t} {int(g.loc[k, t])}" for t in g.columns) for k in g.index[:10]) + ".")
        ids = {t.lower() for t in re.findall(r"[A-Za-z0-9_\-]{3,}", message)}
        hit = d[d["account"].astype(str).str.lower().isin(ids)].head(5)
        for r in hit.itertuples(index=False):
            rr = r._asdict()
            why = ", ".join(_driver(rr[k]) for k in ("driver_1", "driver_2", "driver_3") if k in rr and isinstance(rr[k], str) and rr[k])
            extra = ", ".join(f"{c} = {rr[c]}" for c in d.columns if c not in fixed and c in rr and rr[c] not in ("", None))
            out.append(f"Customer {rr['account']}: chance of churning {rr['churn_probability'] * 100:.0f}% ({rr['risk_tier']} risk), revenue {_num(rr.get(val, ''))}, "
                       f"expected loss {_num(rr.get('expected_value_at_risk', ''))}; renewal due {str(rr.get('as_of_snapshot', ''))[:10]}; reasons: {why or 'several factors'}" + (f"; {extra}" if extra else "") + ".")
    return "\n".join(out)[:9000]


def answer(run, message: str, history: list[dict]) -> str:
    from ..ai.llm import get_provider

    provider = get_provider()
    if provider is None:
        raise RuntimeError("AI isn't configured")
    df = load_customers(run)
    facts = build_facts(run, df, message)
    convo = "\n".join(f"{'User' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:600]}" for m in (history or [])[-6:])
    prompt = f"{SYSTEM}\n\nFACTS:\n{facts}\n\n" + (f"CONVERSATION SO FAR:\n{convo}\n\n" if convo else "") + f"User question: {message}\n\nAnswer:"
    text = provider.generate(prompt, temperature=0.2, max_tokens=700)
    return (text or "").strip() or "I couldn't produce an answer — try rephrasing the question."
