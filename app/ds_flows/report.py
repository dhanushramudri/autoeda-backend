"""Headline KPIs, stage summaries, narrative (LLM with deterministic fallback) and the Markdown report.

The LLM never produces a number: it is handed the computed facts and only words them. If no LLM is
configured, or its answer cannot be parsed, a template built from the same facts is used. Caveats
(label-definition warning, leakage exclusions) are appended programmatically so they can never be dropped.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .common import pretty_feature

logger = logging.getLogger("autoeda.ds_flows.report")


def fmt_num(x: float | None, pct: bool = False, digits: int = 1) -> str:
    if x is None:
        return "n/a"
    if pct:
        return f"{x * 100:.{digits}f}%"
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:.2f}B"
    if a >= 1e6:
        return f"{x / 1e6:.2f}M"
    if a >= 1e3:
        return f"{x / 1e3:.1f}K"
    return f"{x:.{digits}f}" if a < 100 and x != int(x) else f"{x:,.0f}"


def _label(f: str) -> str:
    return pretty_feature(f)


# ---------------------------------------------------------------------------
# stage summaries (shown on the live timeline)
# ---------------------------------------------------------------------------

def stage_summary(key: str, r: dict) -> tuple[str, list[str]]:
    if key == "discover":
        lc = r["label_counts"]
        s = f"Outcome '{r['label']['column']}' in {r['base_table']} · {lc['churned']:,} churned, {lc['retained']:,} retained, {lc['unlabeled']:,} open"
        logs = [f"{t['name']}: {t['rows']:,} rows — {t['role']}" for t in r["tables"]] + r["log"]
        logs.append(f"Account key {r['link_key']} · period {r['period_col']} · revenue {r['value_col']}")
        return s, logs
    if key == "understand":
        s = f"{r['rows']:,} rows · {r['entities']:,} accounts · churn rate {fmt_num(r['churn_rate'], pct=True)}"
        logs = [f"Detected columns — label: {r['target_col']}, account: {r['entity_col']}, date: {r['date_col']}, value: {r['value_col']}",
                f"{r['positives']:,} churned rows of {r['labeled_rows']:,} labelled"]
        if r.get("is_panel"):
            logs.append(f"Panel data: ~{r['rows_per_entity_median']:.0f} rows per account across {r['snapshots']} snapshots ({r['date_min']} → {r['date_max']})")
        if r.get("quality"):
            logs.append(f"Data quality score {r['quality']['overall']}/100")
        return s, logs + r.get("issues", [])
    if key == "leakage":
        ex = r["excluded"]
        s = f"{len(ex)} leaking column(s) excluded" if ex else "No leakage found"
        logs = [f"Screened {r['candidate_features']} candidate features"] + [f"Excluded {e['feature']}: {e['reason']}" for e in ex]
        logs += [f"Watch {w['feature']}: {w['note']}" for w in r["warnings"][:4]]
        return s, logs
    if key == "eda":
        segs = r.get("segments", [])
        s = f"Base churn rate {fmt_num(r['base_rate'], pct=True)}; {len(segs)} segment cut(s), {len(r.get('driver_bins', []))} driver profiles"
        logs = [f"{x['dimension']} = {x['group']}: {fmt_num(x['churn_rate'], pct=True)} churn ({x['lift']:.1f}× average, n={x['n']:,})" for x in segs[:4]]
        return s, logs
    if key == "hypotheses":
        s = f"{r['supported']} of {r['total']} hypotheses supported (FDR-corrected)"
        logs = [f"Tested on {r['tested_on_rows']:,} rows" + (" — one row per account so repeated snapshots don't inflate significance" if r["one_row_per_entity"] else "")]
        logs += [f"{h['verdict'].upper()}: {h['statement']} ({h['effect_label']}, q={h['q_value']:.3g})" for h in r["hypotheses"][:5]]
        return s, logs
    if key == "features":
        s = f"{r['total_features']} features ({r['created_count']} engineered)"
        return s, [f"{k}: {v}" for k, v in r["kinds"].items()] or ["No engineered features were needed"]
    if key == "select":
        sp = r["split"]
        s = f"{r['selected']} of {r['start_features']} features kept · {sp['kind']} split"
        logs = [f"Train {sp['train_rows']:,} rows / {sp['train_entities']:,} accounts; holdout {sp['holdout_rows']:,} rows / {sp['holdout_entities']:,} unseen accounts",
                f"Dropped: " + ", ".join(f"{v} {k}" for k, v in r["dropped_by_reason"].items())]
        return s, logs
    if key == "models":
        best = r["selected_model"]
        ho = r["holdout_metrics"]
        s = f"{best} selected · holdout AUC {ho['roc_auc']:.3f}, top-decile lift {ho['lift_top10']:.1f}×"
        logs = [f"Quarantined {q['feature']}: {q['reason']}" for q in r.get("quarantined", [])]
        for b in r["leaderboard"]:
            if b.get("cv"):
                logs.append(f"{b['model']}: CV PR-AUC {b['cv']['pr_auc']:.3f}, holdout AUC {b['holdout']['roc_auc']:.3f}" + (" ← selected" if b.get("selected") else ""))
        logs.append(f"Calibration error {r['calibration']['ece_before']:.3f} → {r['calibration']['ece_after']:.3f}")
        return s, logs
    if key == "explain":
        top = r["top_features"][:5]
        s = "Top drivers: " + ", ".join(_label(t["feature"]) for t in top[:3])
        return s, [f"{_label(t['feature'])} — {t['direction']}" for t in top]
    if key == "value":
        hi = next((t for t in r["tiers"] if t["tier"] == "High"), None)
        s = f"{hi['accounts']} high-risk accounts" if hi else "Tiers assigned"
        if hi and r.get("expected_loss_total") is not None:
            s += f" · expected {r['value_column']} at risk {fmt_num(r['expected_loss_total'])}"
        return s, [r["tier_definition"]]
    if key == "validate":
        return f"{r['passed']}/{r['total']} checks passed", [f"{c['status'].upper()}: {c['check']} — {c['detail']}" for c in r["checks"]]
    if key == "build":
        i = r["integrity"]
        s = f"{r['enriched_rows']:,} rows × {r['enriched_columns']} columns · {i['added_columns']} columns added"
        return s, [f"Original rows preserved: {i['rows_preserved']}", f"Original columns & values unchanged: {i['original_values_unchanged']}",
                   f"Account-level file: {r['account_file_rows']:,} accounts"]
    return "", []


# ---------------------------------------------------------------------------
# headline
# ---------------------------------------------------------------------------

def apply_quarantine(results: dict) -> None:
    """Features quarantined by the model stage must not still be presented as findings elsewhere."""
    q = {x["feature"] for x in (results.get("models") or {}).get("quarantined", [])}
    if not q:
        return
    hy = results.get("hypotheses")
    if hy:
        for h in hy["hypotheses"]:
            if h.get("feature") in q:
                h["verdict"] = "quarantined"
                h["note"] = "feature quarantined as likely recorded at or after the decision"
        hy["supported"] = sum(h["verdict"] == "supported" for h in hy["hypotheses"])
    eda = results.get("eda")
    if eda:
        for key in ("driver_bins", "distributions", "outliers", "profile"):
            if eda.get(key):
                eda[key] = [d for d in eda[key] if d["feature"] not in q]
        cor = eda.get("correlation")
        if cor:
            keep = [i for i, f in enumerate(cor["features"]) if f not in q]
            cor["features"] = [cor["features"][i] for i in keep]
            cor["matrix"] = [[cor["matrix"][i][j] for j in keep] for i in keep]


def build_headline(results: dict) -> dict:
    u, m, v = results.get("understand"), results.get("models"), results.get("value")
    ex, val, lk = results.get("explain"), results.get("validate"), results.get("leakage")
    h: dict[str, Any] = {}
    if u:
        h.update(rows=u["rows"], accounts=u["entities"], churn_rate=u["churn_rate"], positives=u["positives"])
    if m:
        ho = m["holdout_metrics"]
        h.update(model=m["selected_model"], roc_auc=ho["roc_auc"], pr_auc=ho["pr_auc"], lift_top10=ho["lift_top10"],
                 recall_top10=ho["recall_top10"], precision_at_threshold=m["operating_point"]["precision"], recall_at_threshold=m["operating_point"]["recall"])
    if v:
        hi = next((t for t in v["tiers"] if t["tier"] == "High"), None)
        h.update(population=v.get("population"), value_column=v.get("value_column"), total_value=v.get("total_value"), expected_loss_total=v.get("expected_loss_total"),
                 high_risk_accounts=hi["accounts"] if hi else None, high_risk_value=hi.get("value") if hi else None,
                 high_risk_expected_loss=hi.get("expected_loss") if hi else None, accounts_scored=v["accounts_scored"], as_of=v.get("as_of"))
    if ex:
        h["top_drivers"] = [{"feature": t["feature"], "direction": t["direction"], "importance": t["importance"]} for t in ex["top_features"][:6]]
    if val:
        h["validation"] = val["overall"]
        warn = next((c for c in val["checks"] if c["check"] == "Label behaves like churn" and c["status"] != "pass"), None)
        h["label_warning"] = warn["detail"] if warn else None
    if lk:
        h["leakage_excluded"] = [e["feature"] for e in lk["excluded"] if not e["reason"].startswith("implausibly")]
    probe = (lk or {}).get("probe")
    qs = ([(x["feature"], probe["auc_with"]) for x in probe["removed"]] if probe else []) + [(q["feature"], q["auc_with"]) for q in ((m or {}).get("quarantined") or [])]
    if qs:
        h["quarantined"] = [f for f, _a in qs]
        h["suspicious_auc"] = qs[0][1]
    d = results.get("discover")
    if d:
        h["data_used"] = {"base_table": d["base_table"], "linked": [a["table"] for a in d.get("attached", [])],
                          "outcome_column": d["label"]["column"], "churned": d["label_counts"]["churned"],
                          "retained": d["label_counts"]["retained"], "open": d["label_counts"]["unlabeled"]}
    return h


# ---------------------------------------------------------------------------
# narrative
# ---------------------------------------------------------------------------

# Only drivers with an accurate, specific remedy get an action. A driver with no rule is still listed in
# "Drivers"; it just isn't given an invented recommendation.
_ACTION_RULES = [
    (r"auto[_ ]?renew", "Move these accounts onto auto-renewal and confirm their payment method before the renewal date."),
    (r"suggested[_ ]?leave|desire[_ ]?to[_ ]?cancel|switching|competitor",
     "The customer has signalled they may leave: assign an owner for a save call before the renewal date."),
    (r"accreditation|engagement", "Follow up on accreditation progress and contractor engagement with these accounts."),
    (r"complain|dissatisf|negative[_ ]?(customer|experience)", "Resolve open complaints and service issues before the renewal conversation."),
    (r"overdue|dunning|late[_ ]?payment|payment[_ ]?(issue|failed)", "Clear overdue or failed payments early so billing friction doesn't turn into cancellation."),
    (r"upsell|downsell|expansion", "Review the account's recent commercial changes with the account owner before renewal."),
]


def _action_for(feature: str) -> str | None:
    for rx, act in _ACTION_RULES:
        if re.search(rx, feature, re.I):
            return act
    return None


def _template_narrative(h: dict, hyps: list[dict]) -> dict:
    parts = []
    pop = h.get("population") or "accounts"
    if h.get("high_risk_accounts") is not None and h.get("expected_loss_total") is not None:
        parts.append(f"{h['accounts_scored']:,} {pop} scored. {h['high_risk_accounts']:,} are high risk, holding {fmt_num(h['high_risk_value'])} {h['value_column']}, "
                     f"of which {fmt_num(h['high_risk_expected_loss'])} is expected to be lost. Expected loss across all {pop}: {fmt_num(h['expected_loss_total'])}.")
    elif h.get("accounts_scored") is not None:
        parts.append(f"{h['accounts_scored']:,} {pop} scored; {h.get('high_risk_accounts') or 0:,} are high risk.")
    if h.get("model"):
        parts.append(f"{h['model']}: AUC {h['roc_auc']:.2f} on accounts it never saw. The riskiest 10% churn at {h['lift_top10']:.1f}x the average rate "
                     f"and contain {h['recall_top10'] * 100:.0f}% of all churners.")
    if h.get("top_drivers"):
        parts.append("Top drivers: " + "; ".join(f"{_label(d['feature'])}: {d['direction']}" for d in h["top_drivers"][:3]) + ".")
    seen, actions = set(), []
    for d in h.get("top_drivers", []):
        act = _action_for(d["feature"])
        if act is None or act in seen:
            continue
        seen.add(act)
        actions.append({"driver": f"{_label(d['feature'])}: {d['direction']}", "action": act})
        if len(actions) >= 5:
            break
    return {"executive_summary": " ".join(parts), "actions": actions, "source": "template"}


def _display_facts(h: dict, hyps: list[dict], tiers: list[dict]) -> dict:
    vc = h.get("value_column")
    f: dict[str, Any] = {
        "accounts_analysed": f"{h.get('accounts', 0):,}", "rows_analysed": f"{h.get('rows', 0):,}",
        "churn_rate": fmt_num(h.get("churn_rate"), pct=True), "model_chosen": h.get("model"),
        "roc_auc_on_unseen_accounts": f"{h['roc_auc']:.2f}" if h.get("roc_auc") else None,
        "riskiest_10pct_churn_rate_vs_average": f"{h['lift_top10']:.1f} times" if h.get("lift_top10") else None,
        "share_of_churners_in_riskiest_10pct": fmt_num(h.get("recall_top10"), pct=True, digits=0),
        "data_as_of": h.get("as_of"), "validation_result": h.get("validation"),
        "top_drivers": [{"driver": _label(d["feature"]), "direction": d["direction"]} for d in h.get("top_drivers", [])[:5]],
        "supported_hypotheses": [x["statement"] for x in hyps if x["verdict"] == "supported"][:5],
        "caveat_to_mention": h.get("label_warning"),
    }
    if vc:
        f.update({
            "value_measure": vc, "total_value_across_accounts": fmt_num(h.get("total_value")),
            "high_risk_accounts": f"{h.get('high_risk_accounts'):,}" if h.get("high_risk_accounts") is not None else None,
            "high_risk_value": fmt_num(h.get("high_risk_value")),
            "probability_weighted_value_at_risk_all_accounts": fmt_num(h.get("expected_loss_total")),
            "probability_weighted_value_at_risk_high_tier": fmt_num(h.get("high_risk_expected_loss")),
        })
    return {k: v for k, v in f.items() if v not in (None, "n/a")}


def _q(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _llm_narrative(h: dict, hyps: list[dict], tiers: list[dict]) -> dict | None:
    try:
        from ..ai.llm import get_provider
        provider = get_provider()
        if provider is None:
            return None
        facts = _display_facts(h, hyps, tiers)
        prompt = (
            "You are a senior data scientist writing for business executives. Below are FACTS computed by a churn analysis. "
            "Write using ONLY these facts — do not invent, round differently, or add any number that is not present.\n\n"
            f"FACTS (JSON):\n{json.dumps(facts, default=str, indent=1)[:6000]}\n\n"
            'Respond with ONLY a JSON object: {"executive_summary": "<=110 words, plain business English, no jargon", '
            '"actions": [{"driver": "<driver name from the facts>", "action": "<one specific retention action, <=30 words>"}]} with 3 to 5 actions.'
        )
        raw = provider.generate(prompt, temperature=0.2, max_tokens=900)
        if not raw:
            return None
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else None
        if not data or not isinstance(data.get("executive_summary"), str) or not isinstance(data.get("actions"), list):
            return None
        data["actions"] = [a for a in data["actions"] if isinstance(a, dict) and a.get("action")][:5]
        data["source"] = "llm"
        return data
    except Exception:
        logger.exception("LLM narrative failed — using template")
        return None


def build_narrative(results: dict, headline: dict) -> dict:
    hyps = (results.get("hypotheses") or {}).get("hypotheses", [])
    tiers = (results.get("value") or {}).get("tiers", [])
    nar = _template_narrative(headline, hyps)
    caveats = []
    if headline.get("label_warning"):
        caveats.append("Label check: " + headline["label_warning"])
    if headline.get("leakage_excluded"):
        caveats.append("Excluded, leaks the outcome: " + ", ".join(headline["leakage_excluded"]) + ".")
    if headline.get("quarantined"):
        caveats.append(f"Removed as implausibly predictive (first model AUC {headline['suspicious_auc']:.2f}): " + ", ".join(headline["quarantined"])
                       + ". Confirm they are known before the renewal.")
    nar["caveats"] = caveats
    return nar


# ---------------------------------------------------------------------------
# markdown report (compatible with app/eda/report_builder: one table max per "## " section)
# ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_markdown(run_title: str, dataset_name: str, results: dict, headline: dict, narrative: dict) -> str:
    h, md = headline, []
    u = results.get("understand") or {}
    md.append(f"## Executive summary\n\n{narrative['executive_summary']}\n\n" + "\n".join(f"- {c}" for c in narrative.get("caveats", [])))

    v = results.get("value")
    if v:
        rows = []
        for t in v["tiers"]:
            rows.append([t["tier"], f"{t['accounts']:,}", fmt_num(t["avg_probability"], pct=True)]
                        + ([fmt_num(t.get("value")), fmt_num(t.get("expected_loss"))] if v.get("value_column") else []))
        heads = ["Risk tier", "Accounts", "Avg churn probability"] + ([f"{v['value_column']}", "Expected at risk"] if v.get("value_column") else [])
        md.append(f"## Revenue at risk\n\n{_table(heads, rows)}\n\n{v['tier_definition']}. Expected at risk = churn probability × {v.get('value_column') or 'value'}, as of {v.get('as_of')}.")

    if u:
        md.append(f"## Data overview\n\n{u['rows']:,} rows and {u['columns']} columns covering {u['entities']:,} accounts"
                  + (f" over {u['snapshots']} snapshots ({u['date_min']} to {u['date_max']})" if u.get("snapshots") else "")
                  + f". {u['positives']:,} rows are labelled as churned ({fmt_num(u['churn_rate'], pct=True)}). "
                  + (f"Data quality score: {u['quality']['overall']}/100." if u.get("quality") else ""))

    lk = results.get("leakage")
    if lk:
        rows = [[e["feature"], e["reason"]] for e in lk["excluded"]] or [["—", "No leaking columns found"]]
        md.append(f"## Data leakage safeguards\n\n{_table(['Column', 'Why it was excluded'], rows)}")

    hy = results.get("hypotheses")
    if hy:
        rows = [[x["statement"], x["verdict"], x["effect_label"], _q(x['q_value'])] for x in hy["hypotheses"] if x["verdict"] != "quarantined"][:8]
        md.append(f"## What drives churn\n\n{_table(['Hypothesis', 'Verdict', 'Effect size', 'q-value'], rows)}\n\nTested on {hy['tested_on_rows']:,} rows with {hy['correction']}.")

    m = results.get("models")
    if m:
        rows = []
        for b in m["leaderboard"]:
            if b.get("cv"):
                rows.append([b["model"] + (" ✓" if b.get("selected") else ""), f"{b['cv']['pr_auc']:.3f}", f"{b['holdout']['roc_auc']:.3f}", f"{b['holdout']['pr_auc']:.3f}", f"{b['holdout']['lift_top10']:.1f}×"])
        md.append(f"## Model comparison\n\n{_table(['Model', 'CV PR-AUC', 'Holdout ROC-AUC', 'Holdout PR-AUC', 'Top-10% lift'], rows)}\n\nThe model was chosen on cross-validated PR-AUC and judged on accounts held out from training entirely.")

    ex = results.get("explain")
    if ex:
        rows = [[_label(t["feature"]), t["direction"], f"{t['importance']:.4f}"] for t in ex["top_features"][:10]]
        md.append(f"## Top churn drivers\n\n{_table(['Feature', 'Direction', 'Importance (AUC drop)'], rows)}")

    va = results.get("validate")
    if va:
        rows = [[c["check"], c["status"].upper(), c["detail"]] for c in va["checks"]]
        md.append(f"## Validation\n\n{_table(['Check', 'Result', 'Detail'], rows)}")

    if narrative.get("actions"):
        md.append("## Recommended actions\n\n" + "\n".join(f"- **{a.get('driver', '')}** — {a['action']}" for a in narrative["actions"]))

    b = results.get("build")
    if b:
        rows = [[d["column"], d["description"]] for d in b["dictionary"]]
        md.append(f"## Added columns\n\nOriginal rows and columns are unchanged. Filter ds_is_current_row = 1 for one row per account.\n\n{_table(['Added column', 'Meaning'], rows)}")

    md.append("## Notes\n\n- Expected loss = churn probability x the value column; it does not say when revenue is lost.\n"
              "- Drivers are associations, not proof of cause. A driver can also be a reaction to risk (for example a discount offered to a customer who had already threatened to leave).\n"
              "- Hypothesis tests use one row per account; p-values are corrected for multiple testing (Benjamini-Hochberg).")
    return "\n\n".join(md) + "\n"
