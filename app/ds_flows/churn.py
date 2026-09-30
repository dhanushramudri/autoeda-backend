"""
Churn flow — the stages of a typical data-science churn engagement, end to end:

  understand -> leakage audit -> EDA -> hypotheses -> feature engineering ->
  feature selection -> model selection/training -> explainability ->
  risk tiers & revenue at risk -> validation -> deliverables

Every stage is a top-level function `stage_*(ctx) -> (result, artifacts)` so it can run in the
shared process pool (see app/process_pool.py). `ctx` carries the raw frame (`df`), the chosen
column `roles`, run `params`, earlier stage `results` (JSON) and `art` (DataFrames/models).

All statistics and metrics come from pandas / scipy / scikit-learn — nothing is LLM-generated.

Evaluation design (important for panel data such as weekly account snapshots): the same account
appears in many rows with a near-identical label, so a random row split would leak the account's
identity across train/test and inflate every metric. We therefore always split and cross-validate
BY ENTITY (held-out accounts are never seen in training).
"""

from __future__ import annotations

import io
import os
import re
import time
import warnings
from datetime import date as _date
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..eda.ts_columns import parse_time_column
from .common import clean, encode_target, pretty_feature

warnings.filterwarnings("ignore")

SEED = 42
META = ("__y", "__date", "__entity", "__row")
# Training rows are sampled by account above this (50k keeps a 4 GB server safe; override with env DS_FLOW_MAX_TRAIN_ROWS).
MAX_TRAIN_ROWS = int(os.environ.get("DS_FLOW_MAX_TRAIN_ROWS", 50_000))

_LEAK_NAME = re.compile(r"(^|_)(future|next|post|after|forward|outcome|churned|cancel|cancelled|canceled|lost|renewed|retained)(_|$)", re.I)
_ID_LIKE = re.compile(r"(_id|^id|_ref|_key|_uuid)$", re.I)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _auc_fast(x: np.ndarray, y: np.ndarray) -> float | None:
    """ROC-AUC of a single feature via rank-sum (NaNs dropped)."""
    m = ~np.isnan(x)
    x, y = x[m], y[m]
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 < 5 or n0 < 5 or np.nanmin(x) == np.nanmax(x):
        return None
    r = stats.rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out.tolist()


def _feature_frame(work: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    """Numeric (and bool) candidate features, minus meta/excluded/id-like columns."""
    cols = {}
    n = max(len(work), 1)
    for c in work.columns:
        if c in META or c in excluded:
            continue
        s = work[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            v = _num(s)
            nu = v.nunique()
            if nu < 2:
                continue
            if _ID_LIKE.search(str(c)) and nu / n > 0.9:
                continue
            cols[c] = v
    return pd.DataFrame(cols, index=work.index)


def _labeled(work: pd.DataFrame) -> pd.Series:
    return work["__y"].notna()


# ---------------------------------------------------------------------------
# 1. understand
# ---------------------------------------------------------------------------

def stage_understand(ctx):
    df: pd.DataFrame = ctx["df"]
    roles = ctx["roles"]
    tgt, ent, dcol, val = roles.get("target"), roles.get("entity"), roles.get("date"), roles.get("value")
    if not tgt or tgt not in df.columns:
        raise ValueError("No churn label column selected — pick the column that marks churned accounts (0/1)")

    work = df.reset_index(drop=True).copy()
    # numbers stored as text ("" for blanks) -> numeric, so they can be used as features
    for c in list(work.columns):
        if c == tgt or c == ent or c == dcol or not (pd.api.types.is_object_dtype(work[c]) or pd.api.types.is_string_dtype(work[c])):
            continue
        t = work[c].astype(str).str.strip()
        blank = t.eq("") | t.str.lower().isin(["nan", "none", "null"])
        if (~blank).sum() < 30:
            continue
        num = pd.to_numeric(t.where(~blank), errors="coerce")
        if num.notna().sum() >= 0.9 * (~blank).sum() and num.nunique() > 2:
            work[c] = num
    work["__y"] = encode_target(work[tgt], roles.get("positive")).values
    work["__row"] = np.arange(len(work))
    work["__entity"] = work[ent].astype(str).values if ent and ent in work.columns else np.arange(len(work)).astype(str)
    dates = None
    if dcol and dcol in work.columns:
        d = parse_time_column(work[dcol])
        if d.notna().mean() >= 0.5:
            dates = d
    work["__date"] = dates.values if dates is not None else pd.NaT

    lab = _labeled(work)
    positives = int((work.loc[lab, "__y"] == 1).sum())
    n_ent = int(work["__entity"].nunique())
    rows_per_entity = work.groupby("__entity").size()
    is_panel = bool(ent) and float(rows_per_entity.median()) > 1
    snapshots = int(work["__date"].nunique()) if dates is not None else None
    freq_days = None
    if dates is not None and snapshots and snapshots > 1:
        ud = np.sort(work["__date"].dropna().unique())
        freq_days = float(np.median(np.diff(ud).astype("timedelta64[D]").astype(float)))

    issues: list[str] = []
    if positives < 50:
        issues.append(f"Only {positives} churned rows — too few for a reliable model (aim for 100+).")
    elif positives < 100:
        issues.append(f"Only {positives} churned rows — results will be noisy.")
    if (~lab).sum():
        issues.append(f"{int((~lab).sum())} rows have no churn label; they are scored by the final model but not used for training.")
    if dcol and dates is None:
        issues.append(f"Could not parse '{dcol}' as dates — time-based checks skipped.")

    quality = None
    try:
        from ..eda.quality_score import run_quality_score
        sample = df if len(df) <= 20_000 else df.sample(20_000, random_state=SEED)
        q = run_quality_score(sample)
        quality = {k: q[k] for k in ("overall", "completeness", "consistency", "uniqueness")}
        quality["issues"] = [i["description"] for i in q["issues"][:6]]
    except Exception:
        pass

    miss = df.isna().mean().sort_values(ascending=False)
    result = {
        "target_col": tgt, "entity_col": ent, "date_col": dcol, "value_col": val,
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "entities": n_ent, "is_panel": is_panel,
        "rows_per_entity_median": float(rows_per_entity.median()),
        "snapshots": snapshots,
        # quantiles, so a handful of garbage dates (e.g. year 2050) don't define the reported period
        "date_min": work["__date"].quantile(0.001) if dates is not None else None,
        "date_max": work["__date"].quantile(0.995) if dates is not None else None,
        "snapshot_freq_days": freq_days,
        "labeled_rows": int(lab.sum()), "unlabeled_rows": int((~lab).sum()),
        "positives": positives, "churn_rate": float(work.loc[lab, "__y"].mean()) if lab.any() else None,
        "quality": quality,
        "missing_top": [{"column": c, "pct": round(float(p * 100), 1)} for c, p in miss.head(8).items() if p > 0],
        "issues": issues,
    }
    return clean(result), {"work": work}


# ---------------------------------------------------------------------------
# 2. leakage audit
# ---------------------------------------------------------------------------

def stage_leakage(ctx):
    work: pd.DataFrame = ctx["art"]["work"]
    roles = ctx["roles"]
    structural = {c for c in (roles.get("target"), roles.get("entity"), roles.get("date")) if c}
    structural |= set(ctx["params"].get("exclude_columns") or [])
    lab = _labeled(work)
    y = work.loc[lab, "__y"].to_numpy()

    feats = _feature_frame(work, structural)
    excluded: dict[str, str] = {c: "this is the outcome itself — it was used to build the churn label" for c in (ctx["params"].get("exclude_columns") or []) if c in work.columns}
    warns: list[dict] = []

    # (a) names that describe the future / the outcome itself
    by_name = [c for c in feats.columns if _LEAK_NAME.search(str(c))]
    for c in by_name:
        excluded[c] = "name indicates post-outcome information (future / outcome)"

    # (b) columns that are algebraic functions of an excluded column, e.g. change = future - current
    if by_name:
        cols = list(feats.columns)[:400]
        Xm = feats[cols].to_numpy(dtype=float)
        # only difference-style columns can be "outcome minus baseline"; tested on the rows where
        # the derived column is non-zero so the many no-change rows can't fake a match
        derived_name = re.compile(r"(change|diff|delta|growth|gap|net|movement)", re.I)
        for ci, c in enumerate(cols):
            if c in excluded or not derived_name.search(str(c)):
                continue
            cv = Xm[:, ci]
            for L in by_name:
                lv = feats[L].to_numpy(dtype=float)
                base_ok = ~(np.isnan(cv) | np.isnan(lv)) & (cv != 0)
                if base_ok.sum() < 50:
                    continue
                resid = (lv - cv)[:, None]  # would equal the baseline column if c = L - baseline
                with np.errstate(invalid="ignore"):
                    match = np.abs(Xm - resid) <= 1e-6 * (1 + np.abs(Xm))
                share = match[base_ok].mean(axis=0)
                share[[ci, cols.index(L) if L in cols else ci]] = 0
                j = int(np.argmax(share))
                if share[j] >= 0.7:
                    excluded[c] = f"derived from outcome-leaking '{L}' ({c} = {L} - {cols[j]} on {share[j]*100:.0f}% of changed rows)"
                    break

    # (c) statistical screen: a single feature that nearly separates the classes
    uni_rows = []
    for c in feats.columns:
        a = _auc_fast(feats[c].to_numpy(dtype=float)[lab.to_numpy()], y)
        if a is None:
            continue
        miss = float(feats[c].isna().mean())
        uni_rows.append({"feature": c, "auc": a, "strength": abs(a - 0.5) * 2, "direction": "higher" if a >= 0.5 else "lower", "missing_pct": miss * 100})
    uni = pd.DataFrame(uni_rows).sort_values("strength", ascending=False) if uni_rows else pd.DataFrame(columns=["feature", "auc", "strength", "direction", "missing_pct"])
    for r in uni.itertuples():
        if r.feature in excluded:
            continue
        s = max(r.auc, 1 - r.auc)
        if s >= 0.90:
            excluded[r.feature] = f"single-feature AUC {s:.2f} — almost certainly encodes the outcome"
        elif s >= 0.80:
            warns.append({"feature": r.feature, "note": f"single-feature AUC {s:.2f} — verify it is known before churn happens"})

    # (d) missingness that reveals the label
    yy = work["__y"]
    for c in feats.columns:
        if c in excluded:
            continue
        m = feats[c].isna()
        if 0.02 < m.mean() < 0.98:
            r1 = m[lab & (yy == 1)].mean()
            r0 = m[lab & (yy == 0)].mean()
            if abs(r1 - r0) >= 0.6:
                excluded[c] = f"only filled in for one outcome (missing in {r1*100:.0f}% of churned vs {r0*100:.0f}% of retained rows)"
            elif abs(r1 - r0) > 0.35:
                warns.append({"feature": c, "note": f"missing in {r1*100:.0f}% of churned vs {r0*100:.0f}% of retained rows — missingness may reveal the outcome"})

    # (e) text/categorical columns that (almost) state the outcome, e.g. a status column listing churn reasons
    from scipy.stats import chi2_contingency
    labeled_idx = work.index[lab]
    samp = labeled_idx if len(labeled_idx) <= 60_000 else pd.Index(np.random.RandomState(SEED).choice(labeled_idx, 60_000, replace=False))
    ys = work.loc[samp, "__y"]
    cat_checked = 0
    for c in work.columns:
        if c in META or c in structural or c in excluded:
            continue
        col = work[c]
        if not (pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col)):
            continue
        nu = col.nunique(dropna=True)
        if not (2 <= nu <= 80):
            continue
        cat_checked += 1
        tab = pd.crosstab(col.loc[samp].fillna("(missing)").astype(str), ys)
        tab = tab[tab.sum(axis=1) >= 5]
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2 = chi2_contingency(tab, correction=False)[0]
        n_ = tab.values.sum()
        v = float(np.sqrt(chi2 / (n_ * (min(tab.shape) - 1)))) if n_ else 0.0
        if v >= 0.6:
            excluded[c] = f"category alone predicts the outcome (Cramér's V {v:.2f}) — it describes what happened, not what will happen"
        elif v >= 0.45:
            warns.append({"feature": c, "note": f"very strong association with the outcome (Cramér's V {v:.2f}) — verify it is known before the outcome"})

    result = {
        "candidate_features": int(feats.shape[1]), "categorical_checked": cat_checked,
        "excluded": [{"feature": k, "reason": v} for k, v in excluded.items() if not v.startswith("this is the outcome")],
        "warnings": warns[:12],
        "top_univariate": uni.head(25).to_dict(orient="records"),
    }
    return clean(result), {"excluded": excluded, "uni": uni}


# ---------------------------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------------------------

def _onehot_groups(work: pd.DataFrame, feats: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in feats.columns:
        if feats[c].dropna().isin([0, 1]).all() and "_" in str(c):
            groups.setdefault(str(c).split("_", 1)[0], []).append(c)
    return {k: v for k, v in groups.items() if len(v) >= 2 and k in ("segment", "region", "tier", "plan", "industry", "product", "cohort", "size")}


def stage_eda(ctx):
    work: pd.DataFrame = ctx["art"]["work"]
    excluded = ctx["art"]["excluded"]
    uni: pd.DataFrame = ctx["art"]["uni"]
    roles = ctx["roles"]
    lab = _labeled(work)
    base = float(work.loc[lab, "__y"].mean())
    L = work[lab]
    feats = _feature_frame(work, set(excluded) | {c for c in roles.values() if isinstance(c, str)})

    out: dict[str, Any] = {"base_rate": base}

    if work["__date"].notna().any():
        t = L.groupby("__date")["__y"].agg(["size", "mean"]).reset_index()
        out["churn_by_date"] = [{"date": d, "n": int(n), "rate": float(m)} for d, n, m in t.itertuples(index=False) if n >= 30]

    # churn rate by segment-like dimensions (one-hot groups + low-cardinality text columns)
    seg = []
    for prefix, cols in _onehot_groups(work, feats).items():
        for c in cols:
            m = L[c] == 1 if c in L.columns else feats.loc[L.index, c] == 1
            n = int(m.sum())
            if n >= 30:
                seg.append({"dimension": prefix, "group": str(c).split("_", 1)[1], "n": n, "churn_rate": float(L.loc[m, "__y"].mean())})
    for c in work.columns:
        if c in META or c in excluded or c in roles.values():
            continue
        s = work[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and 2 <= s.nunique() <= 12:
            for g, sub in L.groupby(c):
                if len(sub) >= 30:
                    seg.append({"dimension": str(c), "group": (str(g) if str(g).strip() else "(blank)"), "n": int(len(sub)), "churn_rate": float(sub["__y"].mean())})
    for s_ in seg:
        s_["lift"] = s_["churn_rate"] / base if base else None
    seg = sorted(seg, key=lambda d: -abs((d["lift"] or 1) - 1))
    seen_seg, uniq_seg = set(), []
    for s_ in seg:
        key_ = (s_["n"], round(s_["churn_rate"], 5))
        if key_ in seen_seg:
            continue  # the same rows under another column name
        seen_seg.add(key_)
        uniq_seg.append(s_)
    out["segments"] = uniq_seg[:24]

    # driver profiles: churn rate by quantile bin of the strongest numeric signals
    bins_out = []
    top = [r.feature for r in uni.itertuples() if r.feature not in excluded][:10]
    for c in top:
        v = feats[c] if c in feats.columns else _num(work[c])
        vl = v[lab]
        if vl.nunique() < 3:
            continue
        try:
            q = pd.qcut(vl.rank(method="first"), 5, labels=False, duplicates="drop")
        except Exception:
            continue
        g = pd.DataFrame({"q": q, "y": L["__y"], "v": vl}).dropna()
        rows = []
        for qi, sub in g.groupby("q"):
            rows.append({"bin": int(qi) + 1, "lo": float(sub["v"].min()), "hi": float(sub["v"].max()), "n": int(len(sub)), "churn_rate": float(sub["y"].mean())})
        bins_out.append({"feature": c, "bins": rows})
    out["driver_bins"] = bins_out

    # revenue view
    val = roles.get("value")
    if val and val in work.columns:
        v = _num(work[val])
        vl = v[lab]
        out["value"] = {
            "column": val,
            "total": float(vl.sum()),
            "churned_value_share": float(vl[L["__y"] == 1].sum() / vl.sum()) if vl.sum() else None,
            "mean_retained": float(vl[L["__y"] == 0].mean()),
            "mean_churned": float(vl[L["__y"] == 1].mean()),
            "median_retained": float(vl[L["__y"] == 0].median()),
            "median_churned": float(vl[L["__y"] == 1].median()),
        }

    # ---- EDA visuals for the strongest drivers: distributions by outcome, correlations, outliers, column profile ----
    drv = [r.feature for r in uni.itertuples() if r.feature not in excluded and r.feature in feats.columns][:12]
    if drv:
        Lf = feats.loc[L.index, drv]
        yl = L["__y"].to_numpy()
        dist = []
        for c in drv[:6]:
            v = Lf[c].to_numpy(dtype=float)
            m = ~np.isnan(v)
            if m.sum() < 50:
                continue
            lo, hi = np.nanpercentile(v[m], [1, 99])
            if lo == hi:
                continue
            edges = np.linspace(lo, hi, 21)
            h0, _ = np.histogram(np.clip(v[m & (yl == 0)], lo, hi), bins=edges)
            h1, _ = np.histogram(np.clip(v[m & (yl == 1)], lo, hi), bins=edges)
            n0, n1 = max(int(h0.sum()), 1), max(int(h1.sum()), 1)
            dist.append({"feature": c, "bins": [{"x": float((edges[i] + edges[i + 1]) / 2), "retained": float(h0[i] / n0), "churned": float(h1[i] / n1)} for i in range(20)]})
        out["distributions"] = dist

        cs = Lf.copy()
        cs["__churned"] = yl
        if len(cs) > 20_000:
            cs = cs.sample(20_000, random_state=SEED)
        corr = cs.corr(method="spearman")
        out["correlation"] = {"features": list(corr.columns), "matrix": [[None if pd.isna(x) else float(x) for x in row] for row in corr.to_numpy()]}

        outl, prof = [], []
        for c in drv:
            v = Lf[c].dropna()
            if len(v) < 50:
                continue
            q1, q3 = v.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                outl.append({"feature": c, "outlier_pct": float(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean()),
                             "lower": float(q1 - 1.5 * iqr), "upper": float(q3 + 1.5 * iqr)})
            prof.append({"feature": c, "missing_pct": float(Lf[c].isna().mean() * 100), "mean": float(v.mean()), "std": float(v.std()),
                         "min": float(v.min()), "max": float(v.max()), "skew": float(v.skew()) if len(v) > 2 else None})
        out["outliers"] = outl
        out["profile"] = prof

    out["missing_overall_pct"] = round(float(work.drop(columns=list(META)).isna().mean().mean() * 100), 2)
    return clean(out), {}


# ---------------------------------------------------------------------------
# 4. hypotheses (tested on ONE row per entity so panel rows don't inflate significance)
# ---------------------------------------------------------------------------

def _human(feature: str) -> str:
    return pretty_feature(feature)


def stage_hypotheses(ctx):
    work: pd.DataFrame = ctx["art"]["work"]
    excluded = ctx["art"]["excluded"]
    uni: pd.DataFrame = ctx["art"]["uni"]
    roles = ctx["roles"]
    lab = _labeled(work)
    L = work[lab]
    indep = L.groupby("__entity", group_keys=False).sample(1, random_state=SEED) if L["__entity"].nunique() < len(L) else L
    feats = _feature_frame(work, set(excluded) | {c for c in roles.values() if isinstance(c, str)})
    F = feats.loc[indep.index]
    y = indep["__y"].to_numpy()
    base = float(y.mean())

    hyps: list[dict] = []
    for r in uni.itertuples():
        if r.feature in excluded or r.feature not in F.columns:
            continue
        if len(hyps) >= 14:
            break
        x = F[r.feature]
        m = x.notna().to_numpy()
        if m.sum() < 100:
            continue
        x1 = x[m & (y == 1)]
        x0 = x[m & (y == 0)]
        if len(x1) < 10 or len(x0) < 10:
            continue
        try:
            p = float(stats.mannwhitneyu(x1, x0, alternative="two-sided", method="asymptotic").pvalue)
        except Exception:
            continue
        auc = float(_auc_fast(x.to_numpy(dtype=float), y) or 0.5)
        med = float(x[m].median())
        hi = x[m] > med if x[m].nunique() > 2 else x[m] == x[m].max()
        rate_hi = float(y[m][hi.to_numpy()].mean()) if hi.any() else None
        rate_lo = float(y[m][~hi.to_numpy()].mean()) if (~hi).any() else None
        direction = "higher" if auc >= 0.5 else "lower"
        hyps.append({
            "statement": f"Accounts with {direction} {_human(r.feature)} are more likely to churn",
            "feature": r.feature, "test": "Mann-Whitney U", "p_value": p, "effect": abs(auc - 0.5) * 2,
            "effect_label": f"AUC {max(auc, 1-auc):.2f}", "n": int(m.sum()),
            "churn_rate_high": rate_hi, "churn_rate_low": rate_lo, "direction": direction,
            "median_churned": float(x1.median()), "median_retained": float(x0.median()),
        })

    # categorical / segment hypotheses: chi-square on group membership
    feats_no = feats
    for prefix, cols in _onehot_groups(work, feats_no).items():
        sub = pd.DataFrame({c: F[c] for c in cols if c in F.columns})
        if sub.shape[1] < 2:
            continue
        label = sub.idxmax(axis=1)
        tab = pd.crosstab(label, y)
        if tab.shape[0] < 2 or tab.shape[1] < 2 or tab.values.min() < 0:
            continue
        chi2, p, _, _ = stats.chi2_contingency(tab)
        n = tab.values.sum()
        v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1)))) if n else 0.0
        rates = (tab[1.0] / tab.sum(axis=1)).sort_values(ascending=False)
        hyps.append({
            "statement": f"Churn rate differs across {prefix} groups (highest: {str(rates.index[0]).split('_', 1)[-1]} at {rates.iloc[0]*100:.1f}%)",
            "feature": prefix, "test": "Chi-square", "p_value": float(p), "effect": v, "effect_label": f"Cramér's V {v:.2f}",
            "n": int(n), "churn_rate_high": float(rates.iloc[0]), "churn_rate_low": float(rates.iloc[-1]), "direction": "differs",
        })

    q = _bh([h["p_value"] for h in hyps])
    for h, qv in zip(hyps, q):
        h["q_value"] = qv
        if qv < 0.05 and h["effect"] >= 0.15:
            h["verdict"] = "supported"
        elif qv < 0.05:
            h["verdict"] = "weak"  # statistically significant, but the effect is too small to matter
            h["note"] = "statistically significant but small effect"
        elif h["n"] >= 400 and h["effect"] < 0.05:
            h["verdict"] = "refuted"
        else:
            h["verdict"] = "inconclusive"
    hyps.sort(key=lambda h: (h["q_value"], -h["effect"]))
    seen_stats, unique = set(), []
    for h in hyps:
        sig = (h["n"], round(h["effect"], 4), round(h.get("churn_rate_high") or 0, 4), round(h.get("churn_rate_low") or 0, 4))
        if sig in seen_stats:
            continue  # same statistics as an earlier finding: a duplicate of the same column
        seen_stats.add(sig)
        unique.append(h)
    hyps = unique
    result = {
        "tested_on_rows": int(len(indep)), "base_rate": base, "one_row_per_entity": bool(len(indep) < len(L)),
        "correction": "Benjamini-Hochberg FDR across all tested hypotheses",
        "hypotheses": hyps,
        "supported": int(sum(h["verdict"] == "supported" for h in hyps)), "total": len(hyps),
    }
    return clean(result), {}


# ---------------------------------------------------------------------------
# 5. feature engineering
# ---------------------------------------------------------------------------

def stage_features(ctx):
    work: pd.DataFrame = ctx["art"]["work"]
    excluded = ctx["art"]["excluded"]
    uni: pd.DataFrame = ctx["art"]["uni"]
    roles = ctx["roles"]
    skip = set(excluded) | {c for c in roles.values() if isinstance(c, str)}
    X = _feature_frame(work, skip)
    created: list[dict] = []

    # low-cardinality text columns -> one-hot (top categories only)
    for c in work.columns:
        if c in META or c in skip or c in X.columns:
            continue
        s = work[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and 2 <= s.nunique() <= 15:
            top = s.value_counts().head(8).index
            for g in top:
                name = f"{c}={g}"
                X[name] = (s == g).astype(float)
                created.append({"feature": name, "kind": "one-hot", "note": f"{c} equals '{g}'"})

    # missing-value indicators: informative missingness without dropping rows
    base_cols = list(X.columns)
    for c in base_cols:
        mp = float(X[c].isna().mean())
        if 0.03 <= mp <= 0.7:
            X[f"{c}__missing"] = X[c].isna().astype(float)
            created.append({"feature": f"{c}__missing", "kind": "missing-indicator", "note": f"{c} is missing ({mp*100:.0f}% of rows)"})

    # trajectory features from snapshot history (past-only, so no leakage)
    ent, dt = work["__entity"], work["__date"]
    if dt.notna().any() and ent.nunique() < len(work):
        order = work.sort_values(["__entity", "__date"]).index
        top_num = [r.feature for r in uni.itertuples()
                   if r.feature in base_cols and r.feature not in excluded and X[r.feature].nunique() > 10][:8]
        for c in top_num:
            d = X.loc[order, c].groupby(work.loc[order, "__entity"]).diff()
            X[f"{c}__delta_prev"] = d.reindex(X.index)
            created.append({"feature": f"{c}__delta_prev", "kind": "trajectory", "note": f"change in {c} since the account's previous snapshot"})
        X["snapshots_seen"] = work.loc[order].groupby("__entity").cumcount().reindex(X.index) + 1.0
        created.append({"feature": "snapshots_seen", "kind": "trajectory", "note": "number of snapshots observed for the account so far"})

    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)  # half the memory of float64; models read float32 anyway
    result = {
        "base_features": len(base_cols), "total_features": int(X.shape[1]), "created": created[:40], "created_count": len(created),
        "kinds": {k: int(sum(1 for c in created if c["kind"] == k)) for k in {c["kind"] for c in created}},
    }
    return clean(result), {"X": X}


# ---------------------------------------------------------------------------
# split + selection
# ---------------------------------------------------------------------------

def _split(work: pd.DataFrame):
    """Entity-disjoint 80/20 holdout (stratified random split when there is no entity key)."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    lab_idx = work.index[_labeled(work)]
    y = work.loc[lab_idx, "__y"].to_numpy()
    groups = work.loc[lab_idx, "__entity"].to_numpy()
    if len(pd.unique(groups)) < len(lab_idx):
        best = None
        for seed in range(SEED, SEED + 15):
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(lab_idx, y, groups))
            npos = int(y[te].sum())
            if best is None or npos > best[0]:
                best = (npos, tr, te)
            if npos >= max(10, 0.15 * y.sum()):
                break
        _, tr, te = best
        return lab_idx[tr], lab_idx[te], "entity-disjoint"
    tr, te = train_test_split(np.arange(len(lab_idx)), test_size=0.2, stratify=y, random_state=SEED)
    return lab_idx[tr], lab_idx[te], "stratified-random"


def stage_select(ctx):
    from sklearn.feature_selection import mutual_info_classif

    work: pd.DataFrame = ctx["art"]["work"]
    X: pd.DataFrame = ctx["art"]["X"]
    uni: pd.DataFrame = ctx["art"]["uni"]
    max_features = int(ctx["params"].get("max_features") or 50)
    train_idx, test_idx, split_kind = _split(work)
    if len(train_idx) > MAX_TRAIN_ROWS:
        rng = np.random.RandomState(SEED)
        ents = pd.Series(work.loc[train_idx, "__entity"].unique())
        keep = set(ents.sample(frac=MAX_TRAIN_ROWS / len(train_idx), random_state=SEED))
        train_idx = train_idx[work.loc[train_idx, "__entity"].isin(keep).to_numpy()]

    Xt = X.loc[train_idx]
    yt = work.loc[train_idx, "__y"].to_numpy()
    dropped: list[dict] = []
    keep_cols = list(X.columns)

    def drop(cols, why):
        nonlocal keep_cols
        for c in cols:
            dropped.append({"feature": c, "reason": why})
        keep_cols = [c for c in keep_cols if c not in set(cols)]

    drop([c for c in keep_cols if Xt[c].nunique() < 2], "constant in training data")
    drop([c for c in keep_cols if Xt[c].isna().mean() > 0.7 and not str(c).endswith("__missing")], ">70% missing")

    # near-duplicates: keep whichever has the stronger univariate signal
    auc = {r.feature: r.strength for r in uni.itertuples()}
    sample = Xt[keep_cols].sample(min(len(Xt), 6000), random_state=SEED)
    corr = sample.rank().corr().abs()
    to_drop = set()
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        if a in to_drop:
            continue
        for b in cols[i + 1:]:
            if b in to_drop:
                continue
            if corr.at[a, b] >= 0.97:
                loser = b if auc.get(a, 0) >= auc.get(b, 0) else a
                to_drop.add(loser)
                dropped.append({"feature": loser, "reason": f"near-duplicate of {a if loser == b else b} (|rho| ≥ 0.97)"})
                if loser == a:
                    break
    keep_cols = [c for c in keep_cols if c not in to_drop]

    # rank the survivors by mutual information on the training split only
    Xs = Xt[keep_cols].fillna(Xt[keep_cols].median()).fillna(0)
    s_idx = np.random.RandomState(SEED).choice(len(Xs), size=min(len(Xs), 3500), replace=False)
    mi = mutual_info_classif(Xs.iloc[s_idx], yt[s_idx], random_state=SEED)
    mi_s = pd.Series(mi, index=keep_cols).sort_values(ascending=False)
    selected = list(mi_s.head(max_features).index)
    for c in mi_s.index[max_features:]:
        dropped.append({"feature": c, "reason": f"outside top {max_features} by mutual information"})

    result = {
        "split": {"kind": split_kind, "train_rows": int(len(train_idx)), "holdout_rows": int(len(test_idx)),
                  "train_entities": int(work.loc[train_idx, "__entity"].nunique()), "holdout_entities": int(work.loc[test_idx, "__entity"].nunique()),
                  "holdout_positives": int(work.loc[test_idx, "__y"].sum())},
        "start_features": int(X.shape[1]), "selected": len(selected), "dropped_count": len(dropped),
        "dropped_by_reason": {k: int(v) for k, v in pd.Series([re.sub(r"\(.*|of .*", "", d["reason"]).strip() for d in dropped]).value_counts().items()},
        "top_mutual_information": [{"feature": f, "mi": float(v)} for f, v in mi_s.head(15).items()],
        "dropped_sample": dropped[:25],
    }
    return clean(result), {"selected": selected, "train_idx": train_idx, "test_idx": test_idx, "split_kind": split_kind}


# ---------------------------------------------------------------------------
# 6. models
# ---------------------------------------------------------------------------

def _metrics(y: np.ndarray, p: np.ndarray, top_frac: float = 0.10) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    base = float(y.mean())
    k = max(1, int(round(len(y) * top_frac)))
    top = np.argsort(-p, kind="stable")[:k]
    prec = float(y[top].mean())
    return {
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
        "lift_top10": prec / base if base else None,
        "recall_top10": float(y[top].sum() / y.sum()) if y.sum() else None,
        "precision_top10": prec,
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
    }


def _make_models(n_train: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    trees = 120 if n_train <= 40_000 else 80
    return {
        "Logistic Regression": lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler()),
            ("m", LogisticRegression(C=0.3, max_iter=600, class_weight="balanced")),
        ]),
        "Random Forest": lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("m", RandomForestClassifier(n_estimators=trees, min_samples_leaf=10 if n_train > 40_000 else 5, max_features="sqrt", max_depth=12,
                                         class_weight="balanced_subsample", n_jobs=4, random_state=SEED)),
        ]),
        "Gradient Boosting": lambda: HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=120, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0, early_stopping=False, random_state=SEED),
    }


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False, duplicates="drop")
    tot = 0.0
    for b in np.unique(q):
        m = q == b
        tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


SUSPICIOUS_AUC = 0.95  # churn models this accurate almost always owe it to information recorded at/after the decision
MAX_QUARANTINE = 4


def _run_candidates(Xtr, ytr, Xte, yte, makers, folds, started, budget):
    board, oof_store, fitted = [], {}, {}
    # baseline: always predict the prevalence (AUC 0.5, PR-AUC = base rate)
    base_p = np.full(len(yte), ytr.mean())
    board.append({"model": "Baseline (prevalence)", "baseline": True, "cv": None,
                  "holdout": _metrics(yte, base_p + np.random.RandomState(0).rand(len(yte)) * 1e-9), "train_seconds": 0.0})
    for name, mk in makers.items():
        t0 = time.time()
        try:
            oof = np.zeros(len(ytr))
            per_fold = []
            for f_tr, f_va in folds:
                m = mk()
                m.fit(Xtr.iloc[f_tr], ytr[f_tr])
                oof[f_va] = m.predict_proba(Xtr.iloc[f_va])[:, 1]
                per_fold.append(_metrics(ytr[f_va], oof[f_va]))
            model = mk()
            model.fit(Xtr, ytr)
            ph = model.predict_proba(Xte)[:, 1]
            cvm = {k: float(np.mean([f[k] for f in per_fold if f[k] is not None])) for k in ("roc_auc", "pr_auc", "lift_top10")}
            cvm.update({k + "_std": float(np.std([f[k] for f in per_fold if f[k] is not None])) for k in ("roc_auc", "pr_auc")})
            board.append({"model": name, "cv": cvm, "holdout": _metrics(yte, ph), "train_seconds": round(time.time() - t0, 1)})
            oof_store[name], fitted[name] = (oof, ph), model
        except Exception as e:
            board.append({"model": name, "error": str(e)[:200]})
    if not fitted:
        raise RuntimeError("No candidate model could be trained")
    best = max(fitted, key=lambda n: next(b for b in board if b["model"] == n)["cv"]["pr_auc"])
    return board, oof_store, fitted, best


def _top_culprit(model, Xte: pd.DataFrame, yte: np.ndarray, cols: list[str]) -> tuple[str, float]:
    """Feature whose shuffling hurts held-out AUC the most, and the AUC it costs."""
    from sklearn.metrics import roc_auc_score

    if len(Xte) > 2000:
        s = np.random.RandomState(SEED).choice(len(Xte), 2000, replace=False)
        Xs, ys = Xte.iloc[s], yte[s]
    else:
        Xs, ys = Xte, yte
    rng = np.random.RandomState(SEED)
    base = roc_auc_score(ys, model.predict_proba(Xs)[:, 1])
    best, drop_best = cols[0], -1.0
    for f in cols[:30]:  # cols are ordered by mutual information
        Xp = Xs.copy()
        Xp[f] = rng.permutation(Xp[f].to_numpy())
        d = base - roc_auc_score(ys, model.predict_proba(Xp)[:, 1])
        if d > drop_best:
            best, drop_best = f, d
    return best, float(drop_best)


def stage_models(ctx):
    import joblib
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedGroupKFold

    work: pd.DataFrame = ctx["art"]["work"]
    tr, te = ctx["art"]["train_idx"], ctx["art"]["test_idx"]
    ytr, gtr = work.loc[tr, "__y"].to_numpy(), work.loc[tr, "__entity"].to_numpy()
    yte = work.loc[te, "__y"].to_numpy()
    makers = _make_models(len(tr))
    started = time.time()
    budget = float(ctx["params"].get("time_budget_seconds") or 420)

    n_splits = int(min(4, max(2, ytr.sum() // 10)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(cv.split(ctx["art"]["X"].loc[tr].iloc[:, :1], ytr, gtr))

    sel = list(ctx["art"]["selected"])
    quarantined: list[dict] = []
    for attempt in range(MAX_QUARANTINE + 1):
        X = ctx["art"]["X"][sel]
        Xtr, Xte = X.loc[tr], X.loc[te]
        board, oof_store, fitted, best = _run_candidates(Xtr, ytr, Xte, yte, makers, folds, started, budget)
        ho_auc = next(b for b in board if b["model"] == best)["holdout"]["roc_auc"] or 0.0
        if ho_auc <= SUSPICIOUS_AUC or attempt == MAX_QUARANTINE or len(sel) <= 10:
            break
        # implausibly accurate: quarantine the single most decisive feature and retrain without it
        culprit, cost = _top_culprit(fitted[best], Xte, yte, sel)
        quarantined.append({"feature": culprit, "auc_with": ho_auc, "auc_cost": cost,
                            "reason": f"model reached AUC {ho_auc:.2f} (implausible for churn) and this feature carried the most signal "
                                      f"(shuffling it costs {cost:.2f} AUC) — likely recorded at or after the decision"})
        sel.remove(culprit)

    # model selection: best cross-validated PR-AUC (right metric for imbalanced churn); holdout is only a report card
    for b in board:
        b["selected"] = b["model"] == best
    oof_tr, p_hold = oof_store[best]

    # calibration on training OOF only, judged on the untouched holdout
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof_tr, ytr)
    p_hold_cal = iso.predict(p_hold)
    calib = {
        "ece_before": _ece(yte, p_hold), "ece_after": _ece(yte, p_hold_cal),
        "bins": [{"predicted": float(a), "observed": float(b), "n": int(n)} for a, b, n in _calib_bins(yte, p_hold_cal)],
    }
    # operating threshold chosen on training OOF (maximise F1) — never tuned on the holdout
    prec_grid = []
    for t in np.unique(np.quantile(iso.predict(oof_tr), np.linspace(0.5, 0.99, 50))):
        pred = iso.predict(oof_tr) >= t
        tp = float((pred & (ytr == 1)).sum())
        pr = tp / max(pred.sum(), 1)
        rc = tp / max(ytr.sum(), 1)
        prec_grid.append((2 * pr * rc / max(pr + rc, 1e-9), float(t)))
    thr = max(prec_grid)[1] if prec_grid else 0.5
    hp = p_hold_cal >= thr
    tp = float((hp & (yte == 1)).sum())
    op = {"threshold": thr, "precision": tp / max(hp.sum(), 1), "recall": tp / max(yte.sum(), 1), "flagged_pct": float(hp.mean())}

    # deploy model: refit on every labelled row; score every row
    lab_idx = work.index[_labeled(work)]
    final = makers[best]()
    final.fit(X.loc[lab_idx], work.loc[lab_idx, "__y"].to_numpy())
    raw_all = pd.Series(final.predict_proba(X)[:, 1], index=X.index)

    # rows the final model was trained on get an honest score instead: train rows -> their out-of-fold
    # prediction, holdout rows -> prediction of the train-only model. Unlabelled rows -> final model.
    score = raw_all.copy()
    score_type = pd.Series("final_model", index=X.index)
    score.loc[tr] = oof_tr
    score_type.loc[tr] = "cross_validated"
    score.loc[te] = p_hold
    score_type.loc[te] = "held_out"
    # training rows dropped by the row cap are scored by the train-only model
    dropped_tr = lab_idx.difference(pd.Index(tr)).difference(pd.Index(te))
    if len(dropped_tr):
        score.loc[dropped_tr] = fitted[best].predict_proba(X.loc[dropped_tr])[:, 1]
        score_type.loc[dropped_tr] = "held_out"
    prob = pd.Series(iso.predict(score.to_numpy()), index=X.index)

    buf = io.BytesIO()
    joblib.dump({"model": final, "features": list(X.columns), "calibrator": iso, "threshold": thr, "name": best}, buf, compress=3)
    model_bytes = buf.getvalue() if buf.tell() <= 60 * 1024 * 1024 else None  # a huge blob is not worth storing in the database

    result = {
        "leaderboard": board, "selected_model": best, "selection_metric": "cross-validated PR-AUC",
        "cv_folds": n_splits, "calibration": calib, "operating_point": op,
        "features_used": len(X.columns), "quarantined": quarantined,
        "holdout_metrics": next(b for b in board if b["model"] == best)["holdout"],
    }
    art = {
        "prob": prob, "score_type": score_type, "model_final": final, "model_train": fitted[best], "best": best,
        "iso": iso, "threshold": thr, "model_bytes": model_bytes, "p_hold_cal": pd.Series(p_hold_cal, index=te),
        "selected": sel,
    }
    return clean(result), art


def _calib_bins(y, p, bins=8):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False, duplicates="drop")
    for b in np.unique(q):
        m = q == b
        yield p[m].mean(), y[m].mean(), int(m.sum())


# ---------------------------------------------------------------------------
# 7. explainability
# ---------------------------------------------------------------------------

def _shap_values(model, X: pd.DataFrame):
    """Per-row risk attributions (higher = pushes churn probability up). Returns ndarray or None."""
    import shap
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        est = model.steps[-1][1]
        Xt = model[:-1].transform(X)
        if hasattr(est, "coef_"):  # logistic regression: exact linear contributions in standardised space
            return Xt * est.coef_[0]
        ex = shap.TreeExplainer(est)
        sv = ex.shap_values(Xt, check_additivity=False)
    else:
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(X, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def stage_explain(ctx):
    from sklearn.inspection import permutation_importance

    work: pd.DataFrame = ctx["art"]["work"]
    sel = ctx["art"]["selected"]
    X: pd.DataFrame = ctx["art"]["X"][sel]
    te = ctx["art"]["test_idx"]
    model_train = ctx["art"]["model_train"]
    model_final = ctx["art"]["model_final"]
    prob: pd.Series = ctx["art"]["prob"]

    Xte, yte = X.loc[te], work.loc[te, "__y"].to_numpy()
    from sklearn.metrics import roc_auc_score

    if len(Xte) > 2000:
        s = np.random.RandomState(SEED).choice(len(Xte), 2000, replace=False)
        Xs, ys = Xte.iloc[s], yte[s]
    else:
        Xs, ys = Xte, yte
    # permutation importance on held-out accounts: how much ROC-AUC drops when one feature is shuffled
    rng = np.random.RandomState(SEED)
    base_auc = roc_auc_score(ys, model_train.predict_proba(Xs)[:, 1])
    rows_imp = []
    for f in sel[:20]:  # sel is ordered by mutual information
        drops = []
        for _ in range(2):
            Xp = Xs.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(base_auc - roc_auc_score(ys, model_train.predict_proba(Xp)[:, 1]))
        rows_imp.append((f, float(np.mean(drops)), float(np.std(drops))))
    imp = pd.DataFrame(rows_imp, columns=["feature", "importance", "std"]).sort_values("importance", ascending=False)

    lab = _labeled(work)
    dirs = {}
    for f in imp.head(25)["feature"]:
        r = X.loc[lab, f].corr(prob[lab], method="spearman")
        dirs[f] = "higher → more churn" if (r or 0) > 0.03 else ("lower → more churn" if (r or 0) < -0.03 else "non-monotonic")
    top = [{"feature": r.feature, "importance": float(r.importance), "std": float(r.std), "direction": dirs.get(r.feature)}
           for r in imp.head(20).itertuples() if r.importance > 0]

    # per-row top risk drivers (SHAP where available, else importance-weighted z-scores)
    # rows that get per-row driver text: latest-snapshot rows first (what a CSM acts on), then the riskiest rest.
    # RandomForest SHAP is ~20x slower than boosting, so it gets a smaller budget.
    is_rf = "RandomForest" in type(model_final[-1] if hasattr(model_final, "steps") else model_final).__name__
    cap = 800 if is_rf else 4000
    latest = _current_rows(work)[0]
    lat_sorted = prob.loc[latest].sort_values(ascending=False)
    rows = lat_sorted.head(cap).index
    if len(rows) < cap:
        rest = prob.drop(rows).sort_values(ascending=False).head(cap - len(rows)).index
        rows = rows.append(rest)
    method = "shap"
    try:
        sv = _shap_values(model_final, X.loc[rows])
        if sv is None or sv.shape != (len(rows), len(sel)):
            raise ValueError("unexpected shap shape")
    except Exception:
        method = "importance-weighted z-score"
        z = (X.loc[rows] - X.median()) / (X.std().replace(0, np.nan))
        w = imp.set_index("feature")["importance"].clip(lower=0).reindex(sel).fillna(0).to_numpy()
        sign = np.array([1.0 if (X.loc[lab, f].corr(prob[lab], method="spearman") or 0) >= 0 else -1.0 for f in sel])
        sv = (z.fillna(0).to_numpy() * sign) * w

    drivers = pd.DataFrame(index=rows, columns=["d1", "d2", "d3"], dtype=object)
    Xr = X.loc[rows].to_numpy()
    k = min(3, len(sel))
    top_idx = np.argsort(-sv, axis=1)[:, :k]
    for j in range(k):
        names = []
        for i, col in enumerate(top_idx[:, j]):
            contrib = sv[i, col]
            if contrib <= 0:
                names.append(None)
                continue
            v = Xr[i, col]
            vs = "missing" if pd.isna(v) else (f"{v:.4g}")
            names.append(f"{sel[col]} = {vs}")
        drivers.iloc[:, j] = names

    result = {
        "importance_method": "permutation importance on the held-out accounts (drop in ROC-AUC)",
        "attribution_method": method, "top_features": top,
        "driver_rows": int(len(rows)),
    }
    return clean(result), {"drivers": drivers, "importance": imp}


def _current_rows(work: pd.DataFrame) -> tuple[pd.Index, bool]:
    """The rows worth acting on. If the data has unresolved records (e.g. renewals still open) those are the
    live population — already-decided rows are history. Otherwise: each account's latest row."""
    un = work[work["__y"].isna()]
    if len(un) >= 50:
        dated = un.dropna(subset=["__date"])
        if len(dated):
            idx = dated.loc[dated.groupby("__entity")["__date"].idxmax()].index
            rest = un.index.difference(dated.index)
            rest = un.loc[rest].drop_duplicates("__entity", keep="last").index
            idx = idx.union(rest[~work.loc[rest, "__entity"].isin(work.loc[idx, "__entity"]).to_numpy()])
        else:
            idx = un.drop_duplicates("__entity", keep="last").index
        return pd.Index(idx), True
    return pd.Index(_latest_rows(work)), False


def _latest_rows(work: pd.DataFrame) -> pd.Index:
    if work["__date"].notna().any() and work["__entity"].nunique() < len(work):
        w = work[work["__date"].notna()]
        return w.loc[w.groupby("__entity")["__date"].idxmax()].index
    return work.index


# ---------------------------------------------------------------------------
# 8. risk tiers & value at risk
# ---------------------------------------------------------------------------

def stage_value(ctx):
    work: pd.DataFrame = ctx["art"]["work"]
    prob: pd.Series = ctx["art"]["prob"]
    roles = ctx["roles"]
    te = ctx["art"]["test_idx"]
    p_hold = ctx["art"]["p_hold_cal"]
    val = roles.get("value")
    v = _num(work[val]) if val and val in work.columns else None

    latest_idx, is_open = _current_rows(work)
    lp = prob.loc[latest_idx]
    t_high, t_med = float(lp.quantile(0.90)), float(lp.quantile(0.70))
    base = float(work["__y"].mean())
    t_high = max(t_high, base)  # never call an account "high" below the average churn rate
    t_med = min(t_med, t_high)

    def tier(p):
        return np.where(p >= t_high, "High", np.where(p >= t_med, "Medium", "Low"))

    tiers = pd.Series(tier(prob.to_numpy()), index=prob.index)
    acct = pd.DataFrame({"entity": work.loc[latest_idx, "__entity"], "date": work.loc[latest_idx, "__date"],
                         "prob": lp, "tier": tiers.loc[latest_idx]})
    if v is not None:
        acct["value"] = v.loc[latest_idx]
        acct["expected_loss"] = acct["prob"] * acct["value"].fillna(0)
    tier_rows = []
    for t in ("High", "Medium", "Low"):
        s = acct[acct["tier"] == t]
        row = {"tier": t, "accounts": int(len(s)), "avg_probability": float(s["prob"].mean()) if len(s) else None}
        if v is not None:
            row.update({"value": float(s["value"].sum()), "expected_loss": float(s["expected_loss"].sum())})
        tier_rows.append(row)

    # gain / lift table on untouched holdout accounts
    yte = work.loc[te, "__y"].to_numpy()
    ph = p_hold.to_numpy()
    order = np.argsort(-ph, kind="stable")
    n = len(order)
    gain, cum = [], 0
    tot = max(yte.sum(), 1)
    for d in range(10):
        a, b = int(n * d / 10), int(n * (d + 1) / 10)
        seg = yte[order[a:b]]
        cum += seg.sum()
        gain.append({"decile": d + 1, "n": int(len(seg)), "churners": int(seg.sum()), "churn_rate": float(seg.mean()) if len(seg) else None,
                     "cum_capture": float(cum / tot), "lift": float(seg.mean() / yte.mean()) if len(seg) and yte.mean() else None})

    acct_sorted = acct.sort_values("expected_loss" if v is not None else "prob", ascending=False)
    top_accounts = []
    drv = ctx["art"].get("drivers")
    for idx, r in acct_sorted.head(25).iterrows():
        d = [x for x in (drv.loc[idx].tolist() if drv is not None and idx in drv.index else []) if isinstance(x, str)]
        top_accounts.append({"entity": r["entity"], "date": r["date"], "probability": r["prob"], "tier": r["tier"],
                             "value": r.get("value"), "expected_loss": r.get("expected_loss"), "drivers": d})

    result = {
        "thresholds": {"high": t_high, "medium": t_med},
        "tier_definition": ("High = top 10% by calibrated churn probability (never below the average churn rate); Medium = next 20%; Low = rest — "
                            + ("among open renewals" if is_open else "among accounts' latest records")),
        "accounts_scored": int(len(acct)), "tiers": tier_rows, "gain_table": gain,
        "value_column": val,
        "total_value": float(acct["value"].sum()) if v is not None else None,
        "expected_loss_total": float(acct["expected_loss"].sum()) if v is not None else None,
        "top_accounts": top_accounts,
        "as_of": (work.loc[_labeled(work), "__date"].quantile(0.995) if work.loc[_labeled(work), "__date"].notna().any() else None),
        "population": "open renewals" if is_open else "latest record per account",
        "population_note": ("Tiers and value at risk cover the accounts whose outcome is still open — the ones you can still act on."
                            if is_open else "Tiers and value at risk cover each account's latest record."),
    }
    return clean(result), {"tiers": tiers, "acct": acct, "t_high": t_high, "t_med": t_med}


# ---------------------------------------------------------------------------
# 9. validation
# ---------------------------------------------------------------------------

def stage_validate(ctx):
    work: pd.DataFrame = ctx["art"]["work"]
    m = ctx["results"]["models"]
    sel = ctx["art"]["selected"]
    excluded = ctx["art"]["excluded"]
    board = next(b for b in m["leaderboard"] if b.get("selected"))
    ho, cv = board["holdout"], board["cv"]
    base = ctx["results"]["understand"]["churn_rate"]
    checks: list[dict] = []

    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    leaked = [c for c in sel if c in excluded]
    add("No outcome-leaking features in the model", "pass" if not leaked else "fail",
        f"{sum(1 for v in excluded.values() if not v.startswith('this is the outcome'))} column(s) excluded as leaking the outcome; none used." if not leaked else f"Used excluded: {leaked}")
    add("Evaluated on unseen accounts", "pass", f"Holdout split is {ctx['art']['split_kind']}; train/test never share an account.")
    gap = abs((cv["roc_auc"] or 0) - (ho["roc_auc"] or 0))
    add("Cross-validation agrees with holdout", "pass" if gap < 0.05 else "warn", f"CV AUC {cv['roc_auc']:.3f} vs holdout {ho['roc_auc']:.3f} (gap {gap:.3f}).")
    q = m.get("quarantined") or []
    if q:
        add("Accuracy is plausible", "warn",
            f"First model reached AUC {q[0]['auc_with']:.2f} (implausible for churn). Retrained without " + ", ".join(x["feature"] for x in q)
            + ". Confirm with the data owner that these are known before the renewal.")
    else:
        add("Accuracy is plausible", "pass" if (ho["roc_auc"] or 0) <= SUSPICIOUS_AUC else "warn",
            f"Holdout AUC {ho['roc_auc']:.3f}." if (ho["roc_auc"] or 0) <= SUSPICIOUS_AUC else f"Holdout AUC {ho['roc_auc']:.3f} is very high - verify no feature leaks the outcome.")
    add("Beats the naive baseline", "pass" if (ho["pr_auc"] or 0) > 1.5 * base else "warn",
        f"PR-AUC {ho['pr_auc']:.3f} vs {base:.3f} for predicting the average rate.")
    add("Useful top-decile lift", "pass" if (ho["lift_top10"] or 0) >= 2 else ("warn" if (ho["lift_top10"] or 0) >= 1.3 else "fail"),
        f"The riskiest 10% contain {ho['lift_top10']:.1f}× the average churn rate and capture {ho['recall_top10']*100:.0f}% of churners.")
    cal = m["calibration"]
    add("Probabilities are calibrated", "pass" if cal["ece_after"] <= 0.05 else "warn", f"Expected calibration error {cal['ece_after']:.3f} on holdout (was {cal['ece_before']:.3f} before calibration).")

    # stability across snapshots on holdout accounts
    if work["__date"].nunique() >= 4:
        from sklearn.metrics import roc_auc_score
        te = ctx["art"]["test_idx"]
        d = pd.DataFrame({"date": work.loc[te, "__date"], "y": work.loc[te, "__y"], "p": ctx["art"]["p_hold_cal"]})
        aucs = []
        for dt, s in d.groupby("date"):
            if s["y"].sum() >= 10 and (len(s) - s["y"].sum()) >= 10 and len(s) >= 100:
                aucs.append(roc_auc_score(s["y"], s["p"]))
        if len(aucs) >= 3:
            add("Stable across time", "pass" if min(aucs) >= 0.6 else "warn",
                f"Holdout AUC per period: median {float(np.median(aucs)):.2f}, range {min(aucs):.2f}–{max(aucs):.2f} across {len(aucs)} periods (periods with under 100 rows ignored).")
    # does the label behave like churn? positives should not coincide with growth / larger accounts
    lab = _labeled(work)
    yv = work.loc[lab, "__y"]
    growth_cols = [c for c in work.columns if c not in META and re.search(r"(change|growth|delta|diff)", str(c), re.I)
                   and (pd.api.types.is_numeric_dtype(work[c]) and not pd.api.types.is_bool_dtype(work[c]))]
    concern = None
    for c in growth_cols:
        v = _num(work.loc[lab, c])
        ok = v.notna() & (v != 0)
        pos_v, neg_v = v[ok & (yv == 1)], v[ok & (yv == 0)]
        if len(pos_v) >= 30 and len(neg_v) >= 30 and (pos_v > 0).mean() >= 0.9 and (neg_v > 0).mean() <= 0.6:
            concern = (f"{(pos_v > 0).mean()*100:.0f}% of labelled-positive rows have a positive '{c}' (growth), versus "
                       f"{(neg_v > 0).mean()*100:.0f}% of the rest — this label may mark expansion/upsell rather than churn. Confirm the label definition.")
            break
    val = ctx["roles"].get("value")
    if concern is None and val and val in work.columns:
        vv = _num(work.loc[lab, val])
        mp, mn = vv[yv == 1].mean(), vv[yv == 0].mean()
        if mn and mp / mn > 2:
            concern = f"Labelled-positive accounts are {mp/mn:.1f}× larger in {val} than the rest, which is unusual for churn — confirm the label definition."
    add("Label behaves like churn", "warn" if concern else "pass", concern or "Positives do not coincide with growth or unusually large accounts.")

    imps = [t for t in ((ctx["results"].get("explain") or {}).get("top_features") or []) if t["importance"] > 0]
    if len(imps) >= 3:
        tot = sum(t["importance"] for t in imps)
        top = imps[0]
        share = top["importance"] / tot if tot else 0
        if share >= 0.4:
            add("No single feature dominates", "warn",
                f"{pretty_feature(top['feature'])} carries {share * 100:.0f}% of the model's signal. Confirm with the data owner that it is known before the renewal decision.")
        else:
            add("No single feature dominates", "pass", f"The largest driver carries {share * 100:.0f}% of the signal.")

    pos = ctx["results"]["understand"]["positives"]
    add("Enough churn examples", "pass" if pos >= 100 else ("warn" if pos >= 50 else "fail"), f"{pos} churned rows in the labelled data.")

    order = {"fail": 0, "warn": 1, "pass": 2}
    overall = min((c["status"] for c in checks), key=lambda s: order[s])
    return clean({"checks": checks, "overall": overall, "passed": sum(c["status"] == "pass" for c in checks), "total": len(checks)}), {}


# ---------------------------------------------------------------------------
# 10. deliverables (enriched data in the client's own format + dictionary)
# ---------------------------------------------------------------------------

def stage_build(ctx):
    df: pd.DataFrame = ctx["df"].reset_index(drop=True)
    work: pd.DataFrame = ctx["art"]["work"]
    prob: pd.Series = ctx["art"]["prob"]
    tiers: pd.Series = ctx["art"]["tiers"]
    acct: pd.DataFrame = ctx["art"]["acct"]
    drivers: pd.DataFrame | None = ctx["art"].get("drivers")
    st: pd.Series = ctx["art"]["score_type"]
    thr = ctx["art"]["threshold"]
    roles = ctx["roles"]
    val = roles.get("value")
    best = ctx["art"]["best"]

    out = df.copy()
    orig_cols = list(out.columns)
    P = "ds_"
    taken = set(orig_cols)

    def name(n):
        while n in taken:
            n = "ds2_" + n[len(P):] if n.startswith(P) else "ds2_" + n
        taken.add(n)
        return n

    cols: dict[str, Any] = {}
    dict_rows: list[dict] = []

    def add(colname, series, dtype, desc, values=None):
        cn = name(colname)
        cols[cn] = series.values
        dict_rows.append({"column": cn, "type": dtype, "description": desc, "values": values or ""})

    add("ds_churn_probability", prob.round(4), "float 0-1",
        f"Calibrated probability that the account churns (outcome '{ctx['params'].get('label_column') or 'churn label'}'). Higher = riskier.")
    add("ds_churn_risk_tier", tiers, "text", "Risk bucket from the probability: High = top 10% of the live population (open renewals, or each account's latest record when none are open), Medium = next 20%, Low = the rest.", "High | Medium | Low")
    if work["__date"].notna().any():
        pct = prob.groupby(work["__date"]).rank(pct=True) * 100
    else:
        pct = prob.rank(pct=True) * 100
    add("ds_churn_risk_percentile", pct.round(1), "float 0-100", "Risk rank of this row among all rows of the same snapshot (100 = riskiest).")
    add("ds_churn_predicted_flag", (prob >= thr).astype(int), "int 0/1", f"1 when probability ≥ {thr:.3f}, the operating threshold that maximised F1 on training data.", "0 | 1")
    add("ds_churn_score_type", st, "text", "How this row was scored without seeing its own label: held_out = model trained on other accounts; cross_validated = out-of-fold prediction; final_model = row had no label.", "held_out | cross_validated | final_model")
    for j, cn in enumerate(("d1", "d2", "d3"), start=1):
        s = drivers[cn].reindex(prob.index) if drivers is not None else pd.Series(None, index=prob.index)
        add(f"ds_churn_driver_{j}", s.fillna(""), "text", f"#{j} factor pushing this row's churn risk up, as 'feature = value'. Blank for low-risk rows.")
    if val and val in work.columns and "expected_loss" in acct.columns:
        v = _num(work[val]).fillna(0)
        add("ds_expected_value_at_risk", (prob * v).round(2), "float", f"churn probability × {val} — the expected {val} lost on this row.")
    cur_idx, is_open = _current_rows(work)
    latest = pd.Series(0, index=work.index)
    latest.loc[cur_idx] = 1
    add("ds_is_current_row", latest, "int 0/1",
        "1 for the row to act on: the account's open (unresolved) record, or its latest row when nothing is open. Filter on this for one row per account.", "0 | 1")
    acct_tier = work["__entity"].map(acct.set_index("entity")["tier"].to_dict()) if len(acct) else pd.Series("", index=work.index)
    add("ds_account_risk_tier", acct_tier.fillna(""), "text", "Risk tier of the account's current row (its open renewal, or latest record), repeated on every row of that account. Blank for accounts with no current row.", "High | Medium | Low")
    add("ds_model", pd.Series(f"{best} (AutoEDA churn flow)", index=work.index), "text", "Model that produced the scores.")

    enriched = pd.concat([out, pd.DataFrame(cols, index=out.index)], axis=1)

    # integrity checks (client format preserved)
    integrity = {
        "rows_preserved": bool(len(enriched) == len(df)),
        "original_columns_preserved": bool(list(enriched.columns[: len(orig_cols)]) == orig_cols),
        "original_values_unchanged": bool(enriched[orig_cols].reset_index(drop=True).equals(df[orig_cols].reset_index(drop=True))),
        "added_columns": len(cols),
    }
    integrity["ok"] = all(integrity[k] for k in ("rows_preserved", "original_columns_preserved", "original_values_unchanged"))

    # account-level file: one row per account, ready to drop into a CRM / BI tool
    a = acct.copy()
    a.insert(0, "account", a.pop("entity"))
    a["prob"] = a["prob"].round(4)
    a = a.rename(columns={"prob": "churn_probability", "tier": "risk_tier", "date": "as_of_snapshot", "value": val or "value", "expected_loss": "expected_value_at_risk"})
    if drivers is not None:
        idx = acct.index
        for j, cn in enumerate(("d1", "d2", "d3"), start=1):
            a[f"driver_{j}"] = drivers[cn].reindex(idx).fillna("").values
    a = a.sort_values("expected_value_at_risk" if "expected_value_at_risk" in a.columns else "churn_probability", ascending=False)

    enriched_csv = enriched.to_csv(index=False).encode("utf-8")
    acct_csv = a.to_csv(index=False).encode("utf-8")
    dictionary = pd.DataFrame(dict_rows)
    result = {
        "integrity": integrity, "enriched_rows": int(len(enriched)), "enriched_columns": int(enriched.shape[1]),
        "added": [r["column"] for r in dict_rows], "dictionary": dict_rows,
        "account_file_rows": int(len(a)),
        "preview": clean(enriched[[c for c in cols][:6]].head(8)),
    }
    return clean(result), {
        "enriched_csv": enriched_csv, "accounts_csv": acct_csv,
        "dictionary_csv": dictionary.to_csv(index=False).encode("utf-8"),
    }


STAGES = [
    # key, title, function, needs raw df
    ("understand", "Understand the data", stage_understand, True),
    ("leakage", "Audit for data leakage", stage_leakage, False),
    ("eda", "Explore churn patterns", stage_eda, False),
    ("hypotheses", "Test hypotheses", stage_hypotheses, False),
    ("features", "Engineer features", stage_features, False),
    ("select", "Select features & split", stage_select, False),
    ("models", "Train & compare models", stage_models, False),
    ("explain", "Explain drivers", stage_explain, False),
    ("value", "Risk tiers & revenue at risk", stage_value, False),
    ("validate", "Validate", stage_validate, False),
    ("build", "Build deliverables", stage_build, True),
]
