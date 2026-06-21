"""
correlations.py
───────────────
Production-ready correlation analysis for mixed-type DataFrames.

Supported association measures
───────────────────────────────
Numeric  × Numeric  → Pearson / Spearman / Kendall correlation matrix
                       + VIF (multicollinearity) for Pearson
Categoric × Categoric → Cramér's V  (symmetric, bias-corrected)
                        + Theil's U  (asymmetric, directional)
Numeric  × Categoric → Point-Biserial r  (binary cat)
                        η² (eta-squared)  via one-way ANOVA  (multi-class cat)
                        Rank-Biserial r   (Mann-Whitney, binary only)

All values are clipped to [0, 1] or [-1, 1] where appropriate and returned
as round(·, 4) floats. None signals "not computable" (too few samples, etc.).

NOTE: No columns are ever excluded by this module. Every column is classified
as numeric or categorical and included in all outputs. A column is only skipped
from a specific *computation* when it genuinely cannot be processed (e.g.
all-null, constant with 1 unique value). Unknown dtypes are coerced to string
(categorical) rather than dropped.

The hard MAX_* constants below are now only used for VIF (which is O(n·k²))
and exist purely for runtime safety — they do NOT cause any column to be
excluded from the returned results. All correlation matrices are computed over
all eligible columns regardless of count.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr

# ── Tuneable limits ────────────────────────────────────────────────────────────
# These exist only for computations that are genuinely O(k²) expensive.
# They do NOT cap columns in any returned matrix or top-pairs list.

MAX_VIF_COLS  = 15   # VIF is O(n·k²) — cap to keep it sub-second
MIN_SAMPLES   = 10   # minimum valid rows for any pairwise test


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe(val: Any) -> float | None:
    """Cast to float; return None for NaN / Inf / errors."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None


def _clip1(val: float | None) -> float | None:
    """Clip to [0, 1]."""
    return None if val is None else round(float(np.clip(val, 0.0, 1.0)), 4)


def _valid_pair(s1: pd.Series, s2: pd.Series) -> tuple[pd.Series, pd.Series] | None:
    """Return aligned, dropna pair; None if too few samples."""
    tmp = pd.concat([s1, s2], axis=1).dropna()
    if len(tmp) < MIN_SAMPLES:
        return None
    return tmp.iloc[:, 0], tmp.iloc[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Numeric × Numeric
# ─────────────────────────────────────────────────────────────────────────────

def compute_num_matrix(df: pd.DataFrame, num_cols: list[str], method: str) -> dict:
    """Full correlation matrix + p-values for all numeric columns."""
    df_num = df[num_cols].dropna(axis=1, how="all")
    actual_cols = df_num.columns.tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = df_num.corr(method=method)

    pvals: dict[str, dict[str, float | None]] = {}
    for c in actual_cols:
        pvals[c] = {}
        for d in actual_cols:
            if c == d:
                pvals[c][d] = None
                continue
            pair = _valid_pair(df_num[c], df_num[d])
            if pair is None:
                pvals[c][d] = None
                continue
            try:
                if method == "pearson":
                    _, p = stats.pearsonr(pair[0], pair[1])
                elif method == "spearman":
                    _, p = stats.spearmanr(pair[0], pair[1])
                else:
                    _, p = stats.kendalltau(pair[0], pair[1])
                pvals[c][d] = _safe(p)
            except Exception:
                pvals[c][d] = None

    matrix: dict[str, dict[str, float | None]] = {}
    for col in num_cols:
        if col in corr.columns:
            matrix[col] = {k: _safe(v) for k, v in corr[col].items()}
        else:
            matrix[col] = {c: None for c in actual_cols}

    return {"matrix": matrix, "p_values": pvals, "columns": actual_cols}


def compute_vif(df: pd.DataFrame, num_cols: list[str]) -> list[dict]:
    """
    VIF via numpy QR decomposition.

    Capped at MAX_VIF_COLS because VIF is O(n·k²). When there are more numeric
    columns than the cap, VIF is computed for the first MAX_VIF_COLS only and
    the returned list includes a 'truncated' flag so the caller can surface a
    warning in the UI. This is a compute-time limit only — no columns are
    dropped from any other output.
    """
    if len(num_cols) < 2:
        return []

    truncated = len(num_cols) > MAX_VIF_COLS
    cols_for_vif = num_cols[:MAX_VIF_COLS]

    X = df[cols_for_vif].dropna().values.astype(float)
    if len(X) < 5:
        return []
    Xc = np.column_stack([np.ones(len(X)), X])
    results = []
    for i, col in enumerate(cols_for_vif):
        try:
            col_idx = i + 1
            y = Xc[:, col_idx]
            others = np.delete(Xc, col_idx, axis=1)
            coef, _, _, _ = np.linalg.lstsq(others, y, rcond=None)
            y_pred = others @ coef
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            vif = 1 / (1 - r2 + 1e-12)
            results.append({"column": col, "vif": round(float(vif), 3)})
        except Exception:
            pass

    results.sort(key=lambda x: x["vif"], reverse=True)
    if truncated:
        results.append({
            "column": "__truncated__",
            "vif": None,
            "note": (
                f"VIF computed for first {MAX_VIF_COLS} of {len(num_cols)} numeric columns "
                "to limit runtime. All columns appear in the correlation matrix."
            ),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Categorical × Categorical
# ─────────────────────────────────────────────────────────────────────────────

def _cramers_v(ct: pd.DataFrame) -> float | None:
    """Bias-corrected Cramér's V from a contingency table."""
    try:
        chi2, _, _, _ = chi2_contingency(ct, correction=False)
        n = ct.values.sum()
        r, k = ct.shape
        phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
        k_tilde = k - (k - 1) ** 2 / (n - 1)
        r_tilde = r - (r - 1) ** 2 / (n - 1)
        denom = min(k_tilde - 1, r_tilde - 1)
        if denom <= 0:
            return None
        return _clip1(np.sqrt(phi2 / denom))
    except Exception:
        return None


def _theils_u(x: pd.Series, y: pd.Series) -> float | None:
    """Theil's U (uncertainty coefficient): asymmetric U(X→Y)."""
    try:
        ct = pd.crosstab(x, y)
        n = ct.values.sum()

        def _entropy(s: pd.Series) -> float:
            p = s.value_counts(normalize=True)
            return float(-np.sum(p * np.log2(p + 1e-12)))

        h_y = _entropy(y)
        if h_y == 0:
            return None
        h_y_given_x = 0.0
        for _, row in ct.iterrows():
            row_sum = row.sum()
            if row_sum == 0:
                continue
            p_row = row / row_sum
            h_y_given_x += (row_sum / n) * float(
                -np.sum(p_row[p_row > 0] * np.log2(p_row[p_row > 0] + 1e-12))
            )
        return _clip1((h_y - h_y_given_x) / h_y)
    except Exception:
        return None


def _chi2_pvalue(ct: pd.DataFrame) -> float | None:
    try:
        _, p, _, _ = chi2_contingency(ct, correction=True)
        return _safe(p)
    except Exception:
        return None


def compute_cat_matrix(df: pd.DataFrame, cat_cols: list[str]) -> dict:
    """
    Cramér's V, Theil's U, chi-square p-values for all categorical pairs.
    Columns with only 1 unique value are skipped per-pair (no information),
    but they are still returned in the matrix as None — never dropped entirely.
    """
    cramers: dict[str, dict] = {}
    theils:  dict[str, dict] = {}
    pvals:   dict[str, dict] = {}

    for c1 in cat_cols:
        cramers[c1] = {}
        theils[c1]  = {}
        pvals[c1]   = {}
        for c2 in cat_cols:
            if c1 == c2:
                cramers[c1][c2] = 1.0
                theils[c1][c2]  = 1.0
                pvals[c1][c2]   = None
                continue
            pair = _valid_pair(df[c1], df[c2])
            if pair is None:
                cramers[c1][c2] = theils[c1][c2] = pvals[c1][c2] = None
                continue
            # Skip if either side is constant after dropna
            if pair[0].nunique() < 2 or pair[1].nunique() < 2:
                cramers[c1][c2] = theils[c1][c2] = pvals[c1][c2] = None
                continue
            ct = pd.crosstab(pair[0], pair[1])
            cramers[c1][c2] = _cramers_v(ct)
            theils[c1][c2]  = _theils_u(pair[0], pair[1])
            pvals[c1][c2]   = _chi2_pvalue(ct)

    return {
        "cramers_v": cramers,
        "theils_u":  theils,
        "p_values":  pvals,
        "columns":   cat_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Numeric × Categorical
# ─────────────────────────────────────────────────────────────────────────────

def _eta_squared(num: pd.Series, cat: pd.Series) -> float | None:
    try:
        groups = [g.values for _, g in num.groupby(cat)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            return None
        f_stat, _ = stats.f_oneway(*groups)
        if np.isnan(f_stat):
            return None
        grand_mean = num.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total   = sum((num - grand_mean) ** 2)
        if ss_total == 0:
            return None
        return _clip1(ss_between / ss_total)
    except Exception:
        return None


def _point_biserial(num: pd.Series, cat: pd.Series) -> tuple[float | None, float | None]:
    try:
        unique = cat.unique()
        if len(unique) != 2:
            return None, None
        binary = (cat == unique[1]).astype(int)
        r, p = pointbiserialr(num, binary)
        return _safe(r), _safe(p)
    except Exception:
        return None, None


def _rank_biserial(num: pd.Series, cat: pd.Series) -> float | None:
    try:
        unique = cat.unique()
        if len(unique) != 2:
            return None
        g1 = num[cat == unique[0]].dropna()
        g2 = num[cat == unique[1]].dropna()
        if len(g1) < 3 or len(g2) < 3:
            return None
        u_stat, _ = mannwhitneyu(g1, g2, alternative="two-sided")
        r = 1 - 2 * u_stat / (len(g1) * len(g2))
        return _safe(r)
    except Exception:
        return None


def compute_mixed_matrix(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> dict:
    matrix: dict[str, dict] = {}
    summary: list[dict] = []

    for num_col in num_cols:
        matrix[num_col] = {}
        for cat_col in cat_cols:
            pair = _valid_pair(df[num_col], df[cat_col])
            if pair is None:
                matrix[num_col][cat_col] = None
                continue
            num_s, cat_s = pair
            if cat_s.nunique() < 2:
                matrix[num_col][cat_col] = None
                continue
            n_unique = cat_s.nunique()

            eta        = _eta_squared(num_s, cat_s)
            pb_r, pb_p = _point_biserial(num_s, cat_s)
            rb_r       = _rank_biserial(num_s, cat_s)

            try:
                groups = [g.values for _, g in num_s.groupby(cat_s) if len(g) >= 2]
                _, anova_p = stats.f_oneway(*groups) if len(groups) >= 2 else (None, None)
            except Exception:
                anova_p = None

            cell = {
                "eta_sq":         eta,
                "point_biserial": pb_r,
                "rank_biserial":  rb_r,
                "p_value":        _safe(anova_p) if anova_p is not None else pb_p,
                "n_categories":   int(n_unique),
            }
            matrix[num_col][cat_col] = cell

            best_effect = eta if eta is not None else (abs(pb_r) if pb_r is not None else None)
            if best_effect is not None:
                summary.append({
                    "num_col":        num_col,
                    "cat_col":        cat_col,
                    "eta_sq":         eta,
                    "point_biserial": pb_r,
                    "rank_biserial":  rb_r,
                    "p_value":        _safe(anova_p) if anova_p is not None else pb_p,
                    "n_categories":   int(n_unique),
                    "_sort_key":      best_effect,
                })

    summary.sort(key=lambda x: x["_sort_key"], reverse=True)
    for row in summary:
        row.pop("_sort_key", None)

    return {
        "matrix":    matrix,
        "top_pairs": summary[:25],
        "num_cols":  num_cols,
        "cat_cols":  cat_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top pairs helpers
# ─────────────────────────────────────────────────────────────────────────────

def _top_num_pairs(matrix: dict, cols: list[str]) -> list[dict]:
    pairs = []
    col_list = [c for c in cols if c in matrix]
    for i, c1 in enumerate(col_list):
        for c2 in col_list[i + 1:]:
            val = matrix.get(c1, {}).get(c2)
            if val is not None:
                pairs.append({
                    "col1": c1,
                    "col2": c2,
                    "correlation": val,
                    "abs_correlation": abs(val),
                })
    return sorted(pairs, key=lambda x: x["abs_correlation"], reverse=True)[:25]


def _top_cat_pairs(cramers: dict, cols: list[str]) -> list[dict]:
    pairs = []
    col_list = [c for c in cols if c in cramers]
    for i, c1 in enumerate(col_list):
        for c2 in col_list[i + 1:]:
            val = cramers.get(c1, {}).get(c2)
            if val is not None and c1 != c2:
                pairs.append({"col1": c1, "col2": c2, "cramers_v": val})
    return sorted(pairs, key=lambda x: x["cramers_v"], reverse=True)[:25]


# ─────────────────────────────────────────────────────────────────────────────
# Column profiling  —  NO columns are ever excluded
# ─────────────────────────────────────────────────────────────────────────────

def _profile_columns(df: pd.DataFrame) -> dict:
    """
    Classify every column as numeric or categorical. No column is ever
    put in an 'ignored' list.

    Rules
    ─────
    • bool                          → categorical (binary)
    • numeric dtype                 → numeric
    • object / string / category    → categorical
    • unknown / other dtype         → coerced to string → categorical
                                      (logged in `coerced_cols`)
    • all-null                      → skipped_cols  (genuinely unusable for
                                      any pairwise test, but still reported)

    `cat_cardinality` lets the UI surface a cardinality warning for high-
    cardinality columns without excluding them from analysis.
    """
    num_cols:      list[str] = []
    cat_cols:      list[str] = []
    skipped_cols:  list[str] = []   # truly all-null — cannot compute anything
    coerced_cols:  list[str] = []   # unknown dtype → cast to str → categorical
    cat_cardinality: dict[str, int] = {}

    for col in df.columns:
        series = df[col].dropna()

        # All-null: include in skipped report but nowhere else
        if len(series) == 0:
            skipped_cols.append(col)
            continue

        if pd.api.types.is_bool_dtype(series):
            cat_cols.append(col)
            cat_cardinality[col] = int(series.nunique())

        elif pd.api.types.is_numeric_dtype(series):
            num_cols.append(col)

        elif (
            pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            cat_cols.append(col)
            cat_cardinality[col] = int(series.nunique())

        else:
            # Unknown dtype: coerce to string so the column is still usable.
            # The original DataFrame is not mutated; callers that need the
            # coerced values should cast: df[col].astype(str)
            coerced_cols.append(col)
            cat_cols.append(col)
            cat_cardinality[col] = int(series.astype(str).nunique())

    return {
        "num_cols":        num_cols,
        "cat_cols":        cat_cols,
        "ignored_cols":    [],           # always empty — exclusion is the user's job
        "skipped_cols":    skipped_cols, # all-null columns; reported, not silently dropped
        "coerced_cols":    coerced_cols, # unknown dtype → cast to str → categorical
        "cat_cardinality": cat_cardinality,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run_correlations(df: pd.DataFrame, method: str = "pearson") -> dict:
    """
    Compute all association measures for a mixed-type DataFrame.

    Every column in `df` is included. The only columns absent from a specific
    computation are those that are genuinely all-null (reported in
    `column_profile.skipped_cols`). Unknown dtypes are coerced to string
    (reported in `column_profile.coerced_cols`) and treated as categorical.

    Parameters
    ----------
    df     : Input DataFrame (any mix of numeric and categorical columns)
    method : 'pearson' | 'spearman' | 'kendall'  (numeric×numeric only)

    Returns
    -------
    dict with keys:
      method           – requested method
      column_profile   – {num_cols, cat_cols,
                          ignored_cols  (always []),
                          skipped_cols  (all-null only),
                          coerced_cols  (unknown dtype → str),
                          cat_cardinality}

      # Numeric × Numeric
      matrix           – correlation matrix  {col: {col: r}}
      p_values         – p-value matrix      {col: {col: p}}
      top_pairs        – list of top-25 numeric pairs by |r|
      vif              – VIF scores (Pearson only; capped at MAX_VIF_COLS
                         for runtime; includes a truncation note when capped)

      # Categorical × Categorical
      cramers_v        – bias-corrected Cramér's V matrix
      theils_u         – Theil's U asymmetric matrix [row→col]
      cat_p_values     – chi-square p-values
      cat_top_pairs    – top Cramér's V pairs

      # Numeric × Categorical
      mixed            – full mixed association matrix
      mixed_top_pairs  – top mixed pairs ranked by η²

      # Summary
      insights         – plain-language observations list
    """
    if method not in ("pearson", "spearman", "kendall"):
        method = "pearson"

    profile = _profile_columns(df)

    # Coerce unknown-dtype columns in the working copy so downstream code
    # can treat them uniformly as string/object categorical columns.
    df_work = df.copy()
    for col in profile["coerced_cols"]:
        df_work[col] = df_work[col].astype(str).where(df[col].notna(), other=np.nan)

    num_cols = profile["num_cols"]   # all numeric columns, no cap
    cat_cols = profile["cat_cols"]   # all categorical columns, no cap

    result: dict = {
        "method":         method,
        "column_profile": profile,

        "matrix":     {},
        "p_values":   {},
        "top_pairs":  [],
        "vif":        [],

        "cramers_v":     {},
        "theils_u":      {},
        "cat_p_values":  {},
        "cat_top_pairs": [],

        "mixed":           {},
        "mixed_top_pairs": [],

        "insights": [],
    }

    # ── Numeric × Numeric ─────────────────────────────────────────────────────
    if len(num_cols) >= 2:
        num_result = compute_num_matrix(df_work, num_cols, method)
        result["matrix"]    = num_result["matrix"]
        result["p_values"]  = num_result["p_values"]
        result["top_pairs"] = _top_num_pairs(num_result["matrix"], num_result["columns"])
        if method == "pearson":
            result["vif"] = compute_vif(df_work, num_cols)

    # ── Categorical × Categorical ─────────────────────────────────────────────
    if len(cat_cols) >= 2:
        cat_result = compute_cat_matrix(df_work, cat_cols)
        result["cramers_v"]     = cat_result["cramers_v"]
        result["theils_u"]      = cat_result["theils_u"]
        result["cat_p_values"]  = cat_result["p_values"]
        result["cat_top_pairs"] = _top_cat_pairs(cat_result["cramers_v"], cat_cols)

    # ── Numeric × Categorical ─────────────────────────────────────────────────
    if len(num_cols) >= 1 and len(cat_cols) >= 1:
        mixed_result = compute_mixed_matrix(df_work, num_cols, cat_cols)
        result["mixed"]           = mixed_result["matrix"]
        result["mixed_top_pairs"] = mixed_result["top_pairs"]

    # ── Insights ──────────────────────────────────────────────────────────────
    result["insights"] = _generate_insights(result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Auto-insights
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_LABEL = {"pearson": "Pearson", "spearman": "Spearman rank", "kendall": "Kendall rank"}


def _generate_insights(r: dict) -> list[dict]:
    insights: list[dict] = []
    method_label = _METHOD_LABEL.get(r.get("method", "pearson"), "Pearson")

    # High numeric correlations — a high |r| is only reported as evidence of a
    p_values = r.get("p_values", {})
    for pair in r.get("top_pairs", [])[:5]:
        abs_r = pair["abs_correlation"]
        if abs_r < 0.7:
            continue
        p_val = p_values.get(pair["col1"], {}).get(pair["col2"])
        significant = p_val is not None and p_val < 0.05
        if not significant:
            insights.append({
                "type": "muted",
                "category": "numeric",
                "message": (
                    f"**{pair['col1']}** and **{pair['col2']}** have a high {method_label} "
                    f"coefficient ({pair['correlation']:+.3f}) but it is not statistically significant"
                    + (f" (p={p_val:.3f})" if p_val is not None else " (p-value unavailable)")
                    + " — likely noise from limited overlapping data, not a real relationship."
                ),
            })
        elif abs_r >= 0.9:
            insights.append({
                "type": "warning",
                "category": "numeric",
                "message": (
                    f"Very strong {'positive' if pair['correlation'] > 0 else 'negative'} "
                    f"{method_label} correlation ({pair['correlation']:+.3f}, p={p_val:.3f}) between "
                    f"**{pair['col1']}** and **{pair['col2']}** — possible redundancy."
                ),
            })
        else:
            insights.append({
                "type": "info",
                "category": "numeric",
                "message": (
                    f"Strong {method_label} correlation ({pair['correlation']:+.3f}, p={p_val:.3f}) between "
                    f"**{pair['col1']}** and **{pair['col2']}**."
                ),
            })

    # High VIF
    for v in r.get("vif", [])[:3]:
        if v.get("vif") is None:
            continue  # truncation sentinel row
        if v["vif"] > 10:
            insights.append({
                "type": "warning",
                "category": "vif",
                "message": (
                    f"**{v['column']}** has VIF = {v['vif']:.1f} — high multicollinearity. "
                    "Consider dropping or combining with correlated features."
                ),
            })
        elif v["vif"] > 5:
            insights.append({
                "type": "info",
                "category": "vif",
                "message": (
                    f"**{v['column']}** has VIF = {v['vif']:.1f} — moderate multicollinearity."
                ),
            })

    # High Cramér's V
    cat_p_values = r.get("cat_p_values", {})
    for pair in r.get("cat_top_pairs", [])[:3]:
        if pair["cramers_v"] < 0.7:
            continue
        p_val = cat_p_values.get(pair["col1"], {}).get(pair["col2"])
        significant = p_val is not None and p_val < 0.05
        if not significant:
            insights.append({
                "type": "muted",
                "category": "categorical",
                "message": (
                    f"**{pair['col1']}** and **{pair['col2']}** show a high Cramér's V "
                    f"({pair['cramers_v']:.3f}) but it is not statistically significant"
                    + (f" (p={p_val:.3f})" if p_val is not None else " (p-value unavailable)")
                    + " — likely noise from limited data, not a real association."
                ),
            })
        else:
            insights.append({
                "type": "warning",
                "category": "categorical",
                "message": (
                    f"Strong categorical association (Cramér's V = {pair['cramers_v']:.3f}, p={p_val:.3f}) "
                    f"between **{pair['col1']}** and **{pair['col2']}** — may indicate redundant categories."
                ),
            })

    for pair in r.get("mixed_top_pairs", [])[:3]:
        eta = pair.get("eta_sq")
        if eta is None or eta < 0.14:
            continue
        p_val = pair.get("p_value")
        significant = p_val is not None and p_val < 0.05
        if not significant:
            insights.append({
                "type": "muted",
                "category": "mixed",
                "message": (
                    f"**{pair['cat_col']}** appears to explain {eta * 100:.1f}% of variance in "
                    f"**{pair['num_col']}** (η² = {eta:.3f}) but it is not statistically significant"
                    + (f" (p={p_val:.3f})" if p_val is not None else " (p-value unavailable)")
                    + " — likely noise from limited data or too few samples per group."
                ),
            })
        else:
            insights.append({
                "type": "info",
                "category": "mixed",
                "message": (
                    f"**{pair['cat_col']}** explains {eta * 100:.1f}% of variance in "
                    f"**{pair['num_col']}** (η² = {eta:.3f}, p={p_val:.3f}) — strong group-level difference."
                ),
            })

    # Coerced columns note
    coerced = r.get("column_profile", {}).get("coerced_cols", [])
    if coerced:
        insights.append({
            "type": "info",
            "category": "info",
            "message": (
                f"{len(coerced)} column(s) had an unrecognised dtype and were coerced to "
                "string for categorical analysis: "
                + ", ".join(f"**{c}**" for c in coerced[:5])
                + (" …" if len(coerced) > 5 else "")
                + "."
            ),
        })

    # Skipped columns note (genuinely all-null)
    skipped = r.get("column_profile", {}).get("skipped_cols", [])
    if skipped:
        insights.append({
            "type": "muted",
            "category": "info",
            "message": (
                f"{len(skipped)} column(s) skipped because all values are null: "
                + ", ".join(f"**{c}**" for c in skipped[:5])
                + (" …" if len(skipped) > 5 else "")
                + "."
            ),
        })

    return insights