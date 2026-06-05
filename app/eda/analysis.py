"""
Master EDA Analysis Engine.
Computes all chart data for the /analysis page in a single pass.
Results are cached by dataset content hash.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SAMPLE_THRESHOLD = 100_000
SAMPLE_SIZE = 50_000
MAX_SCATTER_PAIRS = 5
MAX_PAIRPLOT_COLS = 6
MAX_CAT_BARS = 20
MAX_PIE_SLICES = 8


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe(v: Any) -> Any:
    """Convert numpy scalars / NaN / Inf to JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return [_safe(x) for x in v.tolist()]
    return v


def _maybe_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if len(df) > SAMPLE_THRESHOLD:
        return df.sample(SAMPLE_SIZE, random_state=42), True
    return df, False


# ── numeric chart builders ────────────────────────────────────────────────────

def _histogram_kde(series: pd.Series) -> dict:
    """Histogram bins + KDE curve + mean/median lines."""
    from scipy.stats import gaussian_kde

    s = series.dropna()
    if len(s) < 5:
        return {}

    n = len(s)
    # Sturges rule capped at 50
    n_bins = min(max(int(np.ceil(np.log2(n) + 1)), 5), 50)

    counts, bin_edges = np.histogram(s, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # KDE
    try:
        kde_fn = gaussian_kde(s)
        kde_x = np.linspace(s.min(), s.max(), 200)
        kde_y = kde_fn(kde_x)
        # scale to histogram area
        bin_width = bin_edges[1] - bin_edges[0]
        kde_y = kde_y * n * bin_width
    except Exception:
        kde_x, kde_y = np.array([]), np.array([])

    return {
        "bins": [_safe(x) for x in bin_centers.tolist()],
        "counts": [_safe(x) for x in counts.tolist()],
        "kde_x": [_safe(x) for x in kde_x.tolist()],
        "kde_y": [_safe(x) for x in kde_y.tolist()],
        "mean": _safe(float(s.mean())),
        "median": _safe(float(s.median())),
    }


def _box_stats(series: pd.Series) -> dict:
    """Q1/Q2/Q3/whiskers/outliers for box + violin plots."""
    s = series.dropna()
    if len(s) < 5:
        return {}

    q1, q2, q3 = float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = s[(s < lo) | (s > hi)]

    return {
        "min": _safe(float(s[s >= lo].min()) if len(s[s >= lo]) else s.min()),
        "q1": _safe(q1),
        "median": _safe(q2),
        "q3": _safe(q3),
        "max": _safe(float(s[s <= hi].max()) if len(s[s <= hi]) else s.max()),
        "mean": _safe(float(s.mean())),
        "outliers": [_safe(x) for x in outliers.sample(min(200, len(outliers)), random_state=42).tolist()],
    }


def _violin_kde(series: pd.Series) -> dict:
    """KDE curve shaped data for violin plot."""
    from scipy.stats import gaussian_kde

    s = series.dropna()
    if len(s) < 10:
        return {}
    try:
        kde_fn = gaussian_kde(s)
        y = np.linspace(s.min(), s.max(), 200)
        density = kde_fn(y)
        return {
            "y": [_safe(x) for x in y.tolist()],
            "density": [_safe(x) for x in density.tolist()],
        }
    except Exception:
        return {}


def _qq_plot(series: pd.Series) -> dict:
    """Theoretical normal quantiles vs sample quantiles."""
    from scipy import stats

    s = series.dropna()
    if len(s) < 10:
        return {}
    try:
        (osm, osr), (slope, intercept, _) = stats.probplot(s, dist="norm")
        return {
            "theoretical": [_safe(x) for x in osm],
            "sample": [_safe(x) for x in osr],
            "line_x": [_safe(float(min(osm))), _safe(float(max(osm)))],
            "line_y": [
                _safe(slope * float(min(osm)) + intercept),
                _safe(slope * float(max(osm)) + intercept),
            ],
        }
    except Exception:
        return {}


def _ecdf(series: pd.Series) -> dict:
    """Empirical CDF: sorted values + cumulative probability."""
    s = series.dropna().sort_values()
    if len(s) < 5:
        return {}
    # Downsample for large series
    if len(s) > 2000:
        idx = np.round(np.linspace(0, len(s) - 1, 2000)).astype(int)
        s = s.iloc[idx]
    n = len(s)
    ecdf_y = np.arange(1, n + 1) / n
    return {
        "x": [_safe(x) for x in s.tolist()],
        "y": [_safe(x) for x in ecdf_y.tolist()],
        "p25": _safe(float(series.quantile(0.25))),
        "p50": _safe(float(series.quantile(0.50))),
        "p75": _safe(float(series.quantile(0.75))),
    }


def _normality_test(series: pd.Series) -> dict:
    from scipy import stats

    s = series.dropna()
    if len(s) < 8:
        return {"test": "none", "p_value": None, "is_normal": None}
    try:
        if len(s) < 5000:
            stat, p = stats.shapiro(s.sample(min(5000, len(s)), random_state=42))
            test = "shapiro"
        else:
            stat, p = stats.normaltest(s)
            test = "dagostino"
        return {"test": test, "statistic": _safe(stat), "p_value": _safe(p), "is_normal": bool(p > 0.05)}
    except Exception:
        return {"test": "error", "p_value": None, "is_normal": None}


# ── categorical chart builders ────────────────────────────────────────────────

def _bar_chart(series: pd.Series) -> dict:
    counts = series.value_counts()
    total = len(series)
    top = counts.head(MAX_CAT_BARS)
    labels = [str(v) for v in top.index.tolist()]
    values = top.tolist()
    pcts = [(v / total * 100) for v in values]
    return {
        "labels": labels,
        "values": [_safe(v) for v in values],
        "percentages": [_safe(p) for p in pcts],
        "other_count": int(counts.iloc[MAX_CAT_BARS:].sum()) if len(counts) > MAX_CAT_BARS else 0,
        "total_categories": int(len(counts)),
    }


def _pie_data(series: pd.Series) -> dict | None:
    counts = series.value_counts()
    if len(counts) > 15:
        return None  # too many slices
    top = counts.head(MAX_PIE_SLICES)
    other = counts.iloc[MAX_PIE_SLICES:].sum()
    labels = [str(v) for v in top.index.tolist()]
    values = top.tolist()
    if other > 0:
        labels.append(f"Other ({len(counts) - MAX_PIE_SLICES})")
        values.append(int(other))
    total = sum(values)
    pcts = [v / total * 100 for v in values]
    return {
        "labels": labels,
        "values": [_safe(v) for v in values],
        "percentages": [_safe(p) for p in pcts],
    }


def _pareto_data(series: pd.Series) -> dict:
    counts = series.value_counts().head(MAX_CAT_BARS)
    total = counts.sum()
    cumulative = (counts.cumsum() / total * 100).tolist()
    return {
        "labels": [str(v) for v in counts.index.tolist()],
        "values": [_safe(v) for v in counts.tolist()],
        "cumulative_pct": [_safe(p) for p in cumulative],
    }


# ── datetime chart builders ───────────────────────────────────────────────────

def _timeseries_data(df: pd.DataFrame, col: str) -> dict:
    """Auto-aggregate datetime column to appropriate frequency."""
    try:
        ts = pd.to_datetime(df[col], errors="coerce").dropna()
        counts = ts.dt.date.value_counts().sort_index()
        if len(counts) > 500:
            # Resample to week
            s = pd.Series(counts.values, index=pd.DatetimeIndex(counts.index))
            counts = s.resample("W").sum()
        dates = [str(d) for d in counts.index]
        values = [_safe(int(v)) for v in counts.values]
        return {"dates": dates, "values": values}
    except Exception:
        return {}


def _seasonality(df: pd.DataFrame, col: str) -> dict:
    """Hour of day / day of week / month of year distributions."""
    try:
        ts = pd.to_datetime(df[col], errors="coerce").dropna()
        hour_counts = ts.dt.hour.value_counts().sort_index()
        dow_counts = ts.dt.dayofweek.value_counts().sort_index()
        month_counts = ts.dt.month.value_counts().sort_index()

        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        return {
            "by_hour": {
                "labels": [str(h) for h in hour_counts.index.tolist()],
                "values": [int(v) for v in hour_counts.values.tolist()],
            },
            "by_dow": {
                "labels": [dow_names[d] for d in dow_counts.index.tolist()],
                "values": [int(v) for v in dow_counts.values.tolist()],
            },
            "by_month": {
                "labels": [month_names[m - 1] for m in month_counts.index.tolist()],
                "values": [int(v) for v in month_counts.values.tolist()],
            },
        }
    except Exception:
        return {}


# ── multi-column builders ─────────────────────────────────────────────────────

def _correlation_data(df: pd.DataFrame, num_cols: list[str]) -> dict:
    """Full Pearson correlation matrix for numeric columns."""
    if len(num_cols) < 2:
        return {}
    try:
        sub = df[num_cols].dropna(how="all")
        corr = sub.corr(method="pearson")
        labels = corr.columns.tolist()
        z = [[_safe(v) for v in row] for row in corr.values.tolist()]
        return {"labels": labels, "z": z}
    except Exception:
        return {}


def _scatter_pairs(df: pd.DataFrame, num_cols: list[str]) -> list[dict]:
    print(f"[scatter_pairs] called with {len(num_cols)} cols: {num_cols}")  # add this

    if len(num_cols) < 2:
        return []

    from scipy import stats as sp_stats

    # Exclude ID-like columns (>95% unique) — they produce garbage correlations
    n = len(df)
    usable = [c for c in num_cols if df[c].nunique() / n < 0.95]
    print(f"[scatter_pairs] usable cols after ID filter: {usable}")  # add this

    if len(usable) < 2:
        return []

    try:
        corr = df[usable].corr(method="pearson").abs()
    except Exception as e:
        print(f"[scatter_pairs] corr failed: {e}")
        return []

    pairs = []
    seen = set()
    for col in corr.columns:
        for row in corr.index:
            if col == row or (row, col) in seen:
                continue
            seen.add((col, row))
            val = corr.loc[row, col]
            if pd.notna(val):
                pairs.append((col, row, float(val)))

    pairs.sort(key=lambda x: x[2], reverse=True)

    result = []
    for c1, c2, r in pairs[:MAX_SCATTER_PAIRS]:
        joined = df[[c1, c2]].dropna()
        if len(joined) < 10:
            continue
        if len(joined) > 2000:
            joined = joined.sample(2000, random_state=42)
        try:
            slope, intercept, rval, _, _ = sp_stats.linregress(joined[c1], joined[c2])
            x_line = [float(joined[c1].min()), float(joined[c1].max())]
            y_line = [slope * x + intercept for x in x_line]
            r2 = rval ** 2
        except Exception as e:
            print(f"[scatter_pairs] linregress failed for {c1}×{c2}: {e}")
            x_line, y_line, r2 = [], [], None

        result.append({
            "col1": c1, "col2": c2,
            "pearson_r": _safe(r),
            "r2": _safe(r2),
            "x": [_safe(v) for v in joined[c1].tolist()],
            "y": [_safe(v) for v in joined[c2].tolist()],
            "line_x": [_safe(v) for v in x_line],
            "line_y": [_safe(v) for v in y_line],
        })
    return result

def _grouped_box(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> dict:
    if not num_cols:
        return {}
    try:
        n = len(df)
        # Also treat low-cardinality numeric cols as grouping candidates
        low_card_numeric = [
            c for c in num_cols
            if 2 <= df[c].nunique() <= 15
        ]
        grouping_candidates = cat_cols + [
            c for c in low_card_numeric if c not in cat_cols
        ]

        # Pick numeric col with highest variance, excluding ID-like cols
        usable_num = [c for c in num_cols if df[c].nunique() / n > 0.05]
        variances = {c: df[c].var() for c in usable_num if df[c].notna().sum() > 10}
        if not variances:
            return {}
        num_col = max(variances, key=lambda k: variances[k] if not np.isnan(variances[k]) else 0)

        # Pick grouping col with lowest cardinality (2-15 unique values)
        candidates = [c for c in grouping_candidates if 2 <= df[c].nunique() <= 15]
        # Exclude the chosen numeric col itself
        candidates = [c for c in candidates if c != num_col]
        if not candidates:
            return {}
        cat_col = min(candidates, key=lambda c: df[c].nunique())

        groups = {}
        for grp, sub in df[[num_col, cat_col]].dropna().groupby(cat_col):
            s = sub[num_col]
            if len(s) < 5:
                continue
            q1, q2, q3 = float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_mask = (s < lo) | (s > hi)
            groups[str(grp)] = {
                "min": _safe(float(s[s >= lo].min()) if s[s >= lo].shape[0] else s.min()),
                "q1": _safe(q1), "median": _safe(q2), "q3": _safe(q3),
                "max": _safe(float(s[s <= hi].max()) if s[s <= hi].shape[0] else s.max()),
                "outliers": [_safe(x) for x in s[outlier_mask].sample(
                    min(50, outlier_mask.sum()), random_state=42).tolist()],
                "n": int(len(s)),
            }
        if not groups:
            return {}
        return {"numeric_col": num_col, "categorical_col": cat_col, "groups": groups}
    except Exception as e:
        print(f"[grouped_box] failed: {e}")
        return {}
# ── statistical cards ─────────────────────────────────────────────────────────

def _normality_table(df: pd.DataFrame, num_cols: list[str]) -> list[dict]:
    rows = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 8:
            continue
        res = _normality_test(s)
        skew = _safe(float(s.skew())) if len(s) >= 3 else None
        kurt = _safe(float(s.kurtosis())) if len(s) >= 4 else None
        rows.append({
            "column": col, "n": int(len(s)),
            "test": res.get("test"), "p_value": res.get("p_value"),
            "is_normal": res.get("is_normal"),
            "skewness": skew, "kurtosis": kurt,
        })
    return rows


def _outlier_summary(df: pd.DataFrame, num_cols: list[str]) -> list[dict]:
    rows = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({
            "column": col,
            "outlier_count": n_out,
            "outlier_pct": _safe(n_out / len(s) * 100),
            "lower_bound": _safe(float(lo)),
            "upper_bound": _safe(float(hi)),
        })
    rows.sort(key=lambda r: r["outlier_pct"] or 0, reverse=True)
    return rows


def _cardinality_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    n = len(df)
    for col in df.columns:
        uniq = int(df[col].nunique())
        pct = uniq / n * 100 if n > 0 else 0
        if pct > 95:
            flag = "id_like"
        elif uniq == 1:
            flag = "constant"
        elif uniq == 2:
            flag = "binary"
        elif uniq < 10:
            flag = "low_cardinality"
        else:
            flag = "normal"
        rows.append({
            "column": col,
            "unique_count": uniq,
            "unique_pct": _safe(pct),
            "flag": flag,
            "dtype": str(df[col].dtype),
        })
    rows.sort(key=lambda r: r["unique_count"])
    return rows


def _duplicate_info(df: pd.DataFrame) -> dict:
    dup_count = int(df.duplicated().sum())
    return {
        "total_rows": len(df),
        "duplicate_count": dup_count,
        "duplicate_pct": _safe(dup_count / len(df) * 100) if len(df) > 0 else 0,
    }


def _missing_bar(df: pd.DataFrame) -> list[dict]:
    n = len(df)
    rows = []
    for col in df.columns:
        mc = int(df[col].isna().sum())
        rows.append({
            "column": col,
            "missing_count": mc,
            "missing_pct": _safe(mc / n * 100) if n > 0 else 0,
        })
    rows.sort(key=lambda r: r["missing_pct"] or 0, reverse=True)
    return [r for r in rows if r["missing_pct"] and r["missing_pct"] > 0]


# ── master engine ─────────────────────────────────────────────────────────────

def run_full_analysis(df: pd.DataFrame) -> dict:
    import sys
    print("RUN_FULL_ANALYSIS CALLED", file=sys.stderr, flush=True)
    """
    Compute all chart data for the Analysis page.
    Returns a single dict with all chart data keyed by analysis type.
    """
    df_sample, sampled = _maybe_sample(df)

    # Classify columns
    from .profiler import classify_column
    col_types: dict[str, str] = {}
    for col in df.columns:
        try:
            col_types[col] = classify_column(df[col])
        except Exception:
            col_types[col] = "categorical"

    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t in ("categorical", "boolean")]
    dt_cols = [c for c, t in col_types.items() if t == "datetime"]

    # ── Per-column charts are now lazy-loaded via /analysis/column/{col_name} endpoint
    # Skeleton only — no per-column data computed here

    # ── Multi-column analyses ─────────────────────────────────────────────────
    multi = {}
    if len(num_cols) >= 2:
        multi["correlation"] = _correlation_data(df_sample, num_cols)
        multi["scatter_pairs"] = _scatter_pairs(df_sample, num_cols)
    else:
        multi["correlation"] = {}
        multi["scatter_pairs"] = []

    multi["grouped_box"] = _grouped_box(df_sample, num_cols, cat_cols)

    # ── Missing ───────────────────────────────────────────────────────────────
    missing_charts = {
        "bar": _missing_bar(df),  # use full df for accurate missing %
    }

    # ── Statistical cards ─────────────────────────────────────────────────────
    stat_cards = {
        "normality_table": _normality_table(df_sample, num_cols),
        "outlier_summary": _outlier_summary(df_sample, num_cols),
        "cardinality": _cardinality_table(df),
        "duplicates": _duplicate_info(df),
        "missing_bar": _missing_bar(df),
    }

    return {
        "sampled": sampled,
        "sample_size": SAMPLE_SIZE if sampled else len(df),
        "total_rows": len(df),
        "column_types": col_types,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "datetime_cols": dt_cols,
        "multi_column": multi,
        "missing_charts": missing_charts,
        "stat_cards": stat_cards,
    }


# ── On-demand bivariate analysis ─────────────────────────────────────────────

def compute_bivariate_num_num(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Scatter + linear trend for two numeric columns."""
    from scipy import stats as sp_stats
    joined = df[[col1, col2]].dropna()
    if len(joined) < 5:
        return {"error": "Not enough data", "btype": "num_num"}
    if len(joined) > 2000:
        joined = joined.sample(2000, random_state=42)
    try:
        slope, intercept, rval, pval, _ = sp_stats.linregress(joined[col1], joined[col2])
        x_line = [float(joined[col1].min()), float(joined[col1].max())]
        y_line = [slope * x + intercept for x in x_line]
        r2 = float(rval ** 2)
    except Exception:
        x_line, y_line, r2, rval, pval = [], [], None, None, None
    return {
        "btype": "num_num",
        "col1": col1, "col2": col2,
        "x": [_safe(v) for v in joined[col1].tolist()],
        "y": [_safe(v) for v in joined[col2].tolist()],
        "pearson_r": _safe(rval),
        "r2": _safe(r2),
        "p_value": _safe(pval),
        "line_x": [_safe(v) for v in x_line],
        "line_y": [_safe(v) for v in y_line],
        "n": len(joined),
    }


def compute_bivariate_cat_cat(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Cross-tabulation grouped bar for two categorical columns."""
    sub = df[[col1, col2]].dropna()
    if len(sub) < 5:
        return {"error": "Not enough data", "btype": "cat_cat"}

    top1 = sub[col1].value_counts().head(10).index.tolist()
    top2 = sub[col2].value_counts().head(10).index.tolist()
    sub = sub[sub[col1].isin(top1) & sub[col2].isin(top2)]

    crosstab = pd.crosstab(sub[col1], sub[col2])
    cat1_labels = [str(x) for x in crosstab.index.tolist()]
    cat2_labels = [str(x) for x in crosstab.columns.tolist()]

    series = []
    for c2 in cat2_labels:
        series.append({
            "name": c2,
            "values": [int(crosstab.at[c1, c2]) if c1 in crosstab.index and c2 in crosstab.columns else 0
                       for c1 in cat1_labels],
        })

    return {
        "btype": "cat_cat",
        "col1": col1, "col2": col2,
        "cat1_labels": cat1_labels,
        "cat2_labels": cat2_labels,
        "series": series,
        "n": len(sub),
    }


def compute_bivariate_num_cat(df: pd.DataFrame, num_col: str, cat_col: str) -> dict:
    """Grouped box plot for numeric vs categorical."""
    sub = df[[num_col, cat_col]].dropna()
    if len(sub) < 5:
        return {"error": "Not enough data", "btype": "num_cat"}

    top_cats = sub[cat_col].value_counts().head(12).index.tolist()
    sub = sub[sub[cat_col].isin(top_cats)]

    groups = {}
    for grp, gdf in sub.groupby(cat_col):
        s = gdf[num_col]
        if len(s) < 3:
            continue
        q1, q2, q3 = float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (s < lo) | (s > hi)
        groups[str(grp)] = {
            "min": _safe(float(s[s >= lo].min()) if s[s >= lo].shape[0] else s.min()),
            "q1": _safe(q1), "median": _safe(q2), "q3": _safe(q3),
            "max": _safe(float(s[s <= hi].max()) if s[s <= hi].shape[0] else s.max()),
            "outliers": [_safe(x) for x in s[outlier_mask].sample(
                min(50, int(outlier_mask.sum())), random_state=42).tolist()],
            "n": int(len(s)),
        }

    return {
        "btype": "num_cat",
        "numeric_col": num_col, "categorical_col": cat_col,
        "groups": groups,
        "n": len(sub),
    }


def compute_pca(df: pd.DataFrame, num_cols: list, n_components: int = 2) -> dict:
    """PCA on numeric columns — scores, loadings, explained variance."""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA as SkPCA
    except ImportError:
        return {"error": "scikit-learn not installed"}

    usable = [c for c in num_cols if df[c].notna().sum() > 10]
    if len(usable) < 2:
        return {"error": "Need at least 2 numeric columns"}

    sub = df[usable].dropna()
    if len(sub) < 5:
        return {"error": "Not enough rows after dropping nulls"}

    n_components = min(n_components, len(usable), len(sub))
    X = StandardScaler().fit_transform(sub)
    pca = SkPCA(n_components=n_components)
    scores = pca.fit_transform(X)

    if len(scores) > 2000:
        idx = np.random.choice(len(scores), 2000, replace=False)
        scores = scores[idx]

    loadings = pca.components_
    return {
        "n_components": n_components,
        "explained_variance_ratio": [_safe(v) for v in pca.explained_variance_ratio_.tolist()],
        "columns": usable,
        "scores_pc1": [_safe(v) for v in scores[:, 0].tolist()],
        "scores_pc2": [_safe(v) for v in scores[:, 1].tolist()] if n_components >= 2 else [],
        "loadings_pc1": [_safe(v) for v in loadings[0].tolist()],
        "loadings_pc2": [_safe(v) for v in loadings[1].tolist()] if n_components >= 2 else [],
        "n": len(sub),
    }


def compute_scatter3d(df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> dict:
    """3D scatter for three numeric columns."""
    sub = df[[x_col, y_col, z_col]].dropna()
    if len(sub) < 5:
        return {"error": "Not enough data"}
    if len(sub) > 2000:
        sub = sub.sample(2000, random_state=42)
    return {
        "x_col": x_col, "y_col": y_col, "z_col": z_col,
        "x": [_safe(v) for v in sub[x_col].tolist()],
        "y": [_safe(v) for v in sub[y_col].tolist()],
        "z": [_safe(v) for v in sub[z_col].tolist()],
        "n": len(sub),
    }
