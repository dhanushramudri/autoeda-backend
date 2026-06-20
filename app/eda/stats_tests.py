"""On-demand statistical hypothesis tests, run via Scout's run_statistical_test
tool. Separate from correlations.py's embedded chi2/ANOVA (which run for every
column pair as part of the correlation matrix) — these are targeted, single-pair
tests the LLM picks deliberately rather than auto-computed for everything.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def run_statistical_test(
    df: pd.DataFrame,
    test: str,
    column: str,
    group_column: str | None = None,
    group_a: str | None = None,
    group_b: str | None = None,
) -> dict:
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}

    if test == "shapiro":
        values = _clean_numeric(df[column])
        if len(values) < 3:
            return {"error": "Need at least 3 non-null numeric values for a normality test."}
        sample = values.sample(min(len(values), 5000), random_state=0)
        stat, p = stats.shapiro(sample)
        return {
            "test": "shapiro", "column": column, "n": len(sample),
            "statistic": float(stat), "p_value": float(p),
            "interpretation": "Looks normally distributed (fail to reject H0)" if p > 0.05 else "Not normally distributed (reject H0 at p<0.05)",
        }

    if group_column is None or group_column not in df.columns:
        return {"error": f"group_column is required and must exist for test '{test}'."}

    if test == "chi2":
        ct = pd.crosstab(df[column], df[group_column])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return {"error": "Need at least 2 categories in each column for a chi-square test."}
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        return {
            "test": "chi2", "column": column, "group_column": group_column,
            "statistic": float(chi2), "p_value": float(p), "degrees_of_freedom": int(dof),
            "interpretation": "No significant association (fail to reject H0)" if p > 0.05 else "Significant association (reject H0 at p<0.05)",
        }

    if test == "anova":
        groups = [
            _clean_numeric(g[column]) for _, g in df.groupby(group_column)
            if len(_clean_numeric(g[column])) > 0
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            return {"error": "Need at least 2 non-empty groups for ANOVA."}
        stat, p = stats.f_oneway(*groups)
        return {
            "test": "anova", "column": column, "group_column": group_column, "n_groups": len(groups),
            "statistic": float(stat), "p_value": float(p),
            "interpretation": "No significant difference across groups (fail to reject H0)" if p > 0.05 else "Significant difference across groups (reject H0 at p<0.05)",
        }

    # ttest_ind / mannwhitney / ks_2samp all need exactly two named groups.
    if group_a is None or group_b is None:
        return {"error": f"group_a and group_b are required for test '{test}'."}

    mask_a = df[group_column].astype(str) == str(group_a)
    mask_b = df[group_column].astype(str) == str(group_b)
    sample_a = _clean_numeric(df.loc[mask_a, column])
    sample_b = _clean_numeric(df.loc[mask_b, column])
    if len(sample_a) < 2 or len(sample_b) < 2:
        return {"error": f"Need at least 2 values in each group. Found {len(sample_a)} for '{group_a}', {len(sample_b)} for '{group_b}'."}

    if test == "ttest_ind":
        stat, p = stats.ttest_ind(sample_a, sample_b, equal_var=False)
        label = "Welch's t-test"
    elif test == "mannwhitney":
        stat, p = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
        label = "Mann-Whitney U"
    elif test == "ks_2samp":
        stat, p = stats.ks_2samp(sample_a, sample_b)
        label = "Kolmogorov-Smirnov"
    else:
        return {"error": f"Unknown test: {test}"}

    return {
        "test": test, "label": label, "column": column, "group_column": group_column,
        "group_a": group_a, "group_b": group_b,
        "n_a": len(sample_a), "n_b": len(sample_b),
        "mean_a": float(sample_a.mean()), "mean_b": float(sample_b.mean()),
        "statistic": float(stat), "p_value": float(p),
        "interpretation": "No significant difference (fail to reject H0)" if p > 0.05 else "Significant difference (reject H0 at p<0.05)",
    }
