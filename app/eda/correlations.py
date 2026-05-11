import numpy as np
import pandas as pd

MAX_CORR_COLS = 50   # cap correlation matrix size
MAX_VIF_COLS = 15    # VIF is O(n*k^2) — skip when too wide
MAX_CAT_COLS = 8     # Cramér's V pairs cap


def _safe(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Fast Cramér's V via scipy chi2_contingency."""
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    r, k = ct.shape
    return float(np.sqrt(chi2 / max(n * min(k - 1, r - 1), 1)))


def compute_vif(df: pd.DataFrame, num_cols: list[str]) -> list[dict]:
    """VIF via numpy QR decomposition — faster than sklearn LinearRegression loop."""
    if len(num_cols) < 2 or len(num_cols) > MAX_VIF_COLS:
        return []
    X = df[num_cols].dropna().values.astype(float)
    if len(X) < 5:
        return []
    # Add intercept column
    Xc = np.column_stack([np.ones(len(X)), X])
    results = []
    for i, col in enumerate(num_cols):
        try:
            col_idx = i + 1  # offset for intercept
            y = Xc[:, col_idx]
            others = np.delete(Xc, col_idx, axis=1)
            # QR solve is much faster than sklearn fit
            coef, _, _, _ = np.linalg.lstsq(others, y, rcond=None)
            y_pred = others @ coef
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            vif = 1 / (1 - r2 + 1e-12)
            results.append({"column": col, "vif": round(float(vif), 3)})
        except Exception:
            pass
    return sorted(results, key=lambda x: x["vif"], reverse=True)


def run_correlations(df: pd.DataFrame, method: str = "pearson") -> dict:
    all_num = df.select_dtypes(include=np.number).columns.tolist()
    all_cat = df.select_dtypes(include="object").columns.tolist()

    # Limit columns so the correlation matrix stays fast
    num_cols = all_num[:MAX_CORR_COLS]
    cat_cols = all_cat[:MAX_CAT_COLS]

    result: dict = {
        "method": method,
        "matrix": {},
        "top_pairs": [],
        "vif": None,
        "cramers_v": None,
    }

    if len(num_cols) >= 2:
        df_numeric = df[num_cols].dropna(axis=1, how="all")
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = df_numeric.corr(method=method)

        matrix: dict = {}
        for col in num_cols:
            if col in corr.columns:
                matrix[col] = {k: _safe(v) for k, v in corr[col].items()}
            else:
                matrix[col] = {c: None for c in num_cols}
        result["matrix"] = matrix

        pairs = []
        cols_in_corr = [c for c in num_cols if c in corr.columns]
        for i in range(len(cols_in_corr)):
            for j in range(i + 1, len(cols_in_corr)):
                val = _safe(corr.loc[cols_in_corr[i], cols_in_corr[j]])
                if val is not None:
                    pairs.append({
                        "col1": cols_in_corr[i],
                        "col2": cols_in_corr[j],
                        "correlation": val,
                        "abs_correlation": abs(val),
                    })
        result["top_pairs"] = sorted(pairs, key=lambda x: x["abs_correlation"], reverse=True)[:25]

        if method == "pearson":
            result["vif"] = compute_vif(df, num_cols)

    if len(cat_cols) >= 2:
        cv: dict = {}
        for col1 in cat_cols:
            cv[col1] = {}
            for col2 in cat_cols:
                if col1 == col2:
                    cv[col1][col2] = 1.0
                else:
                    try:
                        valid = df[[col1, col2]].dropna()
                        cv[col1][col2] = round(cramers_v(valid[col1], valid[col2]), 4) if len(valid) >= 5 else None
                    except Exception:
                        cv[col1][col2] = None
        result["cramers_v"] = cv

    return result
