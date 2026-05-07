import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _safe(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    n = len(x)
    chi2_stat = 0.0
    row_sums = ct.sum(axis=1).values
    col_sums = ct.sum(axis=0).values
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            expected = row_sums[i] * col_sums[j] / max(n, 1)
            if expected > 0:
                chi2_stat += (ct.iloc[i, j] - expected) ** 2 / expected
    phi2 = chi2_stat / max(n, 1)
    r, k = ct.shape
    return float(np.sqrt(phi2 / max(min(k - 1, r - 1), 1)))


def compute_vif(df: pd.DataFrame, num_cols: list[str]) -> list[dict]:
    if len(num_cols) < 2:
        return []
    X = df[num_cols].dropna()
    if len(X) < 5:
        return []
    results = []
    for col in num_cols:
        try:
            y = X[col].values
            X_others = X.drop(columns=[col]).values
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X_others, y)
            y_pred = reg.predict(X_others)
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            vif = 1 / (1 - r2 + 1e-12)
            results.append({"column": col, "vif": round(float(vif), 3)})
        except Exception:
            pass
    return sorted(results, key=lambda x: x["vif"], reverse=True)


def run_correlations(df: pd.DataFrame, method: str = "pearson") -> dict:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    result: dict = {
        "method": method,
        "matrix": {},
        "top_pairs": [],
        "vif": None,
        "cramers_v": None,
    }

    if len(num_cols) >= 2:
        # Remove columns with all NaN values before correlation
        df_numeric = df[num_cols].dropna(axis=1, how='all')
        
        # Suppress RuntimeWarning for invalid operations (division by zero in stddev)
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = df_numeric.corr(method=method)
        
        matrix: dict = {}
        for col in num_cols:
            if col in corr.columns:
                matrix[col] = {k: _safe(v) for k, v in corr[col].items()}
            else:
                matrix[col] = {c: None for c in num_cols}
        result["matrix"] = matrix

        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                val = _safe(corr.iloc[i, j])
                if val is not None:
                    pairs.append({
                        "col1": num_cols[i],
                        "col2": num_cols[j],
                        "correlation": val,
                        "abs_correlation": abs(val),
                    })
        result["top_pairs"] = sorted(pairs, key=lambda x: x["abs_correlation"], reverse=True)[:25]

        if method == "pearson" and len(num_cols) <= 25:
            result["vif"] = compute_vif(df, num_cols)

    if len(cat_cols) >= 2:
        limited = cat_cols[:10]
        cv: dict = {}
        for col1 in limited:
            cv[col1] = {}
            for col2 in limited:
                if col1 == col2:
                    cv[col1][col2] = 1.0
                else:
                    try:
                        valid = df[[col1, col2]].dropna()
                        if len(valid) >= 5:
                            cv[col1][col2] = round(cramers_v(valid[col1], valid[col2]), 4)
                        else:
                            cv[col1][col2] = None
                    except Exception:
                        cv[col1][col2] = None
        result["cramers_v"] = cv

    return result
