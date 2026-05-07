import numpy as np
import pandas as pd


def run_missing_analysis(df: pd.DataFrame) -> dict:
    missing = df.isnull().sum()
    missing_pct = (missing / max(len(df), 1) * 100).round(2)

    cols_with_missing = [
        {"name": str(col), "count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns
        if missing[col] > 0
    ]

    # Missing co-occurrence correlation
    corr_dict: dict = {}
    if len(cols_with_missing) >= 2:
        missing_cols = [c["name"] for c in cols_with_missing]
        indicator = df[missing_cols].isnull().astype(int)
        corr = indicator.corr().round(3)
        corr_dict = {
            col: {
                k: (None if (v != v) else float(v))
                for k, v in corr[col].items()
            }
            for col in missing_cols
        }

    # MCAR/MAR heuristic
    mcar_indicators: dict = {}
    for col in df.columns:
        if missing[col] == 0:
            continue
        missing_mask = df[col].isnull().astype(int)
        correlated = []
        for other in df.select_dtypes(include=np.number).columns:
            if other == col:
                continue
            try:
                val = float(missing_mask.corr(df[other]))
                if not np.isnan(val) and abs(val) > 0.3:
                    correlated.append(str(other))
            except Exception:
                pass
        mcar_indicators[str(col)] = {
            "likely": "MAR" if correlated else "MCAR",
            "correlated_with": correlated,
        }

    # Imputation suggestions
    suggestions: dict = {}
    for col in df.columns:
        if missing[col] == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            skew = abs(df[col].skew()) if len(df[col].dropna()) > 0 else 0
            suggestions[str(col)] = "median (skewed)" if skew > 1 else "mean (symmetric)"
        else:
            suggestions[str(col)] = "mode (categorical)"

    total_cells = int(len(df)) * int(len(df.columns))
    total_missing = int(missing.sum())

    return {
        "columns": cols_with_missing,
        "total_missing": total_missing,
        "missing_pct": round(total_missing / max(total_cells, 1) * 100, 2),
        "correlation_matrix": corr_dict,
        "mcar_indicators": mcar_indicators,
        "imputation_suggestions": suggestions,
    }
