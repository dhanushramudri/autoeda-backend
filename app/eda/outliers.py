import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _safe(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None


def run_outlier_detection(
    df: pd.DataFrame,
    method: str = "iqr",
    column: str | None = None,
) -> dict:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if column:
        num_cols = [column] if column in num_cols else []

    outlier_summary = []
    all_outlier_indices: set = set()

    if method == "iqr":
        for col in num_cols:
            series = df[col].dropna()
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (df[col] < lower) | (df[col] > upper)
            indices = df.index[mask & df[col].notna()].tolist()
            all_outlier_indices.update(indices)
            outlier_summary.append({
                "name": col,
                "outlier_count": int(mask.sum()),
                "outlier_pct": round(float(mask.sum()) / max(len(series), 1) * 100, 2),
                "bounds": {
                    "lower": _safe(lower),
                    "upper": _safe(upper),
                    "q1": _safe(q1),
                    "q3": _safe(q3),
                },
            })

    elif method == "zscore":
        threshold = 3.0
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 3 or series.std() == 0:
                continue
            z = np.abs(scipy_stats.zscore(series.values))
            outlier_bool = pd.Series(False, index=df.index)
            outlier_bool.loc[series.index[z > threshold]] = True
            indices = df.index[outlier_bool].tolist()
            all_outlier_indices.update(indices)
            outlier_summary.append({
                "name": col,
                "outlier_count": int(outlier_bool.sum()),
                "outlier_pct": round(float(outlier_bool.sum()) / max(len(series), 1) * 100, 2),
                "bounds": {
                    "threshold": threshold,
                    "mean": _safe(series.mean()),
                    "std": _safe(series.std()),
                },
            })

    elif method == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        if num_cols and len(df) >= 10:
            X = df[num_cols].dropna()
            if len(X) >= 10:
                iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
                labels = iso.fit_predict(X)
                outlier_mask = labels == -1
                indices = X.index[outlier_mask].tolist()
                all_outlier_indices.update(indices)
                outlier_summary.append({
                    "name": "multivariate",
                    "outlier_count": int(outlier_mask.sum()),
                    "outlier_pct": round(float(outlier_mask.sum()) / max(len(X), 1) * 100, 2),
                    "bounds": {"contamination": 0.05},
                })

    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        if num_cols and len(df) >= 10:
            X = df[num_cols].dropna()
            if len(X) >= 10:
                X_scaled = StandardScaler().fit_transform(X)
                db = DBSCAN(eps=0.5, min_samples=5)
                labels = db.fit_predict(X_scaled)
                outlier_mask = labels == -1
                indices = X.index[outlier_mask].tolist()
                all_outlier_indices.update(indices)
                outlier_summary.append({
                    "name": "multivariate_dbscan",
                    "outlier_count": int(outlier_mask.sum()),
                    "outlier_pct": round(float(outlier_mask.sum()) / max(len(X), 1) * 100, 2),
                    "bounds": {"eps": 0.5, "min_samples": 5},
                })

    outlier_rows: list = []
    if all_outlier_indices:
        sample_idx = list(all_outlier_indices)[:100]
        outlier_rows = (
            df.loc[sample_idx].head(100).fillna("").astype(str).to_dict(orient="records")
        )

    return {
        "method": method,
        "columns": outlier_summary,
        "outlier_rows": outlier_rows,
        "total_outliers": len(all_outlier_indices),
    }
