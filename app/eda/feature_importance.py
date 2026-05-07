import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder


def run_feature_importance(df: pd.DataFrame, target: str) -> dict:
    if target not in df.columns:
        return {
            "target": target, "problem_type": "unknown",
            "importances": [], "mutual_info": [], "correlations": [],
            "error": f"Column '{target}' not found",
        }

    df_clean = df.dropna(subset=[target]).copy()
    y_raw = df_clean[target]

    # Detect problem type
    if pd.api.types.is_numeric_dtype(y_raw):
        unique_ratio = y_raw.nunique() / max(len(y_raw), 1)
        problem_type = "classification" if (y_raw.nunique() <= 20 and unique_ratio < 0.05) else "regression"
    else:
        problem_type = "classification"

    # Encode target
    if not pd.api.types.is_numeric_dtype(y_raw):
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw.astype(str))
    else:
        y_enc = y_raw.values.astype(float)

    # Prepare features
    feature_cols = [c for c in df_clean.columns if c != target]
    X = df_clean[feature_cols].copy()

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.select_dtypes(include=np.number)
    numeric_medians = X.median()
    X = X.fillna(numeric_medians)

    if X.empty or len(X) < 10:
        return {
            "target": target, "problem_type": problem_type,
            "importances": [], "mutual_info": [], "correlations": [],
            "error": "Insufficient data for feature importance analysis",
        }

    y_aligned = y_enc[:len(X)]

    # Random Forest
    rf_importances = []
    try:
        clf = (
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            if problem_type == "classification"
            else RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        )
        clf.fit(X, y_aligned)
        for feat, imp in zip(X.columns, clf.feature_importances_):
            rf_importances.append({
                "feature": str(feat),
                "importance": round(float(imp), 6),
                "method": "random_forest",
            })
        rf_importances.sort(key=lambda x: x["importance"], reverse=True)
    except Exception:
        pass

    # Mutual Information
    mi_scores = []
    try:
        fn = mutual_info_classif if problem_type == "classification" else mutual_info_regression
        mi = fn(X, y_aligned, random_state=42)
        for feat, score in zip(X.columns, mi):
            mi_scores.append({"feature": str(feat), "score": round(float(score), 6)})
        mi_scores.sort(key=lambda x: x["score"], reverse=True)
    except Exception:
        pass

    # Correlations with target (for regression)
    correlations = []
    if problem_type == "regression":
        y_series = pd.Series(y_aligned, index=X.index)
        for col in X.columns:
            try:
                val = float(X[col].corr(y_series))
                if not np.isnan(val):
                    correlations.append({"feature": str(col), "correlation": round(val, 4)})
            except Exception:
                pass
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "target": target,
        "problem_type": problem_type,
        "importances": rf_importances[:20],
        "mutual_info": mi_scores[:20],
        "correlations": correlations[:20],
    }
