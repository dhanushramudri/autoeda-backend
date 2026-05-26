import warnings as _warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    f_classif, f_regression,
)
from sklearn.preprocessing import LabelEncoder


def _uniform_threshold(n_features: int) -> float:
    """Importance level a feature would have if all were equally important."""
    return 1.0 / max(n_features, 1)


def run_feature_importance(df: pd.DataFrame, target: str) -> dict:
    empty = {
    "target": target,
    "problem_type": "unknown",
    "n_samples": 0,
    "n_features": 0,

    "model_score": None,
    "cv_score_mean": None,
    "cv_score_std": None,

    "class_distribution": None,

    "importances": [],
    "permutation_importances": [],

    "mutual_info": [],
    "correlations": [],
    "anova": [],

    "shap_values": [],

    "feature_meta": [],

    "stability": [],
    "interactions": [],

    "top_features": [],
    "drop_candidates": [],

    "warnings": [],

    "redundant_groups": [],
    "leakage_suspects": [],
}

    if target not in df.columns:
        return {**empty, "error": f"Column '{target}' not found"}

    df_clean = df.dropna(subset=[target]).copy()
    y_raw = df_clean[target]
    n_samples = len(df_clean)

    # ── Problem type detection ────────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(y_raw):
        unique_ratio = y_raw.nunique() / max(n_samples, 1)
        problem_type = (
            "classification"
            if (y_raw.nunique() <= 20 and unique_ratio < 0.05)
            else "regression"
        )
    else:
        problem_type = "classification"

    # ── Class distribution ────────────────────────────────────────────────────
    class_distribution = None
    if problem_type == "classification":
        vc = y_raw.value_counts()
        total = len(y_raw)
        class_distribution = {
            str(k): {"count": int(v), "pct": round(float(v / total * 100), 2)}
            for k, v in vc.head(10).items()
        }

    # ── Encode target ─────────────────────────────────────────────────────────
    if not pd.api.types.is_numeric_dtype(y_raw):
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw.astype(str))
    else:
        y_enc = y_raw.values.astype(float)

    # ── Feature matrix preparation ────────────────────────────────────────────
    feature_cols = [c for c in df_clean.columns if c != target]

    # Capture raw meta BEFORE encoding (for missingness, unique counts)
    raw_meta: dict[str, dict] = {}
    for col in feature_cols:
        s = df_clean[col]
        mc = int(s.isna().sum())
        raw_meta[col] = {
            "missing_pct": round(float(mc / n_samples * 100), 2),
            "missing_count": mc,
            "unique_count": int(s.nunique()),
            "dtype": str(s.dtype),
        }

    X = df_clean[feature_cols].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.select_dtypes(include=np.number)
    X = X.fillna(X.median())

    n_features = len(X.columns)

    if X.empty or n_samples < 10:
        return {
            **empty,
            "problem_type": problem_type,
            "n_samples": n_samples,
            "n_features": n_features,
            "class_distribution": class_distribution,
            "warnings": [{"type": "insufficient_data",
                          "message": "Insufficient data for feature importance analysis.",
                          "level": "danger"}],
            "error": "Insufficient data",
        }

    y_aligned = y_enc[: len(X)]

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_importances: list[dict] = []
    rf_map: dict[str, float] = {}
    model_score = None
    try:
        clf = (
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
            if problem_type == "classification"
            else RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
        )
        clf.fit(X, y_aligned)
        model_score = round(float(clf.oob_score_), 4)
        for feat, imp in zip(X.columns, clf.feature_importances_):
            rf_map[str(feat)] = round(float(imp), 6)
        rf_importances = sorted(
            [{"feature": f, "importance": v} for f, v in rf_map.items()],
            key=lambda x: x["importance"], reverse=True,
        )
    except Exception:
        pass

    # ── Mutual Information ────────────────────────────────────────────────────
    mi_scores: list[dict] = []
    mi_map: dict[str, float] = {}
    try:
        fn = mutual_info_classif if problem_type == "classification" else mutual_info_regression
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            mi = fn(X, y_aligned, random_state=42)
        for feat, score in zip(X.columns, mi):
            mi_map[str(feat)] = round(float(score), 6)
        mi_scores = sorted(
            [{"feature": f, "score": v} for f, v in mi_map.items()],
            key=lambda x: x["score"], reverse=True,
        )
    except Exception:
        pass

    # ── Pearson correlation with target ───────────────────────────────────────
    correlations: list[dict] = []
    corr_map: dict[str, float] = {}
    y_series = pd.Series(y_aligned, index=X.index)
    for col in X.columns:
        try:
            val = float(X[col].corr(y_series))
            if not np.isnan(val):
                corr_map[str(col)] = round(val, 4)
        except Exception:
            pass
    correlations = sorted(
        [{"feature": f, "correlation": v} for f, v in corr_map.items()],
        key=lambda x: abs(x["correlation"]), reverse=True,
    )

    # ── ANOVA F-score (F-classif / F-regression) ──────────────────────────────
    anova: list[dict] = []
    anova_map: dict[str, float] = {}
    try:
        fn_a = f_classif if problem_type == "classification" else f_regression
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            f_scores, _ = fn_a(X, y_aligned)
        for feat, fscore in zip(X.columns, f_scores):
            if not np.isnan(fscore) and not np.isinf(fscore):
                anova_map[str(feat)] = round(float(fscore), 4)
        anova = sorted(
            [{"feature": f, "f_score": v} for f, v in anova_map.items()],
            key=lambda x: x["f_score"], reverse=True,
        )
    except Exception:
        pass

    # ── Ranking maps (1 = best) ───────────────────────────────────────────────
    rf_rank   = {item["feature"]: i + 1 for i, item in enumerate(rf_importances)}
    mi_rank   = {item["feature"]: i + 1 for i, item in enumerate(mi_scores)}
    anova_rank = {item["feature"]: i + 1 for i, item in enumerate(anova)}
    uniform_thr = _uniform_threshold(n_features)

    # ── Feature metadata (combined) ───────────────────────────────────────────
    feature_meta: list[dict] = []
    for col in X.columns:
        col_str = str(col)
        meta = raw_meta.get(col_str, {"missing_pct": 0, "missing_count": 0, "unique_count": 0, "dtype": "float64"})
        rf_imp  = rf_map.get(col_str)
        mi_s    = mi_map.get(col_str)
        corr_v  = corr_map.get(col_str)
        anova_f = anova_map.get(col_str)

        # Combined rank (lower = more important)
        ranks = [r for r in [rf_rank.get(col_str), mi_rank.get(col_str), anova_rank.get(col_str)] if r is not None]
        combined_rank = round(sum(ranks) / len(ranks), 2) if ranks else float(n_features)

        # Recommendation
        low_rf      = (rf_imp  is not None) and rf_imp  < uniform_thr * 0.3
        low_mi      = (mi_s    is not None) and mi_s    < 0.005
        high_missing = meta["missing_pct"] > 30
        strong_rf   = (rf_imp  is not None) and rf_imp  > uniform_thr * 2.5

        if high_missing and low_rf and low_mi:
            recommendation = "drop"
        elif low_rf and low_mi:
            recommendation = "consider_drop"
        elif strong_rf:
            recommendation = "keep_strong"
        else:
            recommendation = "keep"

        feature_meta.append({
            "feature":       col_str,
            "rf_importance": rf_imp,
            "mi_score":      mi_s,
            "correlation":   corr_v,
            "anova_f":       anova_f,
            "combined_rank": combined_rank,
            "missing_pct":   meta["missing_pct"],
            "unique_count":  meta["unique_count"],
            "dtype":         meta["dtype"],
            "recommendation": recommendation,
        })

    feature_meta.sort(key=lambda x: x["combined_rank"])

    # ── Top features & drop candidates ────────────────────────────────────────
    top_source  = rf_importances if rf_importances else mi_scores
    top_features   = [item["feature"] for item in top_source[:5]]
    drop_candidates = [
        fm["feature"] for fm in feature_meta
        if fm["recommendation"] in ("drop", "consider_drop")
    ][:10]

    # ── Auto-generated warnings ───────────────────────────────────────────────
    warnings_list: list[dict] = []

    if problem_type == "classification" and class_distribution:
        pcts = [v["pct"] for v in class_distribution.values()]
        dominant = max(pcts) if pcts else 0
        if dominant > 80:
            warnings_list.append({
                "type": "class_imbalance",
                "message": f"Severe class imbalance: dominant class is {dominant:.1f}%. Use stratified splits or SMOTE.",
                "level": "danger",
            })
        elif dominant > 65:
            warnings_list.append({
                "type": "class_imbalance",
                "message": f"Moderate class imbalance: dominant class is {dominant:.1f}%. Consider oversampling.",
                "level": "warning",
            })

    high_missing_cols = [fm["feature"] for fm in feature_meta if fm["missing_pct"] > 20]
    if high_missing_cols:
        names = ", ".join(high_missing_cols[:3]) + ("…" if len(high_missing_cols) > 3 else "")
        warnings_list.append({
            "type": "high_missing",
            "message": f"{len(high_missing_cols)} feature(s) have >20% missing values: {names}. Impute before modelling.",
            "level": "warning",
        })

    if n_samples < 100:
        warnings_list.append({
            "type": "small_dataset",
            "message": f"Small dataset ({n_samples} rows). Importance estimates may be unstable.",
            "level": "warning",
        })

    if drop_candidates:
        warnings_list.append({
            "type": "low_importance",
            "message": f"{len(drop_candidates)} feature(s) score low on all methods and could be dropped: {', '.join(drop_candidates[:3])}{'…' if len(drop_candidates) > 3 else ''}",
            "level": "info",
        })

    return {
    "target": target,
    "problem_type": problem_type,
    "n_samples": n_samples,
    "n_features": n_features,

    "model_score": model_score,

    "cv_score_mean": None,
    "cv_score_std": None,

    "class_distribution": class_distribution,

    "importances": rf_importances[:20],

    # ADD THIS
    "permutation_importances": [],

    "mutual_info": mi_scores[:20],

    "correlations": correlations[:20],

    "anova": anova[:20],

    "shap_values": [],

    "feature_meta": feature_meta[:30],

    "stability": [],
    "interactions": [],

    "top_features": top_features,
    "drop_candidates": drop_candidates,

    "warnings": warnings_list,

    "redundant_groups": [],
    "leakage_suspects": [],
}
