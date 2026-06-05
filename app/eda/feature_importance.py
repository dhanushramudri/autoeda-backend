import warnings as _warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    f_classif, f_regression,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

def _uniform_threshold(n_features: int) -> float:
    """Importance level a feature would have if all were equally important."""
    return 1.0 / max(n_features, 1)


def run_feature_importance(df: pd.DataFrame, target: str, methods: list[str] | None = None) -> dict:
    """
    Run feature importance analysis with lazy loading support.
    
    Args:
        df: Input dataframe
        target: Target column name
        methods: List of methods to compute. If None, compute only ['rf', 'metadata'].
                 Available: ['rf', 'correlation', 'mi', 'anova', 'permutation', 'shap', 'stability', 'interactions']
    
    Returns:
        dict with computed results. 'computed_methods' field tracks which methods were computed.
    """
    # Default: only load RF and metadata on first call (fast)
    if methods is None:
        methods = ['rf', 'metadata']
    
    methods_set = set(methods)
    
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
        "computed_methods": [],
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
    X = X.astype(np.float32)

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

    # ── Random Forest (FAST - always compute) ─────────────────────────────────
    rf_importances: list[dict] = []
    rf_map: dict[str, float] = {}
    model_score = None
    clf = None
    computed = ["metadata"]
    
    if "rf" in methods_set or "permutation" in methods_set or "shap" in methods_set:
        try:
            clf = (
                RandomForestClassifier(n_estimators=25, max_depth=10, random_state=42, n_jobs=-1, oob_score=True)
                if problem_type == "classification"
                else RandomForestRegressor(n_estimators=25, random_state=42, n_jobs=-1, oob_score=True)
            )
            clf.fit(X, y_aligned)
            model_score = round(float(clf.oob_score_), 4)
            for feat, imp in zip(X.columns, clf.feature_importances_):
                rf_map[str(feat)] = round(float(imp), 6)
            rf_importances = sorted(
                [{"feature": f, "importance": v} for f, v in rf_map.items()],
                key=lambda x: x["importance"], reverse=True,
            )
            if "rf" in methods_set:
                computed.append("rf")
        except Exception:
            pass

    # ── Mutual Information (MEDIUM - optional) ────────────────────────────────
    mi_scores: list[dict] = []
    mi_map: dict[str, float] = {}
    if "mi" in methods_set:
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
            computed.append("mi")
        except Exception:
            pass

    # ── Pearson correlation with target ───────────────────────────────────────
    correlations: list[dict] = []
    corr_map: dict[str, float] = {}
    if "correlation" in methods_set:
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
        computed.append("correlation")

    # ── ANOVA F-score (MEDIUM) ────────────────────────────────────────────────
    anova: list[dict] = []
    anova_map: dict[str, float] = {}
    if "anova" in methods_set:
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
            computed.append("anova")
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

        # Recommendation based on available methods
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

    perm_importances: list[dict] = []
    if "permutation" in methods_set and clf is not None:
        try:
            perm_result = permutation_importance(
                clf, X, y_aligned, n_repeats=5, random_state=42, n_jobs=-1
            )
            for feat, imp, std in zip(X.columns, perm_result.importances_mean, perm_result.importances_std):
                perm_importances.append({
                    "feature": str(feat),
                    "importance": round(float(imp), 6),
                    "std": round(float(std), 6),
                })
            perm_importances.sort(key=lambda x: x["importance"], reverse=True)
            computed.append("permutation")
        except Exception:
            pass

    shap_values_list: list[dict] = []
    if "shap" in methods_set and clf is not None:
        try:
            import shap
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_arr = np.mean([np.abs(s) for s in shap_vals], axis=0)
            else:
                shap_arr = np.abs(shap_vals)
            mean_shap = shap_arr.mean(axis=0)
            for feat, val in zip(X.columns, mean_shap):
                shap_values_list.append({
                    "feature": str(feat),
                    "mean_abs_shap": round(float(val), 6),
                })
            shap_values_list.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
            computed.append("shap")
        except Exception:
            pass

    stability_list: list[dict] = []
    if "stability" in methods_set and n_samples >= 50 and clf is not None:
        try:
            from sklearn.utils import resample
            bootstrap_importances = {col: [] for col in X.columns}
            for _ in range(10):
                X_boot, y_boot = resample(X, y_aligned, random_state=None)
                clf_boot = clf.__class__(**{k: v for k, v in clf.get_params().items() if k != 'n_jobs'})
                clf_boot.fit(X_boot, y_boot)
                for feat, imp in zip(X.columns, clf_boot.feature_importances_):
                    bootstrap_importances[str(feat)].append(imp)
            
            for feat in X.columns:
                imps = bootstrap_importances[str(feat)]
                mean_imp = np.mean(imps)
                std_imp = np.std(imps)
                cv = std_imp / (mean_imp + 1e-10)
                stability_list.append({
                    "feature": str(feat),
                    "mean_importance": round(float(mean_imp), 6),
                    "std_importance": round(float(std_imp), 6),
                    "cv": round(float(cv), 4),
                    "rank_stability": round(float(1 - cv / (cv + 1)), 4),
                })
            computed.append("stability")
        except Exception:
            pass

    interactions: list[dict] = []
    if "interactions" in methods_set and n_features >= 3 and n_samples >= 100 and clf is not None:
        try:
            from itertools import combinations
            top_features_for_interaction = [f["feature"] for f in rf_importances[:8]]
            
            for feat_a, feat_b in combinations(top_features_for_interaction, 2):
                idx_a = list(X.columns).index(feat_a)
                idx_b = list(X.columns).index(feat_b)
                
                a_alone = rf_importances[idx_a]["importance"]
                b_alone = rf_importances[idx_b]["importance"]
                
                combined = a_alone + b_alone + abs(X[feat_a].corr(X[feat_b]) * 0.1)
                
                interaction_score = max(0, combined - a_alone - b_alone)
                
                interactions.append({
                    "feature_a": feat_a,
                    "feature_b": feat_b,
                    "interaction_score": round(float(interaction_score), 6),
                    "a_alone": round(float(a_alone), 6),
                    "b_alone": round(float(b_alone), 6),
                    "combined": round(float(combined), 6),
                })
            computed.append("interactions")
        except Exception:
            pass

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
        "permutation_importances": perm_importances[:20],
        "mutual_info": mi_scores[:20],
        "correlations": correlations[:20],
        "anova": anova[:20],
        "shap_values": shap_values_list[:20],
        "feature_meta": feature_meta[:30],
        "stability": stability_list,
        "interactions": interactions,
        "top_features": top_features,
        "drop_candidates": drop_candidates,
        "warnings": warnings_list,
        "redundant_groups": [],
        "leakage_suspects": [],
        "computed_methods": computed,
    }