import warnings as _warnings
import numpy as np
import pandas as pd
from scipy import stats as _stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

from .correlations import _cramers_v, _eta_squared, _point_biserial

def _uniform_threshold(n_features: int) -> float:
    """Importance level a feature would have if all were equally important."""
    return 1.0 / max(n_features, 1)


EXPENSIVE_METHOD_SAMPLE_CAP = 3000
INTERACTION_SAMPLE_CAP = 300
REDUNDANCY_TOP_K = 20
REDUNDANCY_THRESHOLD = 0.9


def _sample_rows(X: pd.DataFrame, y: np.ndarray, cap: int = EXPENSIVE_METHOD_SAMPLE_CAP, seed: int = 42):
    if len(X) <= cap:
        return X, y
    idx = np.random.RandomState(seed).choice(len(X), size=cap, replace=False)
    return X.iloc[idx], y[idx]


def _assoc(s1: pd.Series, s2: pd.Series, s1_num: bool, s2_num: bool) -> tuple[float | None, float | None]:
    """
    Statistically appropriate association strength between two series, routed by type
    so the result is never a Pearson correlation or F-test computed on an arbitrary
    integer label-encoding of unordered categories.

    Returns (association, f_score):
      numeric x numeric              -> Pearson r (signed),           f_regression-equivalent F
      numeric x categorical (2 lvl)   -> point-biserial r (signed),   same F formula, derived from r
      numeric x categorical (3+ lvl)  -> eta = sqrt(eta-squared) (unsigned), real one-way ANOVA F-statistic
      categorical x categorical       -> Cramer's V (unsigned),       None (chi-square isn't an F-statistic)
    """
    tmp = pd.concat([s1, s2], axis=1).dropna()
    if len(tmp) < 5:
        return None, None
    a, b = tmp.iloc[:, 0], tmp.iloc[:, 1]

    if s1_num and s2_num:
        try:
            r = float(a.corr(b))
        except Exception:
            return None, None
        if np.isnan(r):
            return None, None
        n = len(tmp)
        f = (r ** 2) / max(1 - r ** 2, 1e-12) * (n - 2) if n > 2 else None
        return round(r, 4), (round(float(f), 4) if f is not None else None)

    if s1_num != s2_num:
        num_s, cat_s = (a, b) if s1_num else (b, a)
        n_levels = cat_s.nunique()
        if n_levels < 2:
            return None, None
        if n_levels == 2:
            r, _ = _point_biserial(num_s, cat_s)
            if r is None:
                return None, None
            n = len(tmp)
            f = (r ** 2) / max(1 - r ** 2, 1e-12) * (n - 2) if n > 2 else None
            return r, (round(float(f), 4) if f is not None else None)
        f_stat = None
        try:
            groups = [g.values for _, g in num_s.groupby(cat_s)]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) >= 2:
                f_raw, _ = _stats.f_oneway(*groups)
                if not np.isnan(f_raw):
                    f_stat = float(f_raw)
        except Exception:
            f_stat = None
        eta_sq = _eta_squared(num_s, cat_s)
        eta = round(float(np.sqrt(eta_sq)), 4) if eta_sq is not None else None
        return eta, (round(f_stat, 4) if f_stat is not None else None)

    # categorical x categorical -> Cramer's V; no F-statistic equivalent exists for this pairing
    try:
        ct = pd.crosstab(a, b)
        v = _cramers_v(ct)
        return v, None
    except Exception:
        return None, None


def _group_redundant(pairs: list[tuple[str, str, float]]) -> list[dict]:
    """Cluster features connected by a high pairwise association into redundancy groups
    (connected components over the 'highly associated' graph)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    edge_strength: dict[tuple[str, str], float] = {}
    for a, b, strength in pairs:
        union(a, b)
        key = (a, b) if a < b else (b, a)
        edge_strength[key] = max(edge_strength.get(key, 0.0), strength)

    clusters: dict[str, list[str]] = {}
    for node in parent:
        clusters.setdefault(find(node), []).append(node)

    groups = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        member_set = set(members)
        max_corr = max(
            (s for (a, b), s in edge_strength.items() if a in member_set and b in member_set),
            default=0.0,
        )
        groups.append({"features": sorted(members), "max_correlation": round(float(max_corr), 4)})
    groups.sort(key=lambda g: g["max_correlation"], reverse=True)
    return groups


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
    target_is_numeric = problem_type == "regression"

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
    is_numeric_orig: dict[str, bool] = {
        c: bool(pd.api.types.is_numeric_dtype(df_clean[c])) for c in feature_cols
    }

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
    # Which encoded columns were originally categorical -> tells MI to use the discrete estimator
    discrete_mask = [not is_numeric_orig.get(str(c), True) for c in X.columns]

    # ── Random Forest (FAST - always compute) ─────────────────────────────────
    rf_importances: list[dict] = []
    rf_map: dict[str, float] = {}
    model_score = None
    clf = None
    computed = ["metadata"]

    if ("shap" in methods_set or "stability" in methods_set or
    "interactions" in methods_set or "permutation" in methods_set):
        methods_set.add("rf")

    cv_score_mean = None
    cv_score_std = None
    if "rf" in methods_set or "permutation" in methods_set or "shap" in methods_set:
        try:
            clf = (
                RandomForestClassifier(n_estimators=25, max_depth=10, random_state=42, n_jobs=-1, oob_score=True)
                if problem_type == "classification"
                else RandomForestRegressor(n_estimators=25, max_depth=10, random_state=42, n_jobs=-1, oob_score=True)
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

    # Real held-out k-fold CV, distinct from OOB: OOB still only ever sees this one
    # forest's out-of-bag rows, whereas this refits independently per fold.
    if "rf" in methods_set and clf is not None:
        try:
            from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
            if problem_type == "classification":
                min_class_count = int(pd.Series(y_aligned).value_counts().min())
                n_folds = max(2, min(5, min_class_count))
                splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            else:
                n_folds = max(2, min(5, n_samples // 20))
                splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_clf = clf.__class__(**clf.get_params())
            scores = cross_val_score(cv_clf, X, y_aligned, cv=splitter, n_jobs=-1)
            cv_score_mean = round(float(scores.mean()), 4)
            cv_score_std = round(float(scores.std()), 4)
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
                mi = fn(X, y_aligned, discrete_features=discrete_mask, random_state=42)
            for feat, score in zip(X.columns, mi):
                mi_map[str(feat)] = round(float(score), 6)
            mi_scores = sorted(
                [{"feature": f, "score": v} for f, v in mi_map.items()],
                key=lambda x: x["score"], reverse=True,
            )
            computed.append("mi")
        except Exception:
            pass

    # ── Correlation / ANOVA — type-routed association with target ────────────
    # Computed together: a categorical feature is never Pearson-correlated or
    # F-tested against an arbitrary label-encoding of its own categories.
    correlations: list[dict] = []
    anova: list[dict] = []
    corr_map: dict[str, float] = {}
    anova_map: dict[str, float] = {}
    if "correlation" in methods_set or "anova" in methods_set:
        for col in feature_cols:
            assoc_val, f_val = _assoc(df_clean[col], y_raw, is_numeric_orig[col], target_is_numeric)
            if assoc_val is not None:
                corr_map[col] = assoc_val
            if f_val is not None:
                anova_map[col] = f_val
        if "correlation" in methods_set:
            correlations = sorted(
                [{"feature": f, "correlation": v} for f, v in corr_map.items()],
                key=lambda x: abs(x["correlation"]), reverse=True,
            )
            computed.append("correlation")
        if "anova" in methods_set:
            anova = sorted(
                [{"feature": f, "f_score": v} for f, v in anova_map.items()],
                key=lambda x: x["f_score"], reverse=True,
            )
            computed.append("anova")

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

    # ── Leakage suspects — features with a near-deterministic relationship to the
    # target. This is a statistical heuristic (a "suspect", not a verdict): a feature
    # this strongly associated with the target is either a duplicate/derived column,
    # a direct proxy, or genuinely a near-perfect predictor — only the user has the
    # domain context to tell which. Only runs once a real target association has
    # actually been computed (gated on corr_map, not on the raw 'methods' string).
    leakage_suspects: list[dict] = []
    if corr_map:
        for fm in feature_meta:
            v = fm["correlation"]
            if v is None:
                continue
            av = abs(v)
            if av >= 0.97:
                leakage_suspects.append({
                    "feature": fm["feature"],
                    "reason": f"Near-perfect relationship with target (association={av:.3f}) — likely a duplicate, derived value, or direct proxy of the target.",
                    "severity": "high",
                })
            elif av >= 0.90:
                leakage_suspects.append({
                    "feature": fm["feature"],
                    "reason": f"Unusually strong relationship with target (association={av:.3f}) — verify this column would actually be available at prediction time.",
                    "severity": "medium",
                })
        computed.append("leakage")

    # ── Redundant feature groups — features highly associated with EACH OTHER
    # (not the target), clustered by connected components. Restricted to the
    # features most associated with the target to bound the O(k^2) pairwise cost
    # and to focus on redundancy that actually matters for modelling decisions.
    redundant_groups: list[dict] = []
    if corr_map and n_features >= 2:
        top_k_cols = sorted(corr_map.keys(), key=lambda c: abs(corr_map[c]), reverse=True)[:REDUNDANCY_TOP_K]
        edges: list[tuple[str, str, float]] = []
        for i, fa in enumerate(top_k_cols):
            for fb in top_k_cols[i + 1:]:
                strength, _ = _assoc(df_clean[fa], df_clean[fb], is_numeric_orig[fa], is_numeric_orig[fb])
                if strength is not None and abs(strength) >= REDUNDANCY_THRESHOLD:
                    edges.append((fa, fb, abs(strength)))
        redundant_groups = _group_redundant(edges)
        computed.append("redundancy")

    perm_importances: list[dict] = []
    if "permutation" in methods_set and clf is not None:
        try:
            X_perm, y_perm = _sample_rows(X, y_aligned)
            perm_result = permutation_importance(
                clf, X_perm, y_perm, n_repeats=5, random_state=42, n_jobs=-1
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
            X_shap, _ = _sample_rows(X, y_aligned)
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_shap)
            if isinstance(shap_vals, list):
                # Older shap: list of (n_samples, n_features) arrays, one per class
                shap_arr = np.mean([np.abs(s) for s in shap_vals], axis=0)
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                # Newer shap: single (n_samples, n_features, n_classes) array
                shap_arr = np.abs(shap_vals).mean(axis=2)
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
            X_stab, y_stab = _sample_rows(X, y_aligned)
            bootstrap_importances = {col: [] for col in X.columns}
            for _ in range(10):
                X_boot, y_boot = resample(X_stab, y_stab, random_state=None)
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

    # ── Feature interactions — real SHAP interaction values (Lundberg et al.), not
    # a stand-in formula. shap_interaction_values returns, per pair (i, j), how much
    # of the prediction is attributable to i and j acting together rather than
    # independently — the actual definition of "interaction" the UI claims to show.
    interactions: list[dict] = []
    if "interactions" in methods_set and n_features >= 3 and n_samples >= 100 and clf is not None:
        try:
            import shap
            X_int, _ = _sample_rows(X, y_aligned, cap=INTERACTION_SAMPLE_CAP)
            explainer = shap.TreeExplainer(clf)
            inter_vals = explainer.shap_interaction_values(X_int)
            if isinstance(inter_vals, list):
                inter_arr = np.mean([np.abs(iv) for iv in inter_vals], axis=0)
            elif isinstance(inter_vals, np.ndarray) and inter_vals.ndim == 4:
                inter_arr = np.abs(inter_vals).mean(axis=3)
            else:
                inter_arr = np.abs(inter_vals)
            mean_inter = inter_arr.mean(axis=0)  # (n_features, n_features)

            cols = list(X.columns)
            top_features_for_interaction = [f["feature"] for f in rf_importances[:8]]
            from itertools import combinations
            for feat_a, feat_b in combinations(top_features_for_interaction, 2):
                idx_a = cols.index(feat_a)
                idx_b = cols.index(feat_b)
                a_alone = float(mean_inter[idx_a, idx_a])
                b_alone = float(mean_inter[idx_b, idx_b])
                interaction_score = float(mean_inter[idx_a, idx_b])
                combined = a_alone + b_alone + interaction_score
                interactions.append({
                    "feature_a": feat_a,
                    "feature_b": feat_b,
                    "interaction_score": round(interaction_score, 6),
                    "a_alone": round(a_alone, 6),
                    "b_alone": round(b_alone, 6),
                    "combined": round(combined, 6),
                })
            computed.append("interactions")
        except Exception:
            pass

    perm_map = {item["feature"]: item["importance"] for item in perm_importances}
    shap_map = {item["feature"]: item["mean_abs_shap"] for item in shap_values_list}
    stability_map = {item["feature"]: item["cv"] for item in stability_list}
    for fm in feature_meta:
        fm["permutation_importance"] = perm_map.get(fm["feature"])
        fm["shap_value"] = shap_map.get(fm["feature"])
        fm["stability_score"] = stability_map.get(fm["feature"])

    return {
        "target": target,
        "problem_type": problem_type,
        "n_samples": n_samples,
        "n_features": n_features,
        "model_score": model_score,
        "cv_score_mean": cv_score_mean,
        "cv_score_std": cv_score_std,
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
        "redundant_groups": redundant_groups,
        "leakage_suspects": leakage_suspects,
        "computed_methods": computed,
    }
