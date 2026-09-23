"""The DS Sales Playbook's "mostly-used" model shortlist, generalized to
both classification and regression.

The playbook's own Churn Prediction template gives a concrete heuristic for
which algorithms to even try, keyed on dataset size and how categorical the
features are:

    Data size   Categorical   Suggested algorithms
    <25k rows   low           Logistic Regression, Naive Bayes
    >=25k rows  low           Random Forest, Gradient Boosting, SVM
    >=25k rows  high          CatBoost, SVM, Gradient Boosting

That table is written for a classification (churn) target. Each algorithm
key below maps to its standard regression counterpart so the same shortlist
logic works for a numeric target too — Naive Bayes has no real regression
analogue, so its slot becomes Ridge (the same "simple, regularized linear
baseline" role Naive Bayes plays for classification).
"""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COUNT_THRESHOLD = 5  # not specified by the playbook — our own call
LARGE_DATASET_THRESHOLD = 25_000  # from the playbook table
REGRESSION_UNIQUE_THRESHOLD = 20  # numeric target with more distinct values than this -> regression
SVM_MAX_TRAIN_ROWS = 10_000  # SVM training is O(n^2)-O(n^3) — cap so a laptop can finish
DEFAULT_MAX_ROWS = 1_000  # fast for testing the pipeline end-to-end; raise via MAX_TRAINING_ROWS once you want real accuracy

ALGORITHM_LABELS = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svm": "SVM",
    "catboost": "CatBoost",
}


def determine_problem_type(y: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(y):
        return "classification"
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > REGRESSION_UNIQUE_THRESHOLD:
        return "regression"
    return "classification"


def detect_categorical_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def shortlist_algorithms(n_rows: int, n_categorical: int) -> list[str]:
    """The playbook's table, collapsed to the two axes it actually varies on
    (feature *count* doesn't change the suggestion in their table — only
    dataset size and how categorical it is)."""
    if n_rows < LARGE_DATASET_THRESHOLD:
        return ["logistic_regression", "naive_bayes"]
    if n_categorical > CATEGORICAL_COUNT_THRESHOLD:
        return ["catboost", "svm", "gradient_boosting"]
    return ["random_forest", "gradient_boosting", "svm"]


def _build_estimator(algorithm: str, problem_type: str, cat_feature_idx: list[int]):
    if algorithm == "logistic_regression":
        if problem_type == "classification":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000), {"max_iter": 1000}
        from sklearn.linear_model import LinearRegression
        return LinearRegression(), {}

    if algorithm == "naive_bayes":
        if problem_type == "classification":
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB(), {}
        from sklearn.linear_model import Ridge
        return Ridge(), {}  # closest simple-linear-baseline analogue; NB has no regression form

    if algorithm == "random_forest":
        params = {"n_estimators": 200, "max_depth": None, "random_state": 42, "n_jobs": -1}
        if problem_type == "classification":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params), params
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params), params

    if algorithm == "gradient_boosting":
        params = {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}
        if problem_type == "classification":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**params), params
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**params), params

    if algorithm == "svm":
        params = {"kernel": "rbf", "C": 1.0}
        if problem_type == "classification":
            from sklearn.svm import SVC
            return SVC(probability=False, **params), params
        from sklearn.svm import SVR
        return SVR(**params), params

    if algorithm == "catboost":
        params = {"iterations": 300, "depth": 6, "learning_rate": 0.1, "verbose": False, "random_state": 42}
        if problem_type == "classification":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(cat_features=cat_feature_idx or None, **params), params
        from catboost import CatBoostRegressor
        return CatBoostRegressor(cat_features=cat_feature_idx or None, **params), params

    raise ValueError(f"Unknown algorithm: {algorithm}")


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """get_dummies can produce two columns with the same name — e.g. a
    categorical column containing the literal string "nan" alongside real
    missing values collides with the dummy_na=True indicator column it
    also generates as "<col>_nan". A DataFrame with duplicate column
    labels breaks .reindex() outright, so rename any repeats before that
    ever runs, regardless of which raw column caused it."""
    if not df.columns.duplicated().any():
        return df
    seen: dict[str, int] = {}
    new_cols = []
    for col in df.columns:
        seen[col] = seen.get(col, 0) + 1
        new_cols.append(col if seen[col] == 1 else f"{col}__dup{seen[col]}")
    df = df.copy()
    df.columns = new_cols
    return df


def _one_hot_align(X_train: pd.DataFrame, X_test: pd.DataFrame, categorical_cols: list[str]):
    train_enc = _dedupe_columns(pd.get_dummies(X_train, columns=categorical_cols, dummy_na=True))
    test_enc = _dedupe_columns(pd.get_dummies(X_test, columns=categorical_cols, dummy_na=True))
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)
    return train_enc, test_enc


def _feature_importance(estimator, feature_names: list[str], algorithm: str) -> dict:
    try:
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            values = np.abs(coef[0]) if getattr(coef, "ndim", 1) > 1 else np.abs(coef)
        else:
            return {}
        pairs = sorted(zip(feature_names, values), key=lambda p: -abs(p[1]))[:15]
        return {name: round(float(v), 4) for name, v in pairs}
    except Exception:
        return {}


def train_one(
    df: pd.DataFrame, target: str, algorithm: str, problem_type: str, categorical_cols: list[str],
    max_rows: int = DEFAULT_MAX_ROWS,
):
    """Trains one shortlisted algorithm on a 70:30 split and returns
    (estimator, params, metrics, feature_importance, training_seconds) —
    or raises, which the caller reports as a failed run rather than
    aborting the whole experiment."""
    feature_cols = [c for c in df.columns if c != target]
    work = df.dropna(subset=[target]).copy()

    if max_rows and len(work) > max_rows:
        # A laptop CPU training on the full dataset is exactly what makes
        # this feel "stuck" — a stratified sample keeps the class balance
        # (for classification) representative while keeping every
        # algorithm, not just SVM, fast enough to actually finish.
        if problem_type == "classification":
            try:
                work, _ = train_test_split(work, train_size=max_rows, random_state=42, stratify=work[target])
            except ValueError:
                work = work.sample(n=max_rows, random_state=42)
        else:
            work = work.sample(n=max_rows, random_state=42)

    X = work[feature_cols]
    y = work[target]

    stratify = y if problem_type == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if algorithm == "svm" and len(X_train) > SVM_MAX_TRAIN_ROWS:
        # SVM's training cost blows up on a laptop well before the other
        # shortlisted algorithms notice the extra rows — subsample just this
        # one, not the whole experiment's comparison.
        X_train = X_train.sample(n=SVM_MAX_TRAIN_ROWS, random_state=42)
        y_train = y_train.loc[X_train.index]

    t0 = time.time()

    if algorithm == "catboost":
        for c in categorical_cols:
            X_train[c] = X_train[c].astype(str).fillna("__missing__")
            X_test[c] = X_test[c].astype(str).fillna("__missing__")
        for c in [c for c in feature_cols if c not in categorical_cols]:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(X_train[c].median() if pd.api.types.is_numeric_dtype(X_train[c]) else 0)
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(X_train[c].median() if pd.api.types.is_numeric_dtype(X_train[c]) else 0)
        cat_idx = [feature_cols.index(c) for c in categorical_cols]
        estimator, params = _build_estimator(algorithm, problem_type, cat_idx)
        estimator.fit(X_train[feature_cols], y_train)
        preds = estimator.predict(X_test[feature_cols])
        feature_names = feature_cols
    else:
        # Must be computed from the ORIGINAL feature list, before encoding —
        # get_dummies renames categorical columns to "col_value" dummies, so
        # comparing against X_train_enc.columns after the fact wrongly
        # classifies every dummy column as "numeric" too. That silently
        # sends 0/1 dummy columns into StandardScaler below, and scaling a
        # near-constant column (common with rare categories in a small
        # sample) divides by ~zero variance and produces NaN/inf.
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]
        X_train_enc, X_test_enc = _one_hot_align(X_train, X_test, categorical_cols)
        for c in numeric_cols:
            median = pd.to_numeric(X_train_enc[c], errors="coerce").median()
            median = median if pd.notna(median) else 0  # guard an all-NaN column in this sample
            X_train_enc[c] = pd.to_numeric(X_train_enc[c], errors="coerce").fillna(median)
            X_test_enc[c] = pd.to_numeric(X_test_enc[c], errors="coerce").fillna(median)

        if algorithm in ("logistic_regression", "naive_bayes", "svm"):
            scaler = StandardScaler()
            X_train_enc[numeric_cols] = scaler.fit_transform(X_train_enc[numeric_cols])
            X_test_enc[numeric_cols] = scaler.transform(X_test_enc[numeric_cols])

        estimator, params = _build_estimator(algorithm, problem_type, [])
        estimator.fit(X_train_enc, y_train)
        preds = estimator.predict(X_test_enc)
        feature_names = list(X_train_enc.columns)

    training_seconds = round(time.time() - t0, 2)

    metrics: dict = {}
    if problem_type == "classification":
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        metrics["accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
        metrics["precision"] = round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4)
        metrics["f1"] = round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4)
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        metrics["r2"] = round(float(r2_score(y_test, preds)), 4)
        metrics["rmse"] = round(float(mean_squared_error(y_test, preds) ** 0.5), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_test, preds)), 4)

    feature_importance = _feature_importance(estimator, feature_names, algorithm)
    return estimator, params, metrics, feature_importance, training_seconds
