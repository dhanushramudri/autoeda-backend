"""Quick Model (Experimental) — a zero-shot baseline via Google's TabFM.

TabFM needs no training loop: it reads a small "context" of labeled rows and
predicts on held-out rows in a single forward pass. That makes it a fast way to
get "here's a number to beat" without writing any model code — but it comes
with real constraints worth stating plainly:

  - Pretrained weights are under a non-commercial research license. This is
    for internal capability evaluation only — never wire this into a
    client-facing or production path.
  - Hard limit: max 10 classes for classification (a TabFM architectural
    constraint, not something we impose).
  - Context window: capped at MAX_CONTEXT_ROWS by design — TabFM's own
    guidance is that large tables should be sampled before inference, not fed
    in full.
  - "Fast" is relative to writing your own model, not to the rest of AutoEDA:
    loading the model + running inference took ~18 minutes end-to-end on a
    modest CPU box in testing. See app/tabfm_pool.py for why this runs in its
    own single-worker process pool.
"""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MAX_CONTEXT_ROWS = 100
MAX_TEST_ROWS = 50
MAX_CLASSES = 10
REGRESSION_UNIQUE_THRESHOLD = 20  # numeric target w/ more distinct values than this -> regression

_MODEL_CACHE: dict[str, object] = {}


def _get_cached_model(model_type: str):
    """Module-level cache — the dedicated single-worker pool guarantees this
    process is reused across calls, so the ~6.5GB weights load exactly once."""
    if model_type not in _MODEL_CACHE:
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        if model_type == "regression":
            _MODEL_CACHE[model_type] = tabfm_v1_0_0.load(model_type="regression")
        else:
            _MODEL_CACHE[model_type] = tabfm_v1_0_0.load()
    return _MODEL_CACHE[model_type]


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v if isinstance(v, (int, float, str, bool)) or v is None else str(v)


def determine_task_type(y: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(y):
        return "classification"
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > REGRESSION_UNIQUE_THRESHOLD:
        return "regression"
    return "classification"


def run_tabfm_quick_model(df: pd.DataFrame, target: str) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    work = df.dropna(subset=[target]).copy()
    feature_cols = [c for c in work.columns if c != target]
    if not feature_cols:
        raise ValueError("No feature columns available besides the target")
    if len(work) < 10:
        raise ValueError("Need at least 10 non-null target rows to build a train/test split")

    y = work[target]
    task_type = determine_task_type(y)

    if task_type == "classification":
        n_classes = int(y.nunique())
        if n_classes < 2:
            raise ValueError("Target needs at least 2 distinct classes for classification")
        if n_classes > MAX_CLASSES:
            raise ValueError(
                f"Target has {n_classes} classes — TabFM supports a maximum of {MAX_CLASSES} "
                "classes for classification. Pick a different target or bucket this one first."
            )

    X = work[feature_cols]
    n_train = min(MAX_CONTEXT_ROWS, max(int(len(work) * 0.8), 2))
    n_test = min(MAX_TEST_ROWS, len(work) - n_train)
    if n_test < 1:
        raise ValueError("Not enough rows to build a train/test split for a quick model")

    stratify = y if task_type == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=n_train, test_size=n_test, random_state=42, stratify=stratify,
        )
    except ValueError:
        # stratify fails if some class has fewer members than the split needs
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=n_train, test_size=n_test, random_state=42,
        )

    from tabfm import TabFMClassifier, TabFMRegressor

    t0 = time.time()
    if task_type == "classification":
        model = _get_cached_model("classification")
        estimator = TabFMClassifier(model=model)
    else:
        model = _get_cached_model("regression")
        estimator = TabFMRegressor(model=model)

    estimator.fit(X_train, y_train.to_numpy())
    preds = estimator.predict(X_test)
    elapsed = time.time() - t0

    result: dict = {
        "task_type": task_type,
        "target": target,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "elapsed_seconds": round(elapsed, 1),
    }

    y_test_arr = y_test.to_numpy()
    if task_type == "classification":
        from sklearn.metrics import accuracy_score, f1_score
        result["accuracy"] = round(float(accuracy_score(y_test_arr, preds)), 4)
        result["f1_macro"] = round(float(f1_score(y_test_arr, preds, average="macro")), 4)
        result["classes"] = sorted((str(c) for c in y.unique()))
    else:
        from sklearn.metrics import mean_squared_error, r2_score
        result["r2"] = round(float(r2_score(y_test_arr, preds)), 4)
        result["rmse"] = round(float(mean_squared_error(y_test_arr, preds) ** 0.5), 4)

    sample_n = min(10, len(X_test))
    result["sample_predictions"] = [
        {"actual": _jsonable(a), "predicted": _jsonable(p)}
        for a, p in zip(y_test_arr[:sample_n], preds[:sample_n])
    ]
    return result
