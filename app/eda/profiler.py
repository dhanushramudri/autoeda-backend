import numpy as np
import pandas as pd

MAX_FULL_ROWS = 500_000
SAMPLE_ROWS = 100_000


def classify_column(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 1:
            return "constant"
        unique_ratio = series.nunique() / max(len(series), 1)
        if unique_ratio > 0.95 and series.nunique() > 100:
            return "id_like"
        return "numeric"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        if series.nunique() <= 1:
            return "constant"
        unique_ratio = series.nunique() / max(len(series), 1)
        if unique_ratio > 0.95 and series.nunique() > 100:
            return "id_like"
        avg_len = series.dropna().astype(str).str.len().mean() if len(series.dropna()) > 0 else 0
        if avg_len > 50:
            return "text"
        return "categorical"
    return "categorical"


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) or np.isinf(f) else round(f, 6)
    except Exception:
        return None


def profile_column(series: pd.Series) -> dict:
    result = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "semantic_type": classify_column(series),
        "unique_count": int(series.nunique()),
        "unique_pct": round(series.nunique() / max(len(series), 1) * 100, 2),
        "missing_count": int(series.isnull().sum()),
        "missing_pct": round(series.isnull().sum() / max(len(series), 1) * 100, 2),
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "std": None,
        "skewness": None,
        "kurtosis": None,
        "top_values": [],
    }

    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) > 0:
            result["min"] = _safe_float(clean.min())
            result["max"] = _safe_float(clean.max())
            result["mean"] = _safe_float(clean.mean())
            result["median"] = _safe_float(clean.median())
            result["std"] = _safe_float(clean.std())
            result["skewness"] = _safe_float(clean.skew())
            result["kurtosis"] = _safe_float(clean.kurtosis())

    vc = series.value_counts().head(5)
    result["top_values"] = [
        {"value": str(v), "count": int(c), "pct": round(c / max(len(series), 1) * 100, 2)}
        for v, c in vc.items()
    ]
    return result


def run_profile(df: pd.DataFrame) -> dict:
    from concurrent.futures import ThreadPoolExecutor as _TPE

    sampled = len(df) > MAX_FULL_ROWS
    work_df = df.sample(SAMPLE_ROWS, random_state=42) if sampled else df

    # Duplicate check on a sample to keep it fast on large datasets
    dup_df = df.sample(min(50_000, len(df)), random_state=42) if len(df) > 50_000 else df
    dup_count = int(dup_df.duplicated().sum())

    # Profile columns concurrently
    cols = list(work_df.columns)
    with _TPE(max_workers=min(8, len(cols) or 1)) as pool:
        profiles = list(pool.map(lambda c: profile_column(work_df[c]), cols))

    return {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 3),
        "file_size_bytes": None,
        "duplicate_count": dup_count,
        "duplicate_pct": round(dup_count / max(len(dup_df), 1) * 100, 2),
        "sampled": sampled,
        "sample_size": SAMPLE_ROWS if sampled else int(len(df)),
        "columns": profiles,
    }
