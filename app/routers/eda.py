import json
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.orm import defer as sa_defer

from ..auth import get_current_active_user
from ..cache import get_cached_result, store_result
from ..dataset_access import assert_dataset_access
from ..database import get_db
from ..models.dataset import Dataset
from ..models.user import User
from ..process_pool import AnalysisCrashed, AnalysisTimeout, run_isolated
from ..schemas.eda import (
    CorrelationResult,
    DistributionResult,
    FeatureImportanceResult,
    MissingResult,
    OutlierResult,
    ProfileResult,
    QualityScore,
    SmartCleanResult,
    TextResult,
    TimeSeriesResult,
)

router = APIRouter(prefix="/datasets", tags=["eda"])

# Shared bounds for the "row_limit" query param exposed on every analysis endpoint —
# lets the UI choose how many rows to pull from connector-backed sources (Databricks,
# SQL DBs, APIs, cloud storage) instead of the previous hardcoded 50K cap.
ROW_LIMIT_MIN = 1_000
ROW_LIMIT_MAX = 2_000_000


def _run_isolated(fn, *args, **kwargs):
    """Run a heavy EDA computation in the shared process pool.

    Keeps CPU/memory-heavy work out of the request thread so it can't starve
    other users' requests, and so a crash/OOM in one analysis only fails this
    one request instead of taking down the whole API process.
    """
    try:
        return run_isolated(fn, *args, **kwargs)
    except AnalysisTimeout as e:
        raise HTTPException(status_code=504, detail=str(e))
    except AnalysisCrashed as e:
        raise HTTPException(status_code=503, detail=str(e))


def _get_authorized_dataset(dataset_id: int, current_user: User, db: Session) -> Dataset:
    # file_data is the entire raw uploaded file stored as a LargeBinary
    # column — a plain query eagerly loads it into memory even when it's
    # never read (the common case: _load_df's fast path reads from
    # ds.file_path on local disk and never touches ds.file_data at all).
    # Deferring is transparent: if _load_df's fallback branch DOES need it,
    # SQLAlchemy lazy-loads it on first access, same result either way —
    # this only avoids paying for it upfront when it's not needed. Same
    # pattern already used deliberately in routers/warehouse.py. This
    # dataset+the row's DataFrame can both stay alive for a long-running
    # Auto EDA run's entire duration (see auto_eda_orchestrator.py's
    # `loaded` dict), so avoiding a redundant multi-megabyte blob per
    # dataset matters more here than it looks like a one-line change.
    ds = db.query(Dataset).options(sa_defer(Dataset.file_data)).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_access(ds, current_user, db)
    return ds


def _load_df(ds: Dataset, row_limit: Optional[int] = None):
    """row_limit only affects connector-backed sources (Databricks, SQL DBs, APIs,
    cloud storage) — file datasets always load what was actually uploaded."""
    import os
    import pandas as pd
    from ..connectors.file_connector import FileConnector, load_from_bytes
    from ..connectors.db_connector import DBConnector
    from ..connectors.api_connector import RESTAPIConnector
    from ..connectors.cloud_connector import CloudConnector

    config = json.loads(ds.source_config or "{}")

    if ds.source_type == "file":
        filename = os.path.basename(ds.file_path or "") if ds.file_path else ""
        # Use local disk if available (fast), fall back to DB bytes
        if ds.file_path and os.path.exists(ds.file_path):
            config["file_path"] = ds.file_path
            return FileConnector().load_data(config)
        elif ds.file_data:
            return load_from_bytes(ds.file_data, filename, config)
        else:
            raise FileNotFoundError(f"No file data available for dataset {ds.id}")
    elif ds.source_type in ("postgresql", "mysql", "sqlite", "mssql"):
        config["db_type"] = ds.source_type
        return DBConnector().load_data(config, limit=row_limit)
    elif ds.source_type == "mongodb":
        config["db_type"] = "mongodb"
        return DBConnector().load_data(config, limit=row_limit)
    elif ds.source_type == "databricks":
        from ..databricks_loader import load_databricks_dataframe
        return load_databricks_dataframe(ds, config, limit=row_limit)
    elif ds.source_type == "rest_api":
        return RESTAPIConnector().load_data(config, limit=row_limit)
    elif ds.source_type in ("s3", "azure", "gcs"):
        config["cloud_type"] = ds.source_type
        return CloudConnector().load_data(config, limit=row_limit)
    else:
        raise ValueError(f"Unsupported source_type: {ds.source_type}")


def _try_databricks_profile_pushdown(ds: Dataset) -> Optional[dict]:
    """Full-table profile via Spark SQL aggregates — no row limit. Returns None on
    any failure so the caller can fall back to the sampled pandas path."""
    if ds.source_type != "databricks":
        return None
    try:
        import json as _json
        from ..databricks_loader import get_databricks_pushdown_context
        from ..connectors.db_connector import DBConnector

        config = _json.loads(ds.source_config or "{}")
        cfg, source_sql = get_databricks_pushdown_context(ds, config)
        return DBConnector().compute_databricks_profile_full(cfg, source_sql)
    except Exception:
        return None


@router.get("/{dataset_id}/profile", response_model=ProfileResult)
def get_profile(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "profile"}
    cached = get_cached_result(db, dataset_id, "profile", cache_key, ds.content_hash or "")
    if cached:
        return ProfileResult(**cached)

    try:
        pushdown_result = _try_databricks_profile_pushdown(ds)
        if pushdown_result is not None:
            store_result(db, dataset_id, "profile", cache_key, pushdown_result, ds.content_hash or "")
            return ProfileResult(**pushdown_result)

        df = _load_df(ds)
        from ..eda.profiler import run_profile
        result = _run_isolated(run_profile, df)
        if ds.file_path:
            import os
            result["file_size_bytes"] = os.path.getsize(ds.file_path)
        store_result(db, dataset_id, "profile", cache_key, result, ds.content_hash or "")
        return ProfileResult(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




def _missing_result_from_profile(profile: dict) -> dict:
    """Derive the Missing Values view from an already-computed full-table profile —
    avoids a second Databricks round-trip. Co-occurrence correlation and MCAR
    detection need row-level data so they're left empty in pushdown mode."""
    total_rows = profile["total_rows"]
    total_columns = profile["total_columns"]
    cols_with_missing = [
        {"name": c["name"], "count": c["missing_count"], "pct": c["missing_pct"]}
        for c in profile["columns"] if c["missing_count"] > 0
    ]
    total_missing = sum(c["count"] for c in cols_with_missing)
    total_cells = total_rows * max(total_columns, 1)

    suggestions: dict[str, str] = {}
    for c in profile["columns"]:
        if c["missing_count"] == 0:
            continue
        if c["semantic_type"] == "numeric":
            skew = c.get("skewness")
            suggestions[c["name"]] = "median (skewed)" if (skew is not None and abs(skew) > 1) else "mean (symmetric)"
        else:
            suggestions[c["name"]] = "mode (categorical)"

    return {
        "columns": cols_with_missing,
        "total_missing": total_missing,
        "missing_pct": round(total_missing / max(total_cells, 1) * 100, 2),
        "correlation_matrix": {},
        "mcar_indicators": {},
        "imputation_suggestions": suggestions,
    }


@router.get("/{dataset_id}/missing", response_model=MissingResult)
def get_missing(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "missing"}
    cached = get_cached_result(db, dataset_id, "missing", cache_key, ds.content_hash or "")
    if cached:
        return MissingResult(**cached)

    try:
        pushdown_profile = _try_databricks_profile_pushdown(ds)
        if pushdown_profile is not None:
            result = _missing_result_from_profile(pushdown_profile)
            store_result(db, dataset_id, "missing", cache_key, result, ds.content_hash or "")
            return MissingResult(**result)

        df = _load_df(ds)
        from ..eda.missing import run_missing_analysis
        result = _run_isolated(run_missing_analysis, df)
        store_result(db, dataset_id, "missing", cache_key, result, ds.content_hash or "")
        return MissingResult(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/smart-clean", response_model=SmartCleanResult)
def get_smart_clean(
    dataset_id: int,
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Auto-detect cleanable issues (inconsistent casing, whitespace, mixed date
    formats) and return one-click-applyable Transform Studio operations for each."""
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "smart_clean", "row_limit": row_limit}
    cached = get_cached_result(db, dataset_id, "smart_clean", cache_key, ds.content_hash or "")
    if cached:
        return SmartCleanResult(**cached)

    try:
        df = _load_df(ds, row_limit)
        from ..eda.smart_clean import detect_smart_clean_issues
        suggestions = _run_isolated(detect_smart_clean_issues, df)
        result = {"total_rows": len(df), "suggestions": suggestions}
        store_result(db, dataset_id, "smart_clean", cache_key, result, ds.content_hash or "")
        return SmartCleanResult(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/distributions", response_model=DistributionResult)
def get_distributions(
    dataset_id: int,
    column: str,
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "distributions", "column": column, "row_limit": row_limit}
    cached = get_cached_result(db, dataset_id, "distributions", cache_key, ds.content_hash or "")
    if cached:
        return DistributionResult(**cached)

    try:
        df = _load_df(ds, row_limit)
        from ..eda.distributions import run_distribution
        result = _run_isolated(run_distribution, df, column)
        store_result(db, dataset_id, "distributions", cache_key, result, ds.content_hash or "")
        return DistributionResult(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/correlations", response_model=CorrelationResult, response_model_exclude_unset=True)
def get_correlations(
    dataset_id: int,
    method: str = "pearson",
    methods: str | None = None,
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Lazy-loading correlations — mirrors /feature-importance and /timeseries.

    Query params:
    - methods: Optional comma-separated list of groups to compute.
      Default (None): just 'numeric' (numeric×numeric matrix + VIF — fast).
      Available: numeric, categorical, mixed.
    """
    ds = _get_authorized_dataset(dataset_id, current_user, db)

    methods_list = [m.strip().lower() for m in methods.split(",")] if methods else ["numeric"]

    cache_key = {"type": "correlations", "method": method, "methods": sorted(methods_list), "row_limit": row_limit, "v": 3}
    cached = get_cached_result(db, dataset_id, "correlations", cache_key, ds.content_hash or "")
    if cached:
        return CorrelationResult.model_validate(cached)

    try:
        df = _load_df(ds, row_limit)
        from ..eda.correlations import run_correlations
        result = _run_isolated(run_correlations, df, method, methods=methods_list)
        store_result(db, dataset_id, "correlations", cache_key, result, ds.content_hash or "")
        return CorrelationResult.model_validate(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/outliers", response_model=OutlierResult)
def get_outliers(
    dataset_id: int,
    method: str = "iqr",
    column: Optional[str] = None,
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "outliers", "method": method, "column": column, "row_limit": row_limit}
    cached = get_cached_result(db, dataset_id, "outliers", cache_key, ds.content_hash or "")
    if cached:
        return OutlierResult(**cached)

    try:
        df = _load_df(ds, row_limit)
        from ..eda.outliers import run_outlier_detection
        result = _run_isolated(run_outlier_detection, df, method, column)
        store_result(db, dataset_id, "outliers", cache_key, result, ds.content_hash or "")
        return OutlierResult(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# This replaces the existing get_feature_importance endpoint in routes/eda.py

@router.get("/{dataset_id}/feature-importance", response_model=FeatureImportanceResult)
def get_feature_importance(
    dataset_id: int,
    target: str,
    methods: str | None = None,  # NEW: comma-separated list of methods to compute
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get feature importance with lazy loading support.

    Query params:
    - target: Column name to analyze
    - methods: Optional comma-separated list of methods to compute.
      Default (None): Only compute 'rf' and 'metadata' for fast initial load.
      Available: rf, correlation, mi, anova, permutation, shap, stability, interactions
      Example: ?target=price&methods=rf,mi,correlation,permutation
    """
    ds = _get_authorized_dataset(dataset_id, current_user, db)

    if methods:
        methods_list = [m.strip().lower() for m in methods.split(",")]
    else:
        methods_list = ["rf", "metadata"]

    cache_key = {
        "type": "feature_importance",
        "target": target,
        "methods": sorted(methods_list),  # Normalize order for cache consistency
        "row_limit": row_limit,
    }

    cached = get_cached_result(
        db, dataset_id, "feature_importance",
        cache_key, ds.content_hash or ""
    )

    if cached:
        return FeatureImportanceResult(**cached)

    try:
        df = _load_df(ds, row_limit)
        from ..eda.feature_importance import run_feature_importance

        result = _run_isolated(run_feature_importance, df, target, methods=methods_list, timeout=240)
        
        store_result(
            db, dataset_id, "feature_importance",
            cache_key, result, ds.content_hash or ""
        )
        return FeatureImportanceResult(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# NEW ENDPOINT: Get available methods and their estimated compute time
@router.get("/{dataset_id}/feature-importance-methods")
def get_feature_importance_methods(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get list of available importance methods with estimated compute times.
    Useful for building UI progress indicators.
    """
    return {
        "methods": [
            {
                "id": "rf",
                "name": "Random Forest",
                "description": "Impurity-based importance (fastest)",
                "estimated_time_seconds": 10,
                "default": True,
            },
            {
                "id": "correlation",
                "name": "Correlation",
                "description": "Pearson correlation with target",
                "estimated_time_seconds": 5,
                "default": False,
            },
            {
                "id": "mi",
                "name": "Mutual Information",
                "description": "Non-linear statistical dependency",
                "estimated_time_seconds": 20,
                "default": False,
            },
            {
                "id": "anova",
                "name": "ANOVA F-score",
                "description": "F-statistic for feature groups",
                "estimated_time_seconds": 15,
                "default": False,
            },
            {
                "id": "permutation",
                "name": "Permutation Importance",
                "description": "Feature shuffle impact on performance",
                "estimated_time_seconds": 60,
                "default": False,
            },
            {
                "id": "shap",
                "name": "SHAP Values",
                "description": "Shapley value attribution (slowest)",
                "estimated_time_seconds": 120,
                "default": False,
            },
            {
                "id": "stability",
                "name": "Stability Analysis",
                "description": "Bootstrap importance stability",
                "estimated_time_seconds": 45,
                "default": False,
            },
            {
                "id": "interactions",
                "name": "Feature Interactions",
                "description": "Synergistic feature pairs",
                "estimated_time_seconds": 30,
                "default": False,
            },
        ]
    }

@router.get("/{dataset_id}/timeseries", response_model=TimeSeriesResult, response_model_exclude_unset=True)
def get_timeseries(
    dataset_id: int,
    time_col: str,
    value_col: str,
    methods: str | None = None,  # comma-separated; default: "overview" for a fast first render
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Lazy-loading time series analysis — mirrors /feature-importance's pattern.

    Query params:
    - methods: Optional comma-separated list of groups to compute.
      Default (None): just 'overview' (chart + data quality + stats — fast).
      Available: overview, stationarity, decomposition, acf_pacf, anomalies,
      change_points, granger, readiness.
    """
    ds = _get_authorized_dataset(dataset_id, current_user, db)

    methods_list = [m.strip().lower() for m in methods.split(",")] if methods else ["overview"]

    cache_key = {
        "type": "timeseries", "time_col": time_col, "value_col": value_col,
        "methods": sorted(methods_list), "row_limit": row_limit,
    }
    cached = get_cached_result(db, dataset_id, "timeseries", cache_key, ds.content_hash or "")
    if cached:
        return TimeSeriesResult(**cached)

    try:
        df = _load_df(ds, row_limit)
        from ..eda.timeseries import run_timeseries
        result = _run_isolated(run_timeseries, df, time_col, value_col, methods=methods_list)
        store_result(db, dataset_id, "timeseries", cache_key, result, ds.content_hash or "")
        return TimeSeriesResult(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/text", response_model=TextResult)
def get_text_analysis(
    dataset_id: int,
    column: str,
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "text", "column": column, "row_limit": row_limit}
    cached = get_cached_result(db, dataset_id, "text", cache_key, ds.content_hash or "")
    if cached:
        return TextResult(**cached)

    try:
        df = _load_df(ds, row_limit)
        from ..eda.text_analysis import run_text_analysis
        result = _run_isolated(run_text_analysis, df, column)
        store_result(db, dataset_id, "text", cache_key, result, ds.content_hash or "")
        return TextResult(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/quality-score", response_model=QualityScore)
def get_quality_score(
    dataset_id: int,
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)

    try:
        df = _load_df(ds, row_limit)
        from ..eda.quality_score import run_quality_score
        result = _run_isolated(run_quality_score, df)
        return QualityScore(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/analysis")
def get_analysis(
    dataset_id: int,
    force_refresh: bool = False,
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Full EDA analysis — all chart data in one call. Cached per dataset version."""
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "analysis", "row_limit": row_limit}

    if force_refresh:
        from ..models.dataset import EDAResult
        db.query(EDAResult).filter(
            EDAResult.dataset_id == dataset_id,
            EDAResult.analysis_type == "analysis",
        ).delete()
        db.commit()

    cached = get_cached_result(db, dataset_id, "analysis", cache_key, ds.content_hash or "")
    if cached:
        # Validate cache has scatter_pairs — if not, it's stale from the old broken route
        multi = cached.get("multi_column", {})
        if "scatter_pairs" not in multi:
            cached = None  # force recompute

    if cached:
        return cached

    try:
        df = _load_df(ds, row_limit)
        from ..eda.analysis import run_full_analysis
        result = _run_isolated(run_full_analysis, df, timeout=240)
        store_result(db, dataset_id, "analysis", cache_key, result, ds.content_hash or "")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/analysis/column/{col_name}")
def get_analysis_column(
    dataset_id: int,
    col_name: str,
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Chart data for a single column (lazy-loading support)."""
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    try:
        df = _load_df(ds, row_limit)
        from ..eda.analysis import (
            _histogram_kde, _box_stats, _violin_kde, _qq_plot, _ecdf,
            _normality_test, _bar_chart, _pie_data, _pareto_data,
            _safe, _maybe_sample,
        )
        from ..eda.profiler import classify_column

        df_sample, sampled = _maybe_sample(df)
        if col_name not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{col_name}' not found")

        col_type = classify_column(df[col_name])
        s = df_sample[col_name]

        if col_type == "numeric":
            result = {
                "col_name": col_name, "col_type": col_type, "sampled": sampled,
                "histogram_kde": _histogram_kde(s),
                "box": _box_stats(s),
                "violin": _violin_kde(s),
                "qq": _qq_plot(s),
                "ecdf": _ecdf(s),
                "normality": _normality_test(s.dropna()),
                "skewness": _safe(float(s.skew())) if s.dropna().shape[0] >= 3 else None,
                "kurtosis": _safe(float(s.kurtosis())) if s.dropna().shape[0] >= 4 else None,
            }
        elif col_type in ("categorical", "boolean"):
            sd = s.dropna()
            result = {
                "col_name": col_name, "col_type": col_type, "sampled": sampled,
                "bar": _bar_chart(sd),
                "pie": _pie_data(sd),
                "pareto": _pareto_data(sd),
            }
        else:
            result = {"col_name": col_name, "col_type": col_type, "sampled": sampled}

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/bivariate")
def get_bivariate(
    dataset_id: int,
    col1: str,
    col2: str,
    btype: str = "num_num",
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """On-demand bivariate analysis for a specific column pair."""
    if btype not in ("num_num", "cat_cat", "num_cat"):
        raise HTTPException(status_code=400, detail="btype must be num_num, cat_cat, or num_cat")
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "bivariate", "col1": col1, "col2": col2, "btype": btype, "row_limit": row_limit}
    cached = get_cached_result(db, dataset_id, "bivariate", cache_key, ds.content_hash or "")
    if cached:
        return cached
    try:
        df = _load_df(ds, row_limit)
        from ..eda.analysis import (
            compute_bivariate_num_num, compute_bivariate_cat_cat, compute_bivariate_num_cat
        )
        if btype == "num_num":
            result = _run_isolated(compute_bivariate_num_num, df, col1, col2)
        elif btype == "cat_cat":
            result = _run_isolated(compute_bivariate_cat_cat, df, col1, col2)
        else:
            result = _run_isolated(compute_bivariate_num_cat, df, col1, col2)
        store_result(db, dataset_id, "bivariate", cache_key, result, ds.content_hash or "")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/pca")
def get_pca(
    dataset_id: int,
    n_components: int = 2,
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """PCA on all numeric columns."""
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "pca", "n_components": n_components, "row_limit": row_limit}
    cached = get_cached_result(db, dataset_id, "pca", cache_key, ds.content_hash or "")
    if cached:
        return cached
    try:
        df = _load_df(ds, row_limit)
        from ..eda.analysis import compute_pca
        from ..eda.profiler import classify_column
        num_cols = [c for c in df.columns if classify_column(df[c]) == "numeric"]
        result = _run_isolated(compute_pca, df, num_cols, n_components)
        store_result(db, dataset_id, "pca", cache_key, result, ds.content_hash or "")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/scatter3d")
def get_scatter3d(
    dataset_id: int,
    x: str,
    y: str,
    z: str,
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """3D scatter for three numeric columns."""
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "scatter3d", "x": x, "y": y, "z": z, "row_limit": row_limit}
    cached = get_cached_result(db, dataset_id, "scatter3d", cache_key, ds.content_hash or "")
    if cached:
        return cached
    try:
        df = _load_df(ds, row_limit)
        from ..eda.analysis import compute_scatter3d
        result = _run_isolated(compute_scatter3d, df, x, y, z)
        store_result(db, dataset_id, "scatter3d", cache_key, result, ds.content_hash or "")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/transform/preview")
def transform_preview(
    dataset_id: int,
    operations: dict,
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Preview transformations without saving."""
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    try:
        df = _load_df(ds, row_limit)
        
        # Apply transformations in memory
        ops_list = operations.get("operations", [])
        for op in ops_list:
            op_type = op.get("type")
            column = op.get("column")
            
            if op_type == "drop" and column:
                if column in df.columns:
                    df = df.drop(columns=[column])
            elif op_type == "rename" and column:
                new_name = op.get("new_name", column)
                if column in df.columns:
                    df = df.rename(columns={column: new_name})
            elif op_type == "fill_missing" and column:
                method = op.get("method", "mean")
                if column in df.columns:
                    if method == "mean" and df[column].dtype in (float, int):
                        df[column] = df[column].fillna(df[column].mean())
                    elif method == "median" and df[column].dtype in (float, int):
                        df[column] = df[column].fillna(df[column].median())
                    elif method == "mode":
                        df[column] = df[column].fillna(df[column].mode()[0] if len(df[column].mode()) > 0 else None)
                    elif method == "custom":
                        fill_value = op.get("fill_value")
                        df[column] = df[column].fillna(fill_value)
            elif op_type == "cast_type" and column:
                new_dtype = op.get("dtype", "object")
                if column in df.columns:
                    try:
                        df[column] = df[column].astype(new_dtype)
                    except Exception:
                        pass
        
        # Return first 50 rows
        preview_data = df.head(50).to_dict(orient="records")
        return {"success": True, "rows": preview_data, "row_count": len(df)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/transform/apply")
def transform_apply(
    dataset_id: int,
    operations: dict,
    row_limit: Optional[int] = Query(None, ge=ROW_LIMIT_MIN, le=ROW_LIMIT_MAX),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Apply transformations and save as a new dataset."""
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    try:
        df = _load_df(ds, row_limit)
        
        # Apply transformations
        ops_list = operations.get("operations", [])
        for op in ops_list:
            op_type = op.get("type")
            column = op.get("column")
            
            if op_type == "drop" and column:
                if column in df.columns:
                    df = df.drop(columns=[column])
            elif op_type == "rename" and column:
                new_name = op.get("new_name", column)
                if column in df.columns:
                    df = df.rename(columns={column: new_name})
            elif op_type == "fill_missing" and column:
                method = op.get("method", "mean")
                if column in df.columns:
                    if method == "mean" and df[column].dtype in (float, int):
                        df[column] = df[column].fillna(df[column].mean())
                    elif method == "median" and df[column].dtype in (float, int):
                        df[column] = df[column].fillna(df[column].median())
                    elif method == "mode":
                        df[column] = df[column].fillna(df[column].mode()[0] if len(df[column].mode()) > 0 else None)
                    elif method == "custom":
                        fill_value = op.get("fill_value")
                        df[column] = df[column].fillna(fill_value)
            elif op_type == "cast_type" and column:
                new_dtype = op.get("dtype", "object")
                if column in df.columns:
                    try:
                        df[column] = df[column].astype(new_dtype)
                    except Exception:
                        pass
        
        # Save as new dataset
        import tempfile
        import os
        import hashlib
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        
        # Calculate hash
        with open(tmp_path, 'rb') as f:
            content_hash = hashlib.md5(f.read()).hexdigest()
        
        # Create new dataset record
        new_dataset = Dataset(
            workspace_id=ds.workspace_id,
            name=f"{ds.name}_transformed",
            description=f"Transformed version of {ds.name}",
            source_type="file",
            file_path=tmp_path,
            content_hash=content_hash,
            status="ready",
            row_count=len(df),
            column_count=len(df.columns),
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        return new_dataset
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


