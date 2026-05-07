import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..cache import get_cached_result, store_result
from ..database import get_db
from ..models.dataset import Dataset
from ..models.user import User
from ..models.workspace import WorkspaceMember
from ..schemas.eda import (
    CorrelationResult,
    DistributionResult,
    FeatureImportanceResult,
    InsightCard,
    MissingResult,
    OutlierResult,
    ProfileResult,
    QualityScore,
    TextResult,
    TimeSeriesResult,
)

router = APIRouter(prefix="/datasets", tags=["eda"])


def _get_authorized_dataset(dataset_id: int, current_user: User, db: Session) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not current_user.is_admin:
        member = db.query(WorkspaceMember).filter(
            WorkspaceMember.workspace_id == ds.workspace_id,
            WorkspaceMember.user_id == current_user.id,
        ).first()
        if not member:
            raise HTTPException(status_code=403, detail="Access denied")
    return ds


def _load_df(ds: Dataset):
    import pandas as pd
    from ..connectors.file_connector import FileConnector
    from ..connectors.db_connector import DBConnector
    from ..connectors.api_connector import RESTAPIConnector
    from ..connectors.cloud_connector import CloudConnector

    config = json.loads(ds.source_config or "{}")

    if ds.source_type == "file":
        config["file_path"] = ds.file_path
        return FileConnector().load_data(config)
    elif ds.source_type in ("postgresql", "mysql", "sqlite", "mssql"):
        config["db_type"] = ds.source_type
        return DBConnector().load_data(config)
    elif ds.source_type == "mongodb":
        config["db_type"] = "mongodb"
        return DBConnector().load_data(config)
    elif ds.source_type == "rest_api":
        return RESTAPIConnector().load_data(config)
    elif ds.source_type in ("s3", "azure", "gcs"):
        config["cloud_type"] = ds.source_type
        return CloudConnector().load_data(config)
    else:
        raise ValueError(f"Unsupported source_type: {ds.source_type}")


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
        df = _load_df(ds)
        from ..eda.profiler import run_profile
        result = run_profile(df)
        if ds.file_path:
            import os
            result["file_size_bytes"] = os.path.getsize(ds.file_path)
        store_result(db, dataset_id, "profile", cache_key, result, ds.content_hash or "")
        return ProfileResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        df = _load_df(ds)
        from ..eda.missing import run_missing_analysis
        result = run_missing_analysis(df)
        store_result(db, dataset_id, "missing", cache_key, result, ds.content_hash or "")
        return MissingResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/distributions", response_model=DistributionResult)
def get_distributions(
    dataset_id: int,
    column: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "distributions", "column": column}
    cached = get_cached_result(db, dataset_id, "distributions", cache_key, ds.content_hash or "")
    if cached:
        return DistributionResult(**cached)

    try:
        df = _load_df(ds)
        from ..eda.distributions import run_distribution
        result = run_distribution(df, column)
        store_result(db, dataset_id, "distributions", cache_key, result, ds.content_hash or "")
        return DistributionResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/correlations", response_model=CorrelationResult)
def get_correlations(
    dataset_id: int,
    method: str = "pearson",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "correlations", "method": method}
    cached = get_cached_result(db, dataset_id, "correlations", cache_key, ds.content_hash or "")
    if cached:
        return CorrelationResult(**cached)

    try:
        df = _load_df(ds)
        from ..eda.correlations import run_correlations
        result = run_correlations(df, method)
        store_result(db, dataset_id, "correlations", cache_key, result, ds.content_hash or "")
        return CorrelationResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/outliers", response_model=OutlierResult)
def get_outliers(
    dataset_id: int,
    method: str = "iqr",
    column: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "outliers", "method": method, "column": column}
    cached = get_cached_result(db, dataset_id, "outliers", cache_key, ds.content_hash or "")
    if cached:
        return OutlierResult(**cached)

    try:
        df = _load_df(ds)
        from ..eda.outliers import run_outlier_detection
        result = run_outlier_detection(df, method, column)
        store_result(db, dataset_id, "outliers", cache_key, result, ds.content_hash or "")
        return OutlierResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/feature-importance", response_model=FeatureImportanceResult)
def get_feature_importance(
    dataset_id: int,
    target: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "feature_importance", "target": target}
    cached = get_cached_result(db, dataset_id, "feature_importance", cache_key, ds.content_hash or "")
    if cached:
        return FeatureImportanceResult(**cached)

    try:
        df = _load_df(ds)
        from ..eda.feature_importance import run_feature_importance
        result = run_feature_importance(df, target)
        store_result(db, dataset_id, "feature_importance", cache_key, result, ds.content_hash or "")
        return FeatureImportanceResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/timeseries", response_model=TimeSeriesResult)
def get_timeseries(
    dataset_id: int,
    time_col: str,
    value_col: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "timeseries", "time_col": time_col, "value_col": value_col}
    cached = get_cached_result(db, dataset_id, "timeseries", cache_key, ds.content_hash or "")
    if cached:
        return TimeSeriesResult(**cached)

    try:
        df = _load_df(ds)
        from ..eda.timeseries import run_timeseries
        result = run_timeseries(df, time_col, value_col)
        store_result(db, dataset_id, "timeseries", cache_key, result, ds.content_hash or "")
        return TimeSeriesResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/text", response_model=TextResult)
def get_text_analysis(
    dataset_id: int,
    column: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "text", "column": column}
    cached = get_cached_result(db, dataset_id, "text", cache_key, ds.content_hash or "")
    if cached:
        return TextResult(**cached)

    try:
        df = _load_df(ds)
        from ..eda.text_analysis import run_text_analysis
        result = run_text_analysis(df, column)
        store_result(db, dataset_id, "text", cache_key, result, ds.content_hash or "")
        return TextResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/quality-score", response_model=QualityScore)
def get_quality_score(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    cache_key = {"type": "quality_score"}
    cached = get_cached_result(db, dataset_id, "quality_score", cache_key, ds.content_hash or "")
    if cached:
        return QualityScore(**cached)

    try:
        df = _load_df(ds)
        from ..eda.quality_score import run_quality_score
        result = run_quality_score(df)
        store_result(db, dataset_id, "quality_score", cache_key, result, ds.content_hash or "")
        return QualityScore(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/insights", response_model=list[InsightCard])
def get_insights(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    try:
        df = _load_df(ds)
        from ..eda.profiler import run_profile
        from ..eda.quality_score import run_quality_score
        from ..eda.correlations import run_correlations
        from ..insights import InsightEngine

        profile = run_profile(df)
        quality = run_quality_score(df)
        correlations = run_correlations(df)

        engine = InsightEngine()
        insights = (
            engine.from_profile(profile)
            + engine.from_correlations(correlations)
            + engine.from_quality_score(quality)
        )
        return [InsightCard(**i) for i in insights[:20]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/transform/preview")
def transform_preview(
    dataset_id: int,
    operations: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Preview transformations without saving."""
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    try:
        df = _load_df(ds)
        
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/transform/apply")
def transform_apply(
    dataset_id: int,
    operations: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Apply transformations and save as a new dataset."""
    ds = _get_authorized_dataset(dataset_id, current_user, db)
    try:
        df = _load_df(ds)
        
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
