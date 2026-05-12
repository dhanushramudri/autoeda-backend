import hashlib
import json
import os
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..database import get_db
from ..models.dataset import Dataset
from ..models.job import BackgroundJob
from ..models.user import User
from ..models.workspace import WorkspaceMember
from ..schemas.dataset import DatasetCreateResponse, DatasetPreview, DatasetResponse
from ..config import settings

router = APIRouter(tags=["datasets"])


def _assert_member(workspace_id: int, user: User, db: Session, roles: list[str] = None):
    if user.is_admin:
        return
    m = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    if not m:
        raise HTTPException(status_code=403, detail="Not a workspace member")
    if roles and m.role not in roles:
        raise HTTPException(status_code=403, detail=f"Requires role: {roles}")


@router.get("/workspaces/{workspace_id}/datasets", response_model=list[DatasetResponse])
def list_datasets(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    return db.query(Dataset).filter(Dataset.workspace_id == workspace_id).order_by(Dataset.created_at.desc()).all()


@router.post(
    "/workspaces/{workspace_id}/datasets",
    response_model=DatasetCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_dataset(
    workspace_id: int,
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    source_type: str = Form(...),
    description: Optional[str] = Form(None),
    config_json: Optional[str] = Form("{}"),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db, ["admin", "analyst"])

    config = json.loads(config_json or "{}")
    dataset_id = None
    file_path = None
    content_hash = None
    file_size = None

    ds = Dataset(
        workspace_id=workspace_id,
        name=name,
        description=description,
        source_type=source_type,
        source_config=json.dumps(config),
        status="processing",
        created_by=current_user.id,
    )
    db.add(ds)
    db.flush()
    dataset_id = ds.id

    if file:
        content = await file.read()
        content_hash = hashlib.md5(content).hexdigest()
        file_size = len(content)

        # Always store bytes in DB — works in any environment
        ds.file_data = content
        ds.content_hash = content_hash
        ds.file_size_bytes = file_size
        ds.file_path = file.filename  # keep original filename for extension detection

        # Also write to local disk as a read cache (best-effort)
        try:
            storage_dir = os.path.join(settings.STORAGE_PATH, str(workspace_id), str(dataset_id))
            os.makedirs(storage_dir, exist_ok=True)
            local_path = os.path.join(storage_dir, file.filename)
            with open(local_path, "wb") as f:
                f.write(content)
            ds.file_path = local_path
        except OSError:
            pass  # no local disk available — DB copy is the source of truth

    db.commit()
    db.refresh(ds)

    job_id = str(uuid.uuid4())
    job = BackgroundJob(
        id=job_id,
        job_type="eda_pipeline",
        status="pending",
        progress=0,
        dataset_id=dataset_id,
        created_by=current_user.id,
    )
    db.add(job)
    db.commit()

    from ..tasks import run_eda_pipeline
    background_tasks.add_task(run_eda_pipeline, job_id, dataset_id, file_path, config)

    resp = DatasetCreateResponse.model_validate(ds)
    resp.job_id = job_id
    return resp


@router.get("/workspaces/{workspace_id}/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    workspace_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    ds = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.workspace_id == workspace_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds


@router.delete("/workspaces/{workspace_id}/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    workspace_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db, ["admin", "analyst"])
    ds = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.workspace_id == workspace_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if ds.file_path and os.path.exists(ds.file_path):
        os.remove(ds.file_path)

    db.delete(ds)
    db.commit()


@router.post("/workspaces/{workspace_id}/datasets/{dataset_id}/refresh", response_model=DatasetCreateResponse)
def refresh_dataset(
    workspace_id: int,
    dataset_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db, ["admin", "analyst"])
    ds = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.workspace_id == workspace_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ds.status = "processing"
    db.commit()

    job_id = str(uuid.uuid4())
    config = json.loads(ds.source_config or "{}")
    job = BackgroundJob(
        id=job_id, job_type="eda_pipeline", status="pending", progress=0,
        dataset_id=dataset_id, created_by=current_user.id,
    )
    db.add(job)
    db.commit()

    from ..tasks import run_eda_pipeline
    background_tasks.add_task(run_eda_pipeline, job_id, dataset_id, ds.file_path, config)

    resp = DatasetCreateResponse.model_validate(ds)
    resp.job_id = job_id
    return resp


@router.get("/workspaces/{workspace_id}/datasets/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    workspace_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    ds = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.workspace_id == workspace_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = _load_dataset_df(ds, limit=100)
        rows = df.head(100).fillna("").astype(str).to_dict(orient="records")
        return DatasetPreview(
            columns=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            rows=rows,
            total_rows=len(df),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset_by_id(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    _assert_member(ds.workspace_id, current_user, db)
    return ds


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset_shorthand(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Shorthand endpoint for dataset preview without requiring workspace_id"""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    _assert_member(ds.workspace_id, current_user, db)

    try:
        df = _load_dataset_df(ds, limit=100)
        rows = df.head(100).fillna("").astype(str).to_dict(orient="records")
        return DatasetPreview(
            columns=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            rows=rows,
            total_rows=len(df),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{dataset_id}/transform")
def transform_dataset(
    dataset_id: int,
    payload: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    from ..connectors.file_connector import FileConnector
    import pandas as pd

    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    _assert_member(ds.workspace_id, current_user, db, ["admin", "analyst"])

    config = json.loads(ds.source_config or "{}")
    config["file_path"] = ds.file_path
    df = _load_dataset_df(ds)
    operations = payload.get("operations", [])

    for op in operations:
        op_type = op.get("type")
        try:
            if op_type == "drop_columns":
                df = df.drop(columns=[c for c in op.get("columns", []) if c in df.columns], errors="ignore")
            elif op_type == "fill_missing":
                col, strategy, val = op.get("column"), op.get("strategy", "mean"), op.get("value")
                if col in df.columns:
                    if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == "mode":
                        df[col] = df[col].fillna(df[col].mode()[0])
                    elif strategy == "constant" and val is not None:
                        df[col] = df[col].fillna(val)
            elif op_type == "encode":
                col, method = op.get("column"), op.get("method", "label")
                if col in df.columns:
                    if method == "onehot":
                        dummies = pd.get_dummies(df[col], prefix=col)
                        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    else:
                        df[col] = df[col].astype("category").cat.codes
            elif op_type == "scale":
                col, method = op.get("column"), op.get("method", "standard")
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if method == "standard":
                        from sklearn.preprocessing import StandardScaler
                        df[[col]] = StandardScaler().fit_transform(df[[col]])
                    else:
                        from sklearn.preprocessing import MinMaxScaler
                        df[[col]] = MinMaxScaler().fit_transform(df[[col]])
            elif op_type == "drop_duplicates":
                df = df.drop_duplicates()
            elif op_type == "drop_outliers":
                col, method = op.get("column"), op.get("method", "iqr")
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if method == "iqr":
                        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                        iqr = q3 - q1
                        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
                    else:
                        from scipy import stats
                        df = df[abs(stats.zscore(df[col].dropna())) < 3]
        except Exception:
            pass

    out_dir = os.path.join(settings.STORAGE_PATH, str(ds.workspace_id), str(dataset_id), "transformed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "transformed.csv")
    df.to_csv(out_path, index=False)

    return {"message": "Transformations applied", "rows": len(df), "columns": len(df.columns), "path": out_path}


@router.get("/datasets/{dataset_id}/export")
def export_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    from fastapi.responses import FileResponse, StreamingResponse
    import io

    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    _assert_member(ds.workspace_id, current_user, db)

    transformed_path = os.path.join(settings.STORAGE_PATH, str(ds.workspace_id), str(dataset_id), "transformed", "transformed.csv")
    if os.path.exists(transformed_path):
        return FileResponse(transformed_path, media_type="text/csv", filename=f"{ds.name}_transformed.csv")

    df = _load_dataset_df(ds)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{ds.name}.csv"'},
    )


def _load_dataset_df(ds: Dataset, limit: int = None):
    import pandas as pd
    from ..connectors.file_connector import FileConnector
    from ..connectors.db_connector import DBConnector
    from ..connectors.api_connector import RESTAPIConnector
    from ..connectors.cloud_connector import CloudConnector

    config = json.loads(ds.source_config or "{}")

    if ds.source_type == "file":
        config["file_path"] = ds.file_path
        return FileConnector().load_data(config, limit=limit)
    elif ds.source_type in ("postgresql", "mysql", "sqlite", "mssql"):
        config["db_type"] = ds.source_type
        return DBConnector().load_data(config, limit=limit)
    elif ds.source_type == "mongodb":
        config["db_type"] = "mongodb"
        return DBConnector().load_data(config, limit=limit)
    elif ds.source_type == "rest_api":
        return RESTAPIConnector().load_data(config, limit=limit)
    elif ds.source_type in ("s3", "azure", "gcs"):
        config["cloud_type"] = ds.source_type
        return CloudConnector().load_data(config, limit=limit)
    else:
        raise ValueError(f"Unsupported source_type: {ds.source_type}")
