import hashlib
import io
import json
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..dataset_access import assert_dataset_access, dataset_visibility_filter
from ..database import get_db
from ..models.dataset import Dataset
from ..models.job import BackgroundJob
from ..models.user import User
from ..models.workspace import WorkspaceMember
from ..s3_attachments import delete_object, get_object_bytes, head_object, new_dataset_upload_key, presign_put
from ..schemas.dataset import (
    DatasetCreateResponse,
    DatasetPreview,
    DatasetResponse,
    DatasetUploadPresignRequest,
    DatasetUploadPresignResponse,
    ImportDatasetRequest,
)

router = APIRouter(tags=["datasets"])

MAX_DATASET_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB


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
    return (
        db.query(Dataset)
        .filter(dataset_visibility_filter(db, workspace_id))
        .order_by(Dataset.created_at.desc())
        .all()
    )


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

        # Store file bytes in DB — works in every environment (no disk required)
        ds.file_data = content
        ds.content_hash = content_hash
        ds.file_size_bytes = file_size
        ds.file_path = file.filename  # original filename kept for extension detection only

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


@router.post(
    "/workspaces/{workspace_id}/datasets/presign-upload",
    response_model=DatasetUploadPresignResponse,
    status_code=status.HTTP_201_CREATED,
)
def presign_dataset_upload(
    workspace_id: int,
    body: DatasetUploadPresignRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db, ["admin", "analyst"])
    if body.file_size_bytes > MAX_DATASET_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 100MB limit")
    if body.file_size_bytes <= 0:
        raise HTTPException(status_code=400, detail="Empty file")

    key = new_dataset_upload_key(workspace_id, body.filename or "dataset")
    return DatasetUploadPresignResponse(
        s3_key=key,
        upload_url=presign_put(key, body.content_type),
    )


@router.post(
    "/workspaces/{workspace_id}/datasets/confirm-upload",
    response_model=DatasetCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def confirm_dataset_upload(
    workspace_id: int,
    background_tasks: BackgroundTasks,
    s3_key: str = Form(...),
    original_filename: str = Form(...),
    name: str = Form(...),
    source_type: str = Form(...),
    description: Optional[str] = Form(None),
    config_json: Optional[str] = Form("{}"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db, ["admin", "analyst"])

    if not s3_key.startswith(f"dataset-uploads/{workspace_id}/"):
        raise HTTPException(status_code=400, detail="Invalid upload reference")

    meta = head_object(s3_key)
    if not meta:
        raise HTTPException(status_code=400, detail="Uploaded file not found in storage — please retry")
    if meta.get("ContentLength", 0) > MAX_DATASET_UPLOAD_BYTES:
        delete_object(s3_key)
        raise HTTPException(status_code=413, detail="File exceeds 100MB limit")

    content = get_object_bytes(s3_key)
    delete_object(s3_key)
    if content is None:
        raise HTTPException(status_code=400, detail="Could not read uploaded file — please retry")

    config = json.loads(config_json or "{}")
    content_hash = hashlib.md5(content).hexdigest()

    ds = Dataset(
        workspace_id=workspace_id,
        name=name,
        description=description,
        source_type=source_type,
        source_config=json.dumps(config),
        status="processing",
        created_by=current_user.id,
        file_data=content,
        content_hash=content_hash,
        file_size_bytes=len(content),
        file_path=original_filename,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    job_id = str(uuid.uuid4())
    job = BackgroundJob(
        id=job_id, job_type="eda_pipeline", status="pending", progress=0,
        dataset_id=ds.id, created_by=current_user.id,
    )
    db.add(job)
    db.commit()

    from ..tasks import run_eda_pipeline
    background_tasks.add_task(run_eda_pipeline, job_id, ds.id, None, config)

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
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_access(ds, current_user, db)
    return ds


@router.delete("/workspaces/{workspace_id}/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    workspace_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_access(ds, current_user, db, ["admin", "analyst"])

    ds_name = ds.name
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
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_access(ds, current_user, db, ["admin", "analyst"])

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
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_access(ds, current_user, db)

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
    assert_dataset_access(ds, current_user, db)
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
    assert_dataset_access(ds, current_user, db)

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


@router.post("/datasets/{dataset_id}/import", response_model=DatasetCreateResponse, status_code=status.HTTP_201_CREATED)
def import_dataset_to_workspace(
    dataset_id: int,
    payload: ImportDatasetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Copy a dataset (e.g. one linked from the Dataset Library) into one of
    the current user's own workspaces — a fresh, independent Dataset row with
    its own copy of the bytes, not a reference to the original."""
    source = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_access(source, current_user, db)
    if not source.file_data:
        raise HTTPException(status_code=400, detail="This dataset has no file data to import")

    _assert_member(payload.workspace_id, current_user, db, ["admin", "analyst"])

    ds = Dataset(
        workspace_id=payload.workspace_id,
        name=source.name,
        description=source.description,
        source_type="file",
        source_config=source.source_config,
        status="processing",
        created_by=current_user.id,
        file_data=source.file_data,
        content_hash=source.content_hash,
        file_size_bytes=source.file_size_bytes,
        file_path=source.file_path,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    job_id = str(uuid.uuid4())
    job = BackgroundJob(
        id=job_id, job_type="eda_pipeline", status="pending", progress=0,
        dataset_id=ds.id, created_by=current_user.id,
    )
    db.add(job)
    db.commit()

    config = json.loads(ds.source_config or "{}")
    from ..tasks import run_eda_pipeline
    background_tasks.add_task(run_eda_pipeline, job_id, ds.id, None, config)

    resp = DatasetCreateResponse.model_validate(ds)
    resp.job_id = job_id
    return resp


@router.post("/datasets/{dataset_id}/transform")
def transform_dataset(
    dataset_id: int,
    payload: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    import pandas as pd

    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_access(ds, current_user, db, ["admin", "analyst"])

    df = _load_dataset_df(ds)
    operations = payload.get("operations", [])

    import numpy as np

    errors_log = []
    for op in operations:
        op_type = op.get("type")
        try:
            # ── Existing ops ──────────────────────────────────────────────────
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
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan)
                    elif strategy == "constant" and val is not None:
                        df[col] = df[col].fillna(val)
                    elif strategy == "ffill":
                        df[col] = df[col].ffill()
                    elif strategy == "bfill":
                        df[col] = df[col].bfill()

            elif op_type == "encode":
                col, method = op.get("column"), op.get("method", "label")
                if col in df.columns:
                    if method == "onehot":
                        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    else:
                        df[col] = df[col].astype("category").cat.codes

            elif op_type == "scale":
                col, method = op.get("column"), op.get("method", "standard")
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if method == "standard":
                        from sklearn.preprocessing import StandardScaler
                        df[[col]] = StandardScaler().fit_transform(df[[col]])
                    elif method == "minmax":
                        from sklearn.preprocessing import MinMaxScaler
                        df[[col]] = MinMaxScaler().fit_transform(df[[col]])
                    elif method == "robust":
                        from sklearn.preprocessing import RobustScaler
                        df[[col]] = RobustScaler().fit_transform(df[[col]])

            elif op_type == "drop_duplicates":
                subset = op.get("subset") or None
                df = df.drop_duplicates(subset=subset)

            elif op_type == "drop_outliers":
                col, method = op.get("column"), op.get("method", "iqr")
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if method == "iqr":
                        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                        iqr = q3 - q1
                        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
                    elif method == "zscore":
                        from scipy import stats
                        mask = np.abs(stats.zscore(df[col].fillna(df[col].median()))) < 3
                        df = df[mask]

            # ── Column ops ───────────────────────────────────────────────────
            elif op_type == "rename_column":
                old, new = op.get("old_name"), op.get("new_name")
                if old in df.columns and new and new not in df.columns:
                    df = df.rename(columns={old: new})

            elif op_type == "cast_type":
                col, to_type = op.get("column"), op.get("to_type")
                if col in df.columns:
                    if to_type == "int":
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    elif to_type == "float":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif to_type == "str":
                        df[col] = df[col].astype(str)
                    elif to_type == "datetime":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    elif to_type == "bool":
                        df[col] = df[col].astype(bool)

            elif op_type == "select_columns":
                cols = [c for c in op.get("columns", []) if c in df.columns]
                if cols:
                    df = df[cols]

            elif op_type == "reorder_columns":
                cols = op.get("columns", [])
                existing = [c for c in cols if c in df.columns]
                rest = [c for c in df.columns if c not in existing]
                df = df[existing + rest]

            elif op_type == "create_column":
                name, expr = op.get("name"), op.get("expression", "")
                if name and expr:
                    # Restricted eval — only column refs and numpy math
                    allowed = {c: df[c] for c in df.columns}
                    allowed.update({"np": np, "abs": abs, "round": round})
                    df[name] = eval(expr, {"__builtins__": {}}, allowed)  # noqa: S307

            # ── Cleaning ops ─────────────────────────────────────────────────
            elif op_type == "cap_outliers":
                col, method = op.get("column"), op.get("method", "iqr")
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if method == "iqr":
                        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    else:
                        lp = op.get("lower_pct", 1)
                        up = op.get("upper_pct", 99)
                        lower, upper = df[col].quantile(lp / 100), df[col].quantile(up / 100)
                    df[col] = df[col].clip(lower=lower, upper=upper)

            elif op_type == "clip":
                col = op.get("column")
                lower, upper = op.get("lower"), op.get("upper")
                if col in df.columns:
                    df[col] = df[col].clip(
                        lower=float(lower) if lower is not None else None,
                        upper=float(upper) if upper is not None else None,
                    )

            elif op_type == "filter_rows":
                col, operator, value = op.get("column"), op.get("operator"), op.get("value")
                if col in df.columns and operator:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        v = float(value)
                    else:
                        v = value
                    op_map = {
                        "eq": df[col] == v, "neq": df[col] != v,
                        "gt": df[col] > v,  "gte": df[col] >= v,
                        "lt": df[col] < v,  "lte": df[col] <= v,
                        "contains": df[col].astype(str).str.contains(str(value), na=False),
                        "not_contains": ~df[col].astype(str).str.contains(str(value), na=False),
                        "startswith": df[col].astype(str).str.startswith(str(value)),
                        "endswith": df[col].astype(str).str.endswith(str(value)),
                    }
                    if operator in op_map:
                        df = df[op_map[operator]]

            elif op_type == "drop_rows_where_null":
                cols = op.get("columns") or list(df.columns)
                df = df.dropna(subset=[c for c in cols if c in df.columns])

            elif op_type == "sample_rows":
                n, frac = op.get("n"), op.get("frac")
                seed = op.get("random_state", 42)
                if n:
                    df = df.sample(n=min(int(n), len(df)), random_state=seed)
                elif frac:
                    df = df.sample(frac=float(frac), random_state=seed)

            elif op_type == "sort_rows":
                by = op.get("by", [])
                ascending = op.get("ascending", [True] * len(by))
                valid = [c for c in by if c in df.columns]
                asc = ascending[:len(valid)] if isinstance(ascending, list) else [ascending] * len(valid)
                if valid:
                    df = df.sort_values(by=valid, ascending=asc).reset_index(drop=True)

            # ── Feature engineering ───────────────────────────────────────────
            elif op_type == "log_transform":
                col, variant = op.get("column"), op.get("variant", "log1p")
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    new_name = op.get("new_name") or f"{col}_{variant}"
                    df[new_name] = np.log1p(df[col]) if variant == "log1p" else np.log(df[col].clip(lower=1e-9))

            elif op_type == "sqrt_transform":
                col = op.get("column")
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    new_name = op.get("new_name") or f"{col}_sqrt"
                    df[new_name] = np.sqrt(df[col].clip(lower=0))

            elif op_type == "bin":
                col = op.get("column")
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    bins = op.get("bins", 5)
                    labels = op.get("labels") or None
                    strategy = op.get("strategy", "cut")
                    new_name = op.get("new_name") or f"{col}_bin"
                    if strategy == "qcut":
                        df[new_name] = pd.qcut(df[col], q=bins, labels=labels, duplicates="drop")
                    else:
                        df[new_name] = pd.cut(df[col], bins=bins, labels=labels)

            elif op_type == "extract_datetime":
                col = op.get("column")
                if col in df.columns:
                    dt = pd.to_datetime(df[col], errors="coerce")
                    for part in op.get("parts", []):
                        if part == "year":    df[f"{col}_year"]    = dt.dt.year
                        elif part == "month": df[f"{col}_month"]   = dt.dt.month
                        elif part == "day":   df[f"{col}_day"]     = dt.dt.day
                        elif part == "hour":  df[f"{col}_hour"]    = dt.dt.hour
                        elif part == "minute":df[f"{col}_minute"]  = dt.dt.minute
                        elif part == "weekday":df[f"{col}_weekday"]= dt.dt.dayofweek
                        elif part == "quarter":df[f"{col}_quarter"]= dt.dt.quarter

            elif op_type == "text_clean":
                col = op.get("column")
                if col in df.columns:
                    s = df[col].astype(str)
                    if op.get("strip", False):
                        s = s.str.strip()
                    if op.get("lowercase", False):
                        s = s.str.lower()
                    if op.get("uppercase", False):
                        s = s.str.upper()
                    rep_from = op.get("replace_from")
                    rep_to = op.get("replace_to", "")
                    if rep_from:
                        s = s.str.replace(rep_from, rep_to, regex=op.get("regex", False))
                    if op.get("remove_special", False):
                        s = s.str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
                    df[col] = s

        except Exception as e:
            errors_log.append({"op": op_type, "error": str(e)})

    # Save transformed result back to DB as CSV bytes
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue()
    ds.file_data = csv_bytes
    ds.file_size_bytes = len(csv_bytes)
    ds.file_path = (ds.file_path or "").rsplit(".", 1)[0] + ".csv" if ds.file_path else "transformed.csv"
    ds.content_hash = hashlib.md5(csv_bytes).hexdigest()
    ds.row_count = len(df)
    ds.column_count = len(df.columns)
    ds.schema_info = json.dumps({col: str(dtype) for col, dtype in df.dtypes.items()})
    db.commit()

    return {
        "message": "Transformations applied",
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "errors": errors_log,
    }


@router.get("/datasets/{dataset_id}/export")
def export_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    from fastapi.responses import StreamingResponse

    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_access(ds, current_user, db)

    # Serve DB bytes directly if available (covers file datasets in all environments)
    if ds.file_data:
        filename = ds.file_path or f"{ds.name}.csv"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "csv"
        media_types = {"csv": "text/csv", "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       "parquet": "application/octet-stream", "json": "application/json"}
        media_type = media_types.get(ext, "application/octet-stream")
        return StreamingResponse(
            iter([ds.file_data]),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{ds.name}.{ext}"'},
        )

    # Fallback: load via connector and stream as CSV
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
    import os
    from ..connectors.file_connector import load_from_bytes
    from ..connectors.db_connector import DBConnector
    from ..connectors.api_connector import RESTAPIConnector
    from ..connectors.cloud_connector import CloudConnector

    config = json.loads(ds.source_config or "{}")

    if ds.source_type == "file":
        if not ds.file_data:
            raise HTTPException(status_code=400, detail=f"Dataset {ds.id} has no file data in database")
        filename = os.path.basename(ds.file_path or "") if ds.file_path else ""
        # Use database bytes only (file-based data stored in DB)
        return load_from_bytes(ds.file_data, filename, config)
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


