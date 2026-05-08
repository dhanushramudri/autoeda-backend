import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..auth import get_current_active_user
from ..models.user import User
from ..models.data_source import DataSource
from ..connectors.registry import get_connector, SOURCE_CATALOG

router = APIRouter(tags=["sources"])

# ── Simple XOR-based obfuscation (no extra deps needed) ────────────────────────
_SECRET = (os.getenv("SECRET_KEY") or "autoeda-secret-key-change-in-prod").encode()


def _encrypt(plain: str) -> str:
    key = (_SECRET * ((len(plain.encode()) // len(_SECRET)) + 1))[:len(plain.encode())]
    encrypted = bytes(a ^ b for a, b in zip(plain.encode(), key))
    import base64
    return base64.b64encode(encrypted).decode()


def _decrypt(enc: str) -> str:
    import base64
    encrypted = base64.b64decode(enc.encode())
    key = (_SECRET * ((len(encrypted) // len(_SECRET)) + 1))[:len(encrypted)]
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class SourceCreate(BaseModel):
    name: str
    description: Optional[str] = None
    source_type: str
    credentials: dict = {}
    config: dict = {}


class SourceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    credentials: Optional[dict] = None
    config: Optional[dict] = None


class TestConnectionRequest(BaseModel):
    source_type: str
    credentials: dict = {}
    config: dict = {}


class ImportDataRequest(BaseModel):
    source_id: int
    dataset_name: str
    workspace_id: int
    limit: Optional[int] = 50000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _source_or_404(source_id: int, workspace_id: str, db: Session) -> DataSource:
    try:
        wid = int(workspace_id)
    except (ValueError, TypeError):
        wid = workspace_id  # type: ignore
    src = db.query(DataSource).filter(
        DataSource.id == source_id,
        DataSource.workspace_id == wid,
    ).first()
    if not src:
        raise HTTPException(status_code=404, detail="Data source not found")
    return src


def _build_connector_config(source: DataSource) -> dict:
    creds = json.loads(_decrypt(source.credentials_enc)) if source.credentials_enc else {}
    cfg = json.loads(source.config) if source.config else {}
    merged = {**cfg, **creds, "db_type": source.source_type, "cloud_type": source.source_type}
    return merged


def _serialize(source: DataSource) -> dict:
    cfg = json.loads(source.config) if source.config else {}
    return {
        "id": source.id,
        "workspace_id": source.workspace_id,
        "name": source.name,
        "description": source.description,
        "source_type": source.source_type,
        "config": cfg,
        "status": source.status,
        "last_tested_at": source.last_tested_at.isoformat() if source.last_tested_at else None,
        "last_error": source.last_error,
        "created_by": source.created_by,
        "created_at": source.created_at.isoformat(),
        "updated_at": source.updated_at.isoformat(),
    }


# ── Source catalog (all connector types) ──────────────────────────────────────

@router.get("/sources/catalog")
def get_catalog(current_user: User = Depends(get_current_active_user)):
    return {"catalog": SOURCE_CATALOG}


# ── CRUD ──────────────────────────────────────────────────────────────────────

@router.get("/workspaces/{workspace_id}/sources")
def list_sources(
    workspace_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        wid = int(workspace_id)
    except (ValueError, TypeError):
        wid = workspace_id  # type: ignore
    sources = db.query(DataSource).filter(DataSource.workspace_id == wid).all()
    return {"sources": [_serialize(s) for s in sources]}


@router.post("/workspaces/{workspace_id}/sources", status_code=201)
def create_source(
    workspace_id: str,
    body: SourceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        wid = int(workspace_id)
    except (ValueError, TypeError):
        wid = workspace_id  # type: ignore

    creds_enc = _encrypt(json.dumps(body.credentials)) if body.credentials else None
    src = DataSource(
        workspace_id=wid,
        name=body.name,
        description=body.description,
        source_type=body.source_type,
        credentials_enc=creds_enc,
        config=json.dumps(body.config) if body.config else None,
        created_by=current_user.id,
    )
    db.add(src)
    db.commit()
    db.refresh(src)
    return _serialize(src)


@router.get("/workspaces/{workspace_id}/sources/{source_id}")
def get_source(
    workspace_id: str,
    source_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return _serialize(_source_or_404(source_id, workspace_id, db))


@router.patch("/workspaces/{workspace_id}/sources/{source_id}")
def update_source(
    workspace_id: str,
    source_id: int,
    body: SourceUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    src = _source_or_404(source_id, workspace_id, db)
    if body.name is not None:
        src.name = body.name
    if body.description is not None:
        src.description = body.description
    if body.credentials is not None:
        src.credentials_enc = _encrypt(json.dumps(body.credentials))
        src.status = "untested"
    if body.config is not None:
        src.config = json.dumps(body.config)
        src.status = "untested"
    src.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(src)
    return _serialize(src)


@router.delete("/workspaces/{workspace_id}/sources/{source_id}", status_code=204)
def delete_source(
    workspace_id: str,
    source_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    src = _source_or_404(source_id, workspace_id, db)
    db.delete(src)
    db.commit()


# ── Connection test ───────────────────────────────────────────────────────────

@router.post("/workspaces/{workspace_id}/sources/{source_id}/test")
def test_existing_source(
    workspace_id: str,
    source_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    src = _source_or_404(source_id, workspace_id, db)
    try:
        connector = get_connector(src.source_type)
        cfg = _build_connector_config(src)
        ok, msg = connector.test_connection(cfg)
    except Exception as exc:
        ok, msg = False, str(exc)

    src.status = "connected" if ok else "failed"
    src.last_tested_at = datetime.now(timezone.utc)
    src.last_error = None if ok else msg
    db.commit()
    return {"ok": ok, "message": msg, "status": src.status}


@router.post("/sources/test")
def test_adhoc_connection(
    body: TestConnectionRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Test a connection without saving — used during wizard."""
    try:
        connector = get_connector(body.source_type)
        merged = {**body.config, **body.credentials, "db_type": body.source_type, "cloud_type": body.source_type}
        ok, msg = connector.test_connection(merged)
    except Exception as exc:
        ok, msg = False, str(exc)
    return {"ok": ok, "message": msg}


# ── Schema browser ─────────────────────────────────────────────────────────────

@router.get("/workspaces/{workspace_id}/sources/{source_id}/schema")
def get_source_schema(
    workspace_id: str,
    source_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    src = _source_or_404(source_id, workspace_id, db)
    try:
        connector = get_connector(src.source_type)
        cfg = _build_connector_config(src)

        if hasattr(connector, "list_tables"):
            tables = connector.list_tables(cfg)
            return {"type": "database", "tables": tables}
        elif hasattr(connector, "list_s3_objects"):
            objects = connector.list_s3_objects(cfg)
            return {"type": "storage", "objects": objects}
        else:
            return {"type": "single", "tables": [cfg.get("table", "data")]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/workspaces/{workspace_id}/sources/{source_id}/schema/{table_name}/columns")
def get_table_columns(
    workspace_id: str,
    source_id: int,
    table_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    src = _source_or_404(source_id, workspace_id, db)
    try:
        connector = get_connector(src.source_type)
        cfg = {**_build_connector_config(src), "table": table_name, "query": None}
        df = connector.load_data(cfg, limit=1)
        columns = [{"name": col, "type": str(dtype)} for col, dtype in df.dtypes.items()]
        return {"table": table_name, "columns": columns}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Preview ───────────────────────────────────────────────────────────────────

@router.get("/workspaces/{workspace_id}/sources/{source_id}/preview")
def preview_source(
    workspace_id: str,
    source_id: int,
    table: Optional[str] = None,
    rows: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    src = _source_or_404(source_id, workspace_id, db)
    try:
        connector = get_connector(src.source_type)
        cfg = _build_connector_config(src)
        if table:
            cfg["table"] = table
            cfg["query"] = None
        df = connector.load_data(cfg, limit=rows)
        return {
            "columns": list(df.columns),
            "rows": [list(r) for r in df.itertuples(index=False)],
            "row_count": len(df),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Import as Dataset ─────────────────────────────────────────────────────────

@router.post("/workspaces/{workspace_id}/sources/{source_id}/import")
def import_as_dataset(
    workspace_id: str,
    source_id: int,
    body: ImportDataRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    from ..models.dataset import Dataset
    import pandas as pd
    import hashlib

    src = _source_or_404(source_id, workspace_id, db)
    try:
        connector = get_connector(src.source_type)
        cfg = _build_connector_config(src)
        df = connector.load_data(cfg, limit=body.limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load data: {exc}")

    # Save as parquet in uploads dir
    uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in body.dataset_name)
    file_path = os.path.abspath(os.path.join(uploads_dir, f"{safe_name}_{src.id}.parquet"))
    df.to_parquet(file_path, index=False)

    content = open(file_path, "rb").read()
    content_hash = hashlib.sha256(content).hexdigest()

    try:
        wid = int(workspace_id)
    except (ValueError, TypeError):
        wid = workspace_id  # type: ignore

    ds = Dataset(
        workspace_id=wid,
        name=body.dataset_name,
        description=f"Imported from {src.name} ({src.source_type})",
        source_type=src.source_type,
        file_path=file_path,
        row_count=len(df),
        column_count=len(df.columns),
        file_size_bytes=os.path.getsize(file_path),
        content_hash=content_hash,
        status="ready",
        created_by=current_user.id,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return {"dataset_id": ds.id, "row_count": len(df), "column_count": len(df.columns)}
