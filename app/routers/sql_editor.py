import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.dataset import Dataset
from ..auth import get_current_active_user
from ..models.user import User

router = APIRouter(tags=["sql_editor"])


def _load_duckdb():
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise HTTPException(status_code=503, detail="DuckDB not installed. Run: pip install duckdb")


def _load_dataset_df(dataset_id: str, db: Session, user: User = None):
    """Load dataset as a pandas DataFrame — always uses DB bytes, never disk paths."""
    import os
    from ..connectors.file_connector import FileConnector, load_from_bytes
    from ..connectors.db_connector import DBConnector
    from ..connectors.api_connector import RESTAPIConnector
    from ..connectors.cloud_connector import CloudConnector
    import json

    try:
        did = int(dataset_id)
    except (ValueError, TypeError):
        did = dataset_id  # type: ignore[assignment]

    ds = db.query(Dataset).filter(Dataset.id == did).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    config = json.loads(ds.source_config or "{}")

    if ds.source_type == "file":
        filename = os.path.basename(ds.file_path or "") if ds.file_path else ""
        # Prefer DB bytes (works everywhere), fall back to local disk only if available
        if ds.file_data:
            return load_from_bytes(ds.file_data, filename, config)
        elif ds.file_path and os.path.exists(ds.file_path):
            config["file_path"] = ds.file_path
            return FileConnector().load_data(config)
        else:
            raise HTTPException(status_code=400, detail=f"Dataset {dataset_id} has no accessible file data")
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
        raise HTTPException(status_code=400, detail=f"Unsupported source type: {ds.source_type}")


def _register_df(con, dataset_id: str, db: Session, user: User = None):
    """Load dataset and register it as 'df' in a DuckDB connection."""
    df = _load_dataset_df(dataset_id, db, user)
    con.register("df", df)


class SqlExecuteRequest(BaseModel):
    sql: str
    limit: int = 1000


class SqlExplainRequest(BaseModel):
    sql: str


@router.post("/datasets/{dataset_id}/sql/execute")
def execute_sql(
    dataset_id: str,
    body: SqlExecuteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    duckdb = _load_duckdb()
    try:
        con = duckdb.connect(database=":memory:")
        _register_df(con, dataset_id, db, current_user)

        sql = body.sql.strip()
        if sql.upper().startswith("SELECT") and "LIMIT" not in sql.upper():
            sql = f"SELECT * FROM ({sql}) __q LIMIT {body.limit}"

        t0 = time.perf_counter()
        result = con.execute(sql)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        columns = [desc[0] for desc in result.description]
        rows: list[list[Any]] = result.fetchall()
        con.close()
        return {
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "elapsed_ms": elapsed_ms,
            "truncated": len(rows) >= body.limit,
        }
    except HTTPException:
        raise
    except Exception as exc:
        return {
            "columns": [], "rows": [], "row_count": 0,
            "elapsed_ms": 0, "truncated": False, "error": str(exc),
        }


@router.post("/datasets/{dataset_id}/sql/explain")
def explain_sql(
    dataset_id: str,
    body: SqlExplainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    duckdb = _load_duckdb()
    try:
        con = duckdb.connect(database=":memory:")
        _register_df(con, dataset_id, db, current_user)
        result = con.execute(f"EXPLAIN {body.sql}")
        rows = result.fetchall()
        con.close()
        plan = "\n".join(str(r[1]) for r in rows if len(r) > 1)
        return {"plan": plan}
    except HTTPException:
        raise
    except Exception as exc:
        return {"plan": "", "error": str(exc)}


@router.get("/datasets/{dataset_id}/sql/schema")
def get_schema(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    duckdb = _load_duckdb()
    try:
        con = duckdb.connect(database=":memory:")
        _register_df(con, dataset_id, db, current_user)
        result = con.execute("DESCRIBE df")
        rows = result.fetchall()
        con.close()
        return {
            "table": "df",
            "columns": [{"name": r[0], "type": r[1]} for r in rows],
        }
    except HTTPException:
        raise
    except Exception as exc:
        return {"table": "df", "columns": [], "error": str(exc)}
