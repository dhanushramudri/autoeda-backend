import re
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..auth import get_current_active_user
from ..models.user import User
from ..models.dataset import Dataset
from ..models.data_source import DataSource
from ..connectors.registry import get_connector

router = APIRouter(tags=["warehouse"])


# ── Slug helpers ───────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]", "_", name).lower()
    slug = re.sub(r"_+", "_", slug).strip("_")
    if slug and slug[0].isdigit():
        slug = "t_" + slug
    return slug or "dataset"


def _table_slug(source_slug: str, table: str) -> str:
    """Namespace a source table: source_slug__table_name"""
    t = re.sub(r"[^a-zA-Z0-9]", "_", table).lower().strip("_")
    return f"{source_slug}__{t}"


# ── DuckDB helpers ─────────────────────────────────────────────────────────────

def _load_duckdb():
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise HTTPException(status_code=503, detail="DuckDB not installed")


def _register_file_view(con, view_name: str, fp: str):
    fp = fp.replace("\\", "/")
    ext = fp.rsplit(".", 1)[-1].lower() if "." in fp else "csv"
    if ext in ("csv", "tsv"):
        con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_csv_auto('{fp}')")
    elif ext == "parquet":
        con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{fp}')")
    elif ext in ("xls", "xlsx"):
        import pandas as pd
        con.register(view_name, pd.read_excel(fp))
    elif ext == "json":
        con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_json_auto('{fp}')")
    else:
        con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_csv_auto('{fp}')")


def _describe_view(con, view_name: str) -> list[dict]:
    try:
        result = con.execute(f"DESCRIBE {view_name}")
        return [{"name": r[0], "type": r[1]} for r in result.fetchall()]
    except Exception:
        return []


def _normalize_wid(workspace_id: str) -> int:
    try:
        return int(workspace_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid workspace id")


def _get_ready_datasets(wid: int, db: Session) -> list[Dataset]:
    return (
        db.query(Dataset)
        .filter(
            Dataset.workspace_id == wid,
            Dataset.file_path.isnot(None),
            Dataset.status == "ready",
        )
        .order_by(Dataset.name)
        .all()
    )


def _get_workspace_sources(wid: int, db: Session) -> list[DataSource]:
    return (
        db.query(DataSource)
        .filter(DataSource.workspace_id == wid)
        .order_by(DataSource.name)
        .all()
    )


def _decrypt_source_config(source: DataSource) -> dict:
    import json, os, base64
    secret = (os.getenv("SECRET_KEY") or "autoeda-secret-key-change-in-prod").encode()
    creds = {}
    if source.credentials_enc:
        try:
            encrypted = base64.b64decode(source.credentials_enc.encode())
            key = (secret * ((len(encrypted) // len(secret)) + 1))[:len(encrypted)]
            raw = bytes(a ^ b for a, b in zip(encrypted, key)).decode()
            creds = json.loads(raw)
        except Exception:
            pass
    cfg = json.loads(source.config) if source.config else {}
    return {**cfg, **creds, "db_type": source.source_type, "cloud_type": source.source_type}


# ── Catalog endpoint ───────────────────────────────────────────────────────────

@router.get("/workspaces/{workspace_id}/warehouse/catalog")
def get_warehouse_catalog(
    workspace_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Returns the full warehouse catalog grouped by source:
    - One section per connected data source (with tables from list_tables)
    - One section for workspace-uploaded datasets
    """
    wid = _normalize_wid(workspace_id)
    duckdb = _load_duckdb()

    sections = []

    # ── Section 1: Uploaded datasets ──────────────────────────────────────────
    datasets = _get_ready_datasets(wid, db)

    workspace_dataset_items = []
    source_dataset_map: dict[int, list] = {}

    for ds in datasets:
        slug = _slugify(ds.name)
        fp = ds.file_path.replace("\\", "/")

        columns = []

        try:
            con = duckdb.connect(":memory:")
            _register_file_view(con, slug, fp)
            columns = _describe_view(con, slug)
            con.close()
        except Exception:
            pass

        item = {
            "slug": slug,
            "name": ds.name,
            "id": ds.id,
            "row_count": ds.row_count,
            "column_count": ds.column_count,
            "source_type": ds.source_type,
            "columns": columns,
        }

        # Imported from source
        if ds.source_id:
            if ds.source_id not in source_dataset_map:
                source_dataset_map[ds.source_id] = []

            source_dataset_map[ds.source_id].append(item)

        # Uploaded dataset
        else:
            workspace_dataset_items.append(item)

    if workspace_dataset_items:
        sections.append({
            "type": "datasets",
            "label": "Workspace Datasets",
            "icon": "upload",
            "status": "ok",
            "items": workspace_dataset_items,
        })

    # ── Section 2+: Connected data sources ────────────────────────────────────
    sources = _get_workspace_sources(wid, db)
    for src in sources:
        source_slug = _slugify(src.name)
        items = source_dataset_map.get(src.id, []).copy()
        error = None

        try:
            connector = get_connector(src.source_type)
            cfg = _decrypt_source_config(src)

            if hasattr(connector, "list_tables"):
                tables = connector.list_tables(cfg)
                for table in tables[:200]:  # cap at 200 tables
                    items.append({
                        "slug": _table_slug(source_slug, table),
                        "table": table,
                        "source_id": src.id,
                        "source_slug": source_slug,
                        "columns": [],  # loaded lazily per-table
                    })
            elif hasattr(connector, "list_s3_objects"):
                objects = connector.list_s3_objects(cfg)
                for obj in objects[:200]:
                    items.append({
                        "slug": _table_slug(source_slug, obj),
                        "table": obj,
                        "source_id": src.id,
                        "source_slug": source_slug,
                        "columns": [],
                    })
        except Exception as exc:
            error = str(exc)

        sections.append({
            "type": "source",
            "id": src.id,
            "label": src.name,
            "source_type": src.source_type,
            "status": src.status,
            "icon": "source",
            "error": error,
            "items": items,
        })

    return {"sections": sections}


# ── Lazy column load for source tables ────────────────────────────────────────

@router.get("/workspaces/{workspace_id}/warehouse/sources/{source_id}/tables/{table_name}/columns")
def get_source_table_columns(
    workspace_id: str,
    source_id: int,
    table_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    wid = _normalize_wid(workspace_id)
    src = db.query(DataSource).filter(
        DataSource.id == source_id,
        DataSource.workspace_id == wid,
    ).first()
    if not src:
        raise HTTPException(status_code=404, detail="Source not found")
    try:
        connector = get_connector(src.source_type)
        cfg = {**_decrypt_source_config(src), "table": table_name, "query": None}
        df = connector.load_data(cfg, limit=1)
        columns = [{"name": col, "type": str(dtype)} for col, dtype in df.dtypes.items()]
        return {"columns": columns}
    except Exception as exc:
        return {"columns": [], "error": str(exc)}


# ── Execute SQL ────────────────────────────────────────────────────────────────

class WarehouseExecuteRequest(BaseModel):
    sql: str
    limit: int = 5000


@router.post("/workspaces/{workspace_id}/warehouse/execute")
def execute_warehouse_sql(
    workspace_id: str,
    body: WarehouseExecuteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    wid = _normalize_wid(workspace_id)
    duckdb = _load_duckdb()

    datasets = _get_ready_datasets(wid, db)
    sources = _get_workspace_sources(wid, db)

    try:
        con = duckdb.connect(":memory:")
        registered: list[str] = []

        # 1. Register all workspace datasets (files → DuckDB views)
        for ds in datasets:
            slug = _slugify(ds.name)
            try:
                _register_file_view(con, slug, ds.file_path)
                registered.append(slug)
            except Exception:
                pass

        # 2. For each data source, detect which table slugs appear in the SQL
        #    and load only those tables on-demand
        sql_upper = body.sql.upper()
        for src in sources:
            source_slug = _slugify(src.name)
            try:
                connector = get_connector(src.source_type)
                cfg = _decrypt_source_config(src)

                if hasattr(connector, "list_tables"):
                    tables = connector.list_tables(cfg)
                elif hasattr(connector, "list_s3_objects"):
                    tables = connector.list_s3_objects(cfg)
                else:
                    tables = []

                for table in tables[:200]:
                    slug = _table_slug(source_slug, table)
                    # Only load if this slug is actually referenced in the SQL
                    if not re.search(r"\b" + re.escape(slug) + r"\b", body.sql, re.IGNORECASE):
                        continue
                    try:
                        table_cfg = {**cfg, "table": table, "collection": table, "query": None}
                        df = connector.load_data(table_cfg, limit=100000)
                        con.register(slug, df)
                        registered.append(slug)
                    except Exception:
                        pass

            except Exception:
                pass

        if not registered:
            return {
                "sql": body.sql, "columns": [], "rows": [],
                "row_count": 0, "elapsed_ms": 0, "truncated": False,
                "error": "No tables could be loaded. Check your data sources and datasets.",
            }

        sql = body.sql.strip().rstrip(";")
        safe_sql = f"SELECT * FROM ({sql}) __q LIMIT {body.limit}"

        t0 = time.perf_counter()
        result = con.execute(safe_sql)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        columns = [d[0] for d in result.description]
        rows: list[list[Any]] = result.fetchall()
        con.close()

        return {
            "sql": body.sql,
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "elapsed_ms": elapsed_ms,
            "truncated": len(rows) >= body.limit,
            "registered_tables": registered,
        }

    except Exception as exc:
        return {
            "sql": body.sql, "columns": [], "rows": [],
            "row_count": 0, "elapsed_ms": 0, "truncated": False, "error": str(exc),
        }


# ── Explain ────────────────────────────────────────────────────────────────────

class WarehouseExplainRequest(BaseModel):
    sql: str


@router.post("/workspaces/{workspace_id}/warehouse/explain")
def explain_warehouse_sql(
    workspace_id: str,
    body: WarehouseExplainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    wid = _normalize_wid(workspace_id)
    datasets = _get_ready_datasets(wid, db)
    duckdb = _load_duckdb()
    try:
        con = duckdb.connect(":memory:")
        for ds in datasets:
            try:
                _register_file_view(con, _slugify(ds.name), ds.file_path)
            except Exception:
                pass
        result = con.execute(f"EXPLAIN {body.sql}")
        rows = result.fetchall()
        con.close()
        plan = "\n".join(str(r[1]) for r in rows if len(r) > 1)
        return {"plan": plan}
    except Exception as exc:
        return {"plan": "", "error": str(exc)}
