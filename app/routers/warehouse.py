import re
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..auth import get_current_active_user
from ..models.user import User
from ..models.workspace import WorkspaceMember
from ..models.dataset import Dataset
from ..models.data_source import DataSource
from ..connectors.registry import get_connector

router = APIRouter(tags=["warehouse"])


def _assert_member(workspace_id: int, user: User, db: Session):
    if user.is_admin:
        return
    m = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    if not m:
        raise HTTPException(status_code=403, detail="Not a workspace member")


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


def _register_dataset(con, view_name: str, ds: Dataset):
    """Register a dataset as a DuckDB view — always loads from DB bytes, never disk."""
    import os
    from ..connectors.file_connector import load_from_bytes
    import json

    config = json.loads(ds.source_config or "{}")
    
    if not ds.file_data:
        raise ValueError(f"Dataset '{ds.name}' has no file data in database")
    
    filename = os.path.basename(ds.file_path or "") if ds.file_path else ""
    # Use database bytes only (file-based data stored in DB)
    df = load_from_bytes(ds.file_data, filename, config)
    con.register(view_name, df)


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


def _get_ready_datasets(wid: int, db: Session, load_data: bool = False) -> list[Dataset]:
    """
    Load file datasets that are ready.
    `load_data=False` defers the heavy file_data blob — use for catalog / slug matching.
    `load_data=True`  fetches everything — use only after you know which rows you need.
    """
    from sqlalchemy import or_
    from sqlalchemy.orm import defer as sa_defer

    q = (
        db.query(Dataset)
        .filter(
            Dataset.workspace_id == wid,
            Dataset.status == "ready",
            or_(Dataset.file_data.isnot(None), Dataset.file_path.isnot(None)),
        )
        .order_by(Dataset.name)
    )
    if not load_data:
        q = q.options(sa_defer(Dataset.file_data))
    return q.all()


def _columns_from_schema_info(ds: Dataset) -> list[dict]:
    """Extract column list from the schema_info JSON stored by the EDA pipeline — zero I/O."""
    if not ds.schema_info:
        return []
    import json
    try:
        schema = json.loads(ds.schema_info)
        return [{"name": col, "type": dtype} for col, dtype in schema.items()]
    except Exception:
        return []


def _slugs_referenced(sql: str, slugs: list[str]) -> set[str]:
    """Return the subset of slugs that appear as identifiers in the SQL."""
    return {s for s in slugs if re.search(r"\b" + re.escape(s) + r"\b", sql, re.IGNORECASE)}


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
    _assert_member(wid, current_user, db)

    sections = []

    # ── Section 1: Uploaded datasets ─────────────────────────────────────────
    # Defer file_data — schema_info has all column info we need, no file I/O.
    datasets = _get_ready_datasets(wid, db, load_data=False)

    workspace_dataset_items = []
    source_dataset_map: dict[int, list] = {}

    for ds in datasets:
        slug = _slugify(ds.name)
        columns = _columns_from_schema_info(ds)

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
    _assert_member(wid, current_user, db)
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
    _assert_member(wid, current_user, db)
    duckdb = _load_duckdb()

    # Step 1: load metadata only (no file_data blob) to compute slugs.
    all_datasets = _get_ready_datasets(wid, db, load_data=False)
    sources = _get_workspace_sources(wid, db)

    # Step 2: find which dataset slugs appear in the SQL.
    slug_to_id = {_slugify(ds.name): ds.id for ds in all_datasets}
    needed_slugs = _slugs_referenced(body.sql, list(slug_to_id.keys()))

    # Step 3: fetch only the needed datasets WITH file_data.
    needed_ids = [slug_to_id[s] for s in needed_slugs]
    if needed_ids:
        datasets_with_data = (
            db.query(Dataset).filter(Dataset.id.in_(needed_ids)).all()
        )
        id_to_ds = {ds.id: ds for ds in datasets_with_data}
    else:
        id_to_ds = {}

    try:
        con = duckdb.connect(":memory:")
        registered: list[str] = []

        # 4. Register only the referenced datasets.
        for slug, ds_id in slug_to_id.items():
            if slug not in needed_slugs:
                continue
            ds = id_to_ds.get(ds_id)
            if not ds:
                continue
            try:
                _register_dataset(con, slug, ds)
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

        # Split on semicolons so multi-statement SQL works correctly.
        # Execute each statement; return the last one that produces rows.
        statements = [s.strip() for s in body.sql.split(";") if s.strip()]

        columns: list[str] = []
        rows: list[list[Any]] = []
        elapsed_ms = 0

        t0 = time.perf_counter()
        for stmt in statements:
            result = con.execute(stmt)
            if result.description:
                columns = [d[0] for d in result.description]
                fetched = result.fetchall()
                # Apply limit to the last SELECT result
                rows = [list(r) for r in fetched[:body.limit]]
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        con.close()

        return {
            "sql": body.sql,
            "columns": columns,
            "rows": rows,
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
    _assert_member(wid, current_user, db)
    duckdb = _load_duckdb()
    all_datasets = _get_ready_datasets(wid, db, load_data=False)
    slug_to_id = {_slugify(ds.name): ds.id for ds in all_datasets}
    needed_slugs = _slugs_referenced(body.sql, list(slug_to_id.keys()))
    needed_ids = [slug_to_id[s] for s in needed_slugs]
    datasets_with_data = db.query(Dataset).filter(Dataset.id.in_(needed_ids)).all() if needed_ids else []
    id_to_ds = {ds.id: ds for ds in datasets_with_data}
    try:
        con = duckdb.connect(":memory:")
        for slug, ds_id in slug_to_id.items():
            if slug not in needed_slugs:
                continue
            ds = id_to_ds.get(ds_id)
            if ds:
                try:
                    _register_dataset(con, slug, ds)
                except Exception:
                    pass
        result = con.execute(f"EXPLAIN {body.sql}")
        rows = result.fetchall()
        con.close()
        plan = "\n".join(str(r[1]) for r in rows if len(r) > 1)
        return {"plan": plan}
    except Exception as exc:
        return {"plan": "", "error": str(exc)}
