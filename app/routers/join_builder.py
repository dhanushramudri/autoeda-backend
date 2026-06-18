import hashlib
import io
from typing import Union

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from ..database import get_db
from ..dataset_access import dataset_visibility_filter
from ..models.dataset import Dataset
from ..models.job import BackgroundJob
from ..models.workspace import WorkspaceMember
from ..auth import get_current_active_user
from ..models.user import User

router = APIRouter(tags=["join_builder"])


def _assert_member(workspace_id: int, user: User, db: Session):
    if user.is_admin:
        return
    m = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    if not m:
        raise HTTPException(status_code=403, detail="Not a workspace member")


# ── Pydantic models ────────────────────────────────────────────────────────────

class JoinCondition(BaseModel):
    source_key: str
    target_key: str


class JoinNode(BaseModel):
    id: Union[str, int]
    label: str

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id(cls, v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return v


class JoinEdge(BaseModel):
    source: Union[str, int]
    target: Union[str, int]
    join_type: str = "INNER"
    source_key: str = ""
    target_key: str = ""
    conditions: list[JoinCondition] = []

    @field_validator("source", "target", mode="before")
    @classmethod
    def coerce_ids(cls, v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return v

    def all_conditions(self) -> list[JoinCondition]:
        seen: set[tuple[str, str]] = set()
        result: list[JoinCondition] = []
        if self.source_key and self.target_key:
            seen.add((self.source_key, self.target_key))
            result.append(JoinCondition(source_key=self.source_key, target_key=self.target_key))
        for c in self.conditions:
            if c.source_key and c.target_key:
                key = (c.source_key, c.target_key)
                if key not in seen:
                    seen.add(key)
                    result.append(c)
        return result


class GenerateSqlRequest(BaseModel):
    nodes: list[JoinNode]
    edges: list[JoinEdge]


class ExecuteJoinRequest(BaseModel):
    nodes: list[JoinNode]
    edges: list[JoinEdge]
    limit: int = 1000


class SaveJoinRequest(BaseModel):
    nodes: list[JoinNode]
    edges: list[JoinEdge]
    name: str
    workspace_id: int


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_duckdb():
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise HTTPException(status_code=503, detail="DuckDB not installed")


def _load_dataset_df(dataset_id: Union[str, int], db: Session, workspace_id: int):
    """Load a dataset as a pandas DataFrame — handles both disk and DB-stored files.

    Scoped to workspace_id (or the global shared-dataset account) so a node
    referencing a dataset from an unrelated workspace can't be used to pull
    or join in data the caller isn't a member of.
    """
    import os
    try:
        did = int(dataset_id)
    except (ValueError, TypeError):
        did = dataset_id  # type: ignore[assignment]

    ds = db.query(Dataset).filter(Dataset.id == did, dataset_visibility_filter(db, workspace_id)).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    import json
    from ..connectors.file_connector import load_from_bytes

    config = json.loads(ds.source_config or "{}")

    if ds.source_type == "file":
        if not ds.file_data:
            raise HTTPException(status_code=400, detail=f"Dataset {dataset_id} has no accessible file data")
        filename = os.path.basename(ds.file_path or "") if ds.file_path else ""
        # Use database bytes only (file-based data stored in DB)
        return load_from_bytes(ds.file_data, filename, config)
    else:
        # DB / API connectors — load via tasks helper
        from ..connectors.db_connector import DBConnector
        from ..connectors.api_connector import RESTAPIConnector
        from ..connectors.cloud_connector import CloudConnector
        src = ds.source_type
        if src in ("postgresql", "mysql", "sqlite", "mssql"):
            config["db_type"] = src
            return DBConnector().load_data(config)
        elif src == "mongodb":
            config["db_type"] = "mongodb"
            return DBConnector().load_data(config)
        elif src == "rest_api":
            return RESTAPIConnector().load_data(config)
        elif src in ("s3", "azure", "gcs"):
            config["cloud_type"] = src
            return CloudConnector().load_data(config)
        raise HTTPException(status_code=400, detail=f"Unsupported source type: {src}")


def _build_sql(nodes: list[JoinNode], edges: list[JoinEdge]) -> tuple[str, dict]:
    """Build SQL using stable aliases — no file paths needed (DuckDB DataFrames)."""
    if not nodes:
        raise HTTPException(status_code=400, detail="No tables on canvas")

    alias_map: dict = {node.id: f"t{i}" for i, node in enumerate(nodes)}

    if not edges:
        alias = alias_map[nodes[0].id]
        return f"SELECT {alias}.* FROM df_{alias} AS {alias}", alias_map

    first = edges[0]
    base = alias_map.get(first.source, "t0")
    join_clauses: list[str] = []

    for edge in edges:
        src = alias_map.get(edge.source, "")
        tgt = alias_map.get(edge.target, "")
        if not src or not tgt:
            continue
        jtype = edge.join_type.upper()
        if jtype not in ("INNER", "LEFT", "RIGHT", "FULL"):
            jtype = "INNER"
        conds = edge.all_conditions()
        on_clause = "ON " + " AND ".join(f"{src}.{c.source_key} = {tgt}.{c.target_key}" for c in conds) if conds else "ON 1=1"
        join_clauses.append(f"{jtype} JOIN df_{tgt} AS {tgt} {on_clause}")

    sql = f"SELECT * FROM df_{base} AS {base}\n" + "\n".join(join_clauses)
    return sql, alias_map


def _execute_sql(nodes: list[JoinNode], edges: list[JoinEdge], db: Session, workspace_id: int, limit: int = 0):
    """Load DataFrames, register with DuckDB, execute SQL. Returns (df_result, sql, columns, rows)."""
    duckdb = _load_duckdb()
    sql, alias_map = _build_sql(nodes, edges)

    con = duckdb.connect(database=":memory:")
    try:
        for node in nodes:
            alias = alias_map[node.id]
            df = _load_dataset_df(node.id, db, workspace_id)
            con.register(f"df_{alias}", df)

        query = f"SELECT * FROM ({sql}) __q" + (f" LIMIT {limit}" if limit > 0 else "")
        result = con.execute(query)
        columns = [d[0] for d in result.description]
        rows = result.fetchall()
        return rows, columns, sql
    finally:
        con.close()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/workspaces/{workspace_id}/join-builder/generate-sql")
def generate_join_sql(
    workspace_id: str,
    body: GenerateSqlRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(int(workspace_id), current_user, db)
    try:
        sql, _ = _build_sql(body.nodes, body.edges)
        return {"sql": sql}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/workspaces/{workspace_id}/join-builder/execute")
def execute_join(
    workspace_id: str,
    body: ExecuteJoinRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    wid = int(workspace_id)
    _assert_member(wid, current_user, db)
    try:
        rows, columns, sql = _execute_sql(body.nodes, body.edges, db, wid, limit=body.limit)
        return {
            "sql": sql,
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "truncated": body.limit > 0 and len(rows) >= body.limit,
        }
    except HTTPException:
        raise
    except Exception as exc:
        return {"sql": "", "columns": [], "rows": [], "row_count": 0, "truncated": False, "error": str(exc)}


@router.post("/workspaces/{workspace_id}/join-builder/save-as-dataset", status_code=201)
def save_join_as_dataset(
    workspace_id: str,
    body: SaveJoinRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Execute the join query, save the full result as a new dataset, trigger EDA."""
    import uuid
    import json
    import pandas as pd

    wid = int(workspace_id)
    _assert_member(wid, current_user, db)
    try:
        rows, columns, sql = _execute_sql(body.nodes, body.edges, db, wid, limit=0)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Join execution failed: {exc}")

    # Convert to DataFrame then CSV bytes
    df = pd.DataFrame(rows, columns=columns)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue()
    content_hash = hashlib.md5(csv_bytes).hexdigest()
    filename = f"{body.name.replace(' ', '_')}.csv"

    # Create dataset record
    ds = Dataset(
        workspace_id=wid,
        name=body.name,
        description=f"Joined dataset — {len(body.nodes)} tables, {len(body.edges)} join(s)",
        source_type="file",
        source_config=json.dumps({"origin": "join_builder", "sql": sql}),
        file_path=filename,
        file_data=csv_bytes,
        file_size_bytes=len(csv_bytes),
        content_hash=content_hash,
        status="processing",
        created_by=current_user.id,
    )
    db.add(ds)
    db.flush()

    job_id = str(uuid.uuid4())
    job = BackgroundJob(
        id=job_id,
        job_type="eda_pipeline",
        status="pending",
        progress=0,
        dataset_id=ds.id,
        created_by=current_user.id,
    )
    db.add(job)
    db.commit()
    db.refresh(ds)

    from ..tasks import run_eda_pipeline
    background_tasks.add_task(run_eda_pipeline, job_id, ds.id, None, {})

    return {"dataset_id": ds.id, "name": ds.name, "job_id": job_id}
