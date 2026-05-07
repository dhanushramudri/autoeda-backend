from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.dataset import Dataset
from ..routers.auth import get_current_user
from ..models.user import User

router = APIRouter(tags=["join_builder"])


class JoinEdge(BaseModel):
    source: str           # node id (dataset_id)
    target: str           # node id (dataset_id)
    join_type: str = "INNER"   # INNER | LEFT | RIGHT | FULL
    source_key: str
    target_key: str


class JoinNode(BaseModel):
    id: str               # dataset_id
    label: str            # dataset name


class GenerateSqlRequest(BaseModel):
    nodes: list[JoinNode]
    edges: list[JoinEdge]


class ExecuteJoinRequest(BaseModel):
    nodes: list[JoinNode]
    edges: list[JoinEdge]
    limit: int = 500


def _load_duckdb():
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise HTTPException(status_code=503, detail="DuckDB not installed")


def _get_file_path(dataset_id: str, db: Session) -> str:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return ds.file_path


def _generate_sql(nodes: list[JoinNode], edges: list[JoinEdge], db: Session) -> tuple[str, dict[str, str]]:
    """Build a SQL SELECT from the node/edge graph. Returns (sql, {node_id: file_path})."""
    if not nodes:
        raise HTTPException(status_code=400, detail="No nodes provided")
    if not edges:
        # Single table, no joins
        node = nodes[0]
        fp = _get_file_path(node.id, db)
        alias = f"t0"
        return f"SELECT {alias}.* FROM df_{alias} AS {alias}", {node.id: fp}

    # Map node id → alias
    alias_map: dict[str, str] = {}
    file_map: dict[str, str] = {}
    for i, node in enumerate(nodes):
        alias = f"t{i}"
        alias_map[node.id] = alias
        file_map[node.id] = _get_file_path(node.id, db)

    # Build FROM clause starting from the first edge's source
    first_edge = edges[0]
    base_alias = alias_map[first_edge.source]

    from_clause = f"df_{base_alias} AS {base_alias}"
    joined = {first_edge.source}

    join_parts: list[str] = []
    for edge in edges:
        src_alias = alias_map.get(edge.source, "")
        tgt_alias = alias_map.get(edge.target, "")
        if not src_alias or not tgt_alias:
            continue
        jtype = edge.join_type.upper()
        if jtype not in ("INNER", "LEFT", "RIGHT", "FULL"):
            jtype = "INNER"
        join_parts.append(
            f"{jtype} JOIN df_{tgt_alias} AS {tgt_alias} "
            f"ON {src_alias}.{edge.source_key} = {tgt_alias}.{edge.target_key}"
        )

    sql = f"SELECT * FROM {from_clause}\n" + "\n".join(join_parts)
    return sql, file_map


@router.post("/workspaces/{workspace_id}/join-builder/generate-sql")
def generate_join_sql(
    workspace_id: str,
    body: GenerateSqlRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        sql, _ = _generate_sql(body.nodes, body.edges, db)
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
    current_user: User = Depends(get_current_user),
):
    duckdb = _load_duckdb()

    try:
        sql, file_map = _generate_sql(body.nodes, body.edges, db)

        con = duckdb.connect(database=":memory:")

        # Register each dataset as a view df_t0, df_t1, …
        alias_map: dict[str, str] = {}
        for i, node in enumerate(body.nodes):
            alias = f"t{i}"
            alias_map[node.id] = alias
            fp = file_map[node.id]
            ext = fp.rsplit(".", 1)[-1].lower() if "." in fp else "csv"
            view_name = f"df_{alias}"
            if ext in ("csv", "tsv"):
                con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_csv_auto('{fp}')")
            elif ext == "parquet":
                con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{fp}')")
            elif ext in ("xls", "xlsx"):
                import pandas as pd
                pandas_df = pd.read_excel(fp)
                con.register(view_name, pandas_df)
            elif ext == "json":
                con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_json_auto('{fp}')")
            else:
                con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_csv_auto('{fp}')")

        # Run with limit
        limited_sql = f"SELECT * FROM ({sql}) __join_result LIMIT {body.limit}"
        result = con.execute(limited_sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        con.close()

        return {
            "sql": sql,
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "truncated": len(rows) >= body.limit,
        }
    except HTTPException:
        raise
    except Exception as exc:
        return {
            "sql": "",
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": str(exc),
        }
