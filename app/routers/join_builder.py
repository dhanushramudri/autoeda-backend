from typing import Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.dataset import Dataset
from ..auth import get_current_active_user
from ..models.user import User

router = APIRouter(tags=["join_builder"])


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
    source_key: str = ""        # primary condition (backwards compat)
    target_key: str = ""
    conditions: list[JoinCondition] = []  # compound ON conditions

    @field_validator("source", "target", mode="before")
    @classmethod
    def coerce_ids(cls, v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return v

    def all_conditions(self) -> list[JoinCondition]:
        """Merge primary key pair + extra conditions, dedup."""
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_duckdb():
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise HTTPException(status_code=503, detail="DuckDB not installed")


def _get_file_path(dataset_id: Union[str, int], db: Session) -> str:
    try:
        did = int(dataset_id)
    except (ValueError, TypeError):
        did = dataset_id  # type: ignore[assignment]
    ds = db.query(Dataset).filter(Dataset.id == did).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    if not ds.file_path:
        raise HTTPException(status_code=400, detail=f"Dataset {dataset_id} has no file attached")
    return ds.file_path.replace("\\", "/")


def _register_view(con, view_name: str, fp: str):
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


def _build_sql_and_maps(
    nodes: list[JoinNode],
    edges: list[JoinEdge],
    db: Session,
) -> tuple[str, dict, dict]:
    if not nodes:
        raise HTTPException(status_code=400, detail="No tables on canvas")

    alias_map: dict = {node.id: f"t{i}" for i, node in enumerate(nodes)}
    file_map: dict = {node.id: _get_file_path(node.id, db) for node in nodes}

    if not edges:
        node = nodes[0]
        alias = alias_map[node.id]
        return f"SELECT {alias}.* FROM df_{alias} AS {alias}", alias_map, file_map

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
        if conds:
            on_parts = [f"{src}.{c.source_key} = {tgt}.{c.target_key}" for c in conds]
            on_clause = "ON " + " AND ".join(on_parts)
        else:
            on_clause = "ON 1=1"
        join_clauses.append(f"{jtype} JOIN df_{tgt} AS {tgt} {on_clause}")

    sql = f"SELECT * FROM df_{base} AS {base}\n" + "\n".join(join_clauses)
    return sql, alias_map, file_map


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/workspaces/{workspace_id}/join-builder/generate-sql")
def generate_join_sql(
    workspace_id: str,
    body: GenerateSqlRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        sql, _, _ = _build_sql_and_maps(body.nodes, body.edges, db)
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
    duckdb = _load_duckdb()
    try:
        sql, alias_map, file_map = _build_sql_and_maps(body.nodes, body.edges, db)
        con = duckdb.connect(database=":memory:")
        for node in body.nodes:
            alias = alias_map[node.id]
            _register_view(con, f"df_{alias}", file_map[node.id])

        result = con.execute(f"SELECT * FROM ({sql}) __q LIMIT {body.limit}")
        columns = [d[0] for d in result.description]
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
        return {"sql": "", "columns": [], "rows": [], "row_count": 0, "truncated": False, "error": str(exc)}
