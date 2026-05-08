import time
import traceback
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


def _get_dataset_file(dataset_id: str, db: Session, user: User = None) -> str:
    try:
        did = int(dataset_id)
    except (ValueError, TypeError):
        did = dataset_id  # type: ignore[assignment]
    ds = db.query(Dataset).filter(Dataset.id == did).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not ds.file_path:
        raise HTTPException(status_code=400, detail="Dataset has no file attached")
    fp = ds.file_path.replace("\\", "/")
    return fp


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
    file_path = _get_dataset_file(dataset_id, db, current_user)

    try:
        con = duckdb.connect(database=":memory:")
        # Load dataset as a view named "df"
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else "csv"
        if ext in ("csv", "tsv"):
            con.execute(f"CREATE VIEW df AS SELECT * FROM read_csv_auto('{file_path}')")
        elif ext == "parquet":
            con.execute(f"CREATE VIEW df AS SELECT * FROM read_parquet('{file_path}')")
        elif ext in ("xls", "xlsx"):
            # DuckDB doesn't read Excel natively — load via pandas then register
            import pandas as pd
            import io
            pandas_df = pd.read_excel(file_path)
            con.register("df", pandas_df)
        elif ext == "json":
            con.execute(f"CREATE VIEW df AS SELECT * FROM read_json_auto('{file_path}')")
        else:
            con.execute(f"CREATE VIEW df AS SELECT * FROM read_csv_auto('{file_path}')")

        sql = body.sql.strip()
        # Apply limit if SELECT without existing LIMIT
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
    except Exception as exc:
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "elapsed_ms": 0,
            "truncated": False,
            "error": str(exc),
        }


@router.post("/datasets/{dataset_id}/sql/explain")
def explain_sql(
    dataset_id: str,
    body: SqlExplainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    duckdb = _load_duckdb()
    file_path = _get_dataset_file(dataset_id, db, current_user)

    try:
        con = duckdb.connect(database=":memory:")
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else "csv"
        if ext in ("csv", "tsv"):
            con.execute(f"CREATE VIEW df AS SELECT * FROM read_csv_auto('{file_path}')")
        elif ext == "parquet":
            con.execute(f"CREATE VIEW df AS SELECT * FROM read_parquet('{file_path}')")
        elif ext == "json":
            con.execute(f"CREATE VIEW df AS SELECT * FROM read_json_auto('{file_path}')")
        else:
            con.execute(f"CREATE VIEW df AS SELECT * FROM read_csv_auto('{file_path}')")

        result = con.execute(f"EXPLAIN {body.sql}")
        rows = result.fetchall()
        con.close()
        plan = "\n".join(str(r[1]) for r in rows if len(r) > 1)
        return {"plan": plan}
    except Exception as exc:
        return {"plan": "", "error": str(exc)}


@router.get("/datasets/{dataset_id}/sql/schema")
def get_schema(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    duckdb = _load_duckdb()
    file_path = _get_dataset_file(dataset_id, db, current_user)

    try:
        con = duckdb.connect(database=":memory:")
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else "csv"
        if ext in ("csv", "tsv"):
            con.execute(f"CREATE VIEW df AS SELECT * FROM read_csv_auto('{file_path}')")
        elif ext == "parquet":
            con.execute(f"CREATE VIEW df AS SELECT * FROM read_parquet('{file_path}')")
        elif ext in ("xls", "xlsx"):
            import pandas as pd
            pandas_df = pd.read_excel(file_path)
            con.register("df", pandas_df)
        elif ext == "json":
            con.execute(f"CREATE VIEW df AS SELECT * FROM read_json_auto('{file_path}')")
        else:
            con.execute(f"CREATE VIEW df AS SELECT * FROM read_csv_auto('{file_path}')")

        result = con.execute("DESCRIBE df")
        rows = result.fetchall()
        con.close()
        return {
            "table": "df",
            "columns": [{"name": r[0], "type": r[1]} for r in rows],
        }
    except Exception as exc:
        return {"table": "df", "columns": [], "error": str(exc)}
