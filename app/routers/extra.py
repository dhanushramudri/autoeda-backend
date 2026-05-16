"""
Extra endpoints: NL Query, Pipeline, Filter Preview, Column Metadata,
Quality Rules, Pivot, Dataset Join, Report, History, Analytics,
Saved Charts, Named Segments, Column Detail.
"""
import json
import re
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..database import get_db
from ..models.column_metadata import ColumnMetadata
from ..models.data_quality_rule import DataQualityRule
from ..models.dataset import Dataset, EDAResult
from ..models.eda_run import EDARunRecord
from ..models.named_segment import NamedSegment
from ..models.pipeline_step import PipelineStep
from ..models.saved_chart import SavedChart
from ..models.user import User
from ..models.workspace import WorkspaceMember, Workspace
from ..ai.nl_query import parse_nl_query_ai as parse_nl_query

router = APIRouter(tags=["extra"])


# ── Auth helpers ─────────────────────────────────────────────────────────────

def _get_ds(dataset_id: int, user: User, db: Session) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    if not user.is_admin:
        m = db.query(WorkspaceMember).filter(
            WorkspaceMember.workspace_id == ds.workspace_id,
            WorkspaceMember.user_id == user.id,
        ).first()
        if not m:
            raise HTTPException(403, "Access denied")
    return ds


def _assert_ws_member(workspace_id: int, user: User, db: Session):
    if user.is_admin:
        return
    m = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    if not m:
        raise HTTPException(403, "Not a workspace member")


def _load_df(ds: Dataset) -> pd.DataFrame:
    import json as _json
    from ..connectors.file_connector import FileConnector
    from ..connectors.db_connector import DBConnector
    from ..connectors.api_connector import RESTAPIConnector
    from ..connectors.cloud_connector import CloudConnector

    config = _json.loads(ds.source_config or "{}")
    if ds.source_type == "file":
        config["file_path"] = ds.file_path
        return FileConnector().load_data(config)
    elif ds.source_type in ("postgresql", "mysql", "sqlite", "mssql", "mongodb"):
        config["db_type"] = ds.source_type
        return DBConnector().load_data(config)
    elif ds.source_type == "rest_api":
        return RESTAPIConnector().load_data(config)
    elif ds.source_type in ("s3", "azure", "gcs"):
        config["cloud_type"] = ds.source_type
        return CloudConnector().load_data(config)
    raise ValueError(f"Unsupported source_type: {ds.source_type}")


def _apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    for f in filters:
        col = f.get("column")
        op = f.get("operator", "equals")
        val = f.get("value")
        if col not in df.columns:
            continue
        try:
            if op == "equals":
                df = df[df[col].astype(str) == str(val)]
            elif op == "not_equals":
                df = df[df[col].astype(str) != str(val)]
            elif op == "gt":
                df = df[pd.to_numeric(df[col], errors="coerce") > float(val)]
            elif op == "gte":
                df = df[pd.to_numeric(df[col], errors="coerce") >= float(val)]
            elif op == "lt":
                df = df[pd.to_numeric(df[col], errors="coerce") < float(val)]
            elif op == "lte":
                df = df[pd.to_numeric(df[col], errors="coerce") <= float(val)]
            elif op == "contains":
                df = df[df[col].astype(str).str.contains(str(val), na=False)]
            elif op == "starts_with":
                df = df[df[col].astype(str).str.startswith(str(val), na=False)]
            elif op == "is_null":
                df = df[df[col].isna()]
            elif op == "is_not_null":
                df = df[df[col].notna()]
        except Exception:
            pass
    return df


def _apply_pipeline(df: pd.DataFrame, steps: list[dict]) -> pd.DataFrame:
    for step in steps:
        op = step.get("operation")
        col = step.get("column")
        params = step.get("params") or {}

        if op == "drop" and col and col in df.columns:
            df = df.drop(columns=[col])
        elif op == "rename" and col and col in df.columns:
            new_name = params.get("new_name", col)
            df = df.rename(columns={col: new_name})
        elif op == "fill_missing" and col and col in df.columns:
            method = params.get("method", "mean")
            if method == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            elif method == "median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif method == "mode":
                mode_vals = df[col].mode()
                if len(mode_vals):
                    df[col] = df[col].fillna(mode_vals[0])
            elif method == "custom":
                df[col] = df[col].fillna(params.get("fill_value", ""))
            elif method == "forward":
                df[col] = df[col].ffill()
            elif method == "backward":
                df[col] = df[col].bfill()
        elif op == "cast_type" and col and col in df.columns:
            try:
                df[col] = df[col].astype(params.get("dtype", "str"))
            except Exception:
                pass
        elif op == "drop_high_missing":
            threshold = params.get("threshold", 50)
            to_drop = [
                c for c in df.columns
                if df[c].isna().mean() * 100 > threshold
            ]
            df = df.drop(columns=to_drop)
        elif op == "normalize" and col and col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                mn, mx = df[col].min(), df[col].max()
                if mx != mn:
                    df[col] = (df[col] - mn) / (mx - mn)
        elif op == "standardize" and col and col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                mu, sigma = df[col].mean(), df[col].std()
                if sigma:
                    df[col] = (df[col] - mu) / sigma
        elif op == "log_transform" and col and col in df.columns:
            import numpy as np
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].apply(lambda x: np.log1p(x) if x >= 0 else x)
        elif op == "one_hot_encode" and col and col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col).astype(int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        elif op == "label_encode" and col and col in df.columns:
            categories = df[col].dropna().unique().tolist()
            mapping = {v: i for i, v in enumerate(sorted(str(x) for x in categories))}
            df[col] = df[col].astype(str).map(mapping)
        elif op == "filter_rows":
            filters_list = params.get("filters", [])
            df = _apply_filters(df, filters_list)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# NL QUERY
# ══════════════════════════════════════════════════════════════════════════════

class NLQueryRequest(BaseModel):
    query: str


@router.post("/datasets/{dataset_id}/nl-query")
def nl_query(
    dataset_id: int,
    body: NLQueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    schema = json.loads(ds.schema_info or "{}")
    columns = list(schema.get("dtypes", {}).keys()) if schema else []
    result = parse_nl_query(body.query, columns)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FILTER PREVIEW
# ══════════════════════════════════════════════════════════════════════════════

class FilterPreviewRequest(BaseModel):
    filters: list[dict[str, Any]]
    limit: int = 100


@router.post("/datasets/{dataset_id}/filter-preview")
def filter_preview(
    dataset_id: int,
    body: FilterPreviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    try:
        df = _load_df(ds)
        filtered = _apply_filters(df, body.filters)
        rows = filtered.head(body.limit).fillna("").to_dict(orient="records")
        return {"rows": rows, "total_matching": len(filtered), "total_rows": len(df)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class PipelineStepIn(BaseModel):
    operation: str
    column: Optional[str] = None
    params: Optional[dict[str, Any]] = None


class PipelineRequest(BaseModel):
    steps: list[PipelineStepIn]


@router.get("/datasets/{dataset_id}/pipeline")
def get_pipeline(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _get_ds(dataset_id, current_user, db)
    steps = (
        db.query(PipelineStep)
        .filter(PipelineStep.dataset_id == dataset_id)
        .order_by(PipelineStep.step_order)
        .all()
    )
    return {
        "steps": [
            {
                "id": s.id,
                "step_order": s.step_order,
                "operation": s.operation,
                "column": s.column,
                "params": json.loads(s.params_json or "{}"),
            }
            for s in steps
        ]
    }


@router.post("/datasets/{dataset_id}/pipeline")
def run_pipeline(
    dataset_id: int,
    body: PipelineRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    try:
        # Persist steps
        db.query(PipelineStep).filter(PipelineStep.dataset_id == dataset_id).delete()
        for i, step in enumerate(body.steps):
            ps = PipelineStep(
                dataset_id=dataset_id,
                step_order=i,
                operation=step.operation,
                column=step.column,
                params_json=json.dumps(step.params or {}),
            )
            db.add(ps)
        db.commit()

        # Apply pipeline
        df = _load_df(ds)
        steps_dicts = [
            {"operation": s.operation, "column": s.column, "params": s.params or {}}
            for s in body.steps
        ]
        result_df = _apply_pipeline(df, steps_dicts)
        preview = result_df.head(50).fillna("").to_dict(orient="records")
        return {
            "applied": len(body.steps),
            "result_preview": preview,
            "row_count": len(result_df),
            "column_count": len(result_df.columns),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/datasets/{dataset_id}/pipeline")
def clear_pipeline(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _get_ds(dataset_id, current_user, db)
    db.query(PipelineStep).filter(PipelineStep.dataset_id == dataset_id).delete()
    db.commit()
    return {"cleared": True}


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN METADATA (tags + notes)
# ══════════════════════════════════════════════════════════════════════════════

class ColumnMetadataRequest(BaseModel):
    tags: list[str] = []
    notes: Optional[str] = None


@router.get("/datasets/{dataset_id}/columns/metadata")
def get_all_column_metadata(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _get_ds(dataset_id, current_user, db)
    rows = db.query(ColumnMetadata).filter(ColumnMetadata.dataset_id == dataset_id).all()
    return [
        {
            "column": r.column_name,
            "tags": json.loads(r.tags_json or "[]"),
            "notes": r.notes,
        }
        for r in rows
    ]


@router.put("/datasets/{dataset_id}/columns/{column_name}/metadata")
def upsert_column_metadata(
    dataset_id: int,
    column_name: str,
    body: ColumnMetadataRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _get_ds(dataset_id, current_user, db)
    row = db.query(ColumnMetadata).filter(
        ColumnMetadata.dataset_id == dataset_id,
        ColumnMetadata.column_name == column_name,
    ).first()
    if row:
        row.tags_json = json.dumps(body.tags)
        row.notes = body.notes
    else:
        row = ColumnMetadata(
            dataset_id=dataset_id,
            column_name=column_name,
            tags_json=json.dumps(body.tags),
            notes=body.notes,
        )
        db.add(row)
    db.commit()
    return {"column": column_name, "tags": body.tags, "notes": body.notes}


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY RULES
# ══════════════════════════════════════════════════════════════════════════════

class RuleIn(BaseModel):
    column_name: Optional[str] = None
    rule_type: str  # not_null | range | regex | unique | allowed_values
    params: dict[str, Any] = {}


class RulesRequest(BaseModel):
    rules: list[RuleIn]


@router.post("/datasets/{dataset_id}/rules")
def save_rules(
    dataset_id: int,
    body: RulesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _get_ds(dataset_id, current_user, db)
    db.query(DataQualityRule).filter(DataQualityRule.dataset_id == dataset_id).delete()
    for rule in body.rules:
        r = DataQualityRule(
            dataset_id=dataset_id,
            column_name=rule.column_name,
            rule_type=rule.rule_type,
            params_json=json.dumps(rule.params),
        )
        db.add(r)
    db.commit()
    return {"saved": len(body.rules)}


@router.get("/datasets/{dataset_id}/rules/results")
def get_rule_results(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    rules = db.query(DataQualityRule).filter(DataQualityRule.dataset_id == dataset_id).all()
    if not rules:
        return {"rules": [], "pass_rate": 1.0}

    try:
        df = _load_df(ds)
    except Exception as e:
        raise HTTPException(500, str(e))

    results = []
    for rule in rules:
        col = rule.column_name
        params = json.loads(rule.params_json or "{}")
        total = len(df)
        fail_mask = pd.Series([False] * total, index=df.index)

        if rule.rule_type == "not_null" and col and col in df.columns:
            fail_mask = df[col].isna()
        elif rule.rule_type == "range" and col and col in df.columns:
            lo, hi = params.get("min"), params.get("max")
            numeric = pd.to_numeric(df[col], errors="coerce")
            if lo is not None:
                fail_mask |= numeric < float(lo)
            if hi is not None:
                fail_mask |= numeric > float(hi)
        elif rule.rule_type == "regex" and col and col in df.columns:
            pattern = params.get("pattern", "")
            try:
                fail_mask = ~df[col].astype(str).str.match(pattern, na=False)
            except re.error:
                pass
        elif rule.rule_type == "unique" and col and col in df.columns:
            fail_mask = df[col].duplicated(keep=False)
        elif rule.rule_type == "allowed_values" and col and col in df.columns:
            allowed = set(str(v) for v in params.get("values", []))
            fail_mask = ~df[col].astype(str).isin(allowed) & df[col].notna()

        fail_count = int(fail_mask.sum())
        fail_pct = round(fail_count / total * 100, 2) if total else 0
        pass_pct = round(100 - fail_pct, 2)
        sample_rows = (
            df[fail_mask].head(5).fillna("").to_dict(orient="records")
            if fail_count else []
        )
        label = (
            f"{col} — {rule.rule_type}" if col else rule.rule_type
        )
        results.append({
            "id": rule.id,
            "column": col,
            "rule_type": rule.rule_type,
            "params": params,
            "label": label,
            "pass_pct": pass_pct,
            "fail_count": fail_count,
            "fail_pct": fail_pct,
            "sample_failing_rows": sample_rows,
        })

    pass_rate = (
        sum(r["pass_pct"] for r in results) / len(results) / 100 if results else 1.0
    )
    return {"rules": results, "pass_rate": round(pass_rate, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# PIVOT TABLE
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/datasets/{dataset_id}/pivot")
def get_pivot(
    dataset_id: int,
    row_col: str,
    col_col: str,
    value_col: str,
    agg_func: str = "sum",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    agg_map = {
        "sum": "sum", "mean": "mean", "count": "count",
        "min": "min", "max": "max",
    }
    agg = agg_map.get(agg_func, "sum")
    try:
        df = _load_df(ds)
        for c in [row_col, col_col, value_col]:
            if c not in df.columns:
                raise HTTPException(400, f"Column '{c}' not found")

        pivot = df.pivot_table(
            index=row_col,
            columns=col_col,
            values=value_col,
            aggfunc=agg,
            fill_value=0,
        )
        return {
            "index": [str(x) for x in pivot.index.tolist()],
            "columns": [str(x) for x in pivot.columns.tolist()],
            "data": pivot.values.tolist(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# DATASET JOIN
# ══════════════════════════════════════════════════════════════════════════════

class JoinRequest(BaseModel):
    dataset_a_id: int
    dataset_b_id: int
    join_type: str = "inner"  # inner | left | right | outer
    keys_a: list[str]
    keys_b: list[str]
    name: Optional[str] = None


@router.post("/workspaces/{workspace_id}/datasets/join")
def join_datasets(
    workspace_id: int,
    body: JoinRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_ws_member(workspace_id, current_user, db)
    ds_a = _get_ds(body.dataset_a_id, current_user, db)
    ds_b = _get_ds(body.dataset_b_id, current_user, db)

    join_map = {"inner": "inner", "left": "left", "right": "right", "outer": "outer", "full": "outer"}
    how = join_map.get(body.join_type, "inner")

    try:
        df_a = _load_df(ds_a)
        df_b = _load_df(ds_b)

        if len(body.keys_a) == 1 and len(body.keys_b) == 1:
            merged = df_a.merge(
                df_b,
                left_on=body.keys_a[0],
                right_on=body.keys_b[0],
                how=how,
                suffixes=("_a", "_b"),
            )
        else:
            merged = df_a.merge(
                df_b,
                left_on=body.keys_a,
                right_on=body.keys_b,
                how=how,
                suffixes=("_a", "_b"),
            )

        # Save as new CSV
        import tempfile, hashlib, os
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv",
            dir=str(ds_a.file_path).rsplit("/", 1)[0] if ds_a.file_path else None,
        ) as tmp:
            merged.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            content_hash = hashlib.md5(f.read()).hexdigest()

        new_name = body.name or f"Join_{ds_a.name}_{ds_b.name}"
        new_ds = Dataset(
            workspace_id=workspace_id,
            name=new_name,
            description=f"Joined: {ds_a.name} + {ds_b.name} ({how})",
            source_type="file",
            file_path=tmp_path,
            content_hash=content_hash,
            status="ready",
            row_count=len(merged),
            column_count=len(merged.columns),
            created_by=current_user.id,
        )
        db.add(new_ds)
        db.commit()
        db.refresh(new_ds)

        from ..schemas.dataset import DatasetResponse
        return DatasetResponse.model_validate(new_ds)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/datasets/{dataset_id}/report", response_class=HTMLResponse)
def generate_report(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    try:
        df = _load_df(ds)
        from ..eda.profiler import run_profile
        from ..eda.quality_score import run_quality_score
        from ..eda.missing import run_missing_analysis
        from ..eda.correlations import run_correlations
        from ..insights import InsightEngine

        profile = run_profile(df)
        quality = run_quality_score(df)
        missing = run_missing_analysis(df)
        correlations = run_correlations(df)

        engine = InsightEngine()
        insights = (
            engine.from_profile(profile)
            + engine.from_correlations(correlations)
            + engine.from_quality_score(quality)
        )

        sev_colors = {"info": "#3b82f6", "warning": "#f59e0b", "danger": "#ef4444"}
        insights_html = "".join(
            f'<div style="border-left:4px solid {sev_colors.get(i["severity"],"#94a3b8")};'
            f'padding:8px 12px;margin:6px 0;background:#f8fafc;border-radius:0 6px 6px 0;">'
            f'<span style="font-size:11px;color:#64748b;text-transform:uppercase;font-weight:600;">'
            f'{i["chart_type"]}</span><br>'
            f'<span style="font-size:13px;color:#1e293b;">{i["insight"]}</span></div>'
            for i in insights[:30]
        )

        col_rows = "".join(
            f"<tr><td>{c['name']}</td><td>{c['dtype']}</td>"
            f"<td>{c['semantic_type']}</td>"
            f"<td>{c['missing_pct']:.1f}%</td>"
            f"<td>{c['unique_count']}</td>"
            f"<td>{c.get('mean', '—') if c.get('mean') is not None else '—'}</td></tr>"
            for c in profile.get("columns", [])
        )

        missing_rows = "".join(
            f"<tr><td>{c['name']}</td><td>{c['count']}</td>"
            f"<td>{c['pct']:.1f}%</td></tr>"
            for c in missing.get("columns", []) if c.get("pct", 0) > 0
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AutoEDA Report — {ds.name}</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         margin:0;padding:0;background:#f1f5f9;color:#1e293b;}}
  .header {{background:linear-gradient(135deg,#1e40af,#3b82f6);
            color:#fff;padding:40px 48px;}}
  .header h1 {{margin:0 0 4px;font-size:28px;}}
  .header p {{margin:0;opacity:.8;font-size:14px;}}
  .container {{max-width:1100px;margin:32px auto;padding:0 24px;}}
  .section {{background:#fff;border-radius:12px;padding:28px;
             margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,.08);}}
  .section h2 {{margin:0 0 20px;font-size:18px;color:#1e293b;
               padding-bottom:12px;border-bottom:1px solid #e2e8f0;}}
  .grid-4 {{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;}}
  .stat {{background:#f8fafc;border-radius:8px;padding:16px;text-align:center;}}
  .stat .val {{font-size:28px;font-weight:700;color:#2563eb;}}
  .stat .lbl {{font-size:12px;color:#64748b;margin-top:4px;}}
  .score-ring {{display:inline-block;width:80px;height:80px;
               border-radius:50%;border:6px solid;line-height:68px;
               text-align:center;font-size:22px;font-weight:700;}}
  table {{width:100%;border-collapse:collapse;font-size:13px;}}
  th {{background:#f8fafc;padding:10px 12px;text-align:left;
       font-weight:600;color:#475569;border-bottom:1px solid #e2e8f0;}}
  td {{padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#334155;}}
  tr:hover td {{background:#f8fafc;}}
  .badge {{display:inline-block;padding:2px 8px;border-radius:999px;
           font-size:11px;font-weight:600;}}
  .quality-grid {{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;}}
  .q-item {{text-align:center;background:#f8fafc;border-radius:8px;padding:12px;}}
  .q-val {{font-size:20px;font-weight:700;}}
  .q-lbl {{font-size:11px;color:#64748b;margin-top:2px;}}
  .footer {{text-align:center;padding:24px;color:#94a3b8;font-size:12px;}}
</style>
</head>
<body>
<div class="header">
  <h1>AutoEDA Report</h1>
  <p>{ds.name} &nbsp;·&nbsp; Generated by Jman Group AutoEDA Platform</p>
</div>
<div class="container">

  <!-- Dataset Stats -->
  <div class="section">
    <h2>Dataset Overview</h2>
    <div class="grid-4">
      <div class="stat"><div class="val">{profile.get("total_rows",0):,}</div><div class="lbl">Rows</div></div>
      <div class="stat"><div class="val">{profile.get("total_columns",0)}</div><div class="lbl">Columns</div></div>
      <div class="stat"><div class="val">{profile.get("duplicate_pct",0):.1f}%</div><div class="lbl">Duplicate Rows</div></div>
      <div class="stat"><div class="val">{profile.get("memory_mb",0):.1f} MB</div><div class="lbl">Memory</div></div>
    </div>
  </div>

  <!-- Quality Score -->
  <div class="section">
    <h2>Data Quality Score</h2>
    <div class="quality-grid">
      <div class="q-item">
        <div class="q-val" style="color:{'#22c55e' if quality.get('overall',0)>=80 else '#f59e0b' if quality.get('overall',0)>=60 else '#ef4444'}">
          {quality.get("overall",0)}
        </div><div class="q-lbl">Overall</div>
      </div>
      <div class="q-item"><div class="q-val">{quality.get("completeness",0)}</div><div class="q-lbl">Completeness</div></div>
      <div class="q-item"><div class="q-val">{quality.get("consistency",0)}</div><div class="q-lbl">Consistency</div></div>
      <div class="q-item"><div class="q-val">{quality.get("uniqueness",0)}</div><div class="q-lbl">Uniqueness</div></div>
      <div class="q-item"><div class="q-val">{quality.get("validity",0)}</div><div class="q-lbl">Validity</div></div>
    </div>
  </div>

  <!-- Insights -->
  <div class="section">
    <h2>Smart Insights ({len(insights)} total)</h2>
    {insights_html}
  </div>

  <!-- Column Profile -->
  <div class="section">
    <h2>Column Profile</h2>
    <table>
      <thead><tr><th>Column</th><th>Dtype</th><th>Semantic Type</th>
      <th>Missing %</th><th>Unique</th><th>Mean</th></tr></thead>
      <tbody>{col_rows}</tbody>
    </table>
  </div>

  <!-- Missing Values -->
  <div class="section">
    <h2>Missing Values</h2>
    {'<p style="color:#22c55e;font-weight:600;">No missing values detected.</p>' if not missing_rows else
     f'<table><thead><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr></thead><tbody>{missing_rows}</tbody></table>'}
  </div>

</div>
<div class="footer">
  Jman Group AutoEDA Platform &nbsp;·&nbsp; {ds.name} &nbsp;·&nbsp;
  Report generated at {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC")}
</div>
</body>
</html>"""
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# EDA HISTORY
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/datasets/{dataset_id}/history")
def get_eda_history(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _get_ds(dataset_id, current_user, db)
    runs = (
        db.query(EDARunRecord)
        .filter(EDARunRecord.dataset_id == dataset_id)
        .order_by(EDARunRecord.run_at.desc())
        .limit(50)
        .all()
    )
    return {
        "runs": [
            {
                "id": r.id,
                "run_at": r.run_at.isoformat(),
                "row_count": r.row_count,
                "col_count": r.col_count,
                "quality_score": r.quality_score,
                "missing_pct": r.missing_pct,
                "triggered_by": r.triggered_by,
            }
            for r in runs
        ]
    }


@router.post("/datasets/{dataset_id}/history/record")
def record_eda_run(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Trigger a fresh EDA run and record results in history."""
    ds = _get_ds(dataset_id, current_user, db)
    try:
        df = _load_df(ds)
        from ..eda.profiler import run_profile
        from ..eda.quality_score import run_quality_score
        from ..eda.missing import run_missing_analysis

        profile = run_profile(df)
        quality = run_quality_score(df)
        missing = run_missing_analysis(df)

        run = EDARunRecord(
            dataset_id=dataset_id,
            row_count=profile.get("total_rows"),
            col_count=profile.get("total_columns"),
            quality_score=quality.get("overall"),
            missing_pct=missing.get("missing_pct"),
            triggered_by=f"user:{current_user.email}",
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return {
            "id": run.id,
            "run_at": run.run_at.isoformat(),
            "row_count": run.row_count,
            "col_count": run.col_count,
            "quality_score": run.quality_score,
            "missing_pct": run.missing_pct,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# WORKSPACE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/workspaces/{workspace_id}/analytics")
def workspace_analytics(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_ws_member(workspace_id, current_user, db)
    datasets = (
        db.query(Dataset)
        .filter(Dataset.workspace_id == workspace_id, Dataset.status == "ready")
        .order_by(Dataset.created_at.desc())
        .all()
    )

    summaries = []
    trends: list[dict] = []

    for ds in datasets:
        # Last EDA run quality score
        last_run = (
            db.query(EDARunRecord)
            .filter(EDARunRecord.dataset_id == ds.id)
            .order_by(EDARunRecord.run_at.desc())
            .first()
        )

        # Get cached quality score
        quality_score = None
        cached_qs = (
            db.query(EDAResult)
            .filter(
                EDAResult.dataset_id == ds.id,
                EDAResult.analysis_type == "quality_score",
            )
            .order_by(EDAResult.computed_at.desc())
            .first()
        )
        if cached_qs:
            try:
                qs_data = json.loads(cached_qs.result_data)
                quality_score = qs_data.get("overall")
            except Exception:
                pass

        missing_pct = None
        cached_missing = (
            db.query(EDAResult)
            .filter(
                EDAResult.dataset_id == ds.id,
                EDAResult.analysis_type == "missing",
            )
            .order_by(EDAResult.computed_at.desc())
            .first()
        )
        if cached_missing:
            try:
                m_data = json.loads(cached_missing.result_data)
                missing_pct = m_data.get("missing_pct")
            except Exception:
                pass

        summaries.append({
            "id": ds.id,
            "name": ds.name,
            "row_count": ds.row_count,
            "column_count": ds.column_count,
            "quality_score": quality_score,
            "missing_pct": missing_pct,
            "status": ds.status,
            "created_at": ds.created_at.isoformat(),
            "last_eda_run": last_run.run_at.isoformat() if last_run else None,
        })

        # Trend: quality over time per dataset
        runs = (
            db.query(EDARunRecord)
            .filter(EDARunRecord.dataset_id == ds.id)
            .order_by(EDARunRecord.run_at)
            .all()
        )
        for r in runs:
            if r.quality_score is not None:
                trends.append({
                    "dataset_id": ds.id,
                    "dataset_name": ds.name,
                    "run_at": r.run_at.isoformat(),
                    "quality_score": r.quality_score,
                })

    worst = (
        min(summaries, key=lambda x: x["quality_score"] or 100)
        if summaries else None
    )
    most_missing = (
        max(summaries, key=lambda x: x["missing_pct"] or 0)
        if summaries else None
    )

    return {
        "datasets": summaries,
        "trends": trends,
        "worst_quality": worst,
        "most_missing": most_missing,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SAVED CHARTS
# ══════════════════════════════════════════════════════════════════════════════

class SavedChartRequest(BaseModel):
    name: str
    chart_type: str
    config: dict[str, Any]


@router.get("/datasets/{dataset_id}/charts/saved")
def list_saved_charts(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    charts = db.query(SavedChart).filter(SavedChart.dataset_id == dataset_id).all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "chart_type": c.chart_type,
            "config": json.loads(c.config_json),
            "created_at": c.created_at.isoformat(),
        }
        for c in charts
    ]


@router.post("/datasets/{dataset_id}/charts/saved")
def save_chart(
    dataset_id: int,
    body: SavedChartRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    chart = SavedChart(
        dataset_id=dataset_id,
        workspace_id=ds.workspace_id,
        name=body.name,
        chart_type=body.chart_type,
        config_json=json.dumps(body.config),
    )
    db.add(chart)
    db.commit()
    db.refresh(chart)
    return {"id": chart.id, "name": chart.name, "chart_type": chart.chart_type}


@router.delete("/datasets/{dataset_id}/charts/saved/{chart_id}")
def delete_saved_chart(
    dataset_id: int,
    chart_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _get_ds(dataset_id, current_user, db)
    chart = db.query(SavedChart).filter(
        SavedChart.id == chart_id,
        SavedChart.dataset_id == dataset_id,
    ).first()
    if not chart:
        raise HTTPException(404, "Chart not found")
    db.delete(chart)
    db.commit()
    return {"deleted": True}


# ══════════════════════════════════════════════════════════════════════════════
# NAMED SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════

class SegmentRequest(BaseModel):
    name: str
    filters: list[dict[str, Any]]


@router.get("/datasets/{dataset_id}/segments")
def list_segments(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _get_ds(dataset_id, current_user, db)
    segs = db.query(NamedSegment).filter(NamedSegment.dataset_id == dataset_id).all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "filters": json.loads(s.filter_config_json),
            "created_at": s.created_at.isoformat(),
        }
        for s in segs
    ]


@router.post("/datasets/{dataset_id}/segments")
def create_segment(
    dataset_id: int,
    body: SegmentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _get_ds(dataset_id, current_user, db)
    seg = NamedSegment(
        dataset_id=dataset_id,
        name=body.name,
        filter_config_json=json.dumps(body.filters),
    )
    db.add(seg)
    db.commit()
    db.refresh(seg)
    return {"id": seg.id, "name": seg.name}


@router.delete("/datasets/{dataset_id}/segments/{segment_id}")
def delete_segment(
    dataset_id: int,
    segment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _get_ds(dataset_id, current_user, db)
    seg = db.query(NamedSegment).filter(
        NamedSegment.id == segment_id,
        NamedSegment.dataset_id == dataset_id,
    ).first()
    if not seg:
        raise HTTPException(404, "Segment not found")
    db.delete(seg)
    db.commit()
    return {"deleted": True}


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN DETAIL PANEL
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/datasets/{dataset_id}/columns/{column_name}/detail")
def get_column_detail(
    dataset_id: int,
    column_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    try:
        df = _load_df(ds)
        if column_name not in df.columns:
            raise HTTPException(404, f"Column '{column_name}' not found")

        col = df[column_name]
        total = len(col)
        missing_count = int(col.isna().sum())
        missing_pct = round(missing_count / total * 100, 2) if total else 0
        unique_count = int(col.nunique())

        stats: dict[str, Any] = {
            "name": column_name,
            "dtype": str(col.dtype),
            "total": total,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
        }

        histogram = None
        if pd.api.types.is_numeric_dtype(col):
            clean = col.dropna()
            if len(clean):
                stats["mean"] = float(clean.mean())
                stats["median"] = float(clean.median())
                stats["std"] = float(clean.std())
                stats["min"] = float(clean.min())
                stats["max"] = float(clean.max())
                stats["q1"] = float(clean.quantile(0.25))
                stats["q3"] = float(clean.quantile(0.75))
                iqr = stats["q3"] - stats["q1"]
                outlier_count = int(((clean < stats["q1"] - 1.5 * iqr) | (clean > stats["q3"] + 1.5 * iqr)).sum())
                stats["outlier_count"] = outlier_count
                try:
                    from scipy.stats import skew, kurtosis
                    stats["skewness"] = float(skew(clean))
                    stats["kurtosis"] = float(kurtosis(clean))
                except Exception:
                    pass
                counts, edges = pd.cut(clean, bins=min(20, unique_count or 20), retbins=True)
                histogram = {
                    "bins": [round(float(e), 4) for e in edges[:-1]],
                    "counts": counts.value_counts(sort=False).tolist(),
                }

        top_values = (
            col.value_counts()
            .head(10)
            .reset_index()
            .rename(columns={"index": "value", column_name: "count"})
            .assign(pct=lambda x: (x["count"] / total * 100).round(2))
            .to_dict(orient="records")
        )

        # Suggested dtype
        suggested_dtype = None
        if col.dtype == object:
            try:
                pd.to_numeric(col.dropna())
                suggested_dtype = "float64"
            except Exception:
                try:
                    pd.to_datetime(col.dropna(), infer_datetime_format=True)
                    suggested_dtype = "datetime64"
                except Exception:
                    pass

        # Metadata
        meta = db.query(ColumnMetadata).filter(
            ColumnMetadata.dataset_id == dataset_id,
            ColumnMetadata.column_name == column_name,
        ).first()

        return {
            "stats": stats,
            "histogram": histogram,
            "top_values": top_values,
            "suggested_dtype": suggested_dtype,
            "tags": json.loads(meta.tags_json) if meta else [],
            "notes": meta.notes if meta else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
