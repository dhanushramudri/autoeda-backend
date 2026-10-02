import io
import json
import logging
import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.orm import defer as sa_defer

from ..auth import get_current_active_user
from ..database import get_db
from ..ds_flows.registry import FLOWS, get_flow
from ..dataset_access import dataset_visibility_filter
from ..models.dataset import Dataset
from ..models.ds_flow import DsFlowRun
from ..models.user import User
from ..models.workspace import WorkspaceMember
from ..process_pool import AnalysisCrashed, AnalysisTimeout, run_isolated

logger = logging.getLogger("autoeda.routers.ds_flows")

router = APIRouter(prefix="/workspaces/{workspace_id}/ds-flows", tags=["ds-flows"])

# Each stage persists progress when it starts and ends, and the slowest (model training) is allowed 20 min,
# so a run with no update for longer than this is presumed dead (server restart / crash).
STALE_AFTER = timedelta(minutes=60)
ACTIVE = ("pending", "running")


class RunCreate(BaseModel):
    flow_key: str = "churn"
    # optional: default is every dataset in the workspace
    dataset_ids: list[int] | None = None


def _assert_member(workspace_id: int, user: User, db: Session):
    if user.is_admin:
        return
    if not db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user.id
    ).first():
        raise HTTPException(status_code=403, detail="Not a workspace member")


def _visible_datasets(workspace_id: int, db: Session, only: list[int] | None = None) -> list[Dataset]:
    """Every ready dataset in the workspace (plus the shared global library)."""
    from sqlalchemy import or_

    q = (db.query(Dataset).options(sa_defer(Dataset.file_data))
         .filter(dataset_visibility_filter(db, workspace_id), Dataset.status == "ready")
         .filter(or_(Dataset.source_config.is_(None), Dataset.source_config.notlike('%"ds_flow_run"%'))))
    if only:
        q = q.filter(Dataset.id.in_(only))
    return q.order_by(Dataset.id).all()


def _get_run(workspace_id: int, run_id: int, db: Session) -> DsFlowRun:
    run = db.query(DsFlowRun).filter(DsFlowRun.id == run_id, DsFlowRun.workspace_id == workspace_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


def _reap_if_stale(run: DsFlowRun, db: Session) -> DsFlowRun:
    if run.status not in ACTIVE:
        return run
    updated = run.updated_at if run.updated_at.tzinfo else run.updated_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - updated > STALE_AFTER:
        run.status = "error"
        run.error = "This run stalled — no progress for over 60 minutes, likely because the server restarted. Please start it again."
        if run.stages_json:
            stages = json.loads(run.stages_json)
            for s in stages:
                if s["status"] == "running":
                    s["status"] = "error"
            run.stages_json = json.dumps(stages)
        db.add(run)
        db.commit()
        db.refresh(run)
    return run


def _j(s):
    return json.loads(s) if s else None


def _summary(run: DsFlowRun) -> dict:
    stages = _j(run.stages_json) or []
    done = sum(1 for s in stages if s["status"] in ("done", "error", "skipped"))
    return {
        "id": run.id, "workspace_id": run.workspace_id, "dataset_id": run.dataset_id, "dataset_name": run.dataset_name,
        "flow_key": run.flow_key, "title": run.title, "status": run.status, "error": run.error,
        "progress": {"done": done, "total": len(stages)}, "headline": _j(run.headline_json),
        "created_at": run.created_at.isoformat(), "updated_at": run.updated_at.isoformat(),
        "has_outputs": bool(run.enriched_csv),
    }


def _full(run: DsFlowRun) -> dict:
    return {
        **_summary(run), "roles": _j(run.roles_json), "params": _j(run.params_json), "stages": _j(run.stages_json) or [],
        "results": _j(run.results_json) or {}, "narrative": _j(run.narrative_json), "markdown": run.markdown,
        "files": {
            "enriched": bool(run.enriched_csv), "accounts": bool(run.accounts_csv),
            "dictionary": bool(run.dictionary_csv), "model": bool(run.model_blob),
        },
        "source_filename": run.source_filename,
        "working_dataset_id": (_j(run.params_json) or {}).get("working_dataset_id"),
        "auto_eda_run_id": (_j(run.params_json) or {}).get("auto_eda_run_id"),
    }


@router.get("/catalog")
def catalog(workspace_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    _assert_member(workspace_id, current_user, db)
    return {"flows": FLOWS}


@router.post("/scan")
def scan(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Automatic opportunity scan over every dataset in the workspace: which table holds the outcome, how the
    tables link, and which offerings the data can support. No input needed."""
    from ..ds_flows.discover import plan_workspace
    from ..ds_flows.runner import unique_table_names
    from .eda import _load_df

    _assert_member(workspace_id, current_user, db)
    datasets = _visible_datasets(workspace_id, db)
    if not datasets:
        raise HTTPException(status_code=400, detail="This workspace has no datasets yet — upload or connect one first.")
    names = unique_table_names(datasets)
    try:
        tables = {names[d.id]: _load_df(d) for d in datasets}
        meta = {names[d.id]: {"dataset_id": d.id} for d in datasets}
        return run_isolated(plan_workspace, tables, meta, timeout=300)
    except AnalysisTimeout as e:
        raise HTTPException(status_code=504, detail=str(e))
    except AnalysisCrashed as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("scan failed")
        raise HTTPException(status_code=500, detail=f"Could not analyse the datasets: {e}")


@router.get("/runs")
def list_runs(
    workspace_id: int,
    dataset_id: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    q = (
        db.query(DsFlowRun)
        .options(
            sa_defer(DsFlowRun.enriched_csv), sa_defer(DsFlowRun.accounts_csv), sa_defer(DsFlowRun.dictionary_csv),
            sa_defer(DsFlowRun.model_blob), sa_defer(DsFlowRun.results_json), sa_defer(DsFlowRun.markdown),
        )
        .filter(DsFlowRun.workspace_id == workspace_id)
    )
    if dataset_id is not None:
        q = q.filter(DsFlowRun.dataset_id == dataset_id)
    return [_summary(_reap_if_stale(r, db)) for r in q.order_by(DsFlowRun.created_at.desc()).limit(50).all()]


@router.post("/runs")
def create_run(
    workspace_id: int,
    payload: RunCreate,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    from ..ds_flows.runner import execute_flow_run, initial_stages

    _assert_member(workspace_id, current_user, db)
    flow = get_flow(payload.flow_key)
    if flow is None:
        raise HTTPException(status_code=404, detail="Unknown flow")
    if flow["status"] != "available":
        raise HTTPException(status_code=400, detail=f"{flow['category']} isn't available yet")
    datasets = _visible_datasets(workspace_id, db, payload.dataset_ids)
    if not datasets:
        raise HTTPException(status_code=400, detail="This workspace has no datasets to analyse")

    active = db.query(DsFlowRun).filter(DsFlowRun.workspace_id == workspace_id, DsFlowRun.flow_key == flow["key"], DsFlowRun.status.in_(ACTIVE)).first()
    if active and _reap_if_stale(active, db).status in ACTIVE:
        raise HTTPException(status_code=409, detail="A run is already in progress in this workspace")

    first = datasets[0]
    run = DsFlowRun(
        workspace_id=workspace_id, dataset_id=first.id, created_by=current_user.id, flow_key=flow["key"],
        title=f"{flow['category']} — {len(datasets)} dataset{'s' if len(datasets) != 1 else ''}",
        dataset_name=first.name if len(datasets) == 1 else f"{len(datasets)} datasets",
        source_filename=os.path.basename(first.file_path or "") or first.name,
        status="pending", roles_json=json.dumps({}),
        params_json=json.dumps({"dataset_ids": [d.id for d in datasets]}),
        stages_json=json.dumps(initial_stages(flow["key"])),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    background.add_task(execute_flow_run, run.id)
    return {"run_id": run.id}


@router.get("/runs/{run_id}")
def get_run(workspace_id: int, run_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    _assert_member(workspace_id, current_user, db)
    run = _get_run(workspace_id, run_id, db)
    return _full(_reap_if_stale(run, db))


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
    history: list[dict] = Field(default_factory=list)


@router.get("/runs/{run_id}/customers")
def customers(workspace_id: int, run_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Every scored customer (risk level, revenue, expected loss, main drivers, segments) as compact rows for the dashboard."""
    import pandas as pd

    from ..ds_flows.chat import load_customers

    _assert_member(workspace_id, current_user, db)
    run = _get_run(workspace_id, run_id, db)
    df = load_customers(run)
    if df is None:
        raise HTTPException(status_code=404, detail="Customer list not ready")
    key = "expected_value_at_risk" if "expected_value_at_risk" in df.columns else "churn_probability"
    df = df.sort_values(key, ascending=False).head(20000)
    rows = df.astype(object).where(pd.notna(df), None).values.tolist()
    return {"columns": list(df.columns), "rows": rows, "truncated": len(df) >= 20000}


@router.post("/runs/{run_id}/chat")
def chat(workspace_id: int, run_id: int, body: ChatIn, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    from ..ai.providers.base import QuotaExceededError
    from ..ds_flows.chat import answer

    _assert_member(workspace_id, current_user, db)
    run = _get_run(workspace_id, run_id, db)
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="The analysis has not finished yet")
    try:
        return {"answer": answer(run, body.message.strip(), body.history)}
    except QuotaExceededError:
        raise HTTPException(status_code=429, detail="The AI service is busy — try again shortly")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail="Could not answer that — please try again")


@router.delete("/runs/{run_id}")
def delete_run(workspace_id: int, run_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    _assert_member(workspace_id, current_user, db)
    run = _get_run(workspace_id, run_id, db)
    aid = (_j(run.params_json) or {}).get("auto_eda_run_id")
    if aid:  # the Auto EDA report made for this run goes with it
        from ..models.auto_eda import AutoEdaRun

        ae = db.query(AutoEdaRun).filter(AutoEdaRun.id == aid).first()
        if ae is not None:
            db.delete(ae)
    wid = (_j(run.params_json) or {}).get("working_dataset_id")
    if wid:  # the hidden working table goes with the run
        ws = db.query(Dataset).filter(Dataset.id == wid).first()
        if ws is not None and '"ds_flow_run"' in (ws.source_config or ""):
            db.delete(ws)
    db.delete(run)
    db.commit()
    return {"message": "Run deleted"}


@router.get("/runs/{run_id}/download")
def download(
    workspace_id: int,
    run_id: int,
    kind: str = Query("enriched", pattern="^(enriched|accounts|dictionary|model|report)$"),
    format: str = Query("csv", pattern="^(csv|xlsx|md|docx)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    run = _get_run(workspace_id, run_id, db)
    stem = os.path.splitext(run.source_filename or f"run_{run.id}")[0]
    flow = run.flow_key

    if kind == "report":
        if not run.markdown:
            raise HTTPException(status_code=404, detail="Report not ready")
        if format == "docx":
            from ..eda.report_builder import build_docx_report

            data = build_docx_report(
                title=run.title or "Data science flow", markdown=run.markdown,
                generated_at=run.updated_at.strftime("%B %d, %Y"), business_context=None,
            )
            return Response(data, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            headers={"Content-Disposition": f'attachment; filename="{stem}_{flow}_report.docx"'})
        return PlainTextResponse(run.markdown, media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="{stem}_{flow}_report.md"'})

    blob = {"enriched": run.enriched_csv, "accounts": run.accounts_csv, "dictionary": run.dictionary_csv, "model": run.model_blob}[kind]
    if not blob:
        raise HTTPException(status_code=404, detail="File not ready")
    if kind == "model":
        return Response(blob, media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{stem}_{flow}_model.joblib"'})
    name = {"enriched": f"{stem}_{flow}_enriched", "accounts": f"{stem}_{flow}_accounts", "dictionary": f"{stem}_{flow}_data_dictionary"}[kind]
    if format == "xlsx":
        import pandas as pd

        buf = io.BytesIO()
        pd.read_csv(io.BytesIO(blob), low_memory=False).to_excel(buf, index=False)
        return Response(buf.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f'attachment; filename="{name}.xlsx"'})
    return Response(blob, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})
