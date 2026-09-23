"""AutoML + experiment tracking.

Training never runs on this server — it runs on the requesting engineer's
own laptop, via a small Docker agent (see /agent, shipped inside this same
backend image so the "Download agent" button in the UI always has
something to serve). This router is just the queue + leaderboard: an
engineer creates an Experiment (dataset + target column), the agent polls
for queued work, downloads the dataset through the existing export
endpoint, trains a shortlist of the DS playbook's "mostly-used" models, and
reports each run back here as it finishes. Keeping compute off this box is
the whole point — model training is exactly the kind of spiky, heavy
workload that would otherwise blow up the EC2 bill.
"""
import io
import json
import logging
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..database import get_db
from ..dataset_access import assert_dataset_access
from ..models.dataset import Dataset
from ..models.experiment import Experiment, ExperimentRun
from ..models.user import User
from ..models.workspace import WorkspaceMember
from ..schemas.experiment import (
    AgentCompleteRequest, AgentQueuedExperiment, AgentRunReport,
    ExperimentCreate, ExperimentOut, ExperimentRunOut,
)

router = APIRouter(tags=["experiments"])
logger = logging.getLogger("autoeda.routers.experiments")

ARTIFACT_DIR = Path("uploads") / "experiments"
AGENT_DIR = Path(__file__).resolve().parent.parent.parent / "agent"


def _assert_member(workspace_id: int, user: User, db: Session):
    if user.is_admin:
        return
    member = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a workspace member")


def _get_experiment(experiment_id: int, db: Session) -> Experiment:
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


def _serialize_run(r: ExperimentRun) -> ExperimentRunOut:
    return ExperimentRunOut(
        id=r.id, algorithm=r.algorithm,
        params=json.loads(r.params_json) if r.params_json else {},
        metrics=json.loads(r.metrics_json) if r.metrics_json else {},
        feature_importance=json.loads(r.feature_importance_json) if r.feature_importance_json else {},
        training_seconds=r.training_seconds, status=r.status, error=r.error,
        has_artifact=bool(r.artifact_path), artifact_filename=r.artifact_filename,
        created_at=r.created_at,
    )


def _serialize(exp: Experiment, db: Session) -> ExperimentOut:
    ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
    creator = db.query(User).filter(User.id == exp.created_by).first()
    runs = (
        db.query(ExperimentRun)
        .filter(ExperimentRun.experiment_id == exp.id)
        .order_by(ExperimentRun.created_at.asc())
        .all()
    )
    return ExperimentOut(
        id=exp.id, workspace_id=exp.workspace_id, dataset_id=exp.dataset_id,
        dataset_name=ds.name if ds else None,
        created_by=exp.created_by, created_by_name=creator.full_name if creator else None,
        name=exp.name, target_column=exp.target_column, problem_type=exp.problem_type,
        excluded_columns=json.loads(exp.excluded_columns_json) if exp.excluded_columns_json else [],
        auto_planned=exp.auto_planned, rationale=exp.rationale,
        engineered_features=json.loads(exp.engineered_features_json) if exp.engineered_features_json else [],
        status=exp.status, error=exp.error,
        created_at=exp.created_at, claimed_at=exp.claimed_at, completed_at=exp.completed_at,
        runs=[_serialize_run(r) for r in runs],
    )


# -- Engineer-facing (browser) -----------------------------------------------

@router.get("/agent/download")
def download_agent(current_user: User = Depends(get_current_active_user)):
    """Zips up the local training agent so the AutoML page can offer a
    one-click download — no git clone, no hunting through the repo."""
    if not AGENT_DIR.exists():
        raise HTTPException(status_code=404, detail="Agent bundle not found on the server")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in AGENT_DIR.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, arcname=f"autoeda-agent/{path.relative_to(AGENT_DIR)}")
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=autoeda-agent.zip"},
    )


@router.post("/workspaces/{workspace_id}/experiments", response_model=ExperimentOut, status_code=201)
def create_experiment(
    workspace_id: int,
    payload: ExperimentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    ds = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_access(ds, current_user, db)

    exp = Experiment(
        workspace_id=workspace_id, dataset_id=ds.id, created_by=current_user.id,
        name=payload.name.strip() or f"Experiment on {ds.name}",
        target_column=payload.target_column, problem_type=payload.problem_type,
        excluded_columns_json=json.dumps(payload.excluded_columns) if payload.excluded_columns else None,
        status="queued",
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return _serialize(exp, db)


@router.post("/workspaces/{workspace_id}/experiments/auto", response_model=ExperimentOut, status_code=201)
def create_auto_experiment(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """No dataset, target, or feature list to pick — an LLM call grounded in
    this workspace's own dataset profiles and validated Hypotheses decides
    all of that, then this just queues the resulting plan for the local
    agent exactly like a manually-created experiment."""
    from ..ai.agent.automl_planner import PlanningError, plan_experiment

    _assert_member(workspace_id, current_user, db)
    try:
        plan = plan_experiment(workspace_id, db)
    except PlanningError as e:
        raise HTTPException(status_code=422, detail=str(e))

    exp = Experiment(
        workspace_id=workspace_id, dataset_id=plan["dataset_id"], created_by=current_user.id,
        name=f'Auto: {plan["target_column"]} on {plan["dataset_name"]}',
        target_column=plan["target_column"], problem_type=plan["problem_type"],
        excluded_columns_json=json.dumps(plan["excluded_columns"]) if plan["excluded_columns"] else None,
        auto_planned=True, rationale=plan["rationale"],
        engineered_features_json=json.dumps(plan["engineered_features"]) if plan["engineered_features"] else None,
        status="queued",
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return _serialize(exp, db)


@router.get("/workspaces/{workspace_id}/experiments", response_model=list[ExperimentOut])
def list_experiments(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    exps = (
        db.query(Experiment)
        .filter(Experiment.workspace_id == workspace_id)
        .order_by(Experiment.created_at.desc())
        .all()
    )
    return [_serialize(e, db) for e in exps]


@router.get("/experiments/{experiment_id}", response_model=ExperimentOut)
def get_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    exp = _get_experiment(experiment_id, db)
    _assert_member(exp.workspace_id, current_user, db)
    return _serialize(exp, db)


@router.delete("/experiments/{experiment_id}", status_code=204)
def delete_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    exp = _get_experiment(experiment_id, db)
    _assert_member(exp.workspace_id, current_user, db)
    if exp.created_by != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Only the creator or an admin can delete this experiment")

    # Delete child runs explicitly rather than relying on the model's
    # ondelete="CASCADE" — SQLite doesn't enforce foreign keys unless a
    # PRAGMA is turned on per-connection, which this app doesn't do, so that
    # cascade is a no-op here. Left alone, orphaned runs stick around and
    # resurface under a *different* experiment once SQLite reuses this
    # deleted row's id (it does that once the table empties out).
    runs = db.query(ExperimentRun).filter(ExperimentRun.experiment_id == experiment_id).all()
    for run in runs:
        if run.artifact_path:
            Path(run.artifact_path).unlink(missing_ok=True)
        db.delete(run)
    db.delete(exp)
    db.commit()


@router.get("/experiments/{experiment_id}/runs/{run_id}/download")
def download_artifact(
    experiment_id: int,
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    exp = _get_experiment(experiment_id, db)
    _assert_member(exp.workspace_id, current_user, db)
    run = db.query(ExperimentRun).filter(ExperimentRun.id == run_id, ExperimentRun.experiment_id == experiment_id).first()
    if not run or not run.artifact_path:
        raise HTTPException(status_code=404, detail="No model artifact for this run")
    path = Path(run.artifact_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file is missing on the server")
    return FileResponse(path, filename=run.artifact_filename or f"{run.algorithm}.joblib", media_type="application/octet-stream")


# -- Agent-facing (the laptop-side Docker container) -------------------------
# All of these are authenticated the same way the browser is (Bearer token
# from POST /api/v1/auth/login with the engineer's own credentials) — the
# agent is just another API client, not a special trust boundary.

@router.get("/agent/experiments/queued", response_model=list[AgentQueuedExperiment])
def agent_list_queued(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    exps = (
        db.query(Experiment)
        .filter(Experiment.created_by == current_user.id, Experiment.status == "queued")
        .order_by(Experiment.created_at.asc())
        .all()
    )
    return [
        AgentQueuedExperiment(
            id=e.id, workspace_id=e.workspace_id, dataset_id=e.dataset_id,
            name=e.name, target_column=e.target_column, problem_type=e.problem_type,
            excluded_columns=json.loads(e.excluded_columns_json) if e.excluded_columns_json else [],
            engineered_features=json.loads(e.engineered_features_json) if e.engineered_features_json else [],
        )
        for e in exps
    ]


@router.post("/agent/experiments/{experiment_id}/claim", response_model=ExperimentOut)
def agent_claim(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    from datetime import datetime, timezone

    exp = _get_experiment(experiment_id, db)
    if exp.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Not your experiment")
    if exp.status != "queued":
        raise HTTPException(status_code=409, detail=f"Experiment is already {exp.status}")
    exp.status = "running"
    exp.claimed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(exp)
    return _serialize(exp, db)


@router.post("/agent/experiments/{experiment_id}/runs", response_model=ExperimentRunOut, status_code=201)
def agent_report_run(
    experiment_id: int,
    payload: AgentRunReport,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    exp = _get_experiment(experiment_id, db)
    if exp.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Not your experiment")

    run = ExperimentRun(
        experiment_id=exp.id, algorithm=payload.algorithm,
        params_json=json.dumps(payload.params), metrics_json=json.dumps(payload.metrics),
        feature_importance_json=json.dumps(payload.feature_importance) if payload.feature_importance else None,
        training_seconds=payload.training_seconds, status=payload.status, error=payload.error,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return _serialize_run(run)


@router.post("/agent/experiments/{experiment_id}/runs/{run_id}/artifact", response_model=ExperimentRunOut)
def agent_upload_artifact(
    experiment_id: int,
    run_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    exp = _get_experiment(experiment_id, db)
    if exp.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Not your experiment")
    run = db.query(ExperimentRun).filter(ExperimentRun.id == run_id, ExperimentRun.experiment_id == experiment_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    exp_dir = ARTIFACT_DIR / str(experiment_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    filename = file.filename or f"{run.algorithm}.joblib"
    dest = exp_dir / f"{run_id}_{filename}"
    with open(dest, "wb") as f:
        f.write(file.file.read())

    run.artifact_path = str(dest)
    run.artifact_filename = filename
    db.commit()
    db.refresh(run)
    return _serialize_run(run)


@router.post("/agent/experiments/{experiment_id}/complete", response_model=ExperimentOut)
def agent_complete(
    experiment_id: int,
    payload: AgentCompleteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    from datetime import datetime, timezone

    exp = _get_experiment(experiment_id, db)
    if exp.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Not your experiment")
    exp.status = payload.status
    exp.error = payload.error
    exp.completed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(exp)
    return _serialize(exp, db)
