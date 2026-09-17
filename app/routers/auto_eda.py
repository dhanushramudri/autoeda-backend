import json
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..database import get_db
from ..models.auto_eda import AutoEdaChatMessage, AutoEdaRun
from ..models.user import User
from ..models.workspace import WorkspaceMember
from ..schemas.auto_eda import AutoEdaChatMessageOut, AutoEdaChatSend, AutoEdaRunCreate, AutoEdaRunOut

logger = logging.getLogger("autoeda.routers.auto_eda")

router = APIRouter(prefix="/workspaces/{workspace_id}/auto-eda", tags=["auto-eda"])

# A run stuck in "running" with no DB update for this long is presumed dead —
# e.g. the backend process restarted mid-run (dev --reload) or crashed. There
# is no live HTTP connection to depend on any more (the whole point of the
# background-task model), so this is the only way such a run ever surfaces
# an error instead of showing "running" forever.
STALE_AFTER = timedelta(minutes=5)


def _assert_member(workspace_id: int, user: User, db: Session):
    if user.is_admin:
        return
    member = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a workspace member")


def _get_run(workspace_id: int, run_id: int, db: Session) -> AutoEdaRun:
    run = db.query(AutoEdaRun).filter(
        AutoEdaRun.id == run_id,
        AutoEdaRun.workspace_id == workspace_id,
    ).first()
    if not run:
        raise HTTPException(status_code=404, detail="Auto EDA run not found")
    return run


def _reap_if_stale(run: AutoEdaRun, db: Session) -> AutoEdaRun:
    if run.status not in ("pending", "running", "pausing"):
        return run
    # SQLite (local dev) drops tzinfo on read-back even though the column is
    # written with datetime.now(timezone.utc) — Postgres (prod) keeps it.
    # Normalize so this comparison is correct in both.
    updated_at = run.updated_at if run.updated_at.tzinfo else run.updated_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - updated_at
    if age > STALE_AFTER:
        run.status = "error"
        run.error = (
            "This run stalled — no progress for over "
            f"{int(STALE_AFTER.total_seconds() // 60)} minutes, likely because "
            "the server restarted or crashed mid-run. Please try again."
        )
        db.add(run)
        db.commit()
        db.refresh(run)
    return run


def _serialize(run: AutoEdaRun) -> AutoEdaRunOut:
    return AutoEdaRunOut(
        id=run.id, workspace_id=run.workspace_id,
        dataset_ids=json.loads(run.dataset_ids_json) if run.dataset_ids_json else [],
        status=run.status, title=run.title, business_context=run.business_context, markdown=run.markdown,
        worklist=json.loads(run.worklist_json) if run.worklist_json else [],
        error=run.error, created_at=run.created_at, updated_at=run.updated_at,
    )


@router.get("/runs", response_model=list[AutoEdaRunOut])
def list_runs(
    workspace_id: int,
    dataset_id: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    rows = (
        db.query(AutoEdaRun)
        .filter(AutoEdaRun.workspace_id == workspace_id)
        .order_by(AutoEdaRun.created_at.desc())
        .all()
    )
    if dataset_id is not None:
        # dataset_ids_json has no native JSON-membership query that works
        # identically on both SQLite (local dev) and Postgres (prod), and
        # run counts per workspace are small — filter in Python instead.
        rows = [r for r in rows if dataset_id in json.loads(r.dataset_ids_json or "[]")]
    return [_serialize(_reap_if_stale(r, db)) for r in rows]


@router.get("/runs/{run_id}", response_model=AutoEdaRunOut)
def get_run(
    workspace_id: int,
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    return _serialize(_reap_if_stale(_get_run(workspace_id, run_id, db), db))


@router.delete("/runs/{run_id}")
def delete_run(
    workspace_id: int,
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    run = _get_run(workspace_id, run_id, db)
    db.delete(run)
    db.commit()
    return {"message": "Auto EDA run deleted"}


@router.get("/runs/{run_id}/download")
def download_run(
    workspace_id: int,
    run_id: int,
    format: str = Query("md", pattern="^(md|docx)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    run = _get_run(workspace_id, run_id, db)

    dataset_ids = json.loads(run.dataset_ids_json or "[]")
    file_tag = "_".join(str(d) for d in dataset_ids) or "all"

    if format == "docx":
        from ..eda.report_builder import build_docx_report

        docx_bytes = build_docx_report(
            title=run.title or "Automated EDA",
            markdown=run.markdown or "",
            generated_at=run.updated_at.strftime("%B %d, %Y"),
            business_context=run.business_context,
        )
        filename = f"auto_eda_{file_tag}_{run.id}.docx"
        return Response(
            docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    filename = f"auto_eda_{file_tag}_{run.id}.md"
    return PlainTextResponse(
        run.markdown or "",
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _run_in_background(
    workspace_id: int, dataset_ids: list[int], run_id: int, user_id: int, resume: bool = False,
    report_title: str | None = None,
):
    """Runs the whole Auto EDA pipeline to completion independent of any HTTP
    connection — progress is persisted to the DB by run_auto_eda_stream's own
    incremental _persist() calls as a side effect of driving the generator,
    not by anything consuming the yielded events here. This is deliberately
    NOT a request-scoped session (that one gets closed as soon as the
    request returns) — see app/tasks.py's run_eda_pipeline for the same
    fresh-SessionLocal-in-a-background-task convention."""
    from ..ai.agent.auto_eda_orchestrator import run_auto_eda_stream
    from ..database import SessionLocal

    db = SessionLocal()
    try:
        run_row = db.query(AutoEdaRun).filter(AutoEdaRun.id == run_id).first()
        user = db.query(User).filter(User.id == user_id).first()
        if run_row is None or user is None:
            return
        for _event in run_auto_eda_stream(
            workspace_id=workspace_id, dataset_ids=dataset_ids, db=db, user=user, run_row=run_row, resume=resume,
            business_context=run_row.business_context, report_title=report_title,
        ):
            pass
    except Exception as e:
        logger.exception("Auto EDA background run %s failed", run_id)
        run_row = db.query(AutoEdaRun).filter(AutoEdaRun.id == run_id).first()
        if run_row and run_row.status not in ("completed", "error"):
            run_row.status = "error"
            run_row.error = str(e)
            db.add(run_row)
            db.commit()
    finally:
        db.close()


@router.post("/run")
def start_auto_eda_run(
    workspace_id: int,
    payload: AutoEdaRunCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)

    run_row = AutoEdaRun(
        workspace_id=workspace_id, dataset_ids_json=json.dumps(payload.dataset_ids),
        created_by=current_user.id, status="pending",
        business_context=(payload.business_context or "").strip() or None,
    )
    db.add(run_row)
    db.commit()
    db.refresh(run_row)

    background_tasks.add_task(
        _run_in_background, workspace_id, payload.dataset_ids, run_row.id, current_user.id, False,
        (payload.report_title or "").strip() or None,
    )
    return {"run_id": run_row.id}


@router.post("/runs/{run_id}/pause")
def pause_auto_eda_run(
    workspace_id: int,
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Signals a running background task to stop after whichever item is
    currently in flight (not mid-computation) — see run_auto_eda_stream's
    own per-item db.refresh(run_row) check, which is what actually notices
    this and settles the run into "paused"."""
    _assert_member(workspace_id, current_user, db)
    run = _get_run(workspace_id, run_id, db)
    if run.status not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"Run is {run.status}, not running — nothing to pause")
    run.status = "pausing"
    db.add(run)
    db.commit()
    return {"message": "Pause requested — will stop after the current step"}


@router.post("/runs/{run_id}/resume")
def resume_auto_eda_run(
    workspace_id: int,
    run_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    run = _get_run(workspace_id, run_id, db)
    if run.status != "paused":
        raise HTTPException(status_code=400, detail=f"Run is {run.status}, not paused — nothing to resume")

    dataset_ids = json.loads(run.dataset_ids_json or "[]")
    run.status = "running"
    db.add(run)
    db.commit()

    background_tasks.add_task(_run_in_background, workspace_id, dataset_ids, run.id, current_user.id, True)
    return {"message": "Resumed"}


@router.post("/runs/{run_id}/approve")
def approve_auto_eda_run(
    workspace_id: int,
    run_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """A run stops at status="planned" right after planning (see
    run_auto_eda_stream's require_approval) so a human can review — and,
    via the chat endpoint, steer — the proposed worklist before anything
    actually executes. This is the human-in-the-loop gate: approving is
    mechanically identical to resuming a paused run (continue from the
    first still-"pending" item), just starting from "planned" instead."""
    _assert_member(workspace_id, current_user, db)
    run = _get_run(workspace_id, run_id, db)
    if run.status != "planned":
        raise HTTPException(status_code=400, detail=f"Run is {run.status}, not awaiting approval")

    dataset_ids = json.loads(run.dataset_ids_json or "[]")
    run.status = "running"
    db.add(run)
    db.commit()

    background_tasks.add_task(_run_in_background, workspace_id, dataset_ids, run.id, current_user.id, True)
    return {"message": "Approved — running now"}


@router.get("/runs/{run_id}/chat", response_model=list[AutoEdaChatMessageOut])
def list_chat_messages(
    workspace_id: int,
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    _get_run(workspace_id, run_id, db)
    rows = (
        db.query(AutoEdaChatMessage)
        .filter(AutoEdaChatMessage.run_id == run_id)
        .order_by(AutoEdaChatMessage.created_at)
        .all()
    )
    return [AutoEdaChatMessageOut.model_validate(r, from_attributes=True) for r in rows]


@router.post("/runs/{run_id}/chat", response_model=AutoEdaChatMessageOut)
def send_chat_message(
    workspace_id: int,
    run_id: int,
    payload: AutoEdaChatSend,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Posts a steering instruction to a live run. Only meaningful while the
    run is actually active — run_auto_eda_stream's own loop is what reads
    and applies unhandled messages (see its per-iteration check), so a
    message posted against a completed/errored run would just sit unread."""
    _assert_member(workspace_id, current_user, db)
    run = _get_run(workspace_id, run_id, db)
    if run.status not in ("pending", "planned", "running", "pausing", "paused"):
        raise HTTPException(status_code=400, detail="This run has finished — start a new one to steer it")
    msg = AutoEdaChatMessage(run_id=run_id, role="user", content=payload.content, applied=False)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return AutoEdaChatMessageOut.model_validate(msg, from_attributes=True)
