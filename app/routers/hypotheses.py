import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..ai.agent.hypothesis_orchestrator import (
    run_hypothesis_generation, run_hypothesis_generation_stream,
    run_hypothesis_validation, run_hypothesis_validation_stream,
)
from ..ai.llm import provider_name
from ..auth import get_current_active_user
from ..database import get_db
from ..models.hypothesis import Hypothesis
from ..models.user import User
from ..models.workspace import WorkspaceMember
from ..s3_attachments import presign_get_inline
from ..schemas.hypotheses import GenerateRequest, HypothesisCreate, HypothesisOut

router = APIRouter(prefix="/workspaces/{workspace_id}/hypotheses", tags=["hypotheses"])
logger = logging.getLogger("autoeda.routers.hypotheses")

# Same providers that can actually see an attached image — see routers/scout.py.
_IMAGE_CAPABLE_PROVIDERS = {"claude", "openai"}
_IMAGE_ATTACH_ERROR = "Image attachments require Claude or OpenAI to be the active provider."


def _assert_member(workspace_id: int, user: User, db: Session):
    if user.is_admin:
        return
    member = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a workspace member")


def _get_hypothesis(workspace_id: int, hypothesis_id: int, db: Session) -> Hypothesis:
    h = db.query(Hypothesis).filter(
        Hypothesis.id == hypothesis_id,
        Hypothesis.workspace_id == workspace_id,
    ).first()
    if not h:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return h


def _serialize(h: Hypothesis) -> HypothesisOut:
    return HypothesisOut(
        id=h.id, workspace_id=h.workspace_id, dataset_id=h.dataset_id,
        origin=h.origin, title=h.title, statement=h.statement, category=h.category,
        status=h.status, verdict=h.verdict, evidence_summary=h.evidence_summary,
        confidence=h.confidence, severity=h.severity,
        columns=json.loads(h.columns_json) if h.columns_json else [],
        tool_trace=json.loads(h.tool_trace_json) if h.tool_trace_json else [],
        image_url=presign_get_inline(h.image_key) if h.image_key else None,
        created_at=h.created_at, updated_at=h.updated_at, validated_at=h.validated_at,
    )


@router.get("", response_model=list[HypothesisOut])
def list_hypotheses(
    workspace_id: int,
    dataset_id: int | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    q = db.query(Hypothesis).filter(Hypothesis.workspace_id == workspace_id)
    if dataset_id is not None:
        q = q.filter(Hypothesis.dataset_id == dataset_id)
    if status is not None:
        q = q.filter(Hypothesis.status == status)
    rows = q.order_by(Hypothesis.created_at.desc()).all()
    return [_serialize(h) for h in rows]


@router.post("", response_model=HypothesisOut)
def create_hypothesis(
    workspace_id: int,
    payload: HypothesisCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    if not payload.statement.strip():
        raise HTTPException(status_code=400, detail="statement cannot be empty")
    if payload.image_key and provider_name() not in _IMAGE_CAPABLE_PROVIDERS:
        raise HTTPException(status_code=400, detail=_IMAGE_ATTACH_ERROR)
    h = Hypothesis(
        workspace_id=workspace_id, dataset_id=payload.dataset_id,
        created_by=current_user.id, origin="user",
        statement=payload.statement.strip(), status="pending",
        image_key=payload.image_key, image_content_type=payload.image_content_type,
    )
    db.add(h)
    db.commit()
    db.refresh(h)
    return _serialize(h)


@router.delete("/{hypothesis_id}")
def delete_hypothesis(
    workspace_id: int,
    hypothesis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    h = _get_hypothesis(workspace_id, hypothesis_id, db)
    db.delete(h)
    db.commit()
    return {"message": "Hypothesis deleted"}


def _persist_generated(workspace_id: int, dataset_id: int | None, items: list[dict], tool_trace: list[dict], db: Session) -> list[Hypothesis]:
    rows = []
    for item in items:
        h = Hypothesis(
            workspace_id=workspace_id, dataset_id=dataset_id, created_by=None, origin="ai",
            title=item.get("title"), statement=item.get("statement", ""), category=item.get("category"),
            status=item.get("status", "supported"), verdict=item.get("verdict"),
            evidence_summary=item.get("evidence_summary"), confidence=item.get("confidence"),
            severity=item.get("severity"), columns_json=json.dumps(item.get("columns", [])),
            tool_trace_json=json.dumps(tool_trace, default=str), validated_at=None,
        )
        db.add(h)
        rows.append(h)
    db.commit()
    for h in rows:
        db.refresh(h)
    return rows


def _run_generate_bg(workspace_id: int, dataset_id: int | None, count: int, user_id: int):
    """Runs generation to completion independent of any HTTP connection — the
    old /generate/stream endpoint died the moment the browser navigated away
    (Starlette cancels a StreamingResponse's generator on client disconnect);
    a plain background task has no such tie to the request lifecycle, same
    convention as Auto EDA's _run_in_background."""
    from ..database import SessionLocal

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            return
        result = run_hypothesis_generation(
            workspace_id=workspace_id, dataset_id=dataset_id, count=count, db=db, user=user,
        )
        if result.get("error"):
            logger.warning("hypothesis generation failed: %s", result["error"])
            return
        _persist_generated(workspace_id, dataset_id, result["hypotheses"], result["tool_trace"], db)
    except Exception:
        logger.exception("hypothesis generation background task crashed")
    finally:
        db.close()


@router.post("/generate")
def generate_hypotheses(
    workspace_id: int,
    payload: GenerateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    background_tasks.add_task(_run_generate_bg, workspace_id, payload.dataset_id, payload.count, current_user.id)
    return {"message": "Generating — new hypotheses will appear in the list as they're found."}


@router.post("/generate/stream")
def generate_hypotheses_stream(
    workspace_id: int,
    payload: GenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)

    def event_stream():
        hypotheses: list[dict] = []
        tool_trace: list[dict] = []
        for event in run_hypothesis_generation_stream(
            workspace_id=workspace_id, dataset_id=payload.dataset_id, count=payload.count, db=db, user=current_user,
        ):
            if event["type"] == "result":
                hypotheses = event["hypotheses"]
                tool_trace = event["tool_trace"]
            yield f"data: {json.dumps(event, default=str)}\n\n"

        if hypotheses:
            rows = _persist_generated(workspace_id, payload.dataset_id, hypotheses, tool_trace, db)
            yield f"data: {json.dumps({'type': 'persisted', 'ids': [h.id for h in rows]})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _apply_validation(h: Hypothesis, result: dict, db: Session) -> Hypothesis:
    from datetime import datetime, timezone
    h.status = result.get("status", "inconclusive")
    h.verdict = result.get("verdict")
    h.evidence_summary = result.get("evidence_summary")
    h.confidence = result.get("confidence")
    h.columns_json = json.dumps(result.get("columns", []))
    h.tool_trace_json = json.dumps(result.get("tool_trace", []), default=str)
    h.validated_at = datetime.now(timezone.utc)
    db.add(h)
    db.commit()
    db.refresh(h)
    return h


def _run_validate_bg(workspace_id: int, hypothesis_id: int, statement: str, dataset_id: int | None, image: dict | None, user_id: int):
    """Same rationale as _run_generate_bg — the investigation now survives
    navigating away, since it's a background task with no dependency on the
    request/response still being open."""
    from ..database import SessionLocal

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        h = db.query(Hypothesis).filter(Hypothesis.id == hypothesis_id).first()
        if user is None or h is None:
            return
        result = run_hypothesis_validation(
            statement=statement, workspace_id=workspace_id, dataset_id=dataset_id, db=db, user=user, image=image,
        )
        db.refresh(h)
        if h.stop_requested:
            # /stop already resolved this hypothesis's visible state —
            # don't clobber it once the (uncancellable) investigation
            # eventually finishes on its own.
            return
        _apply_validation(h, result, db)
    except Exception:
        logger.exception("hypothesis validation background task crashed")
    finally:
        db.close()


@router.post("/{hypothesis_id}/validate", response_model=HypothesisOut)
def validate_hypothesis(
    workspace_id: int,
    hypothesis_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    h = _get_hypothesis(workspace_id, hypothesis_id, db)
    h.status = "validating"
    h.stop_requested = False
    db.add(h)
    db.commit()
    db.refresh(h)

    image = {"key": h.image_key, "media_type": h.image_content_type} if h.image_key else None
    background_tasks.add_task(
        _run_validate_bg, workspace_id, hypothesis_id, h.statement, h.dataset_id, image, current_user.id,
    )
    return _serialize(h)


@router.post("/{hypothesis_id}/stop", response_model=HypothesisOut)
def stop_validation(
    workspace_id: int,
    hypothesis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Best-effort stop: the investigation itself is a blocking tool-calling
    loop with no cooperative cancellation point (same underlying loop Scout
    chat uses), so this can't kill it mid-call the instant it's clicked —
    but it resolves the hypothesis's visible state immediately, and
    _run_validate_bg checks stop_requested before applying its eventual
    result so a late-arriving verdict can't overwrite this."""
    _assert_member(workspace_id, current_user, db)
    h = _get_hypothesis(workspace_id, hypothesis_id, db)
    if h.status != "validating":
        raise HTTPException(status_code=400, detail="This hypothesis isn't being validated right now")
    h.stop_requested = True
    h.status = "inconclusive"
    h.verdict = "Stopped by user before the investigation finished."
    db.add(h)
    db.commit()
    db.refresh(h)
    return _serialize(h)


@router.post("/{hypothesis_id}/validate/stream")
def validate_hypothesis_stream(
    workspace_id: int,
    hypothesis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    h = _get_hypothesis(workspace_id, hypothesis_id, db)
    h.status = "validating"
    db.add(h)
    db.commit()
    statement, dataset_id = h.statement, h.dataset_id
    image = {"key": h.image_key, "media_type": h.image_content_type} if h.image_key else None

    def event_stream():
        outcome: dict | None = None
        error_message: str | None = None
        for event in run_hypothesis_validation_stream(
            statement=statement, workspace_id=workspace_id, dataset_id=dataset_id, db=db, user=current_user,
            image=image,
        ):
            if event["type"] == "result":
                outcome = {**event["hypothesis"], "tool_trace": event["tool_trace"]}
            elif event["type"] == "error":
                error_message = event.get("message")
            yield f"data: {json.dumps(event, default=str)}\n\n"

        row = _get_hypothesis(workspace_id, hypothesis_id, db)
        if outcome is not None:
            row = _apply_validation(row, outcome, db)
        else:
            row.status = "error"
            row.verdict = error_message or "Validation failed — please try again."
            db.add(row)
            db.commit()
        yield f"data: {json.dumps({'type': 'persisted', 'id': row.id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
