import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..ai.agent.hypothesis_orchestrator import (
    run_hypothesis_generation, run_hypothesis_generation_stream,
    run_hypothesis_validation, run_hypothesis_validation_stream,
)
from ..auth import get_current_active_user
from ..database import get_db
from ..models.hypothesis import Hypothesis
from ..models.user import User
from ..models.workspace import WorkspaceMember
from ..schemas.hypotheses import GenerateRequest, HypothesisCreate, HypothesisOut

router = APIRouter(prefix="/workspaces/{workspace_id}/hypotheses", tags=["hypotheses"])


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
    h = Hypothesis(
        workspace_id=workspace_id, dataset_id=payload.dataset_id,
        created_by=current_user.id, origin="user",
        statement=payload.statement.strip(), status="pending",
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


@router.post("/generate", response_model=list[HypothesisOut])
def generate_hypotheses(
    workspace_id: int,
    payload: GenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    result = run_hypothesis_generation(
        workspace_id=workspace_id, dataset_id=payload.dataset_id, count=payload.count, db=db, user=current_user,
    )
    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])
    rows = _persist_generated(workspace_id, payload.dataset_id, result["hypotheses"], result["tool_trace"], db)
    return [_serialize(h) for h in rows]


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


@router.post("/{hypothesis_id}/validate", response_model=HypothesisOut)
def validate_hypothesis(
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

    result = run_hypothesis_validation(
        statement=h.statement, workspace_id=workspace_id, dataset_id=h.dataset_id, db=db, user=current_user,
    )
    h = _apply_validation(h, result, db)
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

    def event_stream():
        outcome: dict | None = None
        for event in run_hypothesis_validation_stream(
            statement=statement, workspace_id=workspace_id, dataset_id=dataset_id, db=db, user=current_user,
        ):
            if event["type"] == "result":
                outcome = {**event["hypothesis"], "tool_trace": event["tool_trace"]}
            yield f"data: {json.dumps(event, default=str)}\n\n"

        row = _get_hypothesis(workspace_id, hypothesis_id, db)
        if outcome is not None:
            row = _apply_validation(row, outcome, db)
        else:
            row.status = "error"
            db.add(row)
            db.commit()
        yield f"data: {json.dumps({'type': 'persisted', 'id': row.id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
