import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..ai.agent.orchestrator import run_agent_turn, run_agent_turn_stream
from ..auth import get_current_active_user
from ..database import get_db
from ..dataset_access import dataset_visibility_filter
from ..models.dataset import Dataset
from ..models.scout import ScoutConversation, ScoutMessage
from ..models.user import User
from ..models.workspace import WorkspaceMember
from ..schemas.scout import ScoutConversationOut, ScoutMessageIn, ScoutMessageOut, ScoutSuggestion, ScoutThread

router = APIRouter(prefix="/workspaces/{workspace_id}/scout", tags=["scout"])


def _assert_member(workspace_id: int, user: User, db: Session):
    if user.is_admin:
        return
    member = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a workspace member")


def _get_conversation(workspace_id: int, conversation_id: int, db: Session) -> ScoutConversation:
    convo = db.query(ScoutConversation).filter(
        ScoutConversation.id == conversation_id,
        ScoutConversation.workspace_id == workspace_id,
    ).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return convo


def _title_from_message(message: str) -> str:
    title = " ".join(message.strip().split())
    return title[:57] + "…" if len(title) > 60 else title


def _serialize_message(m: ScoutMessage) -> ScoutMessageOut:
    return ScoutMessageOut(
        id=m.id, role=m.role, content=m.content, mode=m.mode,
        tool_trace=json.loads(m.tool_trace_json) if m.tool_trace_json else [],
        created_at=m.created_at,
    )


@router.get("/conversations", response_model=list[ScoutConversationOut])
def list_conversations(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    convos = (
        db.query(ScoutConversation)
        .filter(ScoutConversation.workspace_id == workspace_id)
        .order_by(ScoutConversation.updated_at.desc())
        .all()
    )
    return [
        ScoutConversationOut(id=c.id, title=c.title or "New conversation", created_at=c.created_at, updated_at=c.updated_at)
        for c in convos
    ]


@router.post("/conversations", response_model=ScoutConversationOut)
def create_conversation(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    convo = ScoutConversation(workspace_id=workspace_id, created_by=current_user.id)
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return ScoutConversationOut(id=convo.id, title="New conversation", created_at=convo.created_at, updated_at=convo.updated_at)


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    workspace_id: int,
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    convo = _get_conversation(workspace_id, conversation_id, db)
    db.delete(convo)
    db.commit()
    return {"message": "Conversation deleted"}


@router.get("/conversations/{conversation_id}/messages", response_model=ScoutThread)
def get_messages(
    workspace_id: int,
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    convo = _get_conversation(workspace_id, conversation_id, db)
    msgs = (
        db.query(ScoutMessage)
        .filter(ScoutMessage.conversation_id == convo.id)
        .order_by(ScoutMessage.created_at.asc())
        .all()
    )
    return ScoutThread(conversation_id=convo.id, messages=[_serialize_message(m) for m in msgs])


@router.post("/conversations/{conversation_id}/messages", response_model=ScoutMessageOut)
def post_message(
    workspace_id: int,
    conversation_id: int,
    payload: ScoutMessageIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    convo = _get_conversation(workspace_id, conversation_id, db)
    if payload.mode not in ("agent", "chat"):
        raise HTTPException(status_code=400, detail="mode must be 'agent' or 'chat'")
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    history_rows = (
        db.query(ScoutMessage)
        .filter(ScoutMessage.conversation_id == convo.id)
        .order_by(ScoutMessage.created_at.asc())
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in history_rows]

    db.add(ScoutMessage(conversation_id=convo.id, role="user", content=payload.message))
    if not convo.title:
        convo.title = _title_from_message(payload.message)
    db.commit()

    result = run_agent_turn(
        message=payload.message,
        history=history,
        workspace_id=workspace_id,
        db=db,
        user=current_user,
        mode=payload.mode,
    )

    assistant_msg = ScoutMessage(
        conversation_id=convo.id,
        role="assistant",
        content=result["answer"],
        mode=payload.mode,
        tool_trace_json=json.dumps(result["tool_trace"], default=str),
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    # Touch updated_at explicitly so conversation ordering reflects latest activity.
    convo.updated_at = assistant_msg.created_at
    db.add(convo)
    db.commit()
    return _serialize_message(assistant_msg)


@router.post("/conversations/{conversation_id}/messages/stream")
def post_message_stream(
    workspace_id: int,
    conversation_id: int,
    payload: ScoutMessageIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """SSE variant of post_message — streams progress events as they happen
    instead of waiting for the full answer. Persists the user + assistant
    messages exactly like the non-streaming endpoint once the stream ends."""
    _assert_member(workspace_id, current_user, db)
    convo = _get_conversation(workspace_id, conversation_id, db)
    if payload.mode not in ("agent", "chat"):
        raise HTTPException(status_code=400, detail="mode must be 'agent' or 'chat'")
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    history_rows = (
        db.query(ScoutMessage)
        .filter(ScoutMessage.conversation_id == convo.id)
        .order_by(ScoutMessage.created_at.asc())
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in history_rows]

    db.add(ScoutMessage(conversation_id=convo.id, role="user", content=payload.message))
    if not convo.title:
        convo.title = _title_from_message(payload.message)
    db.commit()

    def event_stream():
        full_answer = ""
        tool_trace: list[dict] = []
        for event in run_agent_turn_stream(
            message=payload.message, history=history, workspace_id=workspace_id,
            db=db, user=current_user, mode=payload.mode,
        ):
            if event["type"] == "done":
                full_answer = event["answer"]
                tool_trace = event["tool_trace"]
            yield f"data: {json.dumps(event, default=str)}\n\n"

        assistant_msg = ScoutMessage(
            conversation_id=convo.id, role="assistant",
            content=full_answer or "I wasn't able to find an answer.",
            mode=payload.mode,
            tool_trace_json=json.dumps(tool_trace, default=str),
        )
        db.add(assistant_msg)
        db.commit()
        db.refresh(assistant_msg)

        # Touch updated_at explicitly so conversation ordering reflects latest activity.
        convo.updated_at = assistant_msg.created_at
        db.add(convo)
        db.commit()
        yield f"data: {json.dumps({'type': 'persisted', 'message_id': assistant_msg.id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/suggestions", response_model=list[ScoutSuggestion])
def get_suggestions(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    datasets = (
        db.query(Dataset)
        .filter(dataset_visibility_filter(db, workspace_id), Dataset.status == "ready")
        .order_by(Dataset.created_at.desc())
        .limit(3)
        .all()
    )
    if not datasets:
        return [
            ScoutSuggestion(label="What datasets are available in this workspace?"),
        ]

    suggestions = [ScoutSuggestion(label="What datasets are in this workspace and how do they compare?")]
    for d in datasets[:2]:
        suggestions.append(ScoutSuggestion(label=f'What are the main data quality issues in "{d.name}"?'))
    suggestions.append(ScoutSuggestion(label=f'Find outliers in "{datasets[0].name}" and explain possible causes'))
    return suggestions
