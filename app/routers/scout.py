import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..ai.agent.orchestrator import run_agent_turn, run_agent_turn_stream
from ..ai.llm import provider_name
from ..auth import get_current_active_user
from ..database import get_db
from ..dataset_access import dataset_visibility_filter
from ..models.dataset import Dataset
from ..models.scout import ScoutConversation, ScoutMessage
from ..models.user import User
from ..models.workspace import WorkspaceMember
from ..s3_attachments import presign_get_inline, presign_put
from ..schemas.scout import (
    ScoutConversationOut, ScoutImagePresignRequest, ScoutImagePresignResponse,
    ScoutMessageIn, ScoutMessageOut, ScoutSuggestion, ScoutThread,
)

router = APIRouter(prefix="/workspaces/{workspace_id}/scout", tags=["scout"])

MAX_IMAGE_BYTES = 8 * 1024 * 1024
_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
_IMAGE_ATTACH_ERROR = "Image attachments require the Claude provider to be active."


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
        image_url=presign_get_inline(m.image_key) if m.image_key else None,
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


@router.delete("/conversations/{conversation_id}/messages/{message_id}/truncate")
def truncate_from_message(
    workspace_id: int,
    conversation_id: int,
    message_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Deletes a message and everything after it in the conversation — the
    backend half of 'edit & resend': the client truncates from the message
    being edited, then sends the edited text as a normal new message."""
    _assert_member(workspace_id, current_user, db)
    convo = _get_conversation(workspace_id, conversation_id, db)
    target = (
        db.query(ScoutMessage)
        .filter(ScoutMessage.id == message_id, ScoutMessage.conversation_id == convo.id)
        .first()
    )
    if not target:
        raise HTTPException(status_code=404, detail="Message not found")
    db.query(ScoutMessage).filter(
        ScoutMessage.conversation_id == convo.id,
        ScoutMessage.created_at >= target.created_at,
    ).delete(synchronize_session=False)
    db.commit()
    return {"message": "Truncated"}


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
    if not payload.message.strip() and not payload.image_key:
        raise HTTPException(status_code=400, detail="message cannot be empty")
    if payload.image_key and provider_name() != "claude":
        raise HTTPException(status_code=400, detail=_IMAGE_ATTACH_ERROR)

    history_rows = (
        db.query(ScoutMessage)
        .filter(ScoutMessage.conversation_id == convo.id)
        .order_by(ScoutMessage.created_at.asc())
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in history_rows]

    db.add(ScoutMessage(
        conversation_id=convo.id, role="user", content=payload.message,
        image_key=payload.image_key, image_content_type=payload.image_content_type,
    ))
    if not convo.title:
        convo.title = _title_from_message(payload.message) or "Image attachment"
    db.commit()

    image = {"key": payload.image_key, "media_type": payload.image_content_type} if payload.image_key else None
    result = run_agent_turn(
        message=payload.message,
        history=history,
        workspace_id=workspace_id,
        db=db,
        user=current_user,
        mode=payload.mode,
        image=image,
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
    if not payload.message.strip() and not payload.image_key:
        raise HTTPException(status_code=400, detail="message cannot be empty")
    if payload.image_key and provider_name() != "claude":
        raise HTTPException(status_code=400, detail=_IMAGE_ATTACH_ERROR)

    history_rows = (
        db.query(ScoutMessage)
        .filter(ScoutMessage.conversation_id == convo.id)
        .order_by(ScoutMessage.created_at.asc())
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in history_rows]

    db.add(ScoutMessage(
        conversation_id=convo.id, role="user", content=payload.message,
        image_key=payload.image_key, image_content_type=payload.image_content_type,
    ))
    if not convo.title:
        convo.title = _title_from_message(payload.message) or "Image attachment"
    db.commit()

    image = {"key": payload.image_key, "media_type": payload.image_content_type} if payload.image_key else None

    def event_stream():
        full_answer = ""
        tool_trace: list[dict] = []
        error_message: str | None = None
        for event in run_agent_turn_stream(
            message=payload.message, history=history, workspace_id=workspace_id,
            db=db, user=current_user, mode=payload.mode, image=image,
        ):
            if event["type"] == "done":
                full_answer = event["answer"]
                tool_trace = event["tool_trace"]
            elif event["type"] == "error":
                error_message = event.get("message")
            yield f"data: {json.dumps(event, default=str)}\n\n"

        assistant_msg = ScoutMessage(
            conversation_id=convo.id, role="assistant",
            content=full_answer or error_message or "I wasn't able to find an answer.",
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


@router.post("/images/presign", response_model=ScoutImagePresignResponse)
def presign_scout_image(
    workspace_id: int,
    payload: ScoutImagePresignRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Returns a presigned S3 PUT URL so the browser uploads the image
    directly to S3 — the chat message endpoints only ever receive the
    resulting key, never the image bytes, so requests stay tiny regardless
    of image size (and never hit the Vercel proxy's ~4.5MB body limit)."""
    _assert_member(workspace_id, current_user, db)
    if payload.content_type not in _IMAGE_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {payload.content_type}")
    if payload.size_bytes > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail=f"Image exceeds the {MAX_IMAGE_BYTES // (1024*1024)}MB limit")

    safe_name = payload.filename.replace("/", "_").replace("\\", "_")
    key = f"scout-images/{workspace_id}/{uuid.uuid4().hex}-{safe_name}"
    return ScoutImagePresignResponse(upload_url=presign_put(key, payload.content_type), image_key=key)


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
