import html as _html
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..database import get_db
from ..models.feedback import Feedback, FeedbackComment, FeedbackCommentVote, FeedbackVote
from ..models.user import User

logger = logging.getLogger("autoeda.routers.feedback")
router = APIRouter(tags=["feedback"])

UPLOADS_DIR = Path("uploads/feedback")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MIME = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "video/mp4", "video/webm", "video/quicktime", "video/x-msvideo",
}
MAX_BYTES = 100 * 1024 * 1024  # 100 MB

VALID_STATUSES = {"open", "in_review", "in_progress", "planned", "completed", "closed"}


# ── Schemas ───────────────────────────────────────────────────────────────────

class AttachmentOut(BaseModel):
    path: str
    name: str


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_email: str | None
    feedback_type: str
    rating: int | None
    subject: str | None = None
    message: str
    page: str | None
    attachment_path: str | None = None
    attachment_name: str | None = None
    attachments_json: str | None = None
    attachments: list[AttachmentOut] = []
    status: str = "open"
    upvote_count: int = 0
    user_has_voted: bool = False
    comment_count: int = 0
    created_at: datetime

    @model_validator(mode="after")
    def populate_attachments(self) -> "FeedbackOut":
        if self.attachments_json:
            try:
                self.attachments = [AttachmentOut(**a) for a in json.loads(self.attachments_json)]
                return self
            except Exception:
                pass
        if self.attachment_path:
            self.attachments = [AttachmentOut(path=self.attachment_path, name=self.attachment_name or "attachment")]
        return self


class FeedbackCommentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    feedback_id: int
    user_id: int
    parent_id: int | None = None
    user_email: str
    user_name: str
    content: str
    is_system: bool
    created_at: datetime
    like_count: int = 0
    dislike_count: int = 0
    user_vote: str | None = None  # "like" | "dislike" | None


class AddCommentRequest(BaseModel):
    content: str
    parent_id: int | None = None


class UpdateStatusRequest(BaseModel):
    status: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enrich(rows: list[Feedback], current_user_id: int, db: Session) -> list[FeedbackOut]:
    if not rows:
        return []

    ids = [r.id for r in rows]

    # Vote counts
    vote_rows = (
        db.query(FeedbackVote.feedback_id, func.count().label("cnt"))
        .filter(FeedbackVote.feedback_id.in_(ids))
        .group_by(FeedbackVote.feedback_id)
        .all()
    )
    vote_counts = {r.feedback_id: r.cnt for r in vote_rows}

    # User's own votes
    user_votes = {
        v.feedback_id
        for v in db.query(FeedbackVote.feedback_id)
        .filter(FeedbackVote.feedback_id.in_(ids), FeedbackVote.user_id == current_user_id)
        .all()
    }

    # Comment counts (exclude system messages)
    comment_rows = (
        db.query(FeedbackComment.feedback_id, func.count().label("cnt"))
        .filter(FeedbackComment.feedback_id.in_(ids), FeedbackComment.is_system == False)  # noqa: E712
        .group_by(FeedbackComment.feedback_id)
        .all()
    )
    comment_counts = {r.feedback_id: r.cnt for r in comment_rows}

    result = []
    for row in rows:
        out = FeedbackOut.model_validate(row)
        out.upvote_count = vote_counts.get(row.id, 0)
        out.user_has_voted = row.id in user_votes
        out.comment_count = comment_counts.get(row.id, 0)
        result.append(out)
    return result


# ── Mention helpers ───────────────────────────────────────────────────────────

def _strip_html(html_content: str) -> str:
    return _html.unescape(re.sub(r'<[^>]+>', ' ', html_content))


def _extract_mentioned_names(content: str) -> list[str]:
    """Return unique @mention names found in HTML comment content."""
    text = _strip_html(content)
    # Match @Word or @First Last (up to 3 capitalised words)
    found = re.findall(r'@([A-Za-z][A-Za-z0-9]+(?:\s[A-Za-z][A-Za-z0-9]+){0,2})', text)
    return list({name.strip() for name in found})


# ── Submit ────────────────────────────────────────────────────────────────────

@router.post("/feedback", status_code=201, response_model=FeedbackOut)
async def submit_feedback(
    feedback_type: str = Form(default="general"),
    rating: Optional[int] = Form(default=None),
    subject: Optional[str] = Form(default=None),
    message: str = Form(...),
    page: Optional[str] = Form(default=None),
    attachments: List[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    if len(message.strip()) < 5:
        raise HTTPException(status_code=422, detail="Message must be at least 5 characters.")

    saved: list[dict] = []
    for upload in attachments:
        if not upload.filename:
            continue
        content_type = upload.content_type or ""
        if content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=400, detail=f"File '{upload.filename}': only images and videos are allowed.")
        data = await upload.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"File '{upload.filename}' exceeds 100 MB limit.")
        ext = Path(upload.filename).suffix.lower()
        safe_name = f"{uuid.uuid4().hex}{ext}"
        (UPLOADS_DIR / safe_name).write_bytes(data)
        saved.append({"path": f"feedback/{safe_name}", "name": upload.filename})

    row = Feedback(
        user_email=current_user.email,
        feedback_type=feedback_type,
        rating=int(rating) if rating is not None else None,
        subject=subject.strip() if subject else None,
        message=message.strip(),
        page=page or None,
        attachment_path=saved[0]["path"] if saved else None,
        attachment_name=saved[0]["name"] if saved else None,
        attachments_json=json.dumps(saved) if saved else None,
        status="open",
    )
    db.add(row)
    db.commit()
    db.refresh(row)


    out = FeedbackOut.model_validate(row)
    return out


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/feedback", response_model=list[FeedbackOut])
def list_feedback(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    rows = db.query(Feedback).order_by(Feedback.created_at.desc()).all()
    return _enrich(rows, current_user.id, db)


# ── Vote (toggle) ─────────────────────────────────────────────────────────────

@router.post("/feedback/{feedback_id}/vote")
def toggle_vote(
    feedback_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")

    existing = db.query(FeedbackVote).filter(
        FeedbackVote.feedback_id == feedback_id,
        FeedbackVote.user_id == current_user.id,
    ).first()

    if existing:
        db.delete(existing)
        db.commit()
        voted = False
    else:
        db.add(FeedbackVote(feedback_id=feedback_id, user_id=current_user.id))
        db.commit()
        voted = True

    count = db.query(func.count()).filter(FeedbackVote.feedback_id == feedback_id).scalar()
    return {"voted": voted, "upvote_count": count}


# ── Comments ──────────────────────────────────────────────────────────────────

@router.get("/feedback/{feedback_id}/comments", response_model=list[FeedbackCommentOut])
def get_comments(
    feedback_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")

    rows = (
        db.query(FeedbackComment, User)
        .join(User, FeedbackComment.user_id == User.id)
        .filter(FeedbackComment.feedback_id == feedback_id)
        .order_by(FeedbackComment.created_at.asc())
        .all()
    )

    comment_ids = [c.id for c, _ in rows]

    # Aggregate like/dislike counts per comment
    vote_agg = (
        db.query(FeedbackCommentVote.comment_id, FeedbackCommentVote.vote_type, func.count().label("cnt"))
        .filter(FeedbackCommentVote.comment_id.in_(comment_ids))
        .group_by(FeedbackCommentVote.comment_id, FeedbackCommentVote.vote_type)
        .all()
    ) if comment_ids else []

    like_counts: dict[int, int] = {}
    dislike_counts: dict[int, int] = {}
    for cid, vtype, cnt in vote_agg:
        if vtype == "like":
            like_counts[cid] = cnt
        else:
            dislike_counts[cid] = cnt

    # User's own votes
    user_votes: dict[int, str] = {
        v.comment_id: v.vote_type
        for v in db.query(FeedbackCommentVote)
        .filter(
            FeedbackCommentVote.comment_id.in_(comment_ids),
            FeedbackCommentVote.user_id == current_user.id,
        )
        .all()
    } if comment_ids else {}

    return [
        FeedbackCommentOut(
            id=c.id,
            feedback_id=c.feedback_id,
            user_id=c.user_id,
            parent_id=c.parent_id,
            user_email=u.email,
            user_name=u.full_name or u.email,
            content=c.content,
            is_system=c.is_system,
            created_at=c.created_at,
            like_count=like_counts.get(c.id, 0),
            dislike_count=dislike_counts.get(c.id, 0),
            user_vote=user_votes.get(c.id),
        )
        for c, u in rows
    ]


@router.post("/feedback/{feedback_id}/comments", response_model=FeedbackCommentOut)
def add_comment(
    feedback_id: int,
    payload: AddCommentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")

    if len(payload.content.strip()) < 1:
        raise HTTPException(status_code=422, detail="Comment cannot be empty")

    # Validate parent_id belongs to same feedback post
    if payload.parent_id is not None:
        parent = db.query(FeedbackComment).filter(
            FeedbackComment.id == payload.parent_id,
            FeedbackComment.feedback_id == feedback_id,
        ).first()
        if not parent:
            raise HTTPException(status_code=404, detail="Parent comment not found")

    comment = FeedbackComment(
        feedback_id=feedback_id,
        user_id=current_user.id,
        parent_id=payload.parent_id,
        content=payload.content.strip(),
        is_system=False,
    )
    db.add(comment)
    db.commit()
    db.refresh(comment)



    return FeedbackCommentOut(
        id=comment.id,
        feedback_id=comment.feedback_id,
        user_id=comment.user_id,
        parent_id=comment.parent_id,
        user_email=current_user.email,
        user_name=current_user.full_name or current_user.email,
        content=comment.content,
        is_system=comment.is_system,
        created_at=comment.created_at,
    )


class CommentVoteRequest(BaseModel):
    vote_type: str  # "like" | "dislike"


@router.post("/feedback/{feedback_id}/comments/{comment_id}/vote")
def vote_comment(
    feedback_id: int,
    comment_id: int,
    payload: CommentVoteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    if payload.vote_type not in ("like", "dislike"):
        raise HTTPException(status_code=422, detail="vote_type must be 'like' or 'dislike'")

    comment = db.query(FeedbackComment).filter(
        FeedbackComment.id == comment_id,
        FeedbackComment.feedback_id == feedback_id,
    ).first()
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    existing = db.query(FeedbackCommentVote).filter(
        FeedbackCommentVote.comment_id == comment_id,
        FeedbackCommentVote.user_id == current_user.id,
    ).first()

    if existing:
        if existing.vote_type == payload.vote_type:
            # Toggle off
            db.delete(existing)
            user_vote = None
        else:
            # Switch vote type
            existing.vote_type = payload.vote_type
            user_vote = payload.vote_type
    else:
        db.add(FeedbackCommentVote(comment_id=comment_id, user_id=current_user.id, vote_type=payload.vote_type))
        user_vote = payload.vote_type

    db.commit()

    like_count = db.query(func.count()).filter(
        FeedbackCommentVote.comment_id == comment_id,
        FeedbackCommentVote.vote_type == "like",
    ).scalar()
    dislike_count = db.query(func.count()).filter(
        FeedbackCommentVote.comment_id == comment_id,
        FeedbackCommentVote.vote_type == "dislike",
    ).scalar()

    return {"user_vote": user_vote, "like_count": like_count, "dislike_count": dislike_count}


@router.delete("/feedback/{feedback_id}/comments/{comment_id}", status_code=204)
def delete_comment(
    feedback_id: int,
    comment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    comment = db.query(FeedbackComment).filter(
        FeedbackComment.id == comment_id,
        FeedbackComment.feedback_id == feedback_id,
    ).first()
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    if comment.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not allowed")
    db.delete(comment)
    db.commit()


# ── Status update (admin) ─────────────────────────────────────────────────────

@router.patch("/feedback/{feedback_id}", response_model=FeedbackOut)
def update_status(
    feedback_id: int,
    payload: UpdateStatusRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")

    if payload.status not in VALID_STATUSES:
        raise HTTPException(status_code=422, detail=f"Invalid status. Must be one of: {', '.join(VALID_STATUSES)}")

    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")

    old_status = fb.status
    fb.status = payload.status
    db.commit()

    # Log status change as a system comment
    status_label = payload.status.replace("_", " ").title()
    system_comment = FeedbackComment(
        feedback_id=feedback_id,
        user_id=current_user.id,
        content=f"Status changed from **{old_status.replace('_', ' ').title()}** to **{status_label}**",
        is_system=True,
    )
    db.add(system_comment)
    db.commit()
    db.refresh(fb)

    return _enrich([fb], current_user.id, db)[0]
