import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..database import get_db
from ..models.feedback import Feedback
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
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/feedback", response_model=list[FeedbackOut])
def list_feedback(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return db.query(Feedback).order_by(Feedback.created_at.desc()).all()
