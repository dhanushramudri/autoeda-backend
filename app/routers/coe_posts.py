"""CoE Pulse: a shared feed for the DS practice — newsletters, upcoming
events, technical findings, certification opportunities, and general
resources. Same free-for-all convention as the Delivery Playbooks library
(any authenticated user can post; not gatekept per-workspace) — the whole
point is that nothing shared in Slack/Teams/email ever gets lost again.
"""
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..database import get_db
from ..models.coe_post import CoePost
from ..models.user import User
from ..schemas.coe_post import CoePostCreate, CoePostResponse

router = APIRouter(tags=["coe_posts"])


def _user_name(db: Session, user_id: int | None) -> str | None:
    if user_id is None:
        return None
    u = db.query(User).filter(User.id == user_id).first()
    return u.full_name if u else None


def _serialize(db: Session, p: CoePost) -> CoePostResponse:
    return CoePostResponse(
        id=p.id, category=p.category, title=p.title, content=p.content,
        link_url=p.link_url, event_date=p.event_date,
        tags=json.loads(p.tags_json) if p.tags_json else [],
        created_by=p.created_by, created_by_name=_user_name(db, p.created_by),
        created_at=p.created_at,
    )


@router.get("/coe-posts", response_model=list[CoePostResponse])
def list_coe_posts(
    category: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    q = db.query(CoePost)
    if category:
        q = q.filter(CoePost.category == category)
    # Newest first, but anything with a future event/deadline floats up
    # regardless of when it was posted — that's the whole point of tracking
    # event_date at all (a workshop announced 2 weeks ago is still relevant
    # right up to its registration deadline).
    posts = q.order_by(CoePost.created_at.desc()).all()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    # Upcoming-event posts first (soonest first), then everything else
    # newest-first.
    upcoming_posts = sorted(
        [p for p in posts if p.event_date and (p.event_date.replace(tzinfo=timezone.utc) if p.event_date.tzinfo is None else p.event_date) >= now],
        key=lambda p: p.event_date,
    )
    other_posts = [p for p in posts if p not in upcoming_posts]
    return [_serialize(db, p) for p in upcoming_posts + other_posts]


@router.post("/coe-posts", response_model=CoePostResponse)
def create_coe_post(
    payload: CoePostCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    p = CoePost(
        category=payload.category, title=payload.title.strip(), content=payload.content,
        link_url=payload.link_url or None, event_date=payload.event_date,
        tags_json=json.dumps(payload.tags) if payload.tags else None,
        created_by=current_user.id,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return _serialize(db, p)


@router.delete("/coe-posts/{post_id}", status_code=204)
def delete_coe_post(
    post_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    p = db.query(CoePost).filter(CoePost.id == post_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Post not found")
    if p.created_by != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Only the author or an admin can delete this post")
    db.delete(p)
    db.commit()
