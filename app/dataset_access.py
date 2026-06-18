"""Shared dataset-visibility rules.

Datasets uploaded by GLOBAL_DATASET_EMAIL (see config.py) act as a shared
template/sample library: they show up in every workspace's dataset list and
are fully accessible (view/edit/delete) by any authenticated user, regardless
of workspace membership. Every other dataset stays strictly scoped to its own
workspace's members.

Centralized here rather than duplicated per-router (the usual convention in
this codebase) because an access-control rule that drifts between files is
a real security bug waiting to happen — every router that gates dataset
access should import from here.
"""

from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from .config import settings
from .models.dataset import Dataset
from .models.user import User
from .models.workspace import WorkspaceMember


def global_dataset_user_id(db: Session) -> int | None:
    user = db.query(User).filter(User.email == settings.GLOBAL_DATASET_EMAIL).first()
    return user.id if user else None


def dataset_visibility_filter(db: Session, workspace_id: int):
    """SQLAlchemy filter for listing: a workspace's own datasets plus every
    dataset uploaded by the global-sharing account."""
    gid = global_dataset_user_id(db)
    if gid is not None:
        return or_(Dataset.workspace_id == workspace_id, Dataset.created_by == gid)
    return Dataset.workspace_id == workspace_id


def accessible_datasets_query(db: Session, user: User):
    """All datasets a user can see across every workspace they belong to,
    plus every globally-shared dataset. Used by the dataset-doc hub's
    dataset picker, which is cross-workspace by design."""
    if user.is_admin:
        return db.query(Dataset)
    member_ws_ids = [
        m.workspace_id for m in
        db.query(WorkspaceMember).filter(WorkspaceMember.user_id == user.id).all()
    ]
    gid = global_dataset_user_id(db)
    conditions = [Dataset.workspace_id.in_(member_ws_ids)] if member_ws_ids else []
    if gid is not None:
        conditions.append(Dataset.created_by == gid)
    if not conditions:
        return db.query(Dataset).filter(Dataset.id.is_(None))  # no access to anything
    return db.query(Dataset).filter(or_(*conditions))


def has_dataset_access(ds: Dataset, user: User, db: Session) -> bool:
    """Non-raising version of assert_dataset_access — for filtering a list
    (e.g. linked datasets on an article) rather than rejecting a whole request."""
    try:
        assert_dataset_access(ds, user, db)
        return True
    except HTTPException:
        return False


def assert_dataset_access(ds: Dataset, user: User, db: Session, roles: list[str] | None = None) -> None:
    """Authorize user against ds: allowed if they're a member of ds.workspace_id
    (with the given role, if any), if they're a global admin, or if ds was
    uploaded by the global-sharing account."""
    if user.is_admin:
        return
    gid = global_dataset_user_id(db)
    if gid is not None and ds.created_by == gid:
        return
    member = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == ds.workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Access denied")
    if roles and member.role not in roles:
        raise HTTPException(status_code=403, detail=f"Requires role: {roles}")
