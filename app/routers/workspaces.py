from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..database import get_db
from ..models.dataset import Dataset
from ..models.user import User
from ..models.workspace import Workspace, WorkspaceMember
from ..schemas.workspace import (
    AddMemberRequest,
    WorkspaceCreate,
    WorkspaceMemberInfo,
    WorkspaceResponse,
    WorkspaceUpdate,
)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


def _build_response(ws: Workspace, db: Session) -> WorkspaceResponse:
    members_raw = (
        db.query(WorkspaceMember, User)
        .join(User, WorkspaceMember.user_id == User.id)
        .filter(WorkspaceMember.workspace_id == ws.id)
        .all()
    )
    members = [
        WorkspaceMemberInfo(
            user_id=m.user_id,
            email=u.email,
            full_name=u.full_name,
            role=m.role,
        )
        for m, u in members_raw
    ]
    dataset_count = db.query(Dataset).filter(Dataset.workspace_id == ws.id).count()
    return WorkspaceResponse(
        id=ws.id,
        name=ws.name,
        description=ws.description,
        accent_color=ws.accent_color,
        created_by=ws.created_by,
        created_at=ws.created_at,
        updated_at=ws.updated_at,
        member_count=len(members),
        dataset_count=dataset_count,
        members=members,
    )


@router.get("/", response_model=list[WorkspaceResponse])
def list_workspaces(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    if current_user.is_admin:
        workspaces = db.query(Workspace).all()
    else:
        member_wids = (
            db.query(WorkspaceMember.workspace_id)
            .filter(WorkspaceMember.user_id == current_user.id)
            .subquery()
        )
        workspaces = db.query(Workspace).filter(Workspace.id.in_(member_wids)).all()

    return [_build_response(ws, db) for ws in workspaces]


@router.post("/", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
def create_workspace(
    payload: WorkspaceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ws = Workspace(
        name=payload.name,
        description=payload.description,
        accent_color=payload.accent_color or "#2563eb",
        created_by=current_user.id,
    )
    db.add(ws)
    db.flush()
    member = WorkspaceMember(workspace_id=ws.id, user_id=current_user.id, role="admin")
    db.add(member)
    db.commit()
    db.refresh(ws)
    return _build_response(ws, db)


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    _assert_member(workspace_id, current_user, db)
    return _build_response(ws, db)


@router.put("/{workspace_id}", response_model=WorkspaceResponse)
def update_workspace(
    workspace_id: int,
    payload: WorkspaceUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    _assert_role(workspace_id, current_user, db, ["admin", "analyst"])

    if payload.name is not None:
        ws.name = payload.name
    if payload.description is not None:
        ws.description = payload.description
    if payload.accent_color is not None:
        ws.accent_color = payload.accent_color
    db.commit()
    db.refresh(ws)
    return _build_response(ws, db)


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workspace(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    _assert_role(workspace_id, current_user, db, ["admin"])
    db.delete(ws)
    db.commit()


@router.get("/{workspace_id}/members")
def list_members(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_member(workspace_id, current_user, db)
    rows = (
        db.query(WorkspaceMember, User)
        .join(User, WorkspaceMember.user_id == User.id)
        .filter(WorkspaceMember.workspace_id == workspace_id)
        .all()
    )
    return [
        {
            "id": str(m.id),
            "role": m.role,
            "joined_at": m.joined_at.isoformat() if hasattr(m, "joined_at") and m.joined_at else None,
            "user": {
                "id": str(u.id),
                "email": u.email,
                "full_name": u.full_name,
            },
        }
        for m, u in rows
    ]


@router.post("/{workspace_id}/members", response_model=WorkspaceResponse)
def add_member(
    workspace_id: int,
    payload: AddMemberRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    _assert_role(workspace_id, current_user, db, ["admin"])

    target_user = db.query(User).filter(User.email == payload.email).first()
    if not target_user:
        raise HTTPException(status_code=404, detail=f"User {payload.email} not found")

    existing = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == target_user.id)
        .first()
    )
    if existing:
        existing.role = payload.role
    else:
        member = WorkspaceMember(workspace_id=workspace_id, user_id=target_user.id, role=payload.role)
        db.add(member)
    db.commit()
    return _build_response(ws, db)


@router.delete("/{workspace_id}/members/{member_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_member(
    workspace_id: int,
    member_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _assert_role(workspace_id, current_user, db, ["admin"])
    m = db.query(WorkspaceMember).filter(
        WorkspaceMember.id == member_id,
        WorkspaceMember.workspace_id == workspace_id,
    ).first()
    if not m:
        raise HTTPException(status_code=404, detail="Member not found")
    db.delete(m)
    db.commit()


def _assert_member(workspace_id: int, user: User, db: Session):
    if user.is_admin:
        return
    m = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    if not m:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a workspace member")


def _assert_role(workspace_id: int, user: User, db: Session, roles: list[str]):
    if user.is_admin:
        return
    m = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    if not m or m.role not in roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Requires role: {roles}")
