from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class WorkspaceCreate(BaseModel):
    name: str
    description: Optional[str] = None
    accent_color: Optional[str] = "#2563eb"


class WorkspaceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    accent_color: Optional[str] = None


class WorkspaceMemberInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: int
    email: str
    full_name: str
    role: str


class WorkspaceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: Optional[str] = None
    accent_color: str
    created_by: int
    created_at: datetime
    updated_at: datetime
    member_count: int = 0
    dataset_count: int = 0
    source_count: int = 0
    members: list[WorkspaceMemberInfo] = []


class AddMemberRequest(BaseModel):
    email: str
    role: str = "analyst"
