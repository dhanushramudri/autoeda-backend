from datetime import datetime
from pydantic import BaseModel


class ScoutMessageIn(BaseModel):
    message: str
    mode: str = "agent"  # "agent" | "chat"
    image_key: str | None = None
    image_content_type: str | None = None


class ScoutMessageOut(BaseModel):
    id: int
    role: str
    content: str
    mode: str | None = None
    tool_trace: list[dict] = []
    image_url: str | None = None
    created_at: datetime


class ScoutImagePresignRequest(BaseModel):
    filename: str
    content_type: str
    size_bytes: int


class ScoutImagePresignResponse(BaseModel):
    upload_url: str
    image_key: str


class ScoutConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ScoutThread(BaseModel):
    conversation_id: int
    messages: list[ScoutMessageOut]


class ScoutSuggestion(BaseModel):
    label: str
