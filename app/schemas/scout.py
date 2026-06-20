from datetime import datetime
from pydantic import BaseModel


class ScoutMessageIn(BaseModel):
    message: str
    mode: str = "agent"  # "agent" | "chat"


class ScoutMessageOut(BaseModel):
    id: int
    role: str
    content: str
    mode: str | None = None
    tool_trace: list[dict] = []
    created_at: datetime


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
