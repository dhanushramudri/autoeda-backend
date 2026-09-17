from datetime import datetime
from pydantic import BaseModel, Field


class AutoEdaRunCreate(BaseModel):
    dataset_ids: list[int] = Field(min_length=1)
    business_context: str | None = Field(default=None, max_length=4000)
    # Optional explicit report title — defaults to the workspace name when
    # left blank (see auto_eda_orchestrator.run_auto_eda_stream).
    report_title: str | None = Field(default=None, max_length=200)


class AutoEdaRunOut(BaseModel):
    id: int
    workspace_id: int
    dataset_ids: list[int]
    status: str
    title: str | None
    business_context: str | None
    markdown: str | None
    worklist: list[dict] = []
    error: str | None
    created_at: datetime
    updated_at: datetime


class AutoEdaChatSend(BaseModel):
    content: str = Field(min_length=1, max_length=2000)


class AutoEdaChatMessageOut(BaseModel):
    id: int
    run_id: int
    role: str
    content: str
    created_at: datetime
