from datetime import datetime
from pydantic import BaseModel, Field


class AutoEdaRunCreate(BaseModel):
    dataset_ids: list[int] = Field(min_length=1)
    business_context: str | None = Field(default=None, max_length=4000)
    # Optional explicit report title — defaults to the workspace name when
    # left blank (see auto_eda_orchestrator.run_auto_eda_stream).
    report_title: str | None = Field(default=None, max_length=200)
    # How many worklist items this run may plan/grow to — previously fixed
    # server-side via the AUTO_EDA_MAX_ITEMS env var; now a per-run choice.
    # Left blank falls back to that same env var (see run_auto_eda_stream).
    max_items: int | None = Field(default=None, ge=1, le=100)


class AutoEdaRunOut(BaseModel):
    id: int
    workspace_id: int
    dataset_ids: list[int]
    status: str
    title: str | None
    business_context: str | None
    max_items: int | None = None
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


class AutoEdaRunUpdate(BaseModel):
    """Manual, direct save of the report's raw markdown source — the
    canvas's plain-text edit mode. No AI involved."""
    markdown: str = Field(max_length=2_000_000)


class AutoEdaAiEditRequest(BaseModel):
    """Selection-scoped AI edit: rewrite exactly the highlighted excerpt per
    a free-text instruction, leaving the rest of the report untouched."""
    selected_text: str = Field(min_length=1, max_length=20_000)
    instruction: str = Field(min_length=1, max_length=2000)


class AutoEdaAiEditResponse(BaseModel):
    markdown: str
    replacement: str
