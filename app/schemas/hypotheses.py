from datetime import datetime
from pydantic import BaseModel


class HypothesisCreate(BaseModel):
    statement: str
    dataset_id: int | None = None


class GenerateRequest(BaseModel):
    dataset_id: int | None = None
    count: int = 6


class HypothesisOut(BaseModel):
    id: int
    workspace_id: int
    dataset_id: int | None
    origin: str  # ai | user
    title: str | None
    statement: str
    category: str | None
    status: str
    verdict: str | None
    evidence_summary: str | None
    confidence: str | None
    severity: str | None
    columns: list[str] = []
    tool_trace: list[dict] = []
    created_at: datetime
    updated_at: datetime
    validated_at: datetime | None
