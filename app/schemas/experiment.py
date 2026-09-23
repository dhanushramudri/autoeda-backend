from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ExperimentCreate(BaseModel):
    dataset_id: int
    name: str
    target_column: str
    problem_type: Optional[str] = None  # classification | regression | None = auto-detect
    excluded_columns: list[str] = []  # dropped before training — ID-like / likely-leakage features the engineer unchecked


class ExperimentRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    algorithm: str
    params: dict = {}
    metrics: dict = {}
    feature_importance: dict = {}
    training_seconds: Optional[float] = None
    status: str
    error: Optional[str] = None
    has_artifact: bool = False
    artifact_filename: Optional[str] = None
    created_at: datetime


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    workspace_id: int
    dataset_id: int
    dataset_name: Optional[str] = None
    created_by: int
    created_by_name: Optional[str] = None
    name: str
    target_column: str
    problem_type: Optional[str] = None
    excluded_columns: list[str] = []
    auto_planned: bool = False
    rationale: Optional[str] = None
    engineered_features: list[dict] = []
    status: str
    error: Optional[str] = None
    created_at: datetime
    claimed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    runs: list[ExperimentRunOut] = []


# -- Agent-facing payloads --------------------------------------------------

class AgentQueuedExperiment(BaseModel):
    id: int
    workspace_id: int
    dataset_id: int
    name: str
    target_column: str
    problem_type: Optional[str] = None
    excluded_columns: list[str] = []
    engineered_features: list[dict] = []


class AgentRunReport(BaseModel):
    algorithm: str
    params: dict = {}
    metrics: dict = {}
    feature_importance: dict = {}
    training_seconds: Optional[float] = None
    status: str = "completed"
    error: Optional[str] = None


class AgentCompleteRequest(BaseModel):
    status: str  # completed | failed
    error: Optional[str] = None
