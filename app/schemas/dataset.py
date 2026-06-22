from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    source_type: str
    config: dict[str, Any] = {}


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    workspace_id: int
    name: str
    description: Optional[str] = None
    source_type: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    content_hash: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DatasetPreview(BaseModel):
    columns: list[str]
    dtypes: dict[str, str]
    rows: list[dict[str, Any]]
    total_rows: int


class DatasetCreateResponse(DatasetResponse):
    job_id: Optional[str] = None


class DatasetUploadPresignRequest(BaseModel):
    filename: str
    content_type: Optional[str] = None
    file_size_bytes: int


class DatasetUploadPresignResponse(BaseModel):
    s3_key: str
    upload_url: str
