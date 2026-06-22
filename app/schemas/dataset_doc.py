from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DocCategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None


class DocCategoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: Optional[str] = None
    created_by: int
    created_at: datetime
    article_count: int = 0


class LinkedDataset(BaseModel):
    id: int
    name: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    workspace_id: int


class DocAttachmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    article_id: int
    filename: str
    content_type: Optional[str] = None
    file_size_bytes: int
    uploaded_by: int
    uploaded_by_name: Optional[str] = None
    uploaded_at: datetime


class AttachmentPresignRequest(BaseModel):
    filename: str
    content_type: Optional[str] = None
    file_size_bytes: int


class AttachmentPresignResponse(BaseModel):
    upload_id: int
    upload_url: str


class AttachmentConfirmRequest(BaseModel):
    upload_id: int


class ImportAttachmentRequest(BaseModel):
    workspace_id: int
    name: Optional[str] = None


class DocArticleCreate(BaseModel):
    category_id: int
    title: str
    summary: Optional[str] = None
    content: str = ""
    dataset_ids: list[int] = []


class DocArticleUpdate(BaseModel):
    category_id: Optional[int] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    dataset_ids: Optional[list[int]] = None


class DocArticleListItem(BaseModel):
    id: int
    category_id: int
    title: str
    summary: Optional[str] = None
    content_preview: str = ""
    created_by: int
    created_by_name: Optional[str] = None
    created_at: datetime
    updated_by: Optional[int] = None
    updated_by_name: Optional[str] = None
    updated_at: datetime
    datasets: list[LinkedDataset] = []
    attachment_count: int = 0


class DocArticleResponse(DocArticleListItem):
    content: str = ""
    attachments: list[DocAttachmentResponse] = []
