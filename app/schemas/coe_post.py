from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

COE_POST_CATEGORIES = {"newsletter", "event", "finding", "certification", "resource"}


class CoePostCreate(BaseModel):
    category: str = Field(pattern="^(newsletter|event|finding|certification|resource)$")
    title: str
    content: str = ""
    link_url: Optional[str] = None
    event_date: Optional[datetime] = None
    tags: list[str] = []


class CoePostResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    category: str
    title: str
    content: str
    link_url: Optional[str] = None
    event_date: Optional[datetime] = None
    tags: list[str] = []
    created_by: int
    created_by_name: Optional[str] = None
    created_at: datetime
