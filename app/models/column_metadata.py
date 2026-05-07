from datetime import datetime, timezone
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from ..database import Base


def _now():
    return datetime.now(timezone.utc)


class ColumnMetadata(Base):
    __tablename__ = "column_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id"), nullable=False)
    column_name: Mapped[str] = mapped_column(String(255), nullable=False)
    tags_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
