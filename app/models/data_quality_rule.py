from datetime import datetime, timezone
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from ..database import Base


def _now():
    return datetime.now(timezone.utc)


class DataQualityRule(Base):
    __tablename__ = "data_quality_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id"), nullable=False)
    column_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    rule_type: Mapped[str] = mapped_column(String(50), nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
