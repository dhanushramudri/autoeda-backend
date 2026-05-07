from datetime import datetime, timezone
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from ..database import Base


def _now():
    return datetime.now(timezone.utc)


class PipelineStep(Base):
    __tablename__ = "pipeline_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id"), nullable=False)
    step_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    operation: Mapped[str] = mapped_column(String(50), nullable=False)
    column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
