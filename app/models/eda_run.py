from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from ..database import Base


def _now():
    return datetime.now(timezone.utc)


class EDARunRecord(Base):
    __tablename__ = "eda_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id"), nullable=False)
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    col_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quality_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    missing_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    triggered_by: Mapped[str] = mapped_column(String(50), nullable=False, default="manual")
