from datetime import datetime, timezone
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from ..database import Base


def _now():
    return datetime.now(timezone.utc)


class Hypothesis(Base):
    """A single hypothesis/claim about a workspace's data, AI-generated or
    user-submitted, with its validation status and the real tool-computed
    evidence behind that status. One unified table for both origins — origin
    + nullable created_by distinguishes them rather than splitting into two
    models (mirrors how ScoutMessage stores tool_trace_json as plain Text)."""

    __tablename__ = "hypotheses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=True, index=True)
    created_by: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)  # null = AI-generated

    origin: Mapped[str] = mapped_column(String(10), nullable=False)  # ai | user
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    # pending | validating | supported | refuted | inconclusive | error

    verdict: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(10), nullable=True)  # high | medium | low
    severity: Mapped[str | None] = mapped_column(String(10), nullable=True)  # info | warning | danger
    columns_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_trace_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
