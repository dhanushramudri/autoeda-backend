from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
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
    image_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    image_content_type: Mapped[str | None] = mapped_column(String(120), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Set by /stop while a background validation is in flight — since the
    # actual tool-calling loop is a blocking call with no cooperative
    # cancellation point, this doesn't kill it instantly, but it does make
    # the eventual _apply_validation call a no-op instead of clobbering the
    # "stopped" state the user already sees (see routers/hypotheses.py).
    stop_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
