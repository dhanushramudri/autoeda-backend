from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


def _now():
    return datetime.now(timezone.utc)


class DsFlowRun(Base):
    """One Data Science Flow run (e.g. churn) against a dataset: the stage timeline, the JSON results
    of every stage, the board-level narrative/report and the deliverable files (enriched CSV in the
    client's own format, account-level file, data dictionary, trained model)."""

    __tablename__ = "ds_flow_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)  # no FK: the run stays readable if the dataset is removed
    created_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    flow_key: Mapped[str] = mapped_column(String(40), nullable=False, default="churn")
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    dataset_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # pending | running | completed | error
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    roles_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # [{key, title, status, started_at, finished_at, seconds, summary, logs: [...]}, ...]
    stages_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    results_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    headline_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    narrative_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    enriched_csv: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    accounts_csv: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    dictionary_csv: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    model_blob: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
