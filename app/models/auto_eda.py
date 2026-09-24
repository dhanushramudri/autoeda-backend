from datetime import datetime, timezone
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from ..database import Base


def _now():
    return datetime.now(timezone.utc)


class AutoEdaRun(Base):
    """One autonomous EDA run against a dataset: a growing worklist of
    investigations plus the Markdown report assembled from them so far.
    Both `worklist_json` and `markdown` are updated incrementally as the
    run progresses, so a client can poll/reload mid-run and see live
    progress, not just the final result."""

    __tablename__ = "auto_eda_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    # JSON array of dataset ids this run covers — a run can span the whole
    # workspace (every dataset) or be scoped to just one; no FK since it's
    # a variable-length set, not a single reference. Datasets aren't
    # cascade-deleted here on purpose: an old run should stay readable
    # (title/markdown/worklist) even if a dataset it covered was later
    # removed from the workspace.
    dataset_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    # pending | running | pausing | paused | completed | error
    # "pausing" is a transient signal only — a separate request sets it to ask
    # the background loop to stop after its current item; the loop itself is
    # what settles it into "paused" (see auto_eda_orchestrator.run_auto_eda_stream).

    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Free-text business context pasted in by the user before starting the
    # run — used to prioritize which candidate analyses actually get run
    # (see auto_eda_orchestrator._plan_worklist) and to keep AI captions/
    # follow-ups relevant to what the user actually cares about, rather
    # than mechanically enumerating every column.
    business_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    # User-set cap on how many worklist items this run may plan/grow to
    # (1-100, enforced at the API layer — see schemas/auto_eda.py). Null on
    # rows created before this existed, or if left blank — falls back to
    # settings.AUTO_EDA_MAX_ITEMS (see auto_eda_orchestrator.run_auto_eda_stream).
    max_items: Mapped[int | None] = mapped_column(Integer, nullable=True)
    worklist_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class AutoEdaChatMessage(Base):
    """A chat turn attached to one Auto EDA run — lets a user 'steer' a
    still-running pipeline (e.g. "skip the categorical breakdowns", "focus
    more on revenue"), read by run_auto_eda_stream on each iteration
    alongside its pause check. Only user/assistant roles: user messages are
    steering instructions, assistant messages are the model's short summary
    of what it changed in response (or why it didn't)."""

    __tablename__ = "auto_eda_chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("auto_eda_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Set once run_auto_eda_stream has read and acted on a "user" message —
    # lets the loop find only new, not-yet-applied instructions each check.
    applied: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
