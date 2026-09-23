from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


def _now():
    return datetime.now(timezone.utc)


class Experiment(Base):
    """A model-training request against one dataset/target — the actual
    training runs on the requesting engineer's laptop (via the local Docker
    agent), never on this server. This row is just the queue ticket +
    leaderboard header; ExperimentRun rows are what the agent reports back."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    # classification | regression | null = let the agent auto-detect from the target dtype
    problem_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # Columns excluded before training — ID-like columns and likely-leakage
    # features, either flagged automatically or chosen by the auto-planner
    # (see ai/agent/automl_planner.py), so the agent never blindly trains on
    # every raw column.
    excluded_columns_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Set when this experiment was queued by the auto-planner rather than a
    # human picking dataset/target/features by hand.
    auto_planned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    # [{tool, args, output, reason}, ...] — recipes from the fixed feature
    # engineering toolbox (agent/feature_tools.py) for the agent to apply
    # before training.
    engineered_features_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # queued (waiting for an agent to pick it up) | running | completed | failed
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ExperimentRun(Base):
    """One trained candidate model within an Experiment — one row per
    algorithm the agent tried (per the DS playbook's shortlist heuristic)."""

    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)

    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_importance_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    training_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="completed")  # completed | failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    artifact_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
