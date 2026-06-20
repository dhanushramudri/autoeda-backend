"""add hypotheses table

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-21 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "hypotheses",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=True),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("origin", sa.String(10), nullable=False),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("category", sa.String(20), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("verdict", sa.Text(), nullable=True),
        sa.Column("evidence_summary", sa.Text(), nullable=True),
        sa.Column("confidence", sa.String(10), nullable=True),
        sa.Column("severity", sa.String(10), nullable=True),
        sa.Column("columns_json", sa.Text(), nullable=True),
        sa.Column("tool_trace_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_hypotheses_workspace_id", "hypotheses", ["workspace_id"])
    op.create_index("ix_hypotheses_dataset_id", "hypotheses", ["dataset_id"])


def downgrade() -> None:
    op.drop_index("ix_hypotheses_dataset_id", table_name="hypotheses")
    op.drop_index("ix_hypotheses_workspace_id", table_name="hypotheses")
    op.drop_table("hypotheses")
