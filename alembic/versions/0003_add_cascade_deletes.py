"""add ON DELETE CASCADE to all workspace/dataset foreign keys

Revision ID: 0003
Revises: 0002
Create Date: 2025-01-03 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # workspace_members → workspaces
    op.drop_constraint("workspace_members_workspace_id_fkey", "workspace_members", type_="foreignkey")
    op.create_foreign_key(None, "workspace_members", "workspaces", ["workspace_id"], ["id"], ondelete="CASCADE")

    # datasets → workspaces
    op.drop_constraint("datasets_workspace_id_fkey", "datasets", type_="foreignkey")
    op.create_foreign_key(None, "datasets", "workspaces", ["workspace_id"], ["id"], ondelete="CASCADE")

    # data_sources → workspaces
    op.drop_constraint("data_sources_workspace_id_fkey", "data_sources", type_="foreignkey")
    op.create_foreign_key(None, "data_sources", "workspaces", ["workspace_id"], ["id"], ondelete="CASCADE")

    # saved_charts → workspaces
    op.drop_constraint("saved_charts_workspace_id_fkey", "saved_charts", type_="foreignkey")
    op.create_foreign_key(None, "saved_charts", "workspaces", ["workspace_id"], ["id"], ondelete="CASCADE")

    # eda_results → datasets
    op.drop_constraint("eda_results_dataset_id_fkey", "eda_results", type_="foreignkey")
    op.create_foreign_key(None, "eda_results", "datasets", ["dataset_id"], ["id"], ondelete="CASCADE")

    # background_jobs → datasets
    op.drop_constraint("background_jobs_dataset_id_fkey", "background_jobs", type_="foreignkey")
    op.create_foreign_key(None, "background_jobs", "datasets", ["dataset_id"], ["id"], ondelete="SET NULL")

    # saved_charts → datasets
    op.drop_constraint("saved_charts_dataset_id_fkey", "saved_charts", type_="foreignkey")
    op.create_foreign_key(None, "saved_charts", "datasets", ["dataset_id"], ["id"], ondelete="CASCADE")

    # column_metadata → datasets
    op.drop_constraint("column_metadata_dataset_id_fkey", "column_metadata", type_="foreignkey")
    op.create_foreign_key(None, "column_metadata", "datasets", ["dataset_id"], ["id"], ondelete="CASCADE")

    # data_quality_rules → datasets
    op.drop_constraint("data_quality_rules_dataset_id_fkey", "data_quality_rules", type_="foreignkey")
    op.create_foreign_key(None, "data_quality_rules", "datasets", ["dataset_id"], ["id"], ondelete="CASCADE")

    # eda_runs → datasets
    op.drop_constraint("eda_runs_dataset_id_fkey", "eda_runs", type_="foreignkey")
    op.create_foreign_key(None, "eda_runs", "datasets", ["dataset_id"], ["id"], ondelete="CASCADE")

    # pipeline_steps → datasets
    op.drop_constraint("pipeline_steps_dataset_id_fkey", "pipeline_steps", type_="foreignkey")
    op.create_foreign_key(None, "pipeline_steps", "datasets", ["dataset_id"], ["id"], ondelete="CASCADE")

    # named_segments → datasets
    op.drop_constraint("named_segments_dataset_id_fkey", "named_segments", type_="foreignkey")
    op.create_foreign_key(None, "named_segments", "datasets", ["dataset_id"], ["id"], ondelete="CASCADE")


def downgrade() -> None:
    # Reverting cascade deletes (back to RESTRICT)
    for table, fk, ref_table, col, ref_col in [
        ("workspace_members", None, "workspaces", "workspace_id", "id"),
        ("datasets",          None, "workspaces", "workspace_id", "id"),
        ("data_sources",      None, "workspaces", "workspace_id", "id"),
        ("saved_charts",      None, "workspaces", "workspace_id", "id"),
        ("eda_results",       None, "datasets",   "dataset_id",   "id"),
        ("background_jobs",   None, "datasets",   "dataset_id",   "id"),
        ("saved_charts",      None, "datasets",   "dataset_id",   "id"),
        ("column_metadata",   None, "datasets",   "dataset_id",   "id"),
        ("data_quality_rules",None, "datasets",   "dataset_id",   "id"),
        ("eda_runs",          None, "datasets",   "dataset_id",   "id"),
        ("pipeline_steps",    None, "datasets",   "dataset_id",   "id"),
        ("named_segments",    None, "datasets",   "dataset_id",   "id"),
    ]:
        op.drop_constraint(fk, table, type_="foreignkey")
        op.create_foreign_key(None, table, ref_table, [col], [ref_col])
