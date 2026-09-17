"""add business_context to auto_eda_runs

Revision ID: 0013
Revises: 0012
Create Date: 2026-09-16 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("auto_eda_runs", sa.Column("business_context", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("auto_eda_runs", "business_context")
