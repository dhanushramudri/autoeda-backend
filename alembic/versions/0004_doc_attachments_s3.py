"""add s3_key to doc_attachments, make file_data nullable

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-19 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("doc_attachments", sa.Column("s3_key", sa.String(length=500), nullable=True))
    op.alter_column("doc_attachments", "file_data", existing_type=sa.LargeBinary(), nullable=True)


def downgrade() -> None:
    op.alter_column("doc_attachments", "file_data", existing_type=sa.LargeBinary(), nullable=False)
    op.drop_column("doc_attachments", "s3_key")
