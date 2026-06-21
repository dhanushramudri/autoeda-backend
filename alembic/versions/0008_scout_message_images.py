"""add image_key/image_content_type to scout_messages

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-21 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("scout_messages", sa.Column("image_key", sa.String(length=500), nullable=True))
    op.add_column("scout_messages", sa.Column("image_content_type", sa.String(length=120), nullable=True))


def downgrade() -> None:
    op.drop_column("scout_messages", "image_content_type")
    op.drop_column("scout_messages", "image_key")
