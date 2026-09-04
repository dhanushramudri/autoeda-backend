"""add image_key/image_content_type to hypotheses

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-04 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("hypotheses", sa.Column("image_key", sa.String(length=500), nullable=True))
    op.add_column("hypotheses", sa.Column("image_content_type", sa.String(length=120), nullable=True))


def downgrade() -> None:
    op.drop_column("hypotheses", "image_content_type")
    op.drop_column("hypotheses", "image_key")
