from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import settings

_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

if _is_sqlite:
    _db_path = settings.DATABASE_URL.split("sqlite:///")[-1]
    Path(_db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,      # drops dead connections before use
        pool_recycle=1800,       # recycle connections every 30 min
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from .models import (  # noqa: F401
        user, workspace, dataset, job,
        pipeline_step, column_metadata, data_quality_rule,
        eda_run, saved_chart, named_segment, data_source, feedback,
    )
    # When using Alembic this is a no-op safety net for fresh installs only.
    # Run `alembic upgrade head` for proper migrations.
    Base.metadata.create_all(bind=engine)
    _migrate_feedback_attachments()
    _seed_admin()


def _migrate_feedback_attachments():
    """Add new columns to feedback table.

    Uses a fresh connection per column so a PG 'transaction aborted' state
    from an already-existing column never blocks the next ALTER TABLE.
    """
    from sqlalchemy import text
    cols = [
        ("attachment_path",  "VARCHAR(500)"),
        ("attachment_name",  "VARCHAR(255)"),
        ("attachments_json", "TEXT"),
        ("subject",          "VARCHAR(255)"),
    ]
    for col, col_type in cols:
        with engine.connect() as conn:
            try:
                conn.execute(text(f"ALTER TABLE feedback ADD COLUMN {col} {col_type}"))
                conn.commit()
            except Exception:
                pass  # column already exists — safe to ignore


def _seed_admin():
    from .models.user import User
    from .models.workspace import Workspace, WorkspaceMember
    from .auth import get_password_hash

    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.email == settings.ADMIN_EMAIL).first()
        if admin:
            # Re-hash if the stored hash is corrupted (not a valid bcrypt hash)
            if not admin.hashed_password.startswith("$2"):
                admin.hashed_password = get_password_hash(settings.ADMIN_PASSWORD)
                db.commit()
            return

        admin = User(
            email=settings.ADMIN_EMAIL,
            full_name="Admin User",
            hashed_password=get_password_hash(settings.ADMIN_PASSWORD),
            is_active=True,
            is_admin=True,
        )
        db.add(admin)
        db.flush()

        ws = Workspace(
            name="Demo Workspace",
            description="Default workspace for AutoEDA demonstrations",
            accent_color="#2563eb",
            created_by=admin.id,
        )
        db.add(ws)
        db.flush()

        member = WorkspaceMember(workspace_id=ws.id, user_id=admin.id, role="admin")
        db.add(member)
        db.commit()
    finally:
        db.close()
