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
        eda_run, saved_chart, named_segment, data_source,
    )
    # When using Alembic this is a no-op safety net for fresh installs only.
    # Run `alembic upgrade head` for proper migrations.
    Base.metadata.create_all(bind=engine)
    _seed_admin()


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
