from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
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
