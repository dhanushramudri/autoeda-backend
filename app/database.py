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
        dataset_doc, scout, hypothesis,
    )
    # When using Alembic this is a no-op safety net for fresh installs only.
    # Run `alembic upgrade head` for proper migrations.
    Base.metadata.create_all(bind=engine)
    _migrate_feedback_attachments()
    _migrate_feedback_comment_parent()
    _seed_admin()
    _seed_test_user()
    _seed_microsoft_emails()
    _promote_jman_admins()


def _migrate_feedback_attachments():
    """Add new columns to feedback table only if they are missing.

    Checks column existence via the inspector first — this avoids running
    ALTER TABLE (which acquires an ACCESS EXCLUSIVE lock) on every startup
    when the columns already exist, preventing the startup hang that causes
    all incoming requests to queue as pending.
    """
    from sqlalchemy import inspect, text

    try:
        inspector = inspect(engine)
        existing = {c["name"] for c in inspector.get_columns("feedback")}
    except Exception:
        return  # feedback table doesn't exist yet — create_all will handle it

    cols = [
        ("attachment_path",  "VARCHAR(500)"),
        ("attachment_name",  "VARCHAR(255)"),
        ("attachments_json", "TEXT"),
        ("subject",          "VARCHAR(255)"),
        ("status",           "VARCHAR(30) NOT NULL DEFAULT 'open'"),
    ]
    for col, col_type in cols:
        if col not in existing:
            with engine.connect() as conn:
                conn.execute(text(f"ALTER TABLE feedback ADD COLUMN {col} {col_type}"))
                conn.commit()


def _migrate_feedback_comment_parent():
    from sqlalchemy import inspect, text

    try:
        inspector = inspect(engine)
        existing = {c["name"] for c in inspector.get_columns("feedback_comments")}
    except Exception:
        return

    if "parent_id" not in existing:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE feedback_comments ADD COLUMN parent_id INTEGER REFERENCES feedback_comments(id) ON DELETE CASCADE"))
            conn.commit()


def _promote_jman_admins():
    """Ensure specific @jmangroup.com accounts are always admins."""
    from .models.user import User

    ADMIN_EMAILS = set(settings.admin_emails_list) if settings.ADMIN_EMAILS else {
        "admin@jmangroup.com",
        "autoeda@jmangroup.com",
        "dhanush.r@jmangroup.com",
    }
    db = SessionLocal()
    try:
        for email in ADMIN_EMAILS:
            user = db.query(User).filter(User.email == email).first()
            if user and not user.is_admin:
                user.is_admin = True
        db.commit()
    finally:
        db.close()


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


def _seed_test_user():
    """Create a second non-admin user for testing real-time features.

    Credentials: testuser@autoeda.local / Test@1234
    Added to every existing workspace so both users share the same workspace.
    """
    from .models.user import User
    from .models.workspace import Workspace, WorkspaceMember
    from .auth import get_password_hash

    TEST_EMAIL = "testuser@autoeda.local"
    TEST_PASSWORD = "Test@1234"

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == TEST_EMAIL).first()
        if existing:
            return

        db.commit()
    finally:
        db.close()


def _seed_microsoft_emails():
    """
    Seed Microsoft-enabled emails from the MICROSOFT_EMAILS env var.
    These emails can login via the /auth/microsoft-mock endpoint.
    """
    from .models.user import User
    from .auth import get_password_hash

    if not settings.MICROSOFT_EMAILS:
        return

    db = SessionLocal()
    try:
        for email in settings.microsoft_emails_list:
            existing = db.query(User).filter(User.email == email).first()
            if existing:
                continue

            full_name = email.split("@")[0].replace(".", " ").title()

            # Create user with a dummy password (will use Microsoft auth)
            user = User(
                email=email,
                full_name=full_name,
                hashed_password=get_password_hash("microsoft_auth_placeholder"),
                is_active=True,
                is_admin=False,
            )
            db.add(user)
            db.flush()

        db.commit()
    finally:
        db.close()
