import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# Ensure uploads directory exists before mounting
Path("uploads").mkdir(exist_ok=True)

from .database import init_db
from .process_pool import shutdown_pool
from .routers import auth, datasets, eda, jobs, workspaces, extra, sql_editor, join_builder, sources, warehouse, ai as ai_router, feedback as feedback_router, dataset_docs, coe_posts, scout, hypotheses, auto_eda, experiments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("autoeda")


async def _auto_refresh_scheduler():
    """Polls every 60s for datasets due for a scheduled refresh and re-runs the EDA pipeline."""
    import json
    import uuid
    from datetime import datetime, timezone, timedelta

    from .database import SessionLocal
    from .models.dataset import Dataset
    from .models.job import BackgroundJob
    from .tasks import run_eda_pipeline

    while True:
        try:
            db = SessionLocal()
            try:
                now = datetime.now(timezone.utc)
                due = (
                    db.query(Dataset)
                    .filter(Dataset.refresh_interval_minutes.isnot(None))
                    .filter(Dataset.status != "processing")
                    .all()
                )
                for ds in due:
                    last = ds.updated_at
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)
                    if now < last + timedelta(minutes=ds.refresh_interval_minutes):
                        continue

                    ds.status = "processing"
                    db.commit()

                    job_id = str(uuid.uuid4())
                    cfg = json.loads(ds.source_config or "{}")
                    job = BackgroundJob(
                        id=job_id, job_type="scheduled_refresh", status="pending", progress=0,
                        message=f"Scheduled refresh: {ds.name}",
                        dataset_id=ds.id, created_by=ds.created_by,
                    )
                    db.add(job)
                    db.commit()

                    asyncio.create_task(asyncio.to_thread(run_eda_pipeline, job_id, ds.id, ds.file_path, cfg))
            finally:
                db.close()
        except Exception:
            logger.exception("Auto-refresh scheduler tick failed")
        await asyncio.sleep(60)


def _check_one_live_sync(dataset_id: int):
    """Runs in a worker thread: cheap Delta version check, triggers reload only on an actual change."""
    import json
    import uuid

    from .database import SessionLocal
    from .models.dataset import Dataset
    from .models.data_source import DataSource
    from .models.job import BackgroundJob
    from .connectors.db_connector import DBConnector
    from .routers.sources import _build_connector_config
    from .tasks import run_eda_pipeline

    db = SessionLocal()
    try:
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds or not ds.live_sync_enabled or ds.status == "processing":
            return
        source = db.query(DataSource).filter(DataSource.id == ds.source_id).first()
        if not source:
            return

        catalog, schema, table = ds.source_table.split(".")
        cfg = _build_connector_config(source)
        version_info = DBConnector().get_databricks_table_version(cfg, catalog, schema, table)
        current_version = version_info.get("version")

        if current_version is None or current_version == ds.last_synced_version:
            return  # no change since last check — skip the expensive reload

        ds.last_synced_version = current_version
        ds.status = "processing"
        db.commit()

        job_id = str(uuid.uuid4())
        job = BackgroundJob(
            id=job_id, job_type="live_sync", status="pending", progress=0,
            message=f"Live sync: {ds.name} changed (Delta v{current_version}) — reloading",
            dataset_id=ds.id, created_by=ds.created_by,
        )
        db.add(job)
        db.commit()

        cfg2 = json.loads(ds.source_config or "{}")
        run_eda_pipeline(job_id, ds.id, ds.file_path, cfg2)
    except Exception:
        logger.exception(f"Live sync check failed for dataset {dataset_id}")
    finally:
        db.close()


async def _live_sync_scheduler():
    """Polls every 20s for live-sync datasets — cheap Delta version check, reload only on real change."""
    from .database import SessionLocal
    from .models.dataset import Dataset

    while True:
        try:
            db = SessionLocal()
            try:
                dataset_ids = [
                    d.id for d in db.query(Dataset.id).filter(Dataset.live_sync_enabled.is_(True)).all()
                ]
            finally:
                db.close()

            for dataset_id in dataset_ids:
                asyncio.create_task(asyncio.to_thread(_check_one_live_sync, dataset_id))
        except Exception:
            logger.exception("Live-sync scheduler tick failed")
        await asyncio.sleep(20)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("AutoEDA backend started — DB initialised")
    scheduler_task = asyncio.create_task(_auto_refresh_scheduler())
    live_sync_task = asyncio.create_task(_live_sync_scheduler())
    yield
    scheduler_task.cancel()
    live_sync_task.cancel()
    shutdown_pool()


app = FastAPI(
    title="Jman Group AutoEDA API",
    description="Production-grade Automated EDA Platform — Backend API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3000",
        "https://autoeda-frontend-k7rt.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000)
        logger.info(
            "%s %s %s %dms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


app.add_middleware(RequestLoggingMiddleware)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "code": "INTERNAL_ERROR"},
    )


app.include_router(auth.router, prefix="/api/v1")
app.include_router(workspaces.router, prefix="/api/v1")
app.include_router(datasets.router, prefix="/api/v1")
app.include_router(eda.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(extra.router, prefix="/api/v1")
app.include_router(sql_editor.router, prefix="/api/v1")
app.include_router(join_builder.router, prefix="/api/v1")
app.include_router(sources.router, prefix="/api/v1")
app.include_router(warehouse.router, prefix="/api/v1")
app.include_router(ai_router.router, prefix="/api/v1")
app.include_router(feedback_router.router, prefix="/api/v1")
app.include_router(dataset_docs.router, prefix="/api/v1")
app.include_router(coe_posts.router, prefix="/api/v1")
app.include_router(scout.router, prefix="/api/v1")
app.include_router(hypotheses.router, prefix="/api/v1")
app.include_router(auto_eda.router, prefix="/api/v1")
app.include_router(experiments.router, prefix="/api/v1")
# app.include_router(realtime_router.router, prefix="/api/v1")


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": "2.0.0", "service": "Jman AutoEDA"}
