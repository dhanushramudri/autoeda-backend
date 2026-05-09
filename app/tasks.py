import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

logger = logging.getLogger("autoeda.tasks")

# One shared executor — tune max_workers to your CPU count
_executor = ThreadPoolExecutor(max_workers=4)


def _update_job(db, job_id: str, status: str, progress: int, message: str, error: str = None):
    """Update job in-place using an existing session — no open/close per call."""
    from .models.job import BackgroundJob
    job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
    if job:
        job.status = status
        job.progress = progress
        job.message = message
        if error:
            job.error = error
        job.updated_at = datetime.now(timezone.utc)
        db.commit()


def _load_dataframe(dataset_id: int, file_path: str | None, config: dict):
    """Load DataFrame — uses file_path directly if available, no extra DB hit."""
    if file_path:
        from .connectors.file_connector import FileConnector
        return FileConnector().load_data({"file_path": file_path})

    from .database import SessionLocal
    from .models.dataset import Dataset

    db = SessionLocal()
    try:
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found")

        src = ds.source_type
        cfg = json.loads(ds.source_config or "{}")

        if ds.file_path:
            from .connectors.file_connector import FileConnector
            return FileConnector().load_data({"file_path": ds.file_path})
        elif src in ("postgresql", "mysql", "sqlite", "mssql"):
            from .connectors.db_connector import DBConnector
            cfg["db_type"] = src
            return DBConnector().load_data(cfg)
        elif src == "mongodb":
            from .connectors.db_connector import DBConnector
            cfg["db_type"] = "mongodb"
            return DBConnector().load_data(cfg)
        elif src == "rest_api":
            from .connectors.api_connector import RESTAPIConnector
            return RESTAPIConnector().load_data(cfg)
        elif src in ("s3", "azure", "gcs"):
            from .connectors.cloud_connector import CloudConnector
            cfg["cloud_type"] = src
            return CloudConnector().load_data(cfg)
        else:
            raise ValueError(f"Cannot load dataset with source_type={src}")
    finally:
        db.close()


def _run_analysis_step(name: str, fn, df, dataset_id: int, content_hash: str, result_key: str, result_meta: dict):
    """
    Run a single EDA step and store result.
    Returns (name, result) or raises — called in parallel via executor.
    """
    from .database import SessionLocal
    from .cache import store_result

    result = fn(df)
    db = SessionLocal()
    try:
        store_result(db, dataset_id, result_key, result_meta, result, content_hash)
    finally:
        db.close()
    return name, result


def run_eda_pipeline(job_id: str, dataset_id: int, file_path: str | None, config: dict):
    """
    Optimized EDA pipeline:
    - Single DB session for metadata updates
    - Parallel execution of independent analysis steps
    - No redundant thread spawning
    """
    from .database import SessionLocal
    from .models.dataset import Dataset

    db = SessionLocal()
    try:
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        content_hash = ds.content_hash or "" if ds else ""

        # ── Step 1: Load ──────────────────────────────────────────────
        _update_job(db, job_id, "running", 5, "Loading dataset...")
        df = _load_dataframe(dataset_id, file_path, config)

        # ── Step 2: Metadata (instant) ────────────────────────────────
        _update_job(db, job_id, "running", 15, "Updating metadata...")
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if ds:
            ds.row_count = len(df)
            ds.column_count = len(df.columns)
            ds.status = "processing"
            ds.schema_info = json.dumps({col: str(dtype) for col, dtype in df.dtypes.items()})
            db.commit()

        # ── Step 3: Run all analysis steps in parallel ────────────────
        _update_job(db, job_id, "running", 25, "Running analysis...")

        from .eda.profiler import run_profile
        from .eda.missing import run_missing_analysis
        from .eda.quality_score import run_quality_score
        from .eda.correlations import run_correlations
        from .eda.outliers import run_outlier_detection

        steps = [
            ("profile",       run_profile,                    "profile",       {"type": "profile"}),
            ("missing",       run_missing_analysis,           "missing",       {"type": "missing"}),
            ("quality_score", run_quality_score,              "quality_score", {"type": "quality_score"}),
            ("correlations",  run_correlations,               "correlations",  {"type": "correlations", "method": "pearson"}),
            ("outliers",      lambda df: run_outlier_detection(df, method="iqr"),
                                                              "outliers",      {"type": "outliers", "method": "iqr", "column": None}),
        ]

        futures = {
            _executor.submit(
                _run_analysis_step, name, fn, df, dataset_id, content_hash, key, meta
            ): name
            for name, fn, key, meta in steps
        }

        completed = 0
        progress_per_step = 60 // len(steps)  # 25 → 85 range

        for future in as_completed(futures):
            step_name = futures[future]
            try:
                future.result()
                completed += 1
                _update_job(
                    db, job_id, "running",
                    25 + completed * progress_per_step,
                    f"Finished: {step_name} ({completed}/{len(steps)})"
                )
            except Exception as e:
                logger.warning(f"Step '{step_name}' failed for dataset {dataset_id}: {e}")
                # Don't abort — one bad step shouldn't kill the whole pipeline
                completed += 1

        # ── Step 4: Mark ready ────────────────────────────────────────
        _update_job(db, job_id, "running", 95, "Finalizing...")
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if ds:
            ds.status = "ready"
            db.commit()

        _update_job(db, job_id, "completed", 100, "EDA pipeline complete")
        logger.info(f"EDA pipeline complete for dataset {dataset_id}")

    except Exception as e:
        logger.error(f"EDA pipeline failed for dataset {dataset_id}: {e}", exc_info=True)
        _update_job(db, job_id, "failed", 0, f"Pipeline failed: {str(e)}", error=str(e))
        try:
            ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if ds:
                ds.status = "error"
                ds.error_message = str(e)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()