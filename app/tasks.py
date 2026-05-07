import json
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger("autoeda.tasks")


def _update_job(job_id: str, status: str, progress: int, message: str, error: str = None):
    from .database import SessionLocal
    from .models.job import BackgroundJob

    db = SessionLocal()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
        if job:
            job.status = status
            job.progress = progress
            job.message = message
            if error:
                job.error = error
            job.updated_at = datetime.now(timezone.utc)
            db.commit()
    except Exception as e:
        logger.error(f"Job update error {job_id}: {e}")
    finally:
        db.close()


def _load_dataframe(dataset_id: int, file_path: str | None, config: dict):
    import pandas as pd
    from .database import SessionLocal
    from .models.dataset import Dataset

    db = SessionLocal()
    try:
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found")

        if file_path or ds.file_path:
            from .connectors.file_connector import FileConnector
            fp = file_path or ds.file_path
            return FileConnector().load_data({"file_path": fp})
        elif ds.source_type in ("postgresql", "mysql", "sqlite", "mssql"):
            from .connectors.db_connector import DBConnector
            cfg = json.loads(ds.source_config or "{}")
            cfg["db_type"] = ds.source_type
            return DBConnector().load_data(cfg)
        elif ds.source_type == "mongodb":
            from .connectors.db_connector import DBConnector
            cfg = json.loads(ds.source_config or "{}")
            cfg["db_type"] = "mongodb"
            return DBConnector().load_data(cfg)
        elif ds.source_type == "rest_api":
            from .connectors.api_connector import RESTAPIConnector
            cfg = json.loads(ds.source_config or "{}")
            return RESTAPIConnector().load_data(cfg)
        elif ds.source_type in ("s3", "azure", "gcs"):
            from .connectors.cloud_connector import CloudConnector
            cfg = json.loads(ds.source_config or "{}")
            cfg["cloud_type"] = ds.source_type
            return CloudConnector().load_data(cfg)
        else:
            raise ValueError(f"Cannot load dataset with source_type={ds.source_type}")
    finally:
        db.close()


def run_eda_pipeline(job_id: str, dataset_id: int, file_path: str | None, config: dict):
    def _run():
        from .database import SessionLocal
        from .models.dataset import Dataset
        from .cache import store_result

        db_outer = SessionLocal()
        try:
            ds = db_outer.query(Dataset).filter(Dataset.id == dataset_id).first()
            content_hash = ds.content_hash or "" if ds else ""
        finally:
            db_outer.close()

        try:
            _update_job(job_id, "running", 5, "Loading dataset...")
            df = _load_dataframe(dataset_id, file_path, config)

            _update_job(job_id, "running", 15, "Updating dataset metadata...")
            db = SessionLocal()
            try:
                ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
                if ds:
                    ds.row_count = len(df)
                    ds.column_count = len(df.columns)
                    ds.status = "processing"
                    import json as _json
                    ds.schema_info = _json.dumps({
                        col: str(dtype) for col, dtype in df.dtypes.items()
                    })
                    db.commit()
            finally:
                db.close()

            _update_job(job_id, "running", 25, "Running data profiling...")
            from .eda.profiler import run_profile
            profile = run_profile(df)
            db = SessionLocal()
            store_result(db, dataset_id, "profile", {"type": "profile"}, profile, content_hash)
            db.close()

            _update_job(job_id, "running", 40, "Analyzing missing values...")
            from .eda.missing import run_missing_analysis
            missing = run_missing_analysis(df)
            db = SessionLocal()
            store_result(db, dataset_id, "missing", {"type": "missing"}, missing, content_hash)
            db.close()

            _update_job(job_id, "running", 55, "Computing quality score...")
            from .eda.quality_score import run_quality_score
            quality = run_quality_score(df)
            db = SessionLocal()
            store_result(db, dataset_id, "quality_score", {"type": "quality_score"}, quality, content_hash)
            db.close()

            _update_job(job_id, "running", 70, "Computing correlations...")
            from .eda.correlations import run_correlations
            correlations = run_correlations(df)
            db = SessionLocal()
            store_result(db, dataset_id, "correlations", {"type": "correlations", "method": "pearson"}, correlations, content_hash)
            db.close()

            _update_job(job_id, "running", 85, "Detecting outliers...")
            from .eda.outliers import run_outlier_detection
            outliers = run_outlier_detection(df, method="iqr")
            db = SessionLocal()
            store_result(db, dataset_id, "outliers", {"type": "outliers", "method": "iqr", "column": None}, outliers, content_hash)
            db.close()

            _update_job(job_id, "running", 95, "Finalizing...")
            db = SessionLocal()
            try:
                ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
                if ds:
                    ds.status = "ready"
                    db.commit()
            finally:
                db.close()

            _update_job(job_id, "completed", 100, "EDA pipeline complete")
            logger.info(f"EDA pipeline complete for dataset {dataset_id}")

        except Exception as e:
            logger.error(f"EDA pipeline failed for dataset {dataset_id}: {e}")
            _update_job(job_id, "failed", 0, f"Pipeline failed: {str(e)}", error=str(e))
            db = SessionLocal()
            try:
                ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
                if ds:
                    ds.status = "error"
                    ds.error_message = str(e)
                    db.commit()
            finally:
                db.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
