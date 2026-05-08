from .base import BaseConnector
from .db_connector import DBConnector
from .file_connector import FileConnector
from .cloud_connector import CloudConnector
from .api_connector import RESTAPIConnector

_DB_TYPES = {
    "postgresql", "mysql", "sqlite", "mssql", "redshift",
    "snowflake", "bigquery", "mongodb",
}

_CLOUD_TYPES = {"s3", "azure_blob", "gcs", "google_drive"}

_API_TYPES = {"rest_api", "graphql"}

_FILE_TYPES = {"csv", "excel", "json", "parquet", "tsv"}


def get_connector(source_type: str) -> BaseConnector:
    if source_type in _DB_TYPES:
        return DBConnector()
    if source_type in _CLOUD_TYPES:
        return CloudConnector()
    if source_type in _API_TYPES:
        return RESTAPIConnector()
    if source_type in _FILE_TYPES:
        return FileConnector()
    raise ValueError(f"Unknown source type: {source_type}")


SOURCE_CATALOG = [
    # ── Databases ──────────────────────────────────────────────────────────
    {"id": "postgresql",  "label": "PostgreSQL",   "group": "Databases",     "icon": "database"},
    {"id": "mysql",       "label": "MySQL",         "group": "Databases",     "icon": "database"},
    {"id": "mssql",       "label": "SQL Server",    "group": "Databases",     "icon": "database"},
    {"id": "sqlite",      "label": "SQLite",        "group": "Databases",     "icon": "database"},
    {"id": "redshift",    "label": "Redshift",      "group": "Databases",     "icon": "database"},
    {"id": "snowflake",   "label": "Snowflake",     "group": "Databases",     "icon": "database"},
    {"id": "bigquery",    "label": "BigQuery",      "group": "Databases",     "icon": "database"},
    {"id": "mongodb",     "label": "MongoDB",       "group": "Databases",     "icon": "database"},
    # ── Cloud Storage ──────────────────────────────────────────────────────
    {"id": "s3",          "label": "Amazon S3",     "group": "Cloud Storage", "icon": "cloud"},
    {"id": "azure_blob",  "label": "Azure Blob",    "group": "Cloud Storage", "icon": "cloud"},
    {"id": "gcs",         "label": "Google Cloud Storage", "group": "Cloud Storage", "icon": "cloud"},
    {"id": "google_drive","label": "Google Drive",  "group": "Cloud Storage", "icon": "cloud"},
    # ── Files ──────────────────────────────────────────────────────────────
    {"id": "csv",         "label": "CSV / TSV",     "group": "Files",         "icon": "file"},
    {"id": "excel",       "label": "Excel",         "group": "Files",         "icon": "file"},
    {"id": "json",        "label": "JSON",          "group": "Files",         "icon": "file"},
    {"id": "parquet",     "label": "Parquet",       "group": "Files",         "icon": "file"},
    # ── APIs ───────────────────────────────────────────────────────────────
    {"id": "rest_api",    "label": "REST API",      "group": "APIs",          "icon": "globe"},
    {"id": "graphql",     "label": "GraphQL",       "group": "APIs",          "icon": "globe"},
]
