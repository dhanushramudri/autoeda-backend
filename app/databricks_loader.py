"""Shared Databricks dataset loading — used by every router that reads a
materialized dataset's underlying data (profile, SQL editor, joins, warehouse,
refresh pipeline). Centralized so 'source_type == databricks' only needs to be
handled correctly in one place.
"""
import pandas as pd

from .models.dataset import Dataset


def get_databricks_pushdown_context(ds: Dataset, extra_config: dict) -> tuple[dict, str]:
    """Returns (connector_config, source_sql) for running aggregate SQL directly against
    Databricks — used to compute full-table stats without pulling rows into pandas.
    source_sql is subquery-able: either a quoted catalog.schema.table or an arbitrary
    SELECT (from a SQL-query-imported dataset).
    """
    from .database import SessionLocal
    from .models.data_source import DataSource
    from .routers.sources import _build_connector_config

    if not ds.source_id:
        raise ValueError(f"Dataset {ds.id} has no linked Databricks source")

    session = SessionLocal()
    try:
        source = session.query(DataSource).filter(DataSource.id == ds.source_id).first()
    finally:
        session.close()
    if not source:
        raise ValueError(f"Linked Databricks source for dataset {ds.id} not found")

    cfg = _build_connector_config(source)

    query = extra_config.get("query")
    if query:
        return cfg, query

    table = extra_config.get("table") or ds.source_table
    if not table:
        raise ValueError(f"Dataset {ds.id} has no table or query to push down against")

    parts = table.split(".")
    if len(parts) == 3:
        source_sql = f"`{parts[0]}`.`{parts[1]}`.`{parts[2]}`"
    else:
        source_sql = table
    return cfg, source_sql


def load_databricks_dataframe(ds: Dataset, extra_config: dict, limit: int | None = None) -> pd.DataFrame:
    from .database import SessionLocal
    from .models.data_source import DataSource
    from .routers.sources import _build_connector_config
    from .connectors.db_connector import DBConnector

    if not ds.source_id:
        raise ValueError(f"Dataset {ds.id} has no linked Databricks source")

    session = SessionLocal()
    try:
        source = session.query(DataSource).filter(DataSource.id == ds.source_id).first()
    finally:
        session.close()
    if not source:
        raise ValueError(f"Linked Databricks source for dataset {ds.id} not found")

    cfg = _build_connector_config(source)
    if extra_config.get("query"):
        cfg["query"] = extra_config["query"]
        cfg["table"] = None
    elif extra_config.get("table") or ds.source_table:
        cfg["table"] = extra_config.get("table") or ds.source_table
        cfg["query"] = None

    return DBConnector().load_data(cfg, limit=limit)
