"""Client-chosen storage destination for uploaded file datasets.

By default (a workspace's `storage_destination_source_id` is null — the
normal case, unchanged from before this module existed) uploaded file bytes
are stored the way they always were: in `Dataset.file_data`, our own
database. A workspace can instead point new uploads at a data source it
already owns (its own S3 bucket, or its own Databricks workspace) so we never
hold a copy of that data ourselves. This module is the single place that
decides, on write, which of those two happens, and on read, how to get the
bytes back regardless of where they live.
"""
import base64
import io
import json
import os
from typing import Optional

from sqlalchemy.orm import Session

from .models.data_source import DataSource
from .models.workspace import Workspace

# Only source types we actually know how to write TO (not just read from).
# Snowflake and the rest of the catalog remain read-only sources for now.
DESTINATION_CAPABLE_TYPES = {"s3", "databricks"}

_SECRET = (os.getenv("SECRET_KEY") or "autoeda-secret-key-change-in-prod").encode()


def _decrypt(enc: str) -> str:
    # Mirrors app/routers/sources.py's _decrypt exactly — same XOR scheme,
    # duplicated here rather than imported so this module never depends on
    # a router module.
    encrypted = base64.b64decode(enc.encode())
    key = (_SECRET * ((len(encrypted) // len(_SECRET)) + 1))[:len(encrypted)]
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def _connector_config(source: DataSource) -> dict:
    creds = json.loads(_decrypt(source.credentials_enc)) if source.credentials_enc else {}
    cfg = json.loads(source.config) if source.config else {}
    return {**cfg, **creds, "db_type": source.source_type, "cloud_type": source.source_type}


def get_destination_source(db: Session, workspace_id: int) -> Optional[DataSource]:
    """The workspace's configured destination, or None to keep default behavior."""
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws or not ws.storage_destination_source_id:
        return None
    src = db.query(DataSource).filter(
        DataSource.id == ws.storage_destination_source_id,
        DataSource.workspace_id == workspace_id,
    ).first()
    if not src or src.source_type not in DESTINATION_CAPABLE_TYPES:
        return None
    return src


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def store_uploaded_file(
    db: Session, workspace_id: int, dataset_id: int, filename: str, content: bytes,
) -> dict:
    """Decide where a newly-uploaded file's bytes should live.

    Returns the Dataset fields to assign — either file_data populated and the
    external_* fields left null (default, current behavior), or file_data left
    null with external_storage_type/uri pointing at the client's own source.
    """
    dest = get_destination_source(db, workspace_id)
    if dest is None:
        return {"file_data": content, "external_storage_type": None, "external_storage_uri": None}

    if dest.source_type == "s3":
        import boto3

        cfg = _connector_config(dest)
        client = boto3.client(
            "s3",
            aws_access_key_id=cfg["aws_access_key_id"],
            aws_secret_access_key=cfg["aws_secret_access_key"],
            region_name=cfg.get("region", "us-east-1"),
        )
        prefix = (cfg.get("prefix") or "autoeda-uploads").strip("/")
        key = f"{prefix}/workspace_{workspace_id}/dataset_{dataset_id}_{_safe_name(filename)}"
        client.put_object(Bucket=cfg["bucket"], Key=key, Body=content)
        return {
            "file_data": None,
            "external_storage_type": "s3",
            "external_storage_uri": f"s3://{cfg['bucket']}/{key}",
        }

    if dest.source_type == "databricks":
        from .connectors.db_connector import DBConnector
        from .connectors.file_connector import load_from_bytes

        cfg = _connector_config(dest)
        df = load_from_bytes(content, filename, {})
        catalog = cfg.get("catalog") or "autoeda"
        schema = f"workspace_{workspace_id}"
        table = f"dataset_{dataset_id}"
        DBConnector().write_to_databricks(cfg, df, catalog, schema, table, mode="overwrite")
        return {
            "file_data": None,
            "external_storage_type": "databricks",
            "external_storage_uri": f"databricks://{catalog}.{schema}.{table}",
        }

    # Shouldn't happen — get_destination_source already filters to known types.
    return {"file_data": content, "external_storage_type": None, "external_storage_uri": None}


def fetch_external_bytes(ds) -> bytes:
    """Read a dataset's bytes back from wherever store_uploaded_file put them.

    Takes only the (already session-bound) Dataset — it resolves its own DB
    session via `ds`, so every read call site can use it as a drop-in
    replacement for `ds.file_data` without threading a `db` argument through
    functions that don't already take one.
    """
    from sqlalchemy.orm import object_session

    db = object_session(ds)
    if ds.external_storage_type == "s3":
        import boto3

        src = db.query(DataSource).filter(DataSource.id == ds.source_id).first()
        if not src:
            raise FileNotFoundError(f"Storage destination for dataset {ds.id} no longer exists")
        cfg = _connector_config(src)
        client = boto3.client(
            "s3",
            aws_access_key_id=cfg["aws_access_key_id"],
            aws_secret_access_key=cfg["aws_secret_access_key"],
            region_name=cfg.get("region", "us-east-1"),
        )
        _, _, rest = ds.external_storage_uri.partition("s3://")
        bucket, _, key = rest.partition("/")
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()

    if ds.external_storage_type == "databricks":
        from .connectors.db_connector import DBConnector

        src = db.query(DataSource).filter(DataSource.id == ds.source_id).first()
        if not src:
            raise FileNotFoundError(f"Storage destination for dataset {ds.id} no longer exists")
        cfg = _connector_config(src)
        _, _, fqtn = ds.external_storage_uri.partition("databricks://")
        catalog, schema, table = fqtn.split(".")
        df = DBConnector().load_data({**cfg, "query": f"SELECT * FROM `{catalog}`.`{schema}`.`{table}`"})
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue()

    raise FileNotFoundError(f"Dataset {ds.id} has no recognized external storage")
