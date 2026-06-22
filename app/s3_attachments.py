"""Presigned S3 access for Dataset Library attachments.

Uploads and downloads go directly between the browser and S3 — bypassing the
Vercel proxy in front of this API, which hard-caps request bodies at ~4.5MB
and can't be configured around without a custom domain on the backend.
"""
import uuid

import boto3
from botocore.client import Config as BotoConfig

from .config import settings

PRESIGN_EXPIRES_IN = 3600


def _client():
    return boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
        endpoint_url=f"https://s3.{settings.AWS_REGION}.amazonaws.com",
        config=BotoConfig(signature_version="s3v4"),
    )


def new_object_key(article_id: int, filename: str) -> str:
    safe_name = filename.replace("/", "_").replace("\\", "_")
    return f"doc-attachments/{article_id}/{uuid.uuid4().hex}-{safe_name}"


def new_dataset_upload_key(workspace_id: int, filename: str) -> str:
    """Transient relay key — the uploaded bytes get pulled back into
    Dataset.file_data on confirm, then deleted from S3. Not permanent storage."""
    safe_name = filename.replace("/", "_").replace("\\", "_")
    return f"dataset-uploads/{workspace_id}/{uuid.uuid4().hex}-{safe_name}"


def presign_put(key: str, content_type: str | None) -> str:
    return _client().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.S3_ATTACHMENTS_BUCKET,
            "Key": key,
            "ContentType": content_type or "application/octet-stream",
        },
        ExpiresIn=PRESIGN_EXPIRES_IN,
    )


def presign_get(key: str, filename: str) -> str:
    return _client().generate_presigned_url(
        "get_object",
        Params={
            "Bucket": settings.S3_ATTACHMENTS_BUCKET,
            "Key": key,
            "ResponseContentDisposition": f'attachment; filename="{filename}"',
        },
        ExpiresIn=PRESIGN_EXPIRES_IN,
    )


def presign_get_inline(key: str) -> str:
    """Like presign_get, but without forcing a download — for content meant to
    render directly in the page (e.g. a Scout chat image), not be saved as a file."""
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.S3_ATTACHMENTS_BUCKET, "Key": key},
        ExpiresIn=PRESIGN_EXPIRES_IN,
    )


def put_object_bytes(key: str, data: bytes, content_type: str | None) -> None:
    _client().put_object(
        Bucket=settings.S3_ATTACHMENTS_BUCKET, Key=key, Body=data,
        ContentType=content_type or "application/octet-stream",
    )


def get_object_bytes(key: str) -> bytes | None:
    try:
        return _client().get_object(Bucket=settings.S3_ATTACHMENTS_BUCKET, Key=key)["Body"].read()
    except Exception:
        return None


def head_object(key: str) -> dict | None:
    try:
        return _client().head_object(Bucket=settings.S3_ATTACHMENTS_BUCKET, Key=key)
    except Exception:
        return None


def delete_object(key: str) -> None:
    try:
        _client().delete_object(Bucket=settings.S3_ATTACHMENTS_BUCKET, Key=key)
    except Exception:
        pass
