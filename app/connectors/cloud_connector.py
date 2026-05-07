import io
import json
import os
from typing import Any, Optional

import pandas as pd

from .base import BaseConnector


def _parse_bytes(content: bytes, key_or_name: str, limit: Optional[int] = None) -> pd.DataFrame:
    ext = os.path.splitext(key_or_name)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(io.BytesIO(content))
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(content))
    elif ext == ".json":
        try:
            df = pd.read_json(io.BytesIO(content))
        except Exception:
            data = json.loads(content)
            df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
    else:
        import csv
        sample = content[:4096].decode("utf-8", errors="ignore")
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except Exception:
            sep = ","
        df = pd.read_csv(io.BytesIO(content), sep=sep, low_memory=False)

    return df.head(limit) if limit else df


class CloudConnector(BaseConnector):
    def connect(self, config: dict) -> Any:
        cloud_type = config.get("cloud_type", "s3")
        if cloud_type == "s3":
            import boto3
            return boto3.client(
                "s3",
                aws_access_key_id=config["aws_access_key_id"],
                aws_secret_access_key=config["aws_secret_access_key"],
                region_name=config.get("region", "us-east-1"),
            )
        elif cloud_type == "azure":
            from azure.storage.blob import BlobServiceClient
            return BlobServiceClient.from_connection_string(config["connection_string"])
        elif cloud_type == "gcs":
            from google.cloud import storage
            from google.oauth2 import service_account
            sa_info = json.loads(config["service_account_json"])
            creds = service_account.Credentials.from_service_account_info(sa_info)
            return storage.Client(credentials=creds)
        raise ValueError(f"Unknown cloud_type: {config.get('cloud_type')}")

    def load_data(self, config: dict, limit: Optional[int] = None) -> pd.DataFrame:
        cloud_type = config.get("cloud_type", "s3")

        if cloud_type == "s3":
            client = self.connect(config)
            response = client.get_object(Bucket=config["bucket"], Key=config["key"])
            content = response["Body"].read()
            return _parse_bytes(content, config["key"], limit)

        elif cloud_type == "azure":
            client = self.connect(config)
            blob_client = client.get_blob_client(
                container=config["container"], blob=config["blob_name"]
            )
            content = blob_client.download_blob().readall()
            return _parse_bytes(content, config["blob_name"], limit)

        elif cloud_type == "gcs":
            client = self.connect(config)
            bucket = client.bucket(config["bucket"])
            blob = bucket.blob(config["blob_name"])
            content = blob.download_as_bytes()
            return _parse_bytes(content, config["blob_name"], limit)

        raise ValueError(f"Unknown cloud_type: {cloud_type}")

    def list_s3_objects(self, config: dict) -> list[str]:
        client = self.connect(config)
        paginator = client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=config["bucket"]):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys[:500]
