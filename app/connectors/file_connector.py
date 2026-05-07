import csv
import io
import os
from typing import Optional

import pandas as pd

from .base import BaseConnector


class FileConnector(BaseConnector):
    def connect(self, config: dict):
        path = config.get("file_path")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return path

    def load_data(self, config: dict, limit: Optional[int] = None) -> pd.DataFrame:
        path = self.connect(config)
        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            sep = config.get("delimiter") or self.auto_detect_delimiter(path)
            df = pd.read_csv(path, sep=sep, low_memory=False)
        elif ext in (".xlsx", ".xls"):
            sheet = config.get("sheet_name", 0)
            df = pd.read_excel(path, sheet_name=sheet)
        elif ext == ".json":
            try:
                df = pd.read_json(path)
            except Exception:
                import json
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.json_normalize(data)
                else:
                    raise ValueError("Unsupported JSON structure")
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext in (".tsv", ".txt"):
            df = pd.read_csv(path, sep="\t", low_memory=False)
        else:
            sep = self.auto_detect_delimiter(path)
            df = pd.read_csv(path, sep=sep, low_memory=False)

        if limit:
            df = df.head(limit)
        return df

    def auto_detect_delimiter(self, path: str) -> str:
        try:
            with open(path, "r", errors="ignore") as f:
                sample = f.read(4096)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except Exception:
            return ","

    def get_excel_sheets(self, file_path: str) -> list[str]:
        try:
            xf = pd.ExcelFile(file_path)
            return xf.sheet_names
        except Exception:
            return []
