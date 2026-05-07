from typing import Any, Optional

import pandas as pd
import requests

from .base import BaseConnector


def _extract_by_path(data: Any, path: str) -> Any:
    if not path:
        return data
    for key in path.split("."):
        if isinstance(data, dict):
            data = data.get(key)
        elif isinstance(data, list) and key.isdigit():
            data = data[int(key)]
        else:
            return data
    return data


class RESTAPIConnector(BaseConnector):
    def connect(self, config: dict) -> bool:
        response = requests.request(
            method=config.get("method", "GET"),
            url=config["url"],
            headers=self._build_headers(config),
            auth=self._build_auth(config),
            timeout=10,
        )
        response.raise_for_status()
        return True

    def load_data(self, config: dict, limit: Optional[int] = None) -> pd.DataFrame:
        records = []
        url = config["url"]
        method = config.get("method", "GET").upper()
        headers = self._build_headers(config)
        params = config.get("params") or {}
        body = config.get("body")
        auth = self._build_auth(config)
        json_path = config.get("json_path")
        pagination = config.get("pagination_config") or {}
        max_pages = min(int(pagination.get("max_pages", 10)), 50)
        next_field = pagination.get("next_page_field")

        for page in range(max_pages):
            if method == "GET":
                resp = requests.get(url, headers=headers, params=params, auth=auth, timeout=30)
            else:
                resp = requests.request(method, url, headers=headers, params=params, json=body, auth=auth, timeout=30)
            resp.raise_for_status()

            data = resp.json()
            extracted = _extract_by_path(data, json_path) if json_path else data

            if isinstance(extracted, list):
                records.extend(extracted)
            elif isinstance(extracted, dict):
                records.append(extracted)
            else:
                break

            if not next_field:
                break
            next_url = _extract_by_path(data, next_field)
            if not next_url:
                break
            url = next_url

            if limit and len(records) >= limit:
                break

        if not records:
            return pd.DataFrame()

        df = pd.json_normalize(records) if records else pd.DataFrame()
        if limit:
            df = df.head(limit)
        return df

    def _build_headers(self, config: dict) -> dict:
        headers = dict(config.get("headers") or {})
        auth_type = config.get("auth_type")
        creds = config.get("auth_credentials") or {}
        if auth_type == "bearer":
            headers["Authorization"] = f"Bearer {creds.get('token', '')}"
        elif auth_type == "api_key":
            location = creds.get("location", "header")
            if location == "header":
                headers[creds.get("key_name", "X-API-Key")] = creds.get("key_value", "")
        return headers

    def _build_auth(self, config: dict):
        auth_type = config.get("auth_type")
        creds = config.get("auth_credentials") or {}
        if auth_type == "basic":
            return (creds.get("username", ""), creds.get("password", ""))
        return None

    def test_connection(self, config: dict) -> tuple[bool, str]:
        try:
            self.connect(config)
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)
