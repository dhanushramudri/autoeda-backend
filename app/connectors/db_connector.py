from typing import Any, Optional

import pandas as pd

from .base import BaseConnector

# Safety ceiling for connector row pulls when the caller doesn't specify a
# limit — high enough to cover "give me everything" for the vast majority of
# exploratory tables, low enough to protect the AutoEDA server's memory.
DEFAULT_ROW_LIMIT = 2_000_000


class DBConnector(BaseConnector):

    # ── Databricks REST API helpers (Jobs / Workflows) ─────────────────────────

    def _databricks_rest_base(self, config: dict) -> str:
        host = config["server_hostname"]
        return f"https://{host}/api/2.1"

    def _databricks_rest_headers(self, config: dict) -> dict:
        return {"Authorization": f"Bearer {config.get('access_token')}"}

    def list_databricks_jobs(self, config: dict, limit: int = 50) -> list[dict]:
        import requests
        resp = requests.get(
            f"{self._databricks_rest_base(config)}/jobs/list",
            headers=self._databricks_rest_headers(config),
            params={"limit": limit, "expand_tasks": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        jobs = []
        for j in data.get("jobs", []):
            settings = j.get("settings", {})
            schedule = settings.get("schedule") or {}
            tasks = settings.get("tasks") or []
            jobs.append({
                "job_id": j.get("job_id"),
                "name": settings.get("name", f"Job {j.get('job_id')}"),
                "created_time": j.get("created_time"),
                "creator": (j.get("creator_user_name") or ""),
                "tags": settings.get("tags") or {},
                "task_count": len(tasks) if tasks else 1,
                "schedule": {
                    "cron": schedule.get("quartz_cron_expression"),
                    "timezone": schedule.get("timezone_id"),
                    "paused": schedule.get("pause_status") == "PAUSED",
                } if schedule else None,
                "max_concurrent_runs": settings.get("max_concurrent_runs", 1),
            })
        return jobs

    def list_databricks_active_runs(self, config: dict) -> list[dict]:
        """Lightweight — just active (non-terminal) runs across all jobs, for a badge count."""
        import requests
        resp = requests.get(
            f"{self._databricks_rest_base(config)}/jobs/runs/list",
            headers=self._databricks_rest_headers(config),
            params={"active_only": "true", "limit": 25},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {"run_id": r.get("run_id"), "job_id": r.get("job_id"),
             "life_cycle_state": (r.get("state") or {}).get("life_cycle_state")}
            for r in data.get("runs", [])
        ]

    def list_databricks_job_runs(self, config: dict, job_id: int, limit: int = 10) -> list[dict]:
        import requests
        resp = requests.get(
            f"{self._databricks_rest_base(config)}/jobs/runs/list",
            headers=self._databricks_rest_headers(config),
            params={"job_id": job_id, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        runs = []
        for r in data.get("runs", []):
            state = r.get("state", {})
            start = r.get("start_time")
            end = r.get("end_time")
            duration_ms = (end - start) if (start and end and end > start) else None
            runs.append({
                "run_id": r.get("run_id"),
                "life_cycle_state": state.get("life_cycle_state"),
                "result_state": state.get("result_state"),
                "state_message": state.get("state_message"),
                "start_time": start,
                "end_time": end,
                "duration_ms": duration_ms,
                "run_page_url": r.get("run_page_url"),
                "trigger": r.get("trigger"),
            })
        return runs

    def run_databricks_job_now(self, config: dict, job_id: int, notebook_params: Optional[dict] = None) -> dict:
        import requests
        body: dict = {"job_id": job_id}
        if notebook_params:
            body["notebook_params"] = notebook_params
        resp = requests.post(
            f"{self._databricks_rest_base(config)}/jobs/run-now",
            headers=self._databricks_rest_headers(config),
            json=body,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def cancel_databricks_run(self, config: dict, run_id: int) -> dict:
        import requests
        resp = requests.post(
            f"{self._databricks_rest_base(config)}/jobs/runs/cancel",
            headers=self._databricks_rest_headers(config),
            json={"run_id": run_id},
            timeout=15,
        )
        resp.raise_for_status()
        return {"cancelled": True, "run_id": run_id}

    def get_databricks_run_status(self, config: dict, run_id: int) -> dict:
        import requests
        resp = requests.get(
            f"{self._databricks_rest_base(config)}/jobs/runs/get",
            headers=self._databricks_rest_headers(config),
            params={"run_id": run_id},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state", {})
        return {
            "run_id": data.get("run_id"),
            "life_cycle_state": state.get("life_cycle_state"),
            "result_state": state.get("result_state"),
            "state_message": state.get("state_message"),
            "start_time": data.get("start_time"),
            "end_time": data.get("end_time"),
            "run_page_url": data.get("run_page_url"),
        }

    # ── Databricks helpers ────────────────────────────────────────────────────

    def _databricks_kw(self, config: dict) -> dict:
        kw: dict = dict(
            server_hostname=config["server_hostname"],
            http_path=config["http_path"],
            access_token=config.get("access_token"),
        )
        if config.get("catalog"):
            kw["catalog"] = config["catalog"]
        if config.get("schema"):
            kw["schema"] = config["schema"]
        return kw

    def _load_databricks(self, config: dict, limit: Optional[int] = None) -> pd.DataFrame:
        try:
            from databricks import sql as _db_sql
        except ImportError:
            raise ValueError("databricks-sql-connector not installed. Run: pip install databricks-sql-connector")
        kw = self._databricks_kw(config)
        query = config.get("query")
        if not query:
            table = config.get("table", "")
            if not table:
                return pd.DataFrame()
            limit_n = limit or DEFAULT_ROW_LIMIT
            query = f"SELECT * FROM {table} LIMIT {limit_n}"
        elif limit:
            query = f"SELECT * FROM ({query}) AS _q LIMIT {limit}"
        with _db_sql.connect(**kw) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                cols = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=cols)

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self, config: dict) -> Any:
        db_type = config.get("db_type", "postgresql")
        if db_type == "postgresql":
            import psycopg2
            return psycopg2.connect(
                host=config["host"],
                port=config.get("port", 5432),
                user=config["username"],
                password=config["password"],
                dbname=config["database"],
                connect_timeout=10,
            )
        elif db_type == "mysql":
            import pymysql
            return pymysql.connect(
                host=config["host"],
                port=int(config.get("port", 3306)),
                user=config["username"],
                password=config["password"],
                db=config["database"],
                connect_timeout=10,
            )
        elif db_type == "sqlite":
            import sqlite3
            return sqlite3.connect(config["database"])
        elif db_type == "mssql":
            import pyodbc
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={config['host']},{config.get('port', 1433)};"
                f"DATABASE={config['database']};"
                f"UID={config['username']};"
                f"PWD={config['password']}"
            )
            return pyodbc.connect(conn_str, timeout=10)
        elif db_type == "mongodb":
            from pymongo import MongoClient
            return MongoClient(config["uri"])
        elif db_type == "databricks":
            try:
                from databricks import sql as _db_sql
            except ImportError:
                raise ValueError("databricks-sql-connector not installed. Run: pip install databricks-sql-connector")
            kw = self._databricks_kw(config)
            conn = _db_sql.connect(**kw)
            # Validate credentials immediately with a lightweight query
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            return conn
        elif db_type == "fabric":
            import pyodbc
            # Try ODBC Driver 18 first, fall back to 17
            last_err: Exception | None = None
            for driver in ("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"):
                try:
                    conn_str = (
                        f"DRIVER={{{driver}}};"
                        f"SERVER={config['server']},1433;"
                        f"DATABASE={config['database']};"
                        f"UID={config['username']};"
                        f"PWD={config['password']};"
                        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
                    )
                    return pyodbc.connect(conn_str, timeout=30)
                except pyodbc.Error as e:
                    last_err = e
            raise ConnectionError(f"Microsoft Fabric: ODBC Driver 17/18 for SQL Server not found or connection failed. {last_err}")
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")

    def load_data(self, config: dict, limit: Optional[int] = None) -> pd.DataFrame:
        db_type = config.get("db_type", "postgresql")

        if db_type == "databricks":
            return self._load_databricks(config, limit)

        if db_type == "mongodb":
            client = self.connect(config)
            db = client[config["database"]]
            collection = db[config["collection"]]
            query = config.get("query_filter") or {}
            limit = limit or 10000
            data = list(collection.find(query).limit(limit))
            client.close()
            if data:
                for doc in data:
                    doc.pop("_id", None)
            return pd.DataFrame(data)

        conn = self.connect(config)
        try:
            query = config.get("query")
            if not query:
                table = config.get("table", "unknown_table")
                limit_n = limit or DEFAULT_ROW_LIMIT
                # Fabric / SQL Server use TOP, not LIMIT
                if db_type == "fabric":
                    query = f"SELECT TOP {limit_n} * FROM {table}"
                else:
                    query = f"SELECT * FROM {table} LIMIT {limit_n}"
            elif limit:
                if db_type == "fabric":
                    query = f"SELECT TOP {limit} * FROM ({query}) AS _sub"
                else:
                    query = f"SELECT * FROM ({query}) AS _sub LIMIT {limit}"

            df = pd.read_sql(query, conn)
        finally:
            conn.close()
        return df

    # ── Unity Catalog helpers (Databricks-specific) ───────────────────────────

    def _databricks_connect(self, config: dict):
        try:
            from databricks import sql as _db_sql
        except ImportError:
            raise ValueError("databricks-sql-connector not installed. Run: pip install databricks-sql-connector")
        return _db_sql.connect(**self._databricks_kw(config))

    def write_to_databricks(
        self, config: dict, df: "pd.DataFrame",
        catalog: str, schema: str, table: str, mode: str = "overwrite"
    ) -> dict:
        import math, pandas as pd

        fqtn = f"`{catalog}`.`{schema}`.`{table}`"

        def sql_type(dtype) -> str:
            if pd.api.types.is_bool_dtype(dtype):           return "BOOLEAN"
            if pd.api.types.is_integer_dtype(dtype):        return "BIGINT"
            if pd.api.types.is_float_dtype(dtype):          return "DOUBLE"
            if pd.api.types.is_datetime64_any_dtype(dtype): return "TIMESTAMP"
            return "STRING"

        def fmt_val(v) -> str:
            if v is None:
                return "NULL"
            try:
                if pd.isna(v):
                    return "NULL"
            except Exception:
                pass
            if isinstance(v, bool):
                return "TRUE" if v else "FALSE"
            if isinstance(v, int):
                return str(v)
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return "NULL"
                return repr(v)
            if hasattr(v, "strftime"):
                return f"TIMESTAMP '{v.strftime('%Y-%m-%d %H:%M:%S')}'"
            return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"

        cols_ddl  = ", ".join(f"`{c}` {sql_type(df[c].dtype)}" for c in df.columns)
        col_names = ", ".join(f"`{c}`" for c in df.columns)
        rows_list = df.values.tolist()

        with self._databricks_connect(config) as conn:
            with conn.cursor() as cursor:
                if mode == "overwrite":
                    cursor.execute(f"DROP TABLE IF EXISTS {fqtn}")
                cursor.execute(
                    f"CREATE TABLE IF NOT EXISTS {fqtn} ({cols_ddl}) USING DELTA"
                )
                batch_size = 200
                total = 0
                for i in range(0, len(rows_list), batch_size):
                    batch = rows_list[i : i + batch_size]
                    values_str = ", ".join(
                        "(" + ", ".join(fmt_val(v) for v in row) + ")"
                        for row in batch
                    )
                    cursor.execute(
                        f"INSERT INTO {fqtn} ({col_names}) VALUES {values_str}"
                    )
                    total += len(batch)

        return {
            "rows_written": total,
            "catalog": catalog, "schema": schema, "table": table,
            "fqtn": f"{catalog}.{schema}.{table}", "mode": mode,
        }

    def get_databricks_table_version(self, config: dict, catalog: str, schema: str, table: str) -> dict:
        """Cheap Delta transaction-log read — no data scan, just the latest commit version."""
        fqtn = f"`{catalog}`.`{schema}`.`{table}`"
        with self._databricks_connect(config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DESCRIBE HISTORY {fqtn} LIMIT 1")
                row = cursor.fetchone()
                if not row:
                    return {"version": None, "timestamp": None}
                cols = [d[0] for d in cursor.description]
                data = dict(zip(cols, row))
        return {
            "version": str(data.get("version")),
            "timestamp": str(data.get("timestamp")) if data.get("timestamp") else None,
            "operation": data.get("operation"),
        }

    def list_databricks_catalogs(self, config: dict) -> list[str]:
        with self._databricks_connect(config) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SHOW CATALOGS")
                rows = cursor.fetchall()
        return [r[0] for r in rows]

    def list_databricks_schemas(self, config: dict, catalog: str) -> list[str]:
        with self._databricks_connect(config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SHOW SCHEMAS IN `{catalog}`")
                rows = cursor.fetchall()
        # SHOW SCHEMAS returns (databaseName,) or (namespace, databaseName) depending on version
        # The schema name is always the last non-empty string column
        return [r[-1] if len(r) > 1 else r[0] for r in rows]

    def list_databricks_tables_in_schema(self, config: dict, catalog: str, schema: str) -> list[dict]:
        with self._databricks_connect(config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SHOW TABLES IN `{catalog}`.`{schema}`")
                rows = cursor.fetchall()
                col_names = [d[0].lower() for d in cursor.description]
        results = []
        for r in rows:
            row_dict = dict(zip(col_names, r))
            name = row_dict.get("tablename") or row_dict.get("table_name") or (r[1] if len(r) > 1 else r[0])
            is_temp = bool(row_dict.get("istemporary") or row_dict.get("is_temporary") or False)
            if not is_temp:
                results.append({"name": str(name), "is_temporary": False})
        return results

    def get_databricks_table_stats(self, config: dict, catalog: str, schema: str, table: str) -> dict:
        fqtn = f"`{catalog}`.`{schema}`.`{table}`"
        result: dict = {
            "catalog": catalog, "schema": schema, "table": table,
            "columns": [], "format": None,
            "num_files": None, "size_bytes": None, "row_count": None,
        }
        with self._databricks_connect(config) as conn:
            with conn.cursor() as cursor:
                # Column metadata via information_schema — zero compute
                try:
                    cursor.execute(f"""
                        SELECT column_name, data_type, is_nullable, ordinal_position
                        FROM `{catalog}`.information_schema.columns
                        WHERE table_schema = '{schema}' AND table_name = '{table}'
                        ORDER BY ordinal_position
                    """)
                    result["columns"] = [
                        {"name": r[0], "type": r[1], "nullable": r[2] == "YES"}
                        for r in cursor.fetchall()
                    ]
                except Exception:
                    pass

                # DESCRIBE DETAIL — reads Delta transaction log, zero compute
                try:
                    cursor.execute(f"DESCRIBE DETAIL {fqtn}")
                    detail_row = cursor.fetchone()
                    if detail_row:
                        detail = dict(zip([d[0] for d in cursor.description], detail_row))
                        result["format"] = detail.get("format")
                        result["num_files"] = detail.get("numFiles")
                        result["size_bytes"] = detail.get("sizeInBytes")
                except Exception:
                    pass

                # Row count — fast on Delta tables (uses transaction log stats)
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {fqtn}")
                    row = cursor.fetchone()
                    if row:
                        result["row_count"] = int(row[0])
                except Exception:
                    pass

        return result

    # ── SQL aggregate pushdown (full-table profile, no row limit) ──────────────

    _NUMERIC_TYPE_HINTS = ("int", "bigint", "smallint", "tinyint", "float", "double", "decimal", "long", "real")

    def _is_numeric_sql_type(self, dtype: Optional[str]) -> bool:
        if not dtype:
            return False
        d = dtype.lower()
        return any(hint in d for hint in self._NUMERIC_TYPE_HINTS)

    def compute_databricks_profile_full(self, config: dict, source_sql: str) -> dict:
        """Runs full-table column statistics as Spark SQL aggregates — no row limit,
        no sampling, scales to any table size since the Databricks warehouse does the work."""
        import re

        def safe_col(name: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]", "_", name)

        def as_float(v):
            if v is None:
                return None
            try:
                f = float(v)
                return f if f == f and abs(f) != float("inf") else None
            except (TypeError, ValueError):
                return None

        with self._databricks_connect(config) as conn:
            with conn.cursor() as cursor:
                # Schema discovery — metadata only, zero row scan
                try:
                    cursor.execute(f"DESCRIBE QUERY {source_sql}")
                    rows = cursor.fetchall()
                    col_types = [(r[0], r[1]) for r in rows if r[0] and not str(r[0]).startswith("#")]
                except Exception:
                    cursor.execute(f"SELECT * FROM ({source_sql}) AS _sub_schema LIMIT 0")
                    col_types = [(d[0], None) for d in cursor.description]

                # One combined aggregate query for all columns — single pass over the full table
                agg_exprs = ["COUNT(*) AS __total_rows"]
                for name, dtype in col_types:
                    col, safe = f"`{name}`", safe_col(name)
                    agg_exprs.append(f"COUNT({col}) AS __{safe}_nonnull")
                    agg_exprs.append(f"approx_count_distinct({col}) AS __{safe}_distinct")
                    if self._is_numeric_sql_type(dtype):
                        agg_exprs += [
                            f"MIN({col}) AS __{safe}_min",
                            f"MAX({col}) AS __{safe}_max",
                            f"AVG({col}) AS __{safe}_avg",
                            f"STDDEV({col}) AS __{safe}_std",
                            f"skewness({col}) AS __{safe}_skew",
                            f"kurtosis({col}) AS __{safe}_kurt",
                            f"percentile_approx({col}, 0.5) AS __{safe}_median",
                        ]
                cursor.execute(f"SELECT {', '.join(agg_exprs)} FROM ({source_sql}) AS _sub")
                agg_row = cursor.fetchone()
                agg_cols = [d[0] for d in cursor.description]
                agg = dict(zip(agg_cols, agg_row))

                total_rows = int(agg.get("__total_rows") or 0)

                # Approximate full-row duplicate count — isolated so a failure here
                # (e.g. exotic nested types) doesn't take down the whole profile
                duplicate_count = 0
                try:
                    cursor.execute(
                        f"SELECT {int(total_rows)} - approx_count_distinct(struct(*)) FROM ({source_sql}) AS _subdup"
                    )
                    dup_row = cursor.fetchone()
                    if dup_row and dup_row[0] is not None:
                        duplicate_count = max(int(dup_row[0]), 0)
                except Exception:
                    pass

                columns = []
                for name, dtype in col_types:
                    safe = safe_col(name)
                    nonnull = int(agg.get(f"__{safe}_nonnull") or 0)
                    missing_count = max(total_rows - nonnull, 0)
                    distinct = int(agg.get(f"__{safe}_distinct") or 0)
                    numeric = self._is_numeric_sql_type(dtype)
                    columns.append({
                        "name": name,
                        "dtype": dtype or "unknown",
                        "semantic_type": "numeric" if numeric else "categorical",
                        "unique_count": distinct,
                        "unique_pct": round(distinct / max(total_rows, 1) * 100, 2),
                        "missing_count": missing_count,
                        "missing_pct": round(missing_count / max(total_rows, 1) * 100, 2),
                        "min": as_float(agg.get(f"__{safe}_min")) if numeric else None,
                        "max": as_float(agg.get(f"__{safe}_max")) if numeric else None,
                        "mean": as_float(agg.get(f"__{safe}_avg")) if numeric else None,
                        "median": as_float(agg.get(f"__{safe}_median")) if numeric else None,
                        "std": as_float(agg.get(f"__{safe}_std")) if numeric else None,
                        "skewness": as_float(agg.get(f"__{safe}_skew")) if numeric else None,
                        "kurtosis": as_float(agg.get(f"__{safe}_kurt")) if numeric else None,
                        "top_values": [],
                    })

                # Top-5 values for categorical columns — bounded to avoid unbounded round-trips
                cat_names = [name for name, dtype in col_types if not self._is_numeric_sql_type(dtype)][:25]
                by_name = {c["name"]: c for c in columns}
                for name in cat_names:
                    try:
                        cursor.execute(f"""
                            SELECT `{name}` AS v, COUNT(*) AS cnt
                            FROM ({source_sql}) AS _subtop
                            WHERE `{name}` IS NOT NULL
                            GROUP BY `{name}`
                            ORDER BY cnt DESC
                            LIMIT 5
                        """)
                        for v, c in cursor.fetchall():
                            by_name[name]["top_values"].append({
                                "value": str(v), "count": int(c),
                                "pct": round(c / max(total_rows, 1) * 100, 2),
                            })
                    except Exception:
                        pass

        return {
            "total_rows": total_rows,
            "total_columns": len(col_types),
            "memory_mb": 0.0,
            "file_size_bytes": None,
            "duplicate_count": duplicate_count,
            "duplicate_pct": round(duplicate_count / max(total_rows, 1) * 100, 2),
            "sampled": False,
            "sample_size": total_rows,
            "columns": columns,
        }

    # ── Delta Time Travel ───────────────────────────────────────────────────────

    def list_databricks_table_history(self, config: dict, catalog: str, schema: str, table: str, limit: int = 25) -> list[dict]:
        fqtn = f"`{catalog}`.`{schema}`.`{table}`"
        with self._databricks_connect(config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DESCRIBE HISTORY {fqtn} LIMIT {limit}")
                cols = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
        history = []
        for r in rows:
            data = dict(zip(cols, r))
            history.append({
                "version": data.get("version"),
                "timestamp": str(data.get("timestamp")) if data.get("timestamp") else None,
                "operation": data.get("operation"),
                "user_name": data.get("userName"),
                "operation_parameters": data.get("operationParameters"),
            })
        return history

    def compare_databricks_table_versions(
        self, config: dict, catalog: str, schema: str, table: str,
        version_a: int, version_b: int, sample_rows: int = 20000,
    ) -> dict:
        import pandas as pd

        fqtn = f"`{catalog}`.`{schema}`.`{table}`"

        def load_version(v: int) -> pd.DataFrame:
            with self._databricks_connect(config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT * FROM {fqtn} VERSION AS OF {v} LIMIT {sample_rows}")
                    cols = [d[0] for d in cursor.description]
                    rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=cols)

        def row_count_at(v: int) -> int:
            with self._databricks_connect(config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {fqtn} VERSION AS OF {v}")
                    row = cursor.fetchone()
            return int(row[0]) if row else 0

        df_a = load_version(version_a)
        df_b = load_version(version_b)
        count_a = row_count_at(version_a)
        count_b = row_count_at(version_b)

        cols_a = set(df_a.columns)
        cols_b = set(df_b.columns)
        added = sorted(cols_b - cols_a)
        removed = sorted(cols_a - cols_b)
        common = sorted(cols_a & cols_b)

        column_diffs = []
        for col in common:
            a = df_a[col]
            b = df_b[col]
            dtype_a, dtype_b = str(a.dtype), str(b.dtype)
            null_pct_a = round(float(a.isna().mean() * 100), 2) if len(a) else 0.0
            null_pct_b = round(float(b.isna().mean() * 100), 2) if len(b) else 0.0
            distinct_a = int(a.nunique(dropna=True))
            distinct_b = int(b.nunique(dropna=True))

            entry: dict = {
                "column": col,
                "dtype_a": dtype_a, "dtype_b": dtype_b, "dtype_changed": dtype_a != dtype_b,
                "null_pct_a": null_pct_a, "null_pct_b": null_pct_b,
                "distinct_a": distinct_a, "distinct_b": distinct_b,
                "mean_a": None, "mean_b": None, "min_a": None, "min_b": None, "max_a": None, "max_b": None,
            }
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                try:
                    entry["mean_a"] = round(float(a.mean()), 4) if not a.empty else None
                    entry["mean_b"] = round(float(b.mean()), 4) if not b.empty else None
                    entry["min_a"] = float(a.min()) if not a.empty else None
                    entry["min_b"] = float(b.min()) if not b.empty else None
                    entry["max_a"] = float(a.max()) if not a.empty else None
                    entry["max_b"] = float(b.max()) if not b.empty else None
                except Exception:
                    pass
            column_diffs.append(entry)

        return {
            "version_a": version_a, "version_b": version_b,
            "row_count_a": count_a, "row_count_b": count_b,
            "sampled_rows_a": len(df_a), "sampled_rows_b": len(df_b),
            "columns_added": added, "columns_removed": removed,
            "column_diffs": column_diffs,
        }

    def list_tables(self, config: dict) -> list[str]:
        db_type = config.get("db_type", "postgresql")

        if db_type == "databricks":
            try:
                from databricks import sql as _db_sql
            except ImportError:
                raise ValueError("databricks-sql-connector not installed.")
            kw = self._databricks_kw(config)
            with _db_sql.connect(**kw) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SHOW TABLES")
                    rows = cursor.fetchall()
            # SHOW TABLES returns (databaseName, tableName, isTemporary)
            return [r[1] for r in rows]

        conn = self.connect(config)
        try:
            if db_type == "postgresql":
                with conn.cursor() as cur:
                    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")
                    return [r[0] for r in cur.fetchall()]
            elif db_type == "mysql":
                with conn.cursor() as cur:
                    cur.execute("SHOW TABLES")
                    return [r[0] for r in cur.fetchall()]
            elif db_type == "sqlite":
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                return [r[0] for r in cur.fetchall()]
            elif db_type in ("mssql", "fabric"):
                cur = conn.cursor()
                cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
                return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()
        return []
