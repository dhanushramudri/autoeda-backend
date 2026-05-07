from typing import Any, Optional

import pandas as pd

from .base import BaseConnector


class DBConnector(BaseConnector):
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
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")

    def load_data(self, config: dict, limit: Optional[int] = None) -> pd.DataFrame:
        db_type = config.get("db_type", "postgresql")

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
                if limit:
                    query = f"SELECT * FROM {table} LIMIT {limit}"
                else:
                    query = f"SELECT * FROM {table} LIMIT 50000"
            elif limit:
                query = f"SELECT * FROM ({query}) AS _sub LIMIT {limit}"

            df = pd.read_sql(query, conn)
        finally:
            conn.close()
        return df

    def list_tables(self, config: dict) -> list[str]:
        db_type = config.get("db_type", "postgresql")
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
                import sqlite3
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                return [r[0] for r in cur.fetchall()]
            elif db_type == "mssql":
                cur = conn.cursor()
                cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
                return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()
        return []
