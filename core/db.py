"""Database connectors: Snowflake (real) and DuckDB (local demo with dummy iGaming data).

Both expose the same small interface:
    .dialect      -> sqlglot dialect name used to write/validate SQL
    .label        -> human readable description of what we're connected to
    .tables       -> list[TableInfo] (introspected schema)
    .run(sql, limit) -> (DataFrame, truncated: bool)
"""
from __future__ import annotations

import datetime as _dt
import re
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pandas as pd


@dataclass
class TableInfo:
    full_name: str
    columns: list[tuple[str, str, str]] = field(default_factory=list)  # (name, type, comment)
    comment: str = ""
    row_count: int | None = None

    def to_text(self) -> str:
        head = f"TABLE {self.full_name}"
        if self.row_count is not None:
            head += f"  (~{self.row_count:,} rows)"
        if self.comment:
            head += f"  -- {self.comment}"
        cols = []
        for name, typ, comment in self.columns:
            line = f"    {name} {typ}"
            if comment:
                line += f"  -- {comment}"
            cols.append(line)
        return head + "\n" + "\n".join(cols)


def _clean_frame(rows: list, columns: list[str]) -> pd.DataFrame:
    """Build a DataFrame and turn Decimal / date objects into analysable dtypes."""
    df = pd.DataFrame(rows, columns=columns)
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        first = s.iloc[0]
        if isinstance(first, Decimal):
            df[col] = pd.to_numeric(df[col].map(lambda v: float(v) if v is not None else None))
        elif isinstance(first, (_dt.date, _dt.datetime)) and not isinstance(first, pd.Timestamp):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _ident(name: str) -> str:
    """Return a safe Snowflake identifier (unquoted if simple, quoted otherwise)."""
    name = name.strip()
    if name.startswith('"') and name.endswith('"'):
        return name
    if _IDENT.match(name):
        return name.upper()
    return '"' + name.replace('"', '""') + '"'


class SnowflakeDB:
    dialect = "snowflake"

    def __init__(self, params: dict, schema: str | None = None, statement_timeout_s: int = 300):
        import snowflake.connector  # imported lazily: heavy

        if not params.get("database"):
            raise ValueError("A database name is required so the schema can be read.")
        self.params = {k: v for k, v in params.items() if v not in (None, "")}
        self.conn = snowflake.connector.connect(login_timeout=60, **self.params)
        self.database = _ident(params["database"])
        self.schema = schema.strip() if schema else None
        cur = self.conn.cursor()
        cur.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {int(statement_timeout_s)}")
        cur.close()
        self.label = f"Snowflake · {self.database}" + (f".{self.schema.upper()}" if self.schema else "")
        self.tables = self._introspect()

    def _introspect(self) -> list[TableInfo]:
        db = self.database
        where = "c.table_schema <> 'INFORMATION_SCHEMA'"
        args: list = []
        if self.schema:
            where += " AND UPPER(c.table_schema) = UPPER(%s)"
            args.append(self.schema.strip('"'))
        cur = self.conn.cursor()
        cur.execute(
            f"""SELECT c.table_schema, c.table_name, c.column_name, c.data_type, c.comment
                FROM {db}.INFORMATION_SCHEMA.COLUMNS c
                WHERE {where}
                ORDER BY c.table_schema, c.table_name, c.ordinal_position""",
            args,
        )
        cols = cur.fetchall()
        where_t = "table_schema <> 'INFORMATION_SCHEMA'" + (" AND UPPER(table_schema) = UPPER(%s)" if self.schema else "")
        cur.execute(
            f"SELECT table_schema, table_name, row_count, comment FROM {db}.INFORMATION_SCHEMA.TABLES WHERE {where_t}",
            args,
        )
        meta = {(r[0], r[1]): (r[2], r[3]) for r in cur.fetchall()}
        cur.close()

        tables: dict[tuple[str, str], TableInfo] = {}
        for schema, table, col, typ, comment in cols:
            key = (schema, table)
            if key not in tables:
                rc, tcomment = meta.get(key, (None, None))
                tables[key] = TableInfo(f"{db}.{schema}.{table}", [], tcomment or "", rc)
            tables[key].columns.append((col, typ, comment or ""))
        if not tables:
            raise ValueError("Connected, but no tables were found. Check the database/schema names and your role's grants.")
        return list(tables.values())

    def run(self, sql: str, limit: int = 5000) -> tuple[pd.DataFrame, bool]:
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
            rows = cur.fetchmany(limit + 1)
            columns = [d[0] for d in cur.description]
        finally:
            cur.close()
        truncated = len(rows) > limit
        return _clean_frame(rows[:limit], columns), truncated

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


def _sql_str(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def _backtick(name: str) -> str:
    name = name.strip().strip("`")
    return name if _IDENT.match(name) else "`" + name.replace("`", "``") + "`"


class BigQueryDB:
    """Google BigQuery. Auth with a service-account key, or Application Default Credentials (gcloud)."""

    dialect = "bigquery"

    def __init__(self, project: str | None, dataset: str | None, sa_info: dict | None = None,
                 location: str | None = None, max_gb_billed: float = 10.0, max_tables: int = 300):
        from google.cloud import bigquery

        if sa_info:
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_info(sa_info)
            self.client = bigquery.Client(project=project or sa_info.get("project_id"), credentials=creds,
                                          location=location or None)
        else:
            self.client = bigquery.Client(project=project or None, location=location or None)
        self.project = self.client.project
        self.dataset = (dataset or "").strip() or None
        self.max_bytes = int(max_gb_billed * 1e9)
        self.max_tables = max_tables
        self.label = f"BigQuery · {self.project}" + (f".{self.dataset}" if self.dataset else "")
        self.tables = self._introspect()

    def _introspect(self) -> list[TableInfo]:
        datasets = [self.dataset] if self.dataset else [d.dataset_id for d in self.client.list_datasets(self.project)]
        out: list[TableInfo] = []
        for ds in datasets:
            for item in self.client.list_tables(f"{self.project}.{ds}"):
                if len(out) >= self.max_tables:
                    break
                t = self.client.get_table(item.reference)
                cols = [(f.name, f.field_type + ("[]" if f.mode == "REPEATED" else ""), f.description or "")
                        for f in t.schema]
                out.append(TableInfo(f"`{self.project}.{ds}.{t.table_id}`", cols, t.description or "", t.num_rows))
        if not out:
            raise ValueError("Connected, but no tables were found. Check the project/dataset and your IAM permissions.")
        return out

    def run(self, sql: str, limit: int = 5000) -> tuple[pd.DataFrame, bool]:
        from google.cloud import bigquery

        cfg = bigquery.QueryJobConfig(maximum_bytes_billed=self.max_bytes, use_legacy_sql=False)
        result = self.client.query(sql, job_config=cfg).result(max_results=limit + 1)
        columns = [f.name for f in result.schema]
        rows = [list(r.values()) for r in result]
        return _clean_frame(rows[:limit], columns), len(rows) > limit

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


class DatabricksDB:
    """Databricks SQL warehouse (Unity Catalog) via a personal access token."""

    dialect = "databricks"

    def __init__(self, server_hostname: str, http_path: str, access_token: str, catalog: str, schema: str | None = None):
        from databricks import sql as dbsql

        if not catalog:
            raise ValueError("A catalog name is required so the schema can be read.")
        host = server_hostname.strip().removeprefix("https://").rstrip("/")
        self.conn = dbsql.connect(server_hostname=host, http_path=http_path.strip(), access_token=access_token.strip())
        self.catalog = _backtick(catalog)
        self.schema = (schema or "").strip() or None
        self.label = f"Databricks · {catalog.strip()}" + (f".{self.schema}" if self.schema else "")
        self.tables = self._introspect()

    def _introspect(self) -> list[TableInfo]:
        cat = self.catalog
        where = "table_schema <> 'information_schema'"
        if self.schema:
            where += f" AND lower(table_schema) = lower({_sql_str(self.schema)})"
        cols, _ = self.run(
            f"""SELECT table_schema, table_name, column_name, data_type, comment
                FROM {cat}.information_schema.columns WHERE {where}
                ORDER BY table_schema, table_name, ordinal_position""", limit=200000)
        tmeta, _ = self.run(f"SELECT table_schema, table_name, comment FROM {cat}.information_schema.tables WHERE {where}",
                            limit=100000)
        comments = {(r.table_schema, r.table_name): r.comment for r in tmeta.itertuples()}
        tables: dict[tuple[str, str], TableInfo] = {}
        for r in cols.itertuples():
            key = (r.table_schema, r.table_name)
            if key not in tables:
                full = f"{cat}.{_backtick(r.table_schema)}.{_backtick(r.table_name)}"
                tables[key] = TableInfo(full, [], comments.get(key) or "")
            tables[key].columns.append((r.column_name, r.data_type, r.comment or ""))
        if not tables:
            raise ValueError("Connected, but no tables were found. Check the catalog/schema and your grants.")
        return list(tables.values())

    def run(self, sql: str, limit: int = 5000) -> tuple[pd.DataFrame, bool]:
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
            rows = [tuple(r) for r in cur.fetchmany(limit + 1)]
            columns = [d[0] for d in cur.description]
        finally:
            cur.close()
        return _clean_frame(rows[:limit], columns), len(rows) > limit

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


_DEMO_BUILD_LOCK = threading.Lock()


class DuckDemoDB:
    """Local DuckDB file with dummy iGaming data (see demo/seed.py)."""

    dialect = "duckdb"

    def __init__(self, path: Path):
        import duckdb

        with _DEMO_BUILD_LOCK:  # on a fresh server, several visitors may pick the demo at once
            if not path.exists():
                from demo.seed import build

                tmp = path.with_suffix(".building.duckdb")
                tmp.unlink(missing_ok=True)
                build(tmp)
                tmp.replace(path)
        self.conn = duckdb.connect(str(path), read_only=True)
        self.label = "Demo DuckDB · dummy iGaming data"
        self.tables = self._introspect()

    def _introspect(self) -> list[TableInfo]:
        cols = self.conn.execute(
            """SELECT c.table_name, c.column_name, c.data_type, COALESCE(c.comment, '')
               FROM duckdb_columns() c
               WHERE c.schema_name = 'main' AND NOT c.internal
               ORDER BY c.table_name, c.column_index"""
        ).fetchall()
        tmeta = {
            r[0]: (r[1], r[2] or "")
            for r in self.conn.execute(
                "SELECT table_name, estimated_size, comment FROM duckdb_tables() WHERE schema_name = 'main'"
            ).fetchall()
        }
        tables: dict[str, TableInfo] = {}
        for table, col, typ, comment in cols:
            if table not in tables:
                rc, tcomment = tmeta.get(table, (None, ""))
                tables[table] = TableInfo(table, [], tcomment, rc)
            tables[table].columns.append((col, typ, comment))
        return list(tables.values())

    def run(self, sql: str, limit: int = 5000) -> tuple[pd.DataFrame, bool]:
        cur = self.conn.cursor()  # a cursor is a separate, thread-safe connection
        try:
            res = cur.execute(sql)
            rows = res.fetchmany(limit + 1)
            columns = [d[0] for d in res.description]
        finally:
            cur.close()
        truncated = len(rows) > limit
        return _clean_frame(rows[:limit], columns), truncated

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
