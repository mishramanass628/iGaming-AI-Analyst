"""Read-only guard for LLM-written SQL.

This is a safety net, not a security boundary: always connect with a Snowflake role
that only has SELECT grants.
"""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

_FORBIDDEN_NODES = tuple(
    getattr(exp, n)
    for n in (
        "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Merge",
        "Command", "TruncateTable", "Grant", "Revoke", "Copy", "Use", "Set", "Transaction",
        "Commit", "Rollback", "LoadData",
    )
    if hasattr(exp, n)
)
_QUERY_NODES = tuple(getattr(exp, n) for n in ("Query", "Select", "Union", "Intersect", "Except") if hasattr(exp, n))

_FORBIDDEN_WORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|CALL|PUT|GET|COPY|USE|"
    r"UNDROP|EXECUTE|REMOVE|SET|UNSET|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)


def _strip_comments_and_strings(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate(sql: str, dialect: str) -> tuple[str, str | None, str | None]:
    """Return (cleaned_sql, error, warning). error is None when the query may run."""
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return sql, "The query is empty.", None
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except Exception as e:  # sqlglot can't parse every vendor extension: fall back to a keyword check
        bare = _strip_comments_and_strings(sql)
        if not re.match(r"^\s*(SELECT|WITH)\b", bare, re.I):
            return sql, "Only SELECT / WITH queries are allowed.", None
        if ";" in bare:
            return sql, "Only a single statement is allowed.", None
        m = _FORBIDDEN_WORDS.search(bare)
        if m:
            return sql, f"Write/DDL keyword '{m.group(1).upper()}' is not allowed.", None
        return sql, None, f"Could not fully parse the SQL locally ({str(e).splitlines()[0][:150]}); passed the keyword check."

    if len(statements) != 1:
        return sql, "Only a single statement is allowed.", None
    stmt = statements[0]
    if not isinstance(stmt, _QUERY_NODES):
        return sql, f"Only SELECT / WITH queries are allowed (got {type(stmt).__name__}).", None
    for node_type in _FORBIDDEN_NODES:
        if stmt.find(node_type) is not None:
            return sql, f"{node_type.__name__} statements are not allowed.", None
    return sql, None, None
