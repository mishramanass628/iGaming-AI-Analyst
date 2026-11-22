"""LangGraph workflows.

SQL graph:       retrieve (business docs + schema) -> generate_sql -> validate --(error, <3 tries)--> generate_sql
Analysis graph:  profile (pandas facts) -> retrieve (business docs) -> analyze (LLM insights)

Execution sits between the two graphs on purpose: the user reviews/edits the SQL and clicks Execute.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional, TypedDict

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph

from . import sql_guard
from .docs import DocStore, format_docs
from .llm import content_text

MAX_SQL_ATTEMPTS = 3
DIALECT_NAMES = {"snowflake": "Snowflake SQL", "duckdb": "DuckDB SQL",
                 "bigquery": "BigQuery GoogleSQL (backtick-quoted table names)", "databricks": "Databricks SQL"}


# --------------------------------------------------------------------------- schema context
class SchemaContext:
    """Whole schema for small warehouses; retrieves the most relevant tables for large ones."""

    def __init__(self, tables, embedder=None, full_threshold: int = 30, k: int = 12):
        self.tables = tables
        self.k = k
        self.store = None
        if len(tables) > full_threshold and embedder is not None:
            self.store = InMemoryVectorStore(embedder)
            self.store.add_texts([t.to_text() for t in tables])

    def context(self, question: str) -> str:
        if self.store is None:
            return "\n\n".join(t.to_text() for t in self.tables)
        hits = self.store.similarity_search(question, k=self.k)
        return "\n\n".join(h.page_content for h in hits)


# --------------------------------------------------------------------------- SQL graph
class SQLState(TypedDict, total=False):
    question: str
    doc_context: str
    doc_sources: list[str]
    schema_context: str
    sql: str
    assumptions: str
    error: Optional[str]
    warning: Optional[str]
    attempts: int
    prev_sql: Optional[str]
    exec_error: Optional[str]


SQL_SYSTEM = """You are a senior analytics engineer at an iGaming operator. You turn business questions into ONE read-only {dialect} query.

Rules:
- Use only the tables and columns in the schema provided; reference tables exactly as named there.
- The business documents define the company's metrics (e.g. GGR, NGR, active player, FTD, churn). When a document defines a metric, use that exact definition.
- Read-only: a single SELECT (CTEs allowed). Never modify data.
- Use readable snake_case column aliases. Add ORDER BY for rankings and time series.
- Add LIMIT {limit} unless the result is naturally small (aggregates, a handful of groups).
- If the question is ambiguous, pick the most standard business interpretation and state it.

Reply in exactly this format:
ASSUMPTIONS: <one or two short lines>
```sql
<the query>
```"""


def _extract_sql(text: str) -> tuple[str, str]:
    assumptions = ""
    if m := re.search(r"ASSUMPTIONS:\s*(.+?)(?:```|$)", text, flags=re.S | re.I):
        assumptions = m.group(1).strip()
    if m := re.search(r"```(?:sql)?\s*(.+?)```", text, flags=re.S | re.I):
        return m.group(1).strip(), assumptions
    if m := re.search(r"\b(WITH|SELECT)\b.+", text, flags=re.S | re.I):
        return m.group(0).strip(), assumptions
    return "", assumptions


def build_sql_graph(llm, dialect: str, schema_ctx: SchemaContext, docstore: DocStore | None, row_limit: int = 1000):
    def retrieve(state: SQLState):
        q = state["question"]
        hits = docstore.search(q, k=5) if docstore else []
        return {
            "doc_context": format_docs(hits),
            "doc_sources": sorted({h.metadata["source"] for h in hits}),
            "schema_context": schema_ctx.context(q),
            "attempts": 0,
        }

    def generate_sql(state: SQLState):
        parts = [
            f"DATABASE SCHEMA:\n{state['schema_context']}",
            f"BUSINESS DOCUMENTS (excerpts):\n{state['doc_context']}",
            f"QUESTION: {state['question']}",
        ]
        problem = state.get("error") or state.get("exec_error")
        if problem and state.get("prev_sql"):
            parts.append(
                f"Your previous query failed.\nPREVIOUS QUERY:\n```sql\n{state['prev_sql']}\n```\nERROR: {problem}\n"
                "Fix the query. Re-check table and column names against the schema."
            )
        msgs = [
            SystemMessage(SQL_SYSTEM.format(dialect=DIALECT_NAMES.get(dialect, dialect), limit=row_limit)),
            HumanMessage("\n\n".join(parts)),
        ]
        text = content_text(llm.invoke(msgs).content)
        sql, assumptions = _extract_sql(text)
        return {"sql": sql, "assumptions": assumptions, "attempts": state.get("attempts", 0) + 1,
                "error": None, "exec_error": None}

    def validate(state: SQLState):
        sql, err, warn = sql_guard.validate(state.get("sql", ""), dialect)
        return {"sql": sql, "error": err, "warning": warn, "prev_sql": sql if err else state.get("prev_sql")}

    def route(state: SQLState):
        return "generate_sql" if state.get("error") and state["attempts"] < MAX_SQL_ATTEMPTS else END

    g = StateGraph(SQLState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate_sql", generate_sql)
    g.add_node("validate", validate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate_sql")
    g.add_edge("generate_sql", "validate")
    g.add_conditional_edges("validate", route, ["generate_sql", END])
    return g.compile()


# --------------------------------------------------------------------------- data profiling
def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}" if abs(v) < 1e6 else f"{v:,.0f}"
    return str(v)


def _date_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    for c in df.columns:
        if c not in cols and df[c].dtype == object and re.search(r"date|day|month|week|period|time|year", str(c), re.I):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.9:
                df[c] = parsed
                cols.append(c)
    return cols


_RATIO = re.compile(r"pct|percent|rate|ratio|share|hold|avg|average|mean|arpu|conversion|margin|per_", re.I)


def _is_ratio(col) -> bool:
    return bool(_RATIO.search(str(col)))


def profile_dataframe(df: pd.DataFrame, truncated: bool) -> str:
    """Deterministic facts the LLM must ground its analysis in."""
    df = df.copy()
    lines = [f"Result: {len(df):,} rows x {len(df.columns)} columns" + (" (TRUNCATED at row limit)" if truncated else "")]
    if df.empty:
        return "\n".join(lines + ["The query returned no rows."])

    date_cols = _date_columns(df)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
                and not re.search(r"(^|_)(id|key)$", str(c), re.I)]
    cat_cols = [c for c in df.columns if c not in num_cols and c not in date_cols]

    if len(df) == 1:
        lines.append("Single row: " + ", ".join(f"{c}={_fmt(df.iloc[0][c])}" for c in df.columns))
        return "\n".join(lines)

    lines.append("\nNUMERIC COLUMNS")
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        total = "" if _is_ratio(c) else f"total={_fmt(s.sum())}, "
        line = (f"- {c}: {total}mean={_fmt(s.mean())}, median={_fmt(s.median())}, "
                f"min={_fmt(s.min())}, max={_fmt(s.max())}, std={_fmt(s.std())}")
        if not _is_ratio(c) and (s >= 0).all() and s.sum() > 0 and len(s) >= 10:
            top = s.sort_values(ascending=False)
            n10 = max(1, int(len(s) * 0.1))
            line += f"; top 10% of rows hold {top.iloc[:n10].sum() / s.sum():.1%} of the total"
        if len(s) >= 8 and s.std() > 0:
            z = (s - s.mean()) / s.std()
            n_out = int((z.abs() > 3).sum())
            if n_out:
                line += f"; {n_out} outlier row(s) beyond 3 std"
        lines.append(line)

    if cat_cols:
        lines.append("\nCATEGORICAL COLUMNS")
        additive = [c for c in num_cols if not _is_ratio(c)]
        measure = additive[0] if additive else None
        for c in cat_cols[:6]:
            nun = df[c].nunique(dropna=True)
            if measure and nun <= 200:
                grp = df.groupby(c, dropna=False)[measure].sum().sort_values(ascending=False)
                total = grp.sum()
                top = ", ".join(
                    f"{k}={_fmt(v)}" + (f" ({v / total:.1%})" if total else "") for k, v in grp.head(6).items()
                )
                lines.append(f"- {c}: {nun} distinct; by {measure}: {top}")
            else:
                vc = df[c].value_counts().head(5)
                lines.append(f"- {c}: {nun} distinct; most common: " + ", ".join(f"{k} ({v})" for k, v in vc.items()))

    if date_cols and num_cols:
        d = date_cols[0]
        lines.append(f"\nTIME TREND (by {d})")
        agg = {c: ("mean" if _is_ratio(c) else "sum") for c in num_cols[:4]}
        ts = df.dropna(subset=[d]).groupby(d).agg(agg).sort_index()
        if len(ts) >= 2:
            span_days = (ts.index.max() - ts.index.min()).days
            freq = "D" if span_days <= 62 else ("W" if span_days <= 370 else "MS")
            if len(ts) > 60:
                ts = ts.resample(freq).agg(agg)
            lines.append("- ratio columns are averaged across groups (unweighted); additive columns are summed")
            lines.append(f"- range {ts.index.min().date()} → {ts.index.max().date()}, {len(ts)} periods")
            for c in ts.columns:
                s = ts[c]
                last, prev = s.iloc[-1], s.iloc[-2]
                chg = f"{(last - prev) / abs(prev):+.1%}" if prev else "n/a"
                slope = np.polyfit(np.arange(len(s)), s.values.astype(float), 1)[0] if len(s) >= 3 else 0
                lines.append(
                    f"- {c}: last period={_fmt(last)} vs previous={_fmt(prev)} ({chg}); "
                    f"peak={_fmt(s.max())} on {s.idxmax().date()}; low={_fmt(s.min())} on {s.idxmin().date()}; "
                    f"overall trend {'up' if slope > 0 else 'down' if slope < 0 else 'flat'}"
                )

    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        pairs = []
        for i, a in enumerate(num_cols):
            for b in num_cols[i + 1:]:
                r = corr.loc[a, b]
                if pd.notna(r) and abs(r) >= 0.6:
                    pairs.append(f"{a}~{b} r={r:.2f}")
        if pairs:
            lines.append("\nSTRONG CORRELATIONS: " + "; ".join(pairs[:8]))

    return "\n".join(lines)


def data_sample(df: pd.DataFrame, max_chars: int = 15000) -> str:
    csv = df.head(300).to_csv(index=False)
    if len(csv) > max_chars:
        csv = csv[:max_chars] + "\n...(sample cut)"
    return csv


# --------------------------------------------------------------------------- analysis graph
class AnalysisState(TypedDict, total=False):
    question: str
    sql: str
    df: Any
    truncated: bool
    profile: str
    doc_context: str
    doc_sources: list[str]
    jira_context: str
    jira_tickets: list[dict]
    insights: str


ANALYSIS_SYSTEM = """You are a principal business analyst at an iGaming operator (casino, live casino, sportsbook).
You turn query results into in-depth, decision-ready insights for leadership, reading them against the company's own business documents (KPI definitions, targets, policies, market notes, strategy).

Ground rules:
- Use only numbers that appear in the COMPUTED FACTS or DATA SAMPLE. Never invent figures. Derived figures (shares, differences) are fine if you show what they came from.
- Compare results with targets, thresholds, and policies from the documents whenever they apply, and cite the document as [document name].
- If the documents don't cover something, say so rather than guessing.
- Flag responsible-gambling, compliance, fraud, or bonus-abuse signals when the data hints at them.
- Be specific and quantitative; no generic advice.

Write Markdown with these sections:
### Direct answer
### Key findings
(numbered; each with a concrete number)
### Drivers and segments
### Against business targets and policies
### Anomalies and risks
### Recommended actions
(prioritised; say which team should own each)
### Related past Jira tickets
(only if PAST JIRA TICKETS are provided: list the relevant ones as [KEY](link) with status/resolution and how they
relate: a repeat of a solved issue, an open known problem, or earlier analysis. Skip unrelated ones. Never invent keys.)
### Caveats and data quality
### Suggested follow-up questions
(3 questions the user could ask next)"""


def build_analysis_graph(llm, docstore: DocStore | None, jira=None, embedder=None):
    def profile(state: AnalysisState):
        return {"profile": profile_dataframe(state["df"], state.get("truncated", False))}

    def retrieve(state: AnalysisState):
        if not docstore:
            return {"doc_context": "(no business documents loaded)", "doc_sources": []}
        cols = " ".join(map(str, state["df"].columns))
        hits = docstore.search(f"{state['question']} {cols} targets policy definition", k=8)
        return {"doc_context": format_docs(hits), "doc_sources": sorted({h.metadata["source"] for h in hits})}

    def retrieve_jira(state: AnalysisState):
        if jira is None:
            return {"jira_context": "", "jira_tickets": []}
        from .jira import format_tickets
        try:
            tickets = jira.search(state["question"], k=6, embedder=embedder)
        except Exception as e:
            return {"jira_context": f"(Jira search failed: {e})", "jira_tickets": []}
        return {"jira_context": format_tickets(tickets),
                "jira_tickets": [{"key": t.key, "url": t.url, "summary": t.summary, "status": t.status,
                                  "resolution": t.resolution} for t in tickets]}

    def analyze(state: AnalysisState):
        prompt = (
            f"TODAY: {date.today().isoformat()} (the most recent period in the data may be incomplete)\n\n"
            f"BUSINESS QUESTION: {state['question']}\n\n"
            f"SQL THAT PRODUCED THE DATA:\n```sql\n{state['sql']}\n```\n\n"
            f"COMPUTED FACTS (pandas):\n{state['profile']}\n\n"
            f"DATA SAMPLE (CSV, first rows):\n{data_sample(state['df'])}\n\n"
            f"BUSINESS DOCUMENTS (excerpts):\n{state['doc_context']}"
            + (f"\n\nPAST JIRA TICKETS (search results, may be partly unrelated):\n{state['jira_context']}"
               if state.get("jira_context") else "")
        )
        msg = llm.invoke([SystemMessage(ANALYSIS_SYSTEM), HumanMessage(prompt)])
        return {"insights": content_text(msg.content)}

    g = StateGraph(AnalysisState)
    g.add_node("profile", profile)
    g.add_node("retrieve", retrieve)
    g.add_node("retrieve_jira", retrieve_jira)
    g.add_node("analyze", analyze)
    g.add_edge(START, "profile")
    g.add_edge("profile", "retrieve")
    g.add_edge("retrieve", "retrieve_jira")
    g.add_edge("retrieve_jira", "analyze")
    g.add_edge("analyze", END)
    return g.compile()


def stream_analysis(graph, inputs: dict, final: dict):
    """Yield insight text as it streams; fills `final` with the end state."""
    streamed = False
    for mode, payload in graph.stream(inputs, stream_mode=["messages", "values"]):
        if mode == "messages":
            chunk, meta = payload
            if meta.get("langgraph_node") == "analyze":
                text = content_text(chunk.content)
                if text:
                    streamed = True
                    yield text
        else:
            final.clear()
            final.update(payload)
    if not streamed and final.get("insights"):
        yield final["insights"]
