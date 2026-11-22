"""Autonomous analyst agent (LangGraph tool-calling loop).

The LLM decides what to do: read business documents, explore the schema, run read-only SQL,
check surprising numbers, drill into drivers, and decide when it has enough to answer.

    START -> agent --(tool calls)--> tools --> agent ... --(no tool calls)--> END
                                         \\--(step budget used)--> finalize --> END

Guard rails: read-only SQL guard on every query, a row cap on what the model sees,
a query budget and a step budget. Every tool call is logged for the UI.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import sql_guard
from .docs import format_docs
from .graphs import DIALECT_NAMES, profile_dataframe
from .jira import format_tickets
from .llm import content_text


# --------------------------------------------------------------------------- workspace (tool side effects)
@dataclass
class AgentWorkspace:
    db: Any
    docstore: Any = None
    jira: Any = None          # JiraClient, optional
    embedder: Any = None      # for ranking Jira results by meaning
    max_queries: int = 8
    rows_to_model: int = 60
    fetch_limit: int = 5000
    queries: list[dict] = field(default_factory=list)   # every query the agent ran (for the UI)
    events: list[dict] = field(default_factory=list)    # every tool call, in order (for the UI)

    def table(self, name: str):
        name = name.strip().strip('`"').lower()
        for t in self.db.tables:
            full = t.full_name.replace("`", "").replace('"', "").lower()
            if full == name or full.split(".")[-1] == name.split(".")[-1]:
                return t
        return None


def make_tools(ws: AgentWorkspace):
    @tool
    def list_tables(keyword: str = "") -> str:
        """List the database tables with their descriptions and column names.
        Optionally pass a keyword to only list tables whose name, description or columns contain it."""
        kw = keyword.lower().strip()
        lines = []
        for t in ws.db.tables:
            blob = (t.full_name + t.comment + " ".join(c[0] for c in t.columns)).lower()
            if kw and kw not in blob:
                continue
            rc = f" (~{t.row_count:,} rows)" if t.row_count else ""
            desc = f" - {t.comment}" if t.comment else ""
            lines.append(f"{t.full_name}{rc}{desc}\n    columns: {', '.join(c[0] for c in t.columns)}")
        ws.events.append({"tool": "list_tables", "title": f"Listed tables{f' matching “{kw}”' if kw else ''}",
                          "detail": f"{len(lines)} tables"})
        return "\n".join(lines[:150]) or "No tables match."

    @tool
    def describe_table(table_name: str) -> str:
        """Show one table's columns with types and descriptions, plus a few sample rows.
        Use the table name exactly as list_tables shows it."""
        t = ws.table(table_name)
        if t is None:
            ws.events.append({"tool": "describe_table", "title": f"Looked for table `{table_name}`", "detail": "not found"})
            return f"Table '{table_name}' not found. Call list_tables to see the exact names."
        text = t.to_text()
        if ws.db.dialect != "bigquery":  # LIMIT doesn't reduce BigQuery scan cost, so skip samples there
            try:
                sample, _ = ws.db.run(f"SELECT * FROM {t.full_name} LIMIT 3", 3)
                text += "\nSAMPLE ROWS:\n" + sample.to_csv(index=False)
            except Exception:
                pass
        ws.events.append({"tool": "describe_table", "title": f"Inspected table `{t.full_name}`",
                          "detail": f"{len(t.columns)} columns"})
        return text

    @tool
    def search_business_docs(query: str) -> str:
        """Search the company's business documents (KPI definitions, targets, strategy, policies, market notes).
        Use it to find metric definitions before writing SQL, and targets/policies to judge results against."""
        if not ws.docstore:
            ws.events.append({"tool": "search_business_docs", "title": f"Searched docs: “{query}”", "detail": "no documents loaded"})
            return "No business documents are loaded. Use standard iGaming definitions and say so."
        hits = ws.docstore.search(query, k=4)
        sources = sorted({h.metadata["source"] for h in hits})
        ws.events.append({"tool": "search_business_docs", "title": f"Searched docs: “{query}”",
                          "detail": ", ".join(sources)})
        return format_docs(hits)

    @tool
    def run_sql(sql: str, purpose: str) -> str:
        """Run ONE read-only SELECT query and get the result.
        `purpose` is a short plain-English note on what this query checks (shown to the user).
        Aggregate in SQL (GROUP BY, SUM, COUNT); don't pull raw rows. Large results are cut to the first rows."""
        qid = len(ws.queries) + 1
        entry = {"id": qid, "purpose": purpose, "sql": sql, "df": None, "error": None, "truncated": False, "ms": 0}
        if len(ws.queries) >= ws.max_queries:
            ws.events.append({"tool": "run_sql", "title": "Query refused: budget used up", "detail": purpose})
            return f"Query budget ({ws.max_queries}) is used up. Do not run more queries; write the final answer."
        clean, err, _ = sql_guard.validate(sql, ws.db.dialect)
        entry["sql"] = clean
        if err:
            entry["error"] = f"Blocked by read-only guard: {err}"
        else:
            t0 = time.perf_counter()
            try:
                df, truncated = ws.db.run(clean, ws.fetch_limit)
                entry.update(df=df, truncated=truncated)
            except Exception as e:
                entry["error"] = str(e)[:800]
            entry["ms"] = (time.perf_counter() - t0) * 1000
        ws.queries.append(entry)
        ws.events.append({"tool": "run_sql", "title": f"Q{qid} · {purpose}", "query_id": qid})
        if entry["error"]:
            return f"Q{qid} FAILED: {entry['error']}\nFix the query (check names with describe_table) or try another approach."
        df = entry["df"]
        out = [f"Q{qid}: {len(df):,} rows x {len(df.columns)} columns" + (" (truncated)" if entry["truncated"] else "")]
        if len(df) > 15:
            out.append("FACTS:\n" + profile_dataframe(df, entry["truncated"]))
        shown = df.head(ws.rows_to_model).to_csv(index=False)
        out.append(f"ROWS (first {min(len(df), ws.rows_to_model)}):\n{shown[:6000]}")
        return "\n".join(out)

    tools = [list_tables, describe_table, search_business_docs, run_sql]
    if ws.jira is None:
        return tools

    @tool
    def search_jira_tickets(query: str) -> str:
        """Search the company's Jira for past tickets about the same topic: incidents, data-quality issues,
        campaign or payment problems, bugs, analyses already done, and how they were resolved.
        Pass a short description of the issue (a few keywords work best, e.g. 'Germany slots turnover drop')."""
        try:
            hits = ws.jira.search(query, k=6, embedder=ws.embedder)
        except Exception as e:
            ws.events.append({"tool": "search_jira_tickets", "title": f"Searched Jira: “{query}”", "detail": f"failed: {e}"})
            return f"Jira search failed: {e}"
        ws.events.append({"tool": "search_jira_tickets", "title": f"Searched Jira: “{query}”",
                          "detail": ", ".join(t.key for t in hits) or "no matches",
                          "tickets": [{"key": t.key, "url": t.url, "summary": t.summary, "status": t.status,
                                       "resolution": t.resolution} for t in hits]})
        return format_tickets(hits)

    @tool
    def read_jira_ticket(key: str) -> str:
        """Read one Jira ticket in full (description and comments), e.g. to see the root cause and fix.
        Use the key from search_jira_tickets, like 'OPS-123'."""
        try:
            t = ws.jira.get(key)
        except Exception as e:
            ws.events.append({"tool": "read_jira_ticket", "title": f"Read Jira ticket {key}", "detail": f"failed: {e}"})
            return f"Couldn't read {key}: {e}"
        ws.events.append({"tool": "read_jira_ticket", "title": f"Read Jira ticket {t.key}: {t.summary}",
                          "detail": t.status + (f" / {t.resolution}" if t.resolution else ""),
                          "tickets": [{"key": t.key, "url": t.url, "summary": t.summary, "status": t.status,
                                       "resolution": t.resolution}]})
        return t.full()

    return tools + [search_jira_tickets, read_jira_ticket]


# --------------------------------------------------------------------------- prompt
AGENT_SYSTEM = """You are an autonomous senior data analyst for an iGaming operator (casino, live casino, sportsbook).
You answer business questions by investigating the company's {dialect} database and its business documents with your tools.

How to work:
1. Look up how the business defines the metrics in the question (search_business_docs) and any targets or policies that apply.
2. Find the right tables (list_tables / describe_table). Never guess column names.
3. Run focused, aggregated read-only queries (run_sql). Start with the headline number, then drill into drivers:
   segments (country, channel, affiliate, game category, VIP tier), time trends, and outliers.
4. Sanity-check surprising numbers with a second query before relying on them.
5. Stop when you can answer confidently. You have at most {max_queries} queries; spend them on what matters most.
{jira_block}
Rules: SQL must be read-only. Use only numbers that appear in your query results. Today is {today}; the latest period may be incomplete.
Keep any text between tool calls to one short sentence about what you'll check next.

When you're done, reply WITHOUT calling tools, in Markdown with these sections:
### Direct answer
### Key findings
(numbered, each with a concrete number and the query it came from, e.g. "(Q2)")
### Drivers and segments
### Against business targets and policies
(cite documents as [document name]; say so if no document covers it)
### Anomalies and risks
(including responsible-gambling, compliance or bonus-abuse signals if the data shows any)
### Recommended actions
(prioritised, with the owning team)
{jira_section}### Caveats
### Suggested follow-up questions
(3 questions)"""

JIRA_BLOCK = """6. Check the company's Jira (search_jira_tickets) for past tickets about the same issue: earlier incidents,
   known data problems, root causes and fixes. Read the most relevant ones (read_jira_ticket). Say whether the current
   finding is new, a repeat of a resolved issue, or linked to an open ticket. Never invent ticket keys.
"""
JIRA_SECTION = """### Related past Jira tickets
(each as [KEY](link): what it was, its status/resolution, and how it relates; or say none were found)
"""

FINALIZE_NUDGE = ("You've used your step budget. Do not call any more tools. Write the final answer now from the "
                  "evidence you've gathered, in the required format, and note what you couldn't check.")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    turns: int


def build_agent(llm, ws: AgentWorkspace, max_turns: int = 14):
    tools = make_tools(ws)
    by_name = {t.name: t for t in tools}
    llm_tools = llm.bind_tools(tools)
    system = SystemMessage(AGENT_SYSTEM.format(dialect=DIALECT_NAMES.get(ws.db.dialect, ws.db.dialect),
                                               max_queries=ws.max_queries, today=date.today().isoformat(),
                                               jira_block=JIRA_BLOCK if ws.jira else "",
                                               jira_section=JIRA_SECTION if ws.jira else ""))

    def run_tool_calls(ai: AIMessage, refuse: bool = False) -> list[ToolMessage]:
        out = []
        for call in ai.tool_calls:
            if refuse:
                content = "Not run: the step budget is used up. Write the final answer."
            elif call["name"] not in by_name:
                content = f"Unknown tool '{call['name']}'. Available: {', '.join(by_name)}."
            else:
                try:
                    content = str(by_name[call["name"]].invoke(call["args"]))
                except Exception as e:
                    content = f"Tool error: {e}"
            out.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return out

    def agent(state: AgentState):
        ai = llm_tools.invoke([system] + state["messages"])
        return {"messages": [ai], "turns": state.get("turns", 0) + 1}

    def tools_node(state: AgentState):
        return {"messages": run_tool_calls(state["messages"][-1])}

    def finalize(state: AgentState):
        new = [HumanMessage(FINALIZE_NUDGE)]
        for _ in range(2):
            ai = llm_tools.invoke([system] + state["messages"] + new)
            new.append(ai)
            if not ai.tool_calls:
                break
            new += run_tool_calls(ai, refuse=True)
        return {"messages": new}

    def after_agent(state: AgentState):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    def after_tools(state: AgentState):
        return "finalize" if state.get("turns", 0) >= max_turns else "agent"

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", tools_node)
    g.add_node("finalize", finalize)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", END])
    g.add_conditional_edges("tools", after_tools, ["agent", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()


def final_answer(messages: list, ws: AgentWorkspace | None = None) -> str:
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and not last.tool_calls and content_text(last.content).strip():
        return content_text(last.content).strip()
    # The model stopped without a written answer: summarise the evidence it gathered instead of showing nothing.
    lines = ["**The agent ran out of steps before writing its conclusion.** Here is what it checked:"]
    for q in (ws.queries if ws else []):
        status = (f"error: {q['error'][:120]}" if q["error"] else f"{len(q['df']):,} rows")
        lines.append(f"- **Q{q['id']}** {q['purpose']} ({status})")
    lines.append("\nOpen the queries below to see the results, or ask a narrower follow-up question.")
    return "\n".join(lines)


def thought(ai: AIMessage) -> str:
    """Short text the model wrote alongside tool calls (its stated next step)."""
    return content_text(ai.content).strip() if ai.tool_calls else ""
