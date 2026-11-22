"""iGaming AI Analyst — Streamlit web app.

Flow:  Setup (1 database: Snowflake/BigQuery/Databricks/dummy, 2 LLM API, 3 documents)  ->  Ask (text or voice)  ->  Generate SQL
       ->  Review / edit  ->  Execute  ->  Analyze with the LLM against the business documents.

Run:   streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

import hmac

from core import agent as agentmod
from core import charts, graphs, sql_guard, usage, voice
from core import jira as jiramod
from core import docs as docsmod
from core import llm as llmmod
from core.db import BigQueryDB, DatabricksDB, DuckDemoDB, SnowflakeDB

ROOT = Path(__file__).parent
DEMO_DB = ROOT / "demo" / "igaming_demo.duckdb"
DEMO_DOCS = ROOT / "demo" / "docs"
USAGE_FILE = ROOT / ".usage" / "my_api_usage.json"

DEMO_QUESTIONS = [
    "What is monthly GGR and hold % by country in 2026?",
    "Did the Autumn Spins 2026 campaign pay back? Compare October vs September 2026 active players, GGR and bonus cost.",
    "How does PlayHub Media compare with other affiliates on FTD conversion and bonus cost per FTD?",
    "Are there self-excluded players with activity after their exclusion date?",
    "How did German slots turnover change after July 2026 compared with the first half of the year?",
]
GENERIC_QUESTIONS = [
    "What was GGR by month for the last 6 months?",
    "Top 10 games by GGR last month",
    "FTD conversion by acquisition channel this year",
]

st.set_page_config(page_title="iGaming AI Analyst", page_icon="🎰", layout="wide")
ss = st.session_state
DEFAULTS = dict(
    stage="setup", db=None, llm=None, llm_label="", docstore=None, schema_ctx=None, whisper_size="base",
    sql_meta=None, df=None, truncated=False, exec_ms=None, exec_error=None, executed_sql=None,
    insights=None, insight_sources=[], audio_hash=None, pending_sql=None, is_demo=False,
    docs_done=False, edit_step=None, usage_guard=None, my_api_unlocked=False,
    agent_messages=[], agent_turns=[], agent_ws=None, jira=None, jira_done=False,
)
for k, v in DEFAULTS.items():
    ss.setdefault(k, v)

# Values for widgets must be set before the widget is drawn in a run.
if ss.pending_sql is not None:
    ss["sql_editor"] = ss.pending_sql
    ss.pending_sql = None


@st.cache_resource
def usage_ledger():
    """One daily counter shared by every session of this app process."""
    return usage.DailyLedger(USAGE_FILE)


@st.cache_resource(show_spinner="Loading the embedding model (first run downloads ~70 MB)…")
def embedder():
    return docsmod.get_embedder()


def reset_results():
    ss.df, ss.exec_error, ss.exec_ms, ss.executed_sql, ss.insights, ss.insight_sources = None, None, None, None, None, []


def set_db(db, is_demo: bool):
    if ss.db is not None:
        ss.db.close()
    ss.db = db
    ss.is_demo = is_demo
    ss.schema_ctx = graphs.SchemaContext(db.tables, embedder() if len(db.tables) > 30 else None)
    ss.sql_meta = None
    reset_results()
    ss.agent_messages, ss.agent_turns, ss.agent_ws = [], [], None


# =========================================================================== setup page
DB_PLATFORMS = ["Snowflake", "Google BigQuery", "Databricks SQL", "Dummy iGaming data (demo)"]
LLM_CHOICES = [llmmod.ANTHROPIC, llmmod.OPENAI, llmmod.GEMINI, llmmod.OPENAI_COMPAT, llmmod.MY_API]
DOC_SOURCES = ["Upload documents", "Google Drive link", "Sample iGaming documents"]
SECRETS_EXAMPLE = '''# .streamlit/secrets.toml  (or set the same names as environment variables)
LLM_PROVIDER = "anthropic"        # anthropic | openai | gemini | openai_compatible
LLM_MODEL    = "claude-opus-5-5"
LLM_API_KEY  = "sk-ant-..."
# LLM_BASE_URL = "https://.../v1"  # only for openai_compatible'''


def _secret(name: str):
    try:
        if name in st.secrets and st.secrets[name] not in (None, ""):
            return st.secrets[name]
    except Exception:  # no secrets file
        pass
    return os.environ.get(name)


def _connect_db(factory, is_demo: bool = False):
    try:
        with st.spinner("Connecting and reading the schema…"):
            db = factory()
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return
    set_db(db, is_demo)
    ss.edit_step = None
    st.rerun()


# ---------------------------------------------------------------- step 1: database
def snowflake_form():
    with st.form("snowflake_form"):
        account = st.text_input("Account identifier", placeholder="orgname-accountname  or  xy12345.eu-west-1")
        user = st.text_input("User")
        auth = st.selectbox("Authentication", ["Password or programmatic access token",
                                               "SSO (opens a browser on this machine)", "Password + MFA push"])
        password = st.text_input("Password / token", type="password")
        a, b = st.columns(2)
        role = a.text_input("Role", help="Use a read-only role.")
        warehouse = b.text_input("Warehouse")
        a, b = st.columns(2)
        database = a.text_input("Database")
        schema = b.text_input("Schema (optional)", help="Leave empty to read every schema in the database.")
        submitted = st.form_submit_button("Connect to Snowflake", type="primary")
    if submitted:
        params = dict(account=account.strip(), user=user.strip(), role=role.strip(),
                      warehouse=warehouse.strip(), database=database.strip(), schema=schema.strip())
        if auth.startswith("SSO"):
            params["authenticator"] = "externalbrowser"
        elif auth.startswith("Password + MFA"):
            params["authenticator"] = "username_password_mfa"
            params["password"] = password
        else:
            params["password"] = password
        missing = [k for k in ("account", "user", "database") if not params.get(k)]
        if missing:
            st.error("Please fill in: " + ", ".join(missing))
            return
        _connect_db(lambda: SnowflakeDB(params, schema.strip() or None))


def bigquery_form():
    with st.form("bigquery_form"):
        auth = st.radio("Authentication", ["Service-account key (JSON)", "Application Default Credentials (gcloud on this machine)"])
        sa_file = st.file_uploader("Service-account key (JSON)", type=["json"],
                                   help="Needs BigQuery Data Viewer + BigQuery Job User. Not needed for gcloud credentials.")
        a, b = st.columns(2)
        project = a.text_input("Project ID", help="Optional when the key file contains it.")
        dataset = b.text_input("Dataset (optional)", help="Leave empty to read every dataset in the project.")
        a, b = st.columns(2)
        location = a.text_input("Location (optional)", placeholder="EU, US, europe-west2…")
        max_gb = b.number_input("Max GB billed per query", min_value=0.1, max_value=10000.0, value=10.0, step=1.0,
                                help="Queries that would scan more than this are refused by BigQuery.")
        submitted = st.form_submit_button("Connect to BigQuery", type="primary")
    if submitted:
        sa_info = None
        if auth.startswith("Service"):
            if not sa_file:
                st.error("Upload the service-account key, or switch to Application Default Credentials.")
                return
            try:
                sa_info = json.loads(sa_file.getvalue())
            except Exception:
                st.error("That file isn't valid JSON.")
                return
        elif not project.strip():
            st.error("Enter the project ID.")
            return
        _connect_db(lambda: BigQueryDB(project.strip() or None, dataset.strip() or None, sa_info,
                                       location.strip() or None, float(max_gb)))


def databricks_form():
    with st.form("databricks_form"):
        host = st.text_input("Server hostname", placeholder="adb-1234567890123456.7.azuredatabricks.net")
        http_path = st.text_input("HTTP path", placeholder="/sql/1.0/warehouses/abc123def456")
        token = st.text_input("Personal access token", type="password")
        a, b = st.columns(2)
        catalog = a.text_input("Catalog", placeholder="main")
        schema = b.text_input("Schema (optional)", help="Leave empty to read every schema in the catalog.")
        submitted = st.form_submit_button("Connect to Databricks", type="primary")
    if submitted:
        missing = [n for n, v in [("server hostname", host), ("HTTP path", http_path), ("access token", token),
                                  ("catalog", catalog)] if not v.strip()]
        if missing:
            st.error("Please fill in: " + ", ".join(missing))
            return
        _connect_db(lambda: DatabricksDB(host, http_path, token, catalog.strip(), schema.strip() or None))


def database_step():
    platform = st.selectbox("Database platform", DB_PLATFORMS, key="platform")
    if platform == "Snowflake":
        snowflake_form()
    elif platform == "Google BigQuery":
        bigquery_form()
    elif platform == "Databricks SQL":
        databricks_form()
    else:
        st.caption("A local DuckDB database with ~6,000 dummy players, 40 games, daily activity, deposits and "
                   "withdrawals, bonuses, campaigns and affiliates (Jan 2025 – Nov 2026). No account needed.")
        if st.button("Use dummy data", type="primary"):
            _connect_db(lambda: DuckDemoDB(DEMO_DB), is_demo=True)


# ---------------------------------------------------------------- step 2: LLM
def _connect_llm(provider: str, model: str, api_key: str, base_url: str | None, label: str,
                 guard: usage.UsageGuard | None = None, max_tokens: int = 16000, backup_model: str | None = None,
                 extra_models: list[str] | None = None):
    # backup_model may be a comma-separated list (LLM_BACKUP_MODEL for My API): first = backup, rest = extras.
    names = [m.strip() for m in (backup_model or "").split(",")] + list(extra_models or [])
    names = [m for m in dict.fromkeys(names) if m and m != model]
    backup_model, extra_models = (names[0] if names else None), names[1:]
    try:
        cb = [guard] if guard else None
        build = lambda m: llmmod.make_llm(provider, api_key, m, base_url, max_tokens=max_tokens, callbacks=cb)
        primary = build(model)
        backup = build(backup_model) if backup_model else None
        llm = llmmod.ResilientLLM(primary, model, backup, backup_model or "",
                                  extras=[(build(m), m) for m in extra_models])
        with st.spinner("Testing the model…"):
            llmmod.test_llm(llm)
    except usage.UsageLimitError as e:
        st.warning(str(e))
        return
    except llmmod.TransientLLMError as e:
        st.warning(str(e))
        return
    except Exception as e:
        msg = str(e)
        if "NOT_FOUND" in msg or "not_found" in msg or "model_not_found" in msg or "404" in msg:
            st.error(f"The model **{model}** doesn't exist or isn't available to this key. Check the exact model ID "
                     f"in your provider's console (for Gemini, use 'Load my Gemini models').\n\nDetails: {msg[:300]}")
        else:
            st.error(f"LLM test failed: {msg}")
        return
    ss.llm, ss.usage_guard = llm, guard
    ss.agent_messages, ss.agent_turns, ss.agent_ws = [], [], None  # history may not suit a new model
    fallbacks = llm.fallback_names
    ss.llm_label = label + (f" (fallbacks: {', '.join(fallbacks)})" if fallbacks else "")
    ss.whisper_size = ss.get("whisper_choice", "base")
    ss.edit_step = None
    st.rerun()


def gemini_step():
    """Key first, then pick from the models that key can actually use (avoids 404 'model not found')."""
    st.caption("Get a key at aistudio.google.com → Get API key.")
    api_key = st.text_input("Gemini API key", type="password", key="gemini_key").strip()
    if ss.get("gemini_models_for") != api_key:  # key changed: old list no longer applies
        ss.gemini_models, ss.gemini_models_for = None, None
    if st.button("Load my Gemini models", disabled=not api_key):
        try:
            with st.spinner("Asking Google which models this key can use…"):
                models = llmmod.list_gemini_models(api_key)
        except Exception as e:
            st.error(f"Couldn't list models with this key: {e}")
            return
        if not models:
            st.error("This key can't use any text-generation models. Check the key in Google AI Studio.")
            return
        ss.gemini_models, ss.gemini_models_for = models, api_key
        st.rerun()
    models = ss.get("gemini_models")
    if not models:
        st.caption("Enter your key, then load the models it can use.")
        return
    model = st.selectbox(f"Model ({len(models)} available to your key)", models, key="gemini_model")
    others = [m for m in models if m != model]
    default = next((i + 1 for i, m in enumerate(others) if "flash" in m), 0)
    backup = st.selectbox("Backup model (used automatically if the main model is overloaded)",
                          ["None"] + others, index=default, key=f"gemini_backup_{model}")
    backup = None if backup == "None" else backup
    extras = llmmod.pick_gemini_fallbacks(models, [model, backup or ""])
    use_extras = st.checkbox(
        f"If those are overloaded too, also try: {', '.join(extras)}" if extras else "No other fallback models available",
        value=bool(extras), disabled=not extras, key="gemini_extras",
        help="Google's 'high demand' errors often hit several models at once. Lighter models usually still have "
             "capacity. Each fallback is only used while the models above it are failing.")
    if model.startswith("gemma"):
        st.caption("Tip: Gemma models are open models served on shared capacity and are often busy; a Gemini "
                   "'flash' model is usually more reliable as the main model.")
    if st.button("Connect to LLM", type="primary"):
        _connect_llm(llmmod.GEMINI, model, api_key, None, f"{llmmod.GEMINI} · {model}",
                     backup_model=backup, extra_models=extras if use_extras else None)


def llm_step():
    choice = st.selectbox("API provider", LLM_CHOICES, key="provider")
    st.selectbox("Speech-to-text model (local Whisper, for voice questions)", ["base", "tiny", "small", "medium"],
                 key="whisper_choice", help="Runs on the app's own server, not on the AI provider. Larger = more accurate, slower. Downloaded on first use.")
    if choice == llmmod.MY_API:
        cfg = llmmod.preconfigured(_secret)
        if cfg:
            limits = usage.Limits.from_settings(_secret)
            st.info(f"Uses the API key configured by the app owner: **{cfg['provider']} · {cfg['model']}**. "
                    "Users don't need to enter their own key.")
            st.caption(f"Usage limits: {limits.session_requests} AI requests and {limits.session_tokens:,} tokens per "
                       f"session; {limits.daily_requests} requests and {limits.daily_tokens:,} tokens per day for the "
                       "whole app" + (f"; ${limits.daily_usd:,.2f} per day" if limits.daily_usd else "") + ".")
            password = _secret("MY_API_PASSWORD")
            if password and not ss.my_api_unlocked:
                with st.form("my_api_pw"):
                    entered = st.text_input("Access password for My API", type="password")
                    ok = st.form_submit_button("Unlock", type="primary")
                if ok:
                    if hmac.compare_digest(entered.encode(), str(password).encode()):
                        ss.my_api_unlocked = True
                        st.rerun()
                    st.error("Wrong password.")
                return
            if st.button("Use my API", type="primary"):
                guard = usage.UsageGuard(limits, usage_ledger(), cfg["model"])
                _connect_llm(cfg["provider"], cfg["model"], cfg["api_key"], cfg["base_url"], f"My API · {cfg['model']}",
                             guard=guard, max_tokens=limits.max_output_tokens, backup_model=cfg.get("backup_model"))
        else:
            st.warning("No preconfigured key found. Add one to `.streamlit/secrets.toml` (or environment variables), "
                       "then reload the page:")
            st.code(SECRETS_EXAMPLE, language="toml")
        return
    if choice == llmmod.GEMINI:
        gemini_step()
        return
    with st.form("llm_form"):
        hint = {llmmod.OPENAI: "OpenAI model id"}.get(choice, "model id")
        model = st.text_input("Model", value=llmmod.DEFAULT_MODELS[choice], placeholder=hint)
        api_key = st.text_input("API key", type="password")
        base_url = st.text_input("Base URL", placeholder="https://…/v1") if choice == llmmod.OPENAI_COMPAT else None
        backup = st.text_input("Backup model (optional)", placeholder="e.g. a smaller/faster model from the same provider",
                               help="Used automatically if the main model is overloaded or unavailable.")
        submitted = st.form_submit_button("Connect to LLM", type="primary")
    if submitted:
        if not api_key.strip() and choice != llmmod.OPENAI_COMPAT:
            st.error("Please enter an API key.")
            return
        _connect_llm(choice, model.strip(), api_key.strip(), base_url, f"{choice} · {model.strip()}",
                     backup_model=backup)


# ---------------------------------------------------------------- step 3: documents
def _index_docs(loader):
    with st.status("Loading documents…", expanded=True) as status:
        try:
            docs = loader()
            st.write(f"Embedding {len(docs)} documents…")
            ss.docstore = docsmod.DocStore.build(docs, embedder())
            status.update(label=f"Indexed {len(docs)} documents", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Couldn't load documents", state="error")
            st.error(str(e))
            return
    ss.docs_done, ss.edit_step = True, None
    st.rerun()


def docs_step():
    source = st.radio("Where are your business documents?", DOC_SOURCES, horizontal=True, key="doc_source")
    if source == "Upload documents":
        with st.form("upload_form"):
            uploads = st.file_uploader("Documents (PDF, DOCX, TXT, MD, CSV, XLSX, HTML)", accept_multiple_files=True,
                                       type=[e.strip(".") for e in sorted(docsmod.SUPPORTED)])
            submitted = st.form_submit_button("Upload & index", type="primary")
        if submitted:
            if not uploads:
                st.error("Choose at least one file.")
                return

            def load():
                out = [(f.name, docsmod.parse_bytes(f.name, f.getvalue())) for f in uploads]
                return [(n, t) for n, t in out if t.strip()]
            _index_docs(load)
    elif source == "Google Drive link":
        with st.form("drive_form"):
            link = st.text_input("Google Drive folder or file link", placeholder="https://drive.google.com/drive/folders/…")
            st.caption("Public folders: share as *Anyone with the link → Viewer* (up to 50 files per folder). "
                       "Private folders: share them with a service account's email and upload its key below.")
            sa_file = st.file_uploader("Service-account key (JSON, optional)", type=["json"])
            submitted = st.form_submit_button("Fetch & index", type="primary")
        if submitted:
            if not link.strip():
                st.error("Paste a Google Drive link.")
                return
            sa_info = json.loads(sa_file.getvalue()) if sa_file else None
            _index_docs(lambda: docsmod.load_from_drive(link, sa_info))
    else:
        names = sorted(p.name for p in DEMO_DOCS.glob("*.md"))
        st.caption("Dummy business documents that match the dummy data: " + ", ".join(names))
        if st.button("Use sample documents", type="primary"):
            _index_docs(lambda: docsmod.parse_dir(DEMO_DOCS))
    if st.button("Skip: continue without documents"):
        ss.docstore, ss.docs_done, ss.edit_step = None, True, None
        st.rerun()


# ---------------------------------------------------------------- step 4: Jira (optional)
def jira_step():
    st.caption("Optional. Lets the AI find past Jira tickets about the same issue (incidents, data problems, fixes, "
               "earlier analyses) and link them in its answers. Read-only: the app never creates or edits tickets. "
               "Ticket text you retrieve is sent to the LLM provider you chose in step 2.")
    with st.form("jira_form"):
        kind = st.radio("Jira type", [jiramod.CLOUD, jiramod.SERVER], horizontal=True)
        url = st.text_input("Jira site URL", placeholder="https://your-company.atlassian.net")
        email = st.text_input("Account email (Jira Cloud only)")
        token = st.text_input("API token (Cloud) or personal access token (Server / Data Center)", type="password",
                              help="Cloud: id.atlassian.com → Security → API tokens. Use an account that can only browse.")
        a, b = st.columns(2)
        projects = a.text_input("Project keys to search (optional)", placeholder="OPS, DATA, CRM",
                                help="Leave empty to search every project the account can see.")
        extra = b.text_input("Extra JQL filter (optional)", placeholder='created >= -730d')
        submitted = st.form_submit_button("Connect Jira", type="primary")
    if submitted:
        if not url.strip() or not token.strip() or (kind == jiramod.CLOUD and not email.strip()):
            st.error("Please fill in the site URL, token" + (" and email." if kind == jiramod.CLOUD else "."))
            return
        client = jiramod.JiraClient(url, kind, email, token, [p for p in projects.split(",")], extra)
        try:
            with st.spinner("Checking the Jira connection…"):
                client.test()
        except Exception as e:
            st.error(f"Couldn't connect to Jira: {e}")
            return
        ss.jira, ss.jira_done, ss.edit_step = client, True, None
        st.rerun()
    if st.button("Skip: don't use Jira", key="skip_jira"):
        ss.jira, ss.jira_done, ss.edit_step = None, True, None
        st.rerun()


# ---------------------------------------------------------------- page
def _summary(i: int) -> str:
    if i == 0:
        return f"{ss.db.label} · {len(ss.db.tables)} tables"
    if i == 1:
        return ss.llm_label
    if i == 2:
        if ss.docstore:
            return f"{len(ss.docstore.names)} documents indexed ({ss.docstore.n_chunks} chunks)"
        return "Skipped: no business documents"
    if ss.jira:
        scope = ", ".join(ss.jira.projects) if ss.jira.projects else "all visible projects"
        return f"Jira connected as {ss.jira.user} · {ss.jira.base} · {scope}"
    return "Skipped: Jira not used"


def setup_page():
    st.title("🎰 iGaming AI Analyst")
    st.caption("Four steps: database, LLM, business documents, and optionally Jira. Credentials are kept only in this session's "
               "memory on the server and are never written to disk.")
    steps = [("Choose your database", database_step), ("Choose your LLM API", llm_step),
             ("Add business documents (RAG)", docs_step), ("Link past Jira tickets (optional)", jira_step)]
    done = [ss.db is not None, ss.llm is not None, bool(ss.docs_done), bool(ss.jira_done)]
    for i, (title, render) in enumerate(steps):
        unlocked = i == 0 or all(done[:i])
        icon = "✅" if done[i] else ("👉" if unlocked else "🔒")
        with st.container(border=True):
            st.subheader(f"{icon} Step {i + 1} · {title}")
            if not unlocked:
                st.caption("Complete the previous step first.")
                continue
            if done[i] and ss.edit_step != i:
                c1, c2 = st.columns([6, 1], vertical_alignment="center")
                c1.success(_summary(i))
                if c2.button("Change", key=f"change{i}", width="stretch"):
                    ss.edit_step = i
                    st.rerun()
                continue
            render()
            if done[i] and st.button("Cancel", key=f"cancel{i}"):
                ss.edit_step = None
                st.rerun()
    if all(done):
        if st.button("Continue to workspace →", type="primary", width="stretch"):
            ss.stage, ss.edit_step = "workspace", None
            st.rerun()


# =========================================================================== workspace
def sidebar():
    with st.sidebar:
        st.header("Connections")
        st.markdown(f"🗄️ **Data:** {ss.db.label}")
        st.markdown(f"🤖 **LLM:** {ss.llm_label}")
        if ss.docstore:
            st.markdown(f"📚 **Docs:** {len(ss.docstore.names)} files · {ss.docstore.n_chunks} chunks")
        else:
            st.markdown("📚 **Docs:** none loaded")
        st.markdown(f"🎫 **Jira:** {ss.jira.base.removeprefix('https://')}" if ss.jira else "🎫 **Jira:** not connected")
        if ss.usage_guard:
            usage_panel()
        if st.button("⚙️ Change connections", width="stretch"):
            ss.stage = "setup"
            st.rerun()
        with st.expander(f"Schema · {len(ss.db.tables)} tables"):
            for t in ss.db.tables[:300]:
                st.markdown(f"**{t.full_name}**")
                st.caption(", ".join(c[0] for c in t.columns))
        if ss.docstore:
            with st.expander("Documents"):
                for n in ss.docstore.names:
                    st.write("📄 " + n)


def usage_panel():
    g = ss.usage_guard
    lim, day, s = g.limits, g.ledger.snapshot(), g.session
    st.markdown("**My API usage**")
    st.progress(min(1.0, s.requests / max(1, lim.session_requests)),
                text=f"This session: {s.requests}/{lim.session_requests} requests · {s.tokens:,}/{lim.session_tokens:,} tokens")
    st.progress(min(1.0, day.requests / max(1, lim.daily_requests)),
                text=f"Today (all users): {day.requests}/{lim.daily_requests} requests · {day.tokens:,}/{lim.daily_tokens:,} tokens")
    cost = day.cost(g.model)
    if cost is not None:
        st.caption(f"Estimated spend today: ${cost:,.2f}" + (f" of ${lim.daily_usd:,.2f}" if lim.daily_usd else ""))


def llm_failed(e: Exception, what: str):
    if isinstance(e, (usage.UsageLimitError, llmmod.TransientLLMError)):
        st.warning(str(e))
    else:
        st.error(f"{what}: {e}")


def current_question() -> str:
    if ss.sql_meta and ss.sql_meta.get("question"):
        return ss.sql_meta["question"]
    return (ss.get("question") or "").strip()


def run_sql_graph(inputs: dict, spinner: str):
    graph = graphs.build_sql_graph(ss.llm, ss.db.dialect, ss.schema_ctx, ss.docstore,
                                   row_limit=int(ss.get("row_limit", 1000)))
    with st.spinner(spinner):
        try:
            result = graph.invoke(inputs)
        except Exception as e:
            llm_failed(e, "LLM error")
            return
    ss.sql_meta = {**result, "question": inputs["question"]}
    ss.pending_sql = result.get("sql", "")
    reset_results()
    st.rerun()


def question_input() -> str:
    left, right = st.columns([1, 2], gap="large")
    with left:
        audio = st.audio_input("🎤 Speak your question")
        if audio is not None:
            data = audio.getvalue()
            h = hashlib.md5(data).hexdigest()
            if h != ss.audio_hash:
                ss.audio_hash = h
                try:
                    with st.spinner("Transcribing locally with Whisper…"):
                        text = voice.transcribe(data, ss.whisper_size)
                    if text:
                        ss["question"] = text
                        st.toast("Transcribed: " + text[:80])
                    else:
                        st.warning("No speech detected. Try again a little closer to the mic.")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
    with right:
        st.caption("Try an example")
        for i, q in enumerate(DEMO_QUESTIONS if ss.is_demo else GENERIC_QUESTIONS):
            st.button(q, key=f"ex{i}", on_click=lambda q=q: ss.update(question=q), width="stretch")

    return st.text_area("Your question (typed, or filled in from your voice)", key="question", height=90,
                        placeholder="e.g. What was NGR by market last quarter, and are we on target?")


def guided_ask(question: str):
    if st.button("✨ Generate SQL", type="primary", disabled=not (question or "").strip()):
        run_sql_graph({"question": question.strip()}, "Reading business context and writing SQL…")

    meta = ss.sql_meta
    if meta:
        if meta.get("error"):
            st.error(f"Couldn't produce a valid read-only query after {meta.get('attempts')} tries: {meta['error']}")
        if meta.get("assumptions"):
            st.info("**Assumptions:** " + meta["assumptions"])
        if meta.get("warning"):
            st.warning(meta["warning"])
        if meta.get("doc_sources"):
            with st.expander("📚 Business context used for this query"):
                st.caption("Sources: " + ", ".join(meta["doc_sources"]))
                st.text(meta.get("doc_context", "")[:6000])


def sql_section():
    st.subheader("2 · Review & execute SQL")
    ss.setdefault("sql_editor", "")
    st.text_area("SQL (editable; read-only queries only)", key="sql_editor", height=260)
    c1, c2, _ = st.columns([1, 1, 3])
    c1.number_input("Row limit", min_value=10, max_value=100000, value=5000, step=500, key="row_limit")
    c2.write("")
    run = c2.button("▶️ Execute SQL", type="primary", disabled=not ss.sql_editor.strip(), width="stretch")
    if run:
        sql, err, _warn = sql_guard.validate(ss.sql_editor, ss.db.dialect)
        reset_results()
        if err:
            ss.exec_error = f"Blocked by the read-only guard: {err}"
        else:
            t0 = time.perf_counter()
            try:
                with st.spinner("Running query…"):
                    df, truncated = ss.db.run(sql, int(ss.row_limit))
                ss.df, ss.truncated, ss.executed_sql = df, truncated, sql
                ss.exec_ms = (time.perf_counter() - t0) * 1000
            except Exception as e:
                ss.exec_error = str(e)
    if ss.exec_error:
        st.error(ss.exec_error)
        if st.button("🔧 Ask the LLM to fix this query"):
            q = current_question() or "Fix this query"
            run_sql_graph({"question": q, "prev_sql": ss.sql_editor, "exec_error": ss.exec_error},
                          "Fixing the query…")


def results_section():
    df = ss.df
    st.subheader("3 · Results")
    st.caption(f"{len(df):,} rows · {len(df.columns)} columns · {ss.exec_ms:,.0f} ms"
               + (" · ⚠️ truncated at the row limit" if ss.truncated else ""))
    tab_table, tab_chart = st.tabs(["Table", "Chart"])
    with tab_table:
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode(), "query_result.csv", "text/csv")
    with tab_chart:
        try:
            fig = charts.auto_chart(df)
        except Exception:
            fig = None
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No obvious chart for this result shape.")


def insights_section():
    st.subheader("4 · AI insights")
    if ss.docstore is None:
        st.caption("No business documents loaded: insights won't be compared against targets or policies.")
    if st.button("🧠 Analyze results", type="primary", disabled=ss.df is None or ss.df.empty):
        graph = graphs.build_analysis_graph(ss.llm, ss.docstore, ss.jira, embedder() if ss.jira else None)
        final: dict = {}
        inputs = {"question": current_question() or "Explain this data",
                  "sql": ss.executed_sql, "df": ss.df, "truncated": ss.truncated}
        try:
            with st.container(border=True):
                with st.spinner("Profiling the data, reading business documents and writing insights…"):
                    text = st.write_stream(graphs.stream_analysis(graph, inputs, final))
            ss.insights = text if isinstance(text, str) else "".join(map(str, text))
            ss.insight_sources = final.get("doc_sources", [])
            ss.insight_profile = final.get("profile", "")
            ss.insight_tickets = final.get("jira_tickets", [])
            used = getattr(ss.llm, "last_used", None)
            ss.insight_model = used if used and used != getattr(ss.llm, "primary_name", used) else None
        except Exception as e:
            llm_failed(e, "Analysis failed")
            return
        st.rerun()
    if ss.insights:
        with st.container(border=True):
            st.markdown(ss.insights)
            if ss.insight_sources:
                st.caption("Business documents consulted: " + ", ".join(ss.insight_sources))
            if ss.get("insight_model"):
                st.caption(f"The main model was overloaded, so the backup model **{ss.insight_model}** wrote this.")
        if ss.get("insight_tickets"):
            with st.expander(f"🎫 Jira tickets found for this question ({len(ss.insight_tickets)})"):
                st.markdown(ticket_links(ss.insight_tickets))
        with st.expander("Facts computed from the data (what the LLM was given)"):
            st.text(ss.get("insight_profile", ""))
        st.download_button("Download insights (.md)", ss.insights.encode(), "insights.md", "text/markdown")


AGENT_MODE, GUIDED_MODE = "🤖 Agent (autonomous)", "🧭 Guided (step by step)"


def workspace_page():
    sidebar()
    st.title("🎰 iGaming AI Analyst")
    mode = st.radio("Mode", [AGENT_MODE, GUIDED_MODE], horizontal=True, key="mode",
                    help="Agent: the AI plans its own investigation, runs several read-only queries and writes the "
                         "analysis. Guided: you review and run one query, then ask for insights.")
    if mode == AGENT_MODE:
        agent_page()
        return
    st.subheader("1 · Ask a question")
    guided_ask(question_input())
    st.divider()
    sql_section()
    if ss.df is not None:
        st.divider()
        results_section()
        st.divider()
        insights_section()


# =========================================================================== agent mode
TOOL_ICONS = {"list_tables": "🗂️", "describe_table": "🔍", "search_business_docs": "📚", "run_sql": "🧮", "thought": "💭",
              "search_jira_tickets": "🎫", "read_jira_ticket": "🎫"}


def _query_status(q: dict) -> str:
    return f"failed: {q['error'][:70]}" if q["error"] else f"{len(q['df']):,} rows · {q['ms']:,.0f} ms"


def render_event_line(e: dict, queries: dict):
    """Compact one-line view (used live, inside the status box)."""
    icon = TOOL_ICONS.get(e["tool"], "🔧")
    if e["tool"] == "thought":
        st.markdown(f"{icon} _{e['title']}_")
    elif e.get("query_id"):
        q = queries[e["query_id"]]
        st.markdown(f"{icon} **{e['title']}** · {_query_status(q)}")
        st.code(q["sql"], language="sql")
    else:
        st.markdown(f"{icon} {e['title']}" + (f" · _{e['detail']}_" if e.get("detail") else ""))
        if e.get("tickets"):
            st.markdown(ticket_links(e["tickets"]))


def ticket_links(tickets: list[dict]) -> str:
    rows = []
    for t in tickets:
        state = t["status"] + (f" / {t['resolution']}" if t.get("resolution") else "")
        rows.append(f"- [{t['key']}]({t['url']}) {t['summary']} · _{state}_")
    return "\n".join(rows)


def render_event_full(e: dict, queries: dict, key: str):
    """Detailed view with each query's SQL, table and chart (used in the saved conversation)."""
    if not e.get("query_id"):
        render_event_line(e, queries)
        return
    q = queries[e["query_id"]]
    with st.expander(f"🧮 {e['title']} · {_query_status(q)}"):
        st.code(q["sql"], language="sql")
        if q["error"]:
            st.error(q["error"])
            return
        df = q["df"]
        st.dataframe(df, width="stretch", hide_index=True, height=min(400, 38 + 35 * len(df)))
        if q["truncated"]:
            st.caption("Truncated at the row limit.")
        try:
            fig = charts.auto_chart(df)
        except Exception:
            fig = None
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key=f"chart_{key}_{q['id']}")
        st.download_button("Download CSV", df.to_csv(index=False).encode(), f"Q{q['id']}.csv", "text/csv",
                           key=f"dl_{key}_{q['id']}")


def run_agent(question: str, max_queries: int, max_steps: int, resume: dict | None = None):
    """Run the agent on a question. `resume` continues a run that stopped on a provider error, from its last state."""
    if ss.agent_ws is None:
        ss.agent_ws = agentmod.AgentWorkspace(ss.db, ss.docstore)
    ws = ss.agent_ws
    ws.docstore = ss.docstore
    ws.jira = ss.jira
    ws.embedder = embedder() if ss.jira else None
    if resume:  # same question: keep its query budget, steps and history so far
        ws.max_queries, max_steps = resume["max_queries"], resume["max_steps"]
        start_event, start_query = resume["start_event"], resume["start_query"]
        inputs = {"messages": resume["messages"], "turns": resume["turns"]}
    else:
        ws.max_queries = len(ws.queries) + max_queries  # budget is per question
        start_event, start_query = len(ws.events), len(ws.queries)
        inputs = {"messages": list(ss.agent_messages) + [HumanMessage(question)]}
    graph = agentmod.build_agent(ss.llm, ws, max_turns=max_steps)
    queries = lambda: {q["id"]: q for q in ws.queries}
    final_state, error, shown = None, None, start_event
    with st.status("The agent is investigating…", expanded=True) as status:
        try:
            for mode, payload in graph.stream(inputs, stream_mode=["updates", "values"]):
                if mode == "values":
                    final_state = payload
                    continue
                for node, update in payload.items():
                    if node == "agent" and update.get("messages"):
                        thought = agentmod.thought(update["messages"][-1])
                        if thought:
                            ws.events.append({"tool": "thought", "title": thought})
                for e in ws.events[shown:]:
                    render_event_line(e, queries())
                    if e["tool"] != "thought":
                        status.update(label=f"The agent is investigating… {e['title']}")
                shown = len(ws.events)
            n_q = len(ws.queries) - start_query
            status.update(label=f"Done · {len(ws.events) - start_event} steps · {n_q} queries", state="complete",
                          expanded=False)
        except Exception as e:
            error = e
            status.update(label="The agent stopped", state="error")
    turn = {"question": question, "events": ws.events[start_event:], "query_ids": [q["id"] for q in ws.queries[start_query:]]}
    if error is not None:
        turn["answer"] = None
        turn["error"] = str(error)
        turn["error_kind"] = "warning" if isinstance(error, (usage.UsageLimitError, llmmod.TransientLLMError)) else "error"
        if isinstance(error, llmmod.TransientLLMError):  # provider hiccup: keep progress so the run can continue
            state = final_state or inputs
            turn["resume"] = {"messages": state["messages"], "turns": state.get("turns", 0),
                              "max_queries": ws.max_queries, "max_steps": max_steps,
                              "start_event": start_event, "start_query": start_query}
    else:
        ss.agent_messages = final_state["messages"]
        turn["answer"] = agentmod.final_answer(final_state["messages"], ws)
    ss.agent_turns.append(turn)
    st.rerun()


def agent_page():
    st.caption("Ask a business question. The agent reads your business documents, explores the schema, runs "
               "read-only queries, checks surprising numbers and writes an in-depth analysis. Every step is shown.")
    resume_turn = None
    if ss.pop("agent_resume", False) and ss.agent_turns and ss.agent_turns[-1].get("resume"):
        resume_turn = ss.agent_turns.pop()  # re-run below the earlier turns; its steps are carried over
    ws = ss.agent_ws
    queries = {q["id"]: q for q in ws.queries} if ws else {}
    for i, turn in enumerate(ss.agent_turns):
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            n_q = len(turn["query_ids"])
            tab_answer, tab_steps = st.tabs(["📝 Answer", f"🔎 Steps & queries ({len(turn['events'])} steps, {n_q} queries)"])
            with tab_answer:
                if turn.get("answer"):
                    st.markdown(turn["answer"])
                    st.download_button("Download analysis (.md)", turn["answer"].encode(), f"analysis_{i + 1}.md",
                                       "text/markdown", key=f"dl_answer_{i}")
                elif turn.get("error_kind") == "warning":
                    st.warning(turn["error"])
                    if turn.get("resume") and i == len(ss.agent_turns) - 1:
                        done = len(turn["events"])
                        st.caption(f"The agent's progress is kept ({done} steps, {len(turn['query_ids'])} queries). "
                                   "Continuing picks up from the last completed step.")
                        if st.button("▶️ Continue where it stopped", key=f"resume_{i}", type="primary"):
                            ss.agent_resume = True
                            st.rerun()
                else:
                    st.error(f"The agent failed: {turn.get('error')}")
            with tab_steps:
                for j, e in enumerate(turn["events"]):
                    render_event_full(e, queries, key=f"{i}_{j}")

    st.subheader("Ask a follow-up" if ss.agent_turns else "Ask a question")
    question = question_input()
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1], vertical_alignment="bottom")
    max_queries = c1.slider("Max queries", 2, 15, 8, help="Most read-only queries the agent may run per question.")
    max_steps = c2.slider("Max steps", 4, 25, 14, help="Most reasoning turns per question. Each turn is one LLM call.")
    run = c3.button("🤖 Run agent" if not ss.agent_turns else "🤖 Ask follow-up", type="primary",
                    disabled=not (question or "").strip(), width="stretch")
    if c4.button("🧹 New conversation", disabled=not ss.agent_turns, width="stretch"):
        ss.agent_messages, ss.agent_turns, ss.agent_ws = [], [], None
        st.rerun()
    if ss.usage_guard:
        st.caption(f"Each step is one AI request against your My API limits; this run may use up to {max_steps + 1}.")
    if resume_turn is not None:
        with st.chat_message("user"):
            st.markdown(resume_turn["question"])
        run_agent(resume_turn["question"], 0, 0, resume=resume_turn["resume"])
    elif run:
        run_agent(question.strip(), max_queries, max_steps)


if ss.stage == "workspace" and ss.db is not None and ss.llm is not None:
    workspace_page()
else:
    ss.stage = "setup"
    setup_page()
