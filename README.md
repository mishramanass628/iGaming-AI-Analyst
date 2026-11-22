# iGaming AI Analyst

Streamlit web app: connect **Snowflake**, an **LLM** and a **Google Drive folder of business documents**, then ask questions by text or voice → get SQL → review/edit → **Execute** → **Analyze** for in-depth insights grounded in your documents.

Built with Python, LangGraph and LangChain.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Open http://localhost:8501.

## Deploy on Streamlit Community Cloud (free)

1. Put this folder in a GitHub repository (private is fine). `.streamlit/secrets.toml`, `.venv/`, the demo database
   and usage files are git-ignored, so your key is never pushed.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app** → **Deploy a public app from GitHub**.
3. Pick the repository and branch, set **Main file path** to `app.py`.
4. Open **Advanced settings**:
   - **Python version**: 3.12 (the versions in `requirements.txt` were tested with it).
   - **Secrets**: paste the contents of your local `.streamlit/secrets.toml` (with `LLM_API_KEY` filled in).
     These become `st.secrets`; the app reads them exactly as it does locally.
5. Click **Deploy**. The first build takes several minutes (about 1 GB of packages).

Notes for the hosted app:
- **Anyone with the link can open it.** If "My API" uses your key, set `MY_API_PASSWORD` in the secrets, keep the
  usage limits on, and/or restrict viewers in the app's **Settings → Sharing** (viewers must sign in).
- Resources are limited (around 1–3 GB RAM). Voice works with the `tiny`, `base` or `small` Whisper model;
  `medium` may run out of memory.
- The demo database is rebuilt on the first visit after each restart (a few seconds). The daily usage counter is
  stored on the app's disk, which Streamlit may reset when the app restarts or is redeployed.
- The app sleeps after a period without visitors; the next visitor wakes it (about a minute).
- Your database, Jira and Drive must be reachable from the internet. If they only allow listed IP addresses,
  Community Cloud won't be able to connect (its IPs aren't fixed); use a host with a fixed IP instead.
- After a code change, push to GitHub and the app redeploys automatically.

## Setup page (four steps, each unlocks the next)
1. **Database** (dropdown):
   - **Snowflake**: account, user, password/token (or SSO / MFA), role, warehouse, database, optional schema.
   - **Google BigQuery**: service-account JSON key or gcloud Application Default Credentials, project, optional dataset, max GB billed per query.
   - **Databricks SQL**: server hostname, HTTP path, personal access token, catalog, optional schema.
   - **Dummy iGaming data**: local DuckDB, no account needed.
   Use read-only credentials.
2. **LLM API** (dropdown): Anthropic (default `claude-opus-5-5`), OpenAI, Google Gemini, any OpenAI-compatible endpoint, or **My API (preconfigured key)**.
   **My API** uses your own key so users don't need one. Edit `.streamlit/secrets.toml` (already created, git-ignored),
   uncomment `LLM_API_KEY` and paste your key, then restart the app. Options in that file:
   - `LLM_PROVIDER` (anthropic, openai, gemini, openai_compatible), `LLM_MODEL`, `LLM_BASE_URL`: which model your key calls (default Anthropic, `claude-opus-5-5`).
   - `MY_API_PASSWORD`: optional; people must enter it before using your key.
   - Usage limits, enforced on every LLM call made with your key (`core/usage.py`):

     | Setting | Default | Scope |
     |---|---|---|
     | `MY_API_MAX_REQUESTS_PER_SESSION` | 30 | one browser session |
     | `MY_API_MAX_TOKENS_PER_SESSION` | 300,000 | one browser session |
     | `MY_API_MAX_REQUESTS_PER_DAY` | 300 | whole app, resets midnight UTC |
     | `MY_API_MAX_TOKENS_PER_DAY` | 3,000,000 | whole app |
     | `MY_API_MAX_USD_PER_DAY` | 10.0 | whole app, estimated (Claude models only; 0 = off) |
     | `MY_API_MAX_OUTPUT_TOKENS` | 12,000 | per LLM response |

     Daily totals are saved in `.usage/my_api_usage.json`, so restarts don't reset them. The sidebar shows usage.
     These are app-side limits. Also set a spend limit on the key itself in your provider's console as a hard backstop.

**Reliability:** every LLM call retries automatically on provider overload, rate limits, 5xx errors and timeouts
(backoff 2 s, 5 s, 10 s). If you pick a **backup model** in step 2 (or set `LLM_BACKUP_MODEL` for My API), the app
switches to it when the main model stays unavailable, and the insights note which model wrote them.

3. **Business documents** (choice): upload files, a Google Drive link, or the sample iGaming documents. Can be skipped.
   - Drive, public: share as *Anyone with the link → Viewer* (≤50 files per folder).
   - Drive, private: share the folder with a Google service account and upload its JSON key.
   Supported: PDF, DOCX, TXT, MD, CSV, XLSX, HTML, Google Docs/Sheets/Slides.

4. **Past Jira tickets (optional):** Jira Cloud (site URL, email, API token) or Server / Data Center
   (personal access token), optionally limited to project keys or an extra JQL filter. Read-only.
   - The agent gets `search_jira_tickets` and `read_jira_ticket`, checks whether an issue was raised or solved
     before, and lists related tickets with links in a "Related past Jira tickets" section.
   - Guided mode adds matching tickets to the analysis and lists them under the insights.
   - Search: keywords from the question go to Jira (`text ~` JQL), then results are re-ranked by meaning with
     the local embedding model and weak matches are dropped (`core/jira.py`).
   - Ticket text is sent to the LLM provider you chose; use a Jira account that can only browse what's appropriate.

Credentials live only in the Streamlit session's memory and are never written to disk.

## Two modes in the workspace
- **🤖 Agent (autonomous, default):** a LangGraph tool-calling agent decides its own investigation. Tools:
  `search_business_docs`, `list_tables`, `describe_table`, `run_sql` (read-only guard on every query).
  It looks up metric definitions and targets, explores the schema, runs several aggregated queries, checks
  surprising numbers, drills into drivers, then writes the analysis citing queries (Q1, Q2…) and documents.
  Every step is shown live and saved (SQL, table, chart, CSV per query). Follow-up questions keep the context.
  Budgets per question: max queries (default 8) and max steps (default 14); at the step limit it must conclude.
- **🧭 Guided:** you review and run one generated query, then ask for insights (the original flow).

```
Agent:  START → agent ─(tool calls)→ tools → agent … ─(answer)→ END
                                       └─(step budget used)→ finalize → END
```

## How it works
```
SQL graph:      retrieve (doc RAG + schema) → generate_sql → validate ─(invalid, <3 tries)→ generate_sql
                                     user reviews SQL → ▶ Execute (read-only guard + row limit)
                                     on DB error → 🔧 "Ask the LLM to fix" (same graph, with the error)
Analysis graph: profile (pandas facts) → retrieve (docs: targets, policies, definitions) → analyze (streamed)
```
- `core/db.py`: Snowflake + DuckDB connectors, schema introspection (column comments included).
- `core/docs.py`: Google Drive loading, parsing, chunking, local embeddings (fastembed `bge-small`), vector search.
- `core/jira.py`: read-only Jira client (Cloud and Server), ticket search and ranking.
- `core/agent.py`: the autonomous agent (tools, prompt, loop, budgets).
- `core/graphs.py`: the guided-mode LangGraph workflows, prompts, data profiling.
- `core/sql_guard.py`: sqlglot-based read-only check (SELECT/WITH only, single statement).
- `core/voice.py`: local Whisper speech-to-text (faster-whisper). Audio never leaves the machine.
- `demo/seed.py`: builds the dummy iGaming DuckDB; `demo/docs/`: sample business documents.

For warehouses with more than 30 tables, only the most relevant tables (by embedding similarity) are sent to the LLM.
