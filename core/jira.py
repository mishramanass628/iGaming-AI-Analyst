"""Read-only Jira client: find past tickets related to a business question.

Search strategy (Jira's own text search is literal, so a natural-language question rarely matches well):
1. Pull keywords out of the question and ask Jira for candidates matching any of them (JQL `text ~`).
2. Re-rank the candidates by meaning with the same local embeddings used for business documents.

Supports Jira Cloud (email + API token, REST v3) and Jira Server / Data Center (personal access token, REST v2).
Only GET requests (plus Cloud's read-only JQL search) are made; tickets are never created or changed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import requests

CLOUD, SERVER = "Jira Cloud", "Jira Server / Data Center"
FIELDS = ["summary", "status", "resolution", "resolutiondate", "created", "updated", "issuetype", "priority",
          "assignee", "labels", "components", "project", "description", "comment"]

_STOP = set("""a an the and or but if of to in on at by for with from into over under about as is are was were be been
being do does did done have has had having this that these those it its we our you your they their them he she his her
what which who whom whose why how when where there here than then so such not no yes can could should would will may
might must shall all any some each every more most less least very much many few per vs versus last next this year
month week day today yesterday show tell give me please compare between across did does going since ever past
raised solved issue issues problem problems ticket tickets jira related after before during until within
why happened happen look looking find found get got""".split())


def keywords(text: str, limit: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", text)  # no hyphens: Jira treats "-" as an operator
    out: list[str] = []
    for w in words:
        lw = w.lower().strip("-_")
        if lw in _STOP or lw in out or len(lw) < 3:
            continue
        out.append(lw)
    return out[:limit]


def adf_to_text(node) -> str:
    """Flatten Atlassian Document Format (Jira Cloud rich text) to plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(adf_to_text(n) for n in node)
    t = node.get("type")
    if t == "text":
        return node.get("text", "")
    if t in ("hardBreak",):
        return "\n"
    if t == "mention":
        return node.get("attrs", {}).get("text", "")
    inner = adf_to_text(node.get("content", []))
    if t in ("paragraph", "heading", "listItem", "codeBlock", "blockquote", "tableRow"):
        return inner.strip() + "\n"
    return inner


@dataclass
class Ticket:
    key: str
    url: str
    summary: str
    status: str
    resolution: str
    issue_type: str
    priority: str
    project: str
    created: str
    updated: str
    resolved: str
    assignee: str
    labels: list[str]
    description: str
    comments: list[str] = field(default_factory=list)
    score: float = 0.0

    def brief(self) -> str:
        state = f"{self.status}" + (f" / {self.resolution}" if self.resolution else "")
        when = f"created {self.created[:10]}" + (f", resolved {self.resolved[:10]}" if self.resolved else "")
        desc = re.sub(r"\s+", " ", self.description).strip()[:300]
        last = re.sub(r"\s+", " ", self.comments[-1]).strip()[:300] if self.comments else ""
        lines = [f"{self.key} [{self.issue_type}, {state}, {when}] {self.summary}", f"  link: {self.url}"]
        if desc:
            lines.append(f"  description: {desc}")
        if last:
            lines.append(f"  latest comment: {last}")
        return "\n".join(lines)

    def full(self) -> str:
        parts = [self.brief().split("\n")[0], f"link: {self.url}", f"project: {self.project}; priority: {self.priority}; "
                 f"assignee: {self.assignee or 'unassigned'}; labels: {', '.join(self.labels) or 'none'}",
                 "DESCRIPTION:\n" + (self.description.strip()[:3000] or "(empty)")]
        if self.comments:
            parts.append("COMMENTS (oldest to newest):\n" + "\n---\n".join(c.strip()[:1200] for c in self.comments[-8:]))
        return "\n".join(parts)


class JiraClient:
    def __init__(self, base_url: str, kind: str = CLOUD, email: str = "", token: str = "",
                 projects: list[str] | None = None, extra_jql: str = "", timeout: int = 30):
        self.base = base_url.strip().rstrip("/")
        if not self.base.startswith("http"):
            self.base = "https://" + self.base
        self.kind = kind
        self.api = "/rest/api/3" if kind == CLOUD else "/rest/api/2"
        self.projects = [p.strip().upper() for p in (projects or []) if p.strip()]
        self.extra_jql = extra_jql.strip()
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"Accept": "application/json"})
        if kind == CLOUD:
            self.s.auth = (email.strip(), token.strip())
        else:
            self.s.headers["Authorization"] = f"Bearer {token.strip()}"
        self.user = ""

    # ------------------------------------------------------------------ http
    def _get(self, path: str, params: dict | None = None) -> dict:
        r = self.s.get(self.base + path, params=params, timeout=self.timeout)
        if r.status_code == 401:
            raise PermissionError("Jira rejected the credentials (401). Check the email and API token.")
        if r.status_code == 403:
            raise PermissionError("Jira refused access (403). The account may lack 'Browse projects' permission.")
        if r.status_code >= 400:
            raise RuntimeError(f"Jira error {r.status_code}: {r.text[:300]}")
        return r.json()

    def test(self) -> str:
        me = self._get(f"{self.api}/myself")
        self.user = me.get("displayName") or me.get("name") or me.get("emailAddress") or "connected"
        if self.projects:
            for p in self.projects:
                self._get(f"{self.api}/project/{p}")
        return self.user

    # ------------------------------------------------------------------ search
    def _scope(self) -> str:
        parts = []
        if self.projects:
            parts.append("project in (" + ", ".join(self.projects) + ")")
        if self.extra_jql:
            parts.append(f"({self.extra_jql})")
        return " AND ".join(parts)

    def jql_search(self, jql: str, max_results: int = 50) -> list[Ticket]:
        params = {"jql": jql, "maxResults": max_results, "fields": ",".join(FIELDS)}
        path = f"{self.api}/search/jql" if self.kind == CLOUD else f"{self.api}/search"
        data = self._get(path, params)
        return [self._ticket(i) for i in data.get("issues", [])]

    def search(self, question: str, k: int = 6, embedder=None, candidates: int = 50) -> list[Ticket]:
        kws = keywords(question)
        if not kws:
            return []
        text_clause = "(" + " OR ".join(f'text ~ "{w}"' for w in kws) + ")"
        scope = self._scope()
        jql = f"{text_clause}{' AND ' + scope if scope else ''} ORDER BY updated DESC"
        found = self.jql_search(jql, candidates)
        if not found:
            return []
        if embedder is not None and len(found) > 1:
            docs = [f"{t.summary}\n{t.description[:800]}\n{' '.join(t.comments[-2:])[:400]}" for t in found]
            mat = np.array(embedder.embed_documents(docs))
            q = np.array(embedder.embed_query(question))
            sims = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
            for t, s in zip(found, sims):
                t.score = float(s)
        else:  # keyword overlap fallback
            for t in found:
                blob = f"{t.summary} {t.description}".lower()
                t.score = sum(w in blob for w in kws) / len(kws)
        ranked = sorted(found, key=lambda t: t.score, reverse=True)
        if embedder is not None:  # drop weak matches relative to the best one
            cutoff = max(0.5, ranked[0].score * 0.75)
            ranked = [t for t in ranked if t.score >= cutoff]
        return ranked[:k]

    def get(self, key: str) -> Ticket:
        key = key.strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]+-\d+", key):
            raise ValueError(f"'{key}' is not a Jira issue key (like OPS-123).")
        data = self._get(f"{self.api}/issue/{key}", {"fields": ",".join(FIELDS)})
        return self._ticket(data)

    # ------------------------------------------------------------------ parsing
    def _text(self, value) -> str:
        return adf_to_text(value) if isinstance(value, (dict, list)) else (value or "")

    def _ticket(self, issue: dict) -> Ticket:
        f = issue.get("fields", {}) or {}
        name = lambda v, attr="name": (v or {}).get(attr, "") if isinstance(v, dict) else ""
        comments = [self._text(c.get("body")) for c in ((f.get("comment") or {}).get("comments") or [])]
        return Ticket(
            key=issue.get("key", ""), url=f"{self.base}/browse/{issue.get('key', '')}",
            summary=f.get("summary") or "", status=name(f.get("status")), resolution=name(f.get("resolution")),
            issue_type=name(f.get("issuetype")), priority=name(f.get("priority")), project=name(f.get("project"), "key"),
            created=f.get("created") or "", updated=f.get("updated") or "", resolved=f.get("resolutiondate") or "",
            assignee=name(f.get("assignee"), "displayName"), labels=list(f.get("labels") or []),
            description=self._text(f.get("description")), comments=comments,
        )


def format_tickets(tickets: list[Ticket]) -> str:
    if not tickets:
        return "(no related Jira tickets found)"
    return "\n\n".join(t.brief() for t in tickets)
