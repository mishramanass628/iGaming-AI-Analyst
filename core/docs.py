"""Business documents: load from Google Drive (or uploads), parse, chunk, embed, search (RAG)."""
from __future__ import annotations

import io
import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".html", ".htm", ".xlsx", ".xls"}


# --------------------------------------------------------------------------- embeddings
class FastEmbedder(Embeddings):
    """Local ONNX embeddings via fastembed (no API key, ~70 MB model downloaded once)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        if hasattr(self.model, "query_embed"):
            return next(iter(self.model.query_embed(text))).tolist()
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> FastEmbedder:
    return FastEmbedder()


# --------------------------------------------------------------------------- parsing
def parse_bytes(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if ext == ".docx":
        import docx

        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs if p.text.strip()]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        return "\n".join(parts)
    if ext in (".xlsx", ".xls"):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        return "\n\n".join(f"Sheet: {n}\n{df.head(500).to_csv(index=False)}" for n, df in sheets.items())
    text = data.decode("utf-8", errors="ignore")
    if ext in (".html", ".htm"):
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
    return text


def parse_dir(folder: Path) -> list[tuple[str, str]]:
    docs = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            text = parse_bytes(p.name, p.read_bytes())
            if text.strip():
                docs.append((p.name, text))
    return docs


# --------------------------------------------------------------------------- Google Drive
def parse_drive_link(link: str) -> tuple[str, str]:
    link = link.strip()
    if m := re.search(r"/folders/([\w-]+)", link):
        return "folder", m.group(1)
    if m := re.search(r"/(?:file|document|spreadsheets|presentation)/d/([\w-]+)", link):
        return "file", m.group(1)
    if m := re.search(r"[?&]id=([\w-]+)", link):
        return "unknown", m.group(1)
    if re.fullmatch(r"[\w-]{20,}", link):
        return "unknown", link
    raise ValueError("That doesn't look like a Google Drive folder or file link.")


_EXPORTS = {
    "application/vnd.google-apps.document": ("text/plain", ".txt"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
    "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
}
_FOLDER = "application/vnd.google-apps.folder"


def _load_with_service_account(kind: str, fid: str, sa_info: dict, max_files: int = 300) -> list[tuple[str, str]]:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    creds = service_account.Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    svc = build("drive", "v3", credentials=creds, cache_discovery=False)

    def download(f) -> tuple[str, str] | None:
        mime, name = f["mimeType"], f["name"]
        if mime in _EXPORTS:
            export_mime, ext = _EXPORTS[mime]
            req = svc.files().export_media(fileId=f["id"], mimeType=export_mime)
            name = name + ext
        elif Path(name).suffix.lower() in SUPPORTED:
            req = svc.files().get_media(fileId=f["id"], supportsAllDrives=True)
        else:
            return None
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        text = parse_bytes(name, buf.getvalue())
        return (name, text) if text.strip() else None

    root = svc.files().get(fileId=fid, fields="id,name,mimeType", supportsAllDrives=True).execute()
    if root["mimeType"] != _FOLDER:
        got = download(root)
        return [got] if got else []

    out: list[tuple[str, str]] = []
    queue = [root["id"]]
    while queue and len(out) < max_files:
        folder = queue.pop(0)
        token = None
        while True:
            resp = svc.files().list(
                q=f"'{folder}' in parents and trashed = false",
                fields="nextPageToken, files(id,name,mimeType)",
                pageSize=200, pageToken=token,
                supportsAllDrives=True, includeItemsFromAllDrives=True,
            ).execute()
            for f in resp.get("files", []):
                if f["mimeType"] == _FOLDER:
                    queue.append(f["id"])
                elif len(out) < max_files and (got := download(f)):
                    out.append(got)
            token = resp.get("nextPageToken")
            if not token:
                break
    return out


def _load_public(kind: str, fid: str, link: str) -> list[tuple[str, str]]:
    import gdown

    with tempfile.TemporaryDirectory() as tmp:
        if kind in ("folder", "unknown"):
            try:
                res = gdown.download_folder(id=fid, output=tmp, quiet=True, use_cookies=False)
            except Exception as e:
                res = None
                if kind == "folder":
                    raise RuntimeError(
                        "Couldn't read the folder. Share it as 'Anyone with the link → Viewer', "
                        f"or provide a service-account key. ({e})"
                    ) from e
            if res:
                return parse_dir(Path(tmp))
            if kind == "folder":
                raise RuntimeError("The folder is empty or not shared publicly ('Anyone with the link').")
        if "docs.google.com" in link:  # native Google Doc/Sheet/Slides: gdown exports it
            out = gdown.download(url=link, output=tmp + "/", quiet=True, use_cookies=False)
        else:
            out = gdown.download(id=fid, output=tmp + "/", quiet=True, use_cookies=False)
        if not out:
            raise RuntimeError("Couldn't download the file. Make sure it's shared as 'Anyone with the link'.")
        return parse_dir(Path(tmp))


def load_from_drive(link: str, sa_info: dict | None = None) -> list[tuple[str, str]]:
    kind, fid = parse_drive_link(link)
    if sa_info:
        return _load_with_service_account(kind, fid, sa_info)
    return _load_public(kind, fid, link)


# --------------------------------------------------------------------------- index
@dataclass
class DocStore:
    store: InMemoryVectorStore
    names: list[str]
    n_chunks: int

    @classmethod
    def build(cls, docs: list[tuple[str, str]], embedder: Embeddings) -> "DocStore":
        if not docs:
            raise ValueError("No readable documents found (supported: PDF, DOCX, TXT, MD, CSV, XLSX, Google Docs/Sheets).")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        chunks: list[Document] = []
        for name, text in docs:
            for i, piece in enumerate(splitter.split_text(text)):
                chunks.append(Document(page_content=piece, metadata={"source": name, "chunk": i}))
        store = InMemoryVectorStore(embedder)
        store.add_documents(chunks)
        return cls(store, [n for n, _ in docs], len(chunks))

    def search(self, query: str, k: int = 5) -> list[Document]:
        return self.store.similarity_search(query, k=min(k, self.n_chunks))


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(no business documents loaded)"
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)
