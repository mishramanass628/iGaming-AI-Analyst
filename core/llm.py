"""LLM factory (LangChain chat models) + helpers to pull plain text out of responses."""
from __future__ import annotations

ANTHROPIC = "Anthropic (Claude)"
OPENAI = "OpenAI"
OPENAI_COMPAT = "OpenAI-compatible endpoint"
GEMINI = "Google Gemini"
PROVIDERS = [ANTHROPIC, OPENAI, GEMINI, OPENAI_COMPAT]
DEFAULT_MODELS = {ANTHROPIC: "claude-opus-5-5", OPENAI: "", GEMINI: "", OPENAI_COMPAT: ""}


MY_API = "My API (preconfigured key)"
_PROVIDER_ALIASES = {"anthropic": ANTHROPIC, "claude": ANTHROPIC, "openai": OPENAI,
                     "openai_compatible": OPENAI_COMPAT, "openai-compatible": OPENAI_COMPAT,
                     "gemini": GEMINI, "google": GEMINI}


def preconfigured(get) -> dict | None:
    """Read the app owner's LLM settings. `get(name)` looks in Streamlit secrets, then environment variables.

    LLM_PROVIDER  anthropic | openai | gemini | openai_compatible   (default: anthropic)
    LLM_MODEL     model id (default: claude-opus-5-5 for anthropic)
    LLM_API_KEY   key (ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY also accepted)
    LLM_BASE_URL  only for openai_compatible
    LLM_BACKUP_MODEL  optional model (same provider) used when the main one is overloaded
    """
    provider = _PROVIDER_ALIASES.get(str(get("LLM_PROVIDER") or "anthropic").strip().lower(), ANTHROPIC)
    fallback_keys = {ANTHROPIC: ["ANTHROPIC_API_KEY"], OPENAI: ["OPENAI_API_KEY"],
                     GEMINI: ["GEMINI_API_KEY", "GOOGLE_API_KEY"]}.get(provider, [])
    key = get("LLM_API_KEY") or next((get(k) for k in fallback_keys if get(k)), None)
    model = get("LLM_MODEL") or DEFAULT_MODELS.get(provider, "")
    base_url = get("LLM_BASE_URL")
    if not model or (not key and provider != OPENAI_COMPAT) or (provider == OPENAI_COMPAT and not base_url):
        return None
    return {"provider": provider, "model": model, "api_key": key or "", "base_url": base_url,
            "backup_model": get("LLM_BACKUP_MODEL") or None}


_SKIP_GEMINI = ("embedding", "aqa", "imagen", "image", "tts", "audio", "veo", "learnlm", "robotics", "computer-use")


def list_gemini_models(api_key: str) -> list[str]:
    """Text models this Gemini key can call, stable releases first, newest first."""
    from google import genai

    client = genai.Client(api_key=api_key)
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        name = (m.name or "").removeprefix("models/")
        if "generateContent" in actions and name and not any(s in name for s in _SKIP_GEMINI):
            names.append(name)
    unstable = lambda n: any(t in n for t in ("preview", "exp", "latest"))
    return sorted(sorted(set(names), reverse=True), key=unstable)


def make_llm(provider: str, api_key: str, model: str, base_url: str | None = None, max_tokens: int = 16000,
             callbacks: list | None = None):
    if not model:
        raise ValueError("Enter a model name.")
    if provider == ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        # No temperature: current Claude models reject sampling parameters.
        return ChatAnthropic(model=model, api_key=api_key, max_tokens=max_tokens, timeout=600, max_retries=2,
                             callbacks=callbacks)
    if provider == GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, max_output_tokens=max_tokens,
                                      timeout=600, max_retries=2, callbacks=callbacks)
    from langchain_openai import ChatOpenAI

    kwargs = dict(model=model, api_key=api_key or "not-needed", timeout=600, max_retries=2,
                  max_tokens=max_tokens, callbacks=callbacks)
    if provider == OPENAI_COMPAT:
        if not base_url:
            raise ValueError("Enter the endpoint's base URL.")
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


# --------------------------------------------------------------------------- resilience
_TRANSIENT_MARKERS = (
    "503", "unavailable", "overloaded", "529", "high demand", "502", "504", "500 internal", "internal error",
    "deadline", "timed out", "timeout", "connection", "429", "rate limit", "rate_limit", "resource_exhausted",
)


def is_transient(err: Exception) -> bool:
    """Provider-side hiccups worth retrying: overload, rate limit, 5xx, timeouts, dropped connections."""
    text = f"{type(err).__name__} {err}".lower()
    return any(m in text for m in _TRANSIENT_MARKERS)


class TransientLLMError(RuntimeError):
    pass


class ResilientLLM:
    """Wraps a chat model: retries transient failures with backoff, then switches to a backup model.

    Exposes .invoke like a LangChain chat model, so the LangGraph nodes (and token streaming via callbacks)
    work unchanged. `last_used` says which model produced the last answer.
    """

    def __init__(self, primary, primary_name: str, backup=None, backup_name: str = "",
                 waits: tuple[float, ...] = (2, 5, 10), backup_waits: tuple[float, ...] = (3,), sleep=None):
        import time

        self.primary, self.primary_name = primary, primary_name
        self.backup, self.backup_name = backup, backup_name
        self.waits, self.backup_waits = waits, backup_waits
        self.sleep = sleep or time.sleep
        self.last_used = primary_name

    def bind_tools(self, tools, **kwargs) -> "ResilientLLM":
        """Same retry/backup behaviour, with tools bound to both models (tool_choice stays 'auto')."""
        bound = ResilientLLM(self.primary.bind_tools(tools, **kwargs), self.primary_name,
                             self.backup.bind_tools(tools, **kwargs) if self.backup is not None else None,
                             self.backup_name, self.waits, self.backup_waits, self.sleep)
        bound._parent = self
        return bound

    def _record(self, name: str):
        self.last_used = name
        if getattr(self, "_parent", None) is not None:
            self._parent.last_used = name

    def _plan(self):
        for i in range(len(self.waits) + 1):
            yield self.primary, self.primary_name, (self.waits[i] if i < len(self.waits) else None)
        if self.backup is not None:
            for i in range(len(self.backup_waits) + 1):
                yield self.backup, self.backup_name, (self.backup_waits[i] if i < len(self.backup_waits) else None)

    def invoke(self, input, config=None, **kwargs):
        last = None
        tried = []
        for model, name, wait in self._plan():
            try:
                out = model.invoke(input, config, **kwargs)
                self._record(name)
                return out
            except Exception as e:
                if not is_transient(e):
                    raise
                last = e
                tried.append(name)
                if wait:
                    self.sleep(wait)
        models = ", ".join(dict.fromkeys(tried))
        hint = "" if self.backup is not None else " Tip: add a backup model in step 2 so the app can switch automatically."
        raise TransientLLMError(
            f"The AI provider is overloaded or unavailable right now (tried {len(tried)} times: {models}). "
            f"This is usually temporary; try again in a minute.{hint}\n\nLast error: {str(last)[:200]}"
        ) from last


def content_text(content) -> str:
    """Text from a message/chunk content (string or list of blocks; thinking blocks are skipped)."""
    if isinstance(content, str):
        return content
    out = []
    for block in content or []:
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return "".join(out)


def test_llm(llm) -> str:
    msg = llm.invoke("Connection test. Reply with exactly: OK")
    text = content_text(msg.content).strip()
    if not text:
        raise RuntimeError("The model returned an empty reply.")
    return text
