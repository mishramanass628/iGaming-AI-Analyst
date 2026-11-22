"""Usage limits for the app owner's preconfigured LLM key ("My API").

Every LLM call made with that key passes through UsageGuard (a LangChain callback):
- before the call: refuse it if the session or the whole app is over its limit
- after the call:  add the request and its input/output tokens to both counters

Daily counters are shared by every user of this app process and saved to disk, so a restart
doesn't reset them. They reset at midnight UTC.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

# USD per million tokens (input, output) for an estimate shown to the owner. Unknown models show no estimate.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


class UsageLimitError(RuntimeError):
    pass


@dataclass
class Limits:
    session_requests: int = 30
    session_tokens: int = 300_000
    daily_requests: int = 300
    daily_tokens: int = 3_000_000
    daily_usd: float = 0.0          # 0 = no dollar cap (only works for models in PRICES)
    max_output_tokens: int = 12000

    @classmethod
    def from_settings(cls, get) -> "Limits":
        base = cls()
        def num(name, default, typ):
            v = get(name)
            try:
                return typ(v) if v not in (None, "") else default
            except (TypeError, ValueError):
                return default
        return cls(
            session_requests=num("MY_API_MAX_REQUESTS_PER_SESSION", base.session_requests, int),
            session_tokens=num("MY_API_MAX_TOKENS_PER_SESSION", base.session_tokens, int),
            daily_requests=num("MY_API_MAX_REQUESTS_PER_DAY", base.daily_requests, int),
            daily_tokens=num("MY_API_MAX_TOKENS_PER_DAY", base.daily_tokens, int),
            daily_usd=num("MY_API_MAX_USD_PER_DAY", base.daily_usd, float),
            max_output_tokens=num("MY_API_MAX_OUTPUT_TOKENS", base.max_output_tokens, int),
        )


@dataclass
class Counter:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost(self, model: str) -> float | None:
        if model not in PRICES:
            return None
        pin, pout = PRICES[model]
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class DailyLedger:
    """App-wide daily counter, thread-safe, persisted as JSON."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.day = _today()
        self.counter = Counter()
        try:
            data = json.loads(self.path.read_text())
            if data.get("day") == self.day:
                self.counter = Counter(**data["counter"])
        except Exception:
            pass

    def _roll(self):
        if _today() != self.day:
            self.day, self.counter = _today(), Counter()

    def snapshot(self) -> Counter:
        with self.lock:
            self._roll()
            return Counter(**vars(self.counter))

    def add(self, requests=0, input_tokens=0, output_tokens=0):
        with self.lock:
            self._roll()
            self.counter.requests += requests
            self.counter.input_tokens += input_tokens
            self.counter.output_tokens += output_tokens
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"day": self.day, "counter": vars(self.counter)}))


@dataclass
class UsageGuard(BaseCallbackHandler):
    limits: Limits
    ledger: DailyLedger
    model: str
    session: Counter = field(default_factory=Counter)
    raise_error: bool = True  # make LangChain propagate UsageLimitError instead of logging it
    _est_input: dict = field(default_factory=dict, repr=False)

    def check(self):
        lim, day, s = self.limits, self.ledger.snapshot(), self.session
        if s.requests >= lim.session_requests:
            raise UsageLimitError(f"This session has used its {lim.session_requests} AI requests. "
                                  "Start a new session, or use your own API key in step 2.")
        if s.tokens >= lim.session_tokens:
            raise UsageLimitError(f"This session has used its {lim.session_tokens:,} token allowance. "
                                  "Start a new session, or use your own API key in step 2.")
        if day.requests >= lim.daily_requests or day.tokens >= lim.daily_tokens:
            raise UsageLimitError("The app's daily AI allowance is used up. It resets at midnight UTC; "
                                  "or use your own API key in step 2.")
        cost = day.cost(self.model)
        if lim.daily_usd and cost is not None and cost >= lim.daily_usd:
            raise UsageLimitError(f"The app's daily budget (${lim.daily_usd:,.2f}) is used up. It resets at "
                                  "midnight UTC; or use your own API key in step 2.")

    # LangChain callback hooks -------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self.check()
        chars = sum(len(str(m.content)) for batch in messages for m in batch)
        self._est_input[run_id] = chars // 4  # rough fallback if the provider reports no usage

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self.check()
        self._est_input[run_id] = sum(len(p) for p in prompts) // 4

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        tin = tout = 0
        est_out = 0
        for gens in response.generations:
            for g in gens:
                meta = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                tin += meta.get("input_tokens", 0)
                tout += meta.get("output_tokens", 0)
                est_out += len(getattr(g, "text", "") or "") // 4
        est_in = self._est_input.pop(run_id, 0)
        if not tin and not tout:  # provider gave no usage: count an estimate rather than nothing
            tin, tout = est_in, est_out
        self.session.requests += 1
        self.session.input_tokens += tin
        self.session.output_tokens += tout
        self.ledger.add(1, tin, tout)
