"""Speech-to-text with a local Whisper model (faster-whisper). Audio never leaves the machine."""
from __future__ import annotations

import io
from functools import lru_cache

DOMAIN_PROMPT = (
    "iGaming analytics question. Terms: GGR, NGR, FTD, ARPU, LTV, CAC, hold, RTP, VIP tier, "
    "deposits, withdrawals, bonus cost, wagering, churn, slots, live casino, sportsbook, affiliate."
)


@lru_cache(maxsize=2)
def _model(size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(audio: bytes, size: str = "base") -> str:
    segments, _info = _model(size).transcribe(
        io.BytesIO(audio), beam_size=5, vad_filter=True, initial_prompt=DOMAIN_PROMPT
    )
    return " ".join(s.text.strip() for s in segments).strip()
