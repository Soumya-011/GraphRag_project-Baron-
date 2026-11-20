import asyncio
import logging
import re
import threading
import time
from collections import deque
from functools import lru_cache

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import Runnable

from .config import ROLES, get_settings

log = logging.getLogger("graphrag.llm")
_events = threading.local()  # fallback events of the request running in THIS thread


def reset_fallback_events() -> None:
    """Start of a request: clear this thread's fallback events and rate-limiter wait."""
    _events.items = []
    _events.wait_s = 0.0


def limiter_wait_ms() -> int:
    """Time this request spent waiting for the free-tier rate limiter (not model time)."""
    return int(getattr(_events, "wait_s", 0.0) * 1000)


def _waited(seconds: float) -> None:
    _events.wait_s = getattr(_events, "wait_s", 0.0) + seconds


def fallback_events() -> list[str]:
    return list(getattr(_events, "items", []))


def _note_fallback(role: str, model: str, why: str) -> None:
    if not hasattr(_events, "items"):
        _events.items = []
    msg = f"{role}: primary model failed ({why}); used {model}"
    _events.items.append(msg)
    log.warning(msg)
    try:  # counted in /metrics (lazy import: cache imports this module)
        from .cache import record

        record(f"fallback_{role}")
    except Exception:
        pass


# ---------- error hints ----------
def describe_llm_error(e) -> str:
    """Turn Gemini errors into one actionable line."""
    msg = str(e)
    low = msg.lower()
    if re.search(r"\b(401|403)\b", msg) or "api key not valid" in low or "permission denied" in low:
        return "HTTP 401/403: Gemini API key rejected. Check GEMINI_API_KEY (or the task's *_API_KEY / FALLBACK_API_KEY) in .env."
    if re.search(r"\b429\b", msg) or "resource_exhausted" in low or "quota" in low or "rate limit" in low:
        return "HTTP 429: Gemini free-tier rate limit or daily quota reached. Wait and retry, or lower GEMINI_RPM."
    if re.search(r"\b(500|503)\b", msg) or "overloaded" in low or "unavailable" in low:
        return "Gemini is temporarily overloaded (HTTP 503). Retry in a moment."
    if re.search(r"\b404\b", msg) or "not found" in low or "not supported" in low:
        return "Model not found. Check the *_MODEL names in .env."
    if "timed out" in low or "timeout" in low or "connect" in low:
        return "Cannot reach the Gemini API (network problem or timeout)."
    return msg[:300]


# ---------- client-side rate limit (free tier: requests per minute, per model) ----------
class _RpmLimiter(BaseRateLimiter):
    """Sliding 60 s window: allows bursts up to `rpm`, then waits. No wait for the first calls."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.times: deque[float] = deque()
        self.lock = threading.Lock()

    def _wait_needed(self) -> float:
        with self.lock:
            now = time.monotonic()
            while self.times and now - self.times[0] >= 60:
                self.times.popleft()
            if len(self.times) < self.rpm:
                self.times.append(now)
                return 0.0
            return 60 - (now - self.times[0])

    def acquire(self, *, blocking: bool = True) -> bool:
        while (wait := self._wait_needed()) > 0:
            if not blocking:
                return False
            time.sleep(wait)
            _waited(wait)
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        while (wait := self._wait_needed()) > 0:
            if not blocking:
                return False
            await asyncio.sleep(wait)
            _waited(wait)
        return True


@lru_cache
def _limiter(model: str, api_key: str = ""):
    """One limiter per (account key, model): free-tier quota is counted per project and per model."""
    rpm = get_settings().gemini_rpm
    return _RpmLimiter(rpm) if rpm > 0 else None


# ---------- models ----------
@lru_cache
def build_model(model: str, max_tokens: int, api_key: str = "") -> BaseChatModel:
    """One Gemini chat model. Cached per (model, max_tokens, api_key)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key or s.gemini_api_key,
        temperature=s.llm_temperature,
        # Gemini 2.5 "thinking" tokens count toward this cap, so leave generous room
        max_output_tokens=max(max_tokens * s.llm_token_multiplier, s.llm_min_output_tokens),
        timeout=s.llm_timeout_seconds,
        max_retries=s.llm_max_retries,
        rate_limiter=_limiter(model, api_key),
    )


def _usable(out) -> bool:
    """A chat reply with empty text counts as a failure; structured objects are always usable."""
    if out is None:
        return False
    content = getattr(out, "content", out)
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):  # Gemini can return a list of content parts
        return any((p.get("text") if isinstance(p, dict) else str(p)) for p in content)
    return True


class _Fallback(Runnable):
    """primary -> on error, empty output or an implausible result (accept() is False) -> fallback."""

    def __init__(self, primary, fallback, role: str, model: str, accept=None):
        self.primary, self.fallback = primary, fallback
        self.role, self.model, self.accept = role, model, accept

    def invoke(self, input, config=None, **kwargs):
        first, err = None, None
        try:
            first = self.primary.invoke(input, config, **kwargs)
            if _usable(first) and (self.accept is None or self.accept(first, input)):
                return first
            why = "empty or implausible output"
        except Exception as e:
            err, why = e, describe_llm_error(e)
        _note_fallback(self.role, self.model, why)
        try:
            return self.fallback.invoke(input, config, **kwargs)
        except Exception as fe:
            if first is not None and _usable(first):
                return first  # keep the primary's valid-but-implausible answer
            raise RuntimeError(f"primary: {why} | {self.model}: {describe_llm_error(fe)}") from (err or fe)


def with_fallback(primary, fallback, role: str = "", model: str = "", accept=None):
    return primary if fallback is None else _Fallback(primary, fallback, role, model, accept)


@lru_cache
def get_llm(role: str = "answer", max_tokens: int | None = None):
    """Chat model for one task ('answer', 'extraction', 'router'), with the model fallback."""
    s = get_settings()
    limit = max_tokens or s.llm_max_tokens
    primary = build_model(s.model_for(role), limit, s.api_key_for(role))
    fb = s.fallback_for(role)
    return with_fallback(primary, build_model(fb, limit, s.fallback_key_for(role)) if fb else None, role, fb or "")


# ---------- structured output ----------
def structured_pair(schema, role: str = "extraction", max_tokens: int | None = None):
    """(primary, fallback_or_None) runnables returning a validated Pydantic `schema` object."""
    s = get_settings()
    if max_tokens is None:
        max_tokens = s.extraction_max_tokens if role == "extraction" else s.llm_max_tokens
    primary = build_model(s.model_for(role), max_tokens, s.api_key_for(role)).with_structured_output(schema)
    fb = s.fallback_for(role)
    fallback = (build_model(fb, max_tokens, s.fallback_key_for(role)).with_structured_output(schema)
                if fb else None)
    return primary, fallback


def get_structured_llm(schema, role: str = "router", max_tokens: int | None = None):
    primary, fallback = structured_pair(schema, role, max_tokens)
    fb = get_settings().fallback_for(role)
    return with_fallback(primary, fallback, role, fb or "")


_emb_lock = threading.Lock()


def get_embeddings() -> Embeddings:
    """The embedding model, loaded once. The lock matters: the warm-up thread and the first question can ask at
    the same moment, and lru_cache alone would load the model twice."""
    with _emb_lock:
        return _load_embeddings()


@lru_cache
def _load_embeddings() -> Embeddings:
    """Local CPU embeddings (22M-parameter MiniLM, about 100 MB of RAM). No API quota used."""
    from langchain_huggingface import HuggingFaceEmbeddings

    s = get_settings()
    return HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        model_kwargs={"device": s.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


get_embeddings.cache_clear = _load_embeddings.cache_clear  # tests


# ---------- health ----------
def _mask(key: str) -> str:
    return ("..." + key[-4:]) if len(key) > 8 else "set"


def _check_key(key: str) -> tuple[bool, str | None]:
    """Validates one key by listing models (free, no generation quota)."""
    try:
        r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models",
                      params={"pageSize": 1}, headers={"x-goog-api-key": key}, timeout=8)
        if r.status_code != 200:
            return False, f"Gemini rejected the key ({r.status_code})."
    except httpx.HTTPError as e:
        return False, f"Gemini API not reachable: {type(e).__name__}"
    return True, None


def llm_health() -> dict:
    """One check per distinct key. Keys are shown masked (last 4 characters) and never in full."""
    s = get_settings()
    seen: dict[str, tuple[bool, str | None]] = {}

    def status(key: str):
        if key not in seen:
            seen[key] = _check_key(key)
        return seen[key]

    roles = {}
    for role in ROLES:
        key = s.api_key_for(role)
        ok, error = status(key)
        roles[role] = {"model": s.model_for(role), "ok": ok, "key": _mask(key), **({"error": error} if error else {})}
    fbs: dict[str, dict] = {}
    for role in ROLES:
        fb = s.fallback_for(role)
        if fb:
            key = s.fallback_key_for(role)
            ok, error = status(key)
            fbs[f"{fb}|{key}"] = {"model": fb, "ok": ok, "key": _mask(key), **({"error": error} if error else {})}
    all_ok = all(r["ok"] for r in roles.values()) and all(f["ok"] for f in fbs.values())
    return {"ok": all_ok, "roles": roles,
            "fallback": {"enabled": bool(fbs), "model": ", ".join(sorted({f["model"] for f in fbs.values()})),
                         "details": list(fbs.values())}}
