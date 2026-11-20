"""Cheap checks that run BEFORE and AFTER the LLM. Pure functions, no state."""
import re
from functools import lru_cache

import numpy as np

from .config import get_config, get_settings
from .extract import norm
from .llm import get_embeddings


class InputRejected(ValueError):
    """Empty or too-long input: the caller should answer HTTP 422."""


# ---------- 1. input validation + injection patterns ----------
def check_input(question: str) -> tuple[str, bool]:
    """Returns (clean_question, blocked). Raises InputRejected for empty/too long."""
    q = re.sub(r"\s+", " ", question or "").strip()
    limit = get_settings().max_question_chars
    if not q:
        raise InputRejected("Question is empty")
    if len(q) > limit:
        raise InputRejected(f"Question is longer than {limit} characters")
    n = norm(q)
    patterns = [norm(p) for p in get_config()["guardrails"]["blocked_patterns"]]
    return q, any(p in n for p in patterns)


# ---------- 2a. small talk ----------
@lru_cache
def _smalltalk_vocab() -> set[str]:
    words: set[str] = set()
    for phrase in get_config()["guardrails"]["smalltalk"]:
        words |= set(norm(phrase).split())
    return words


def is_smalltalk(question: str) -> bool:
    """True only if EVERY word is small-talk vocabulary ("hi priya nair" is not small talk)."""
    toks = norm(question).split()
    return bool(toks) and len(toks) <= 5 and all(t in _smalltalk_vocab() for t in toks)


# ---------- 2b. off-topic (embedding similarity to example set) ----------
@lru_cache
def _off_topic_vectors() -> np.ndarray:
    examples = get_config()["guardrails"]["off_topic_examples"]
    return np.array(get_embeddings().embed_documents(examples))


def off_topic_score(question: str) -> float:
    v = np.array(get_embeddings().embed_query(question))
    return float(np.max(_off_topic_vectors() @ v))  # embeddings are normalised: dot = cosine


def is_off_topic(question: str) -> bool:
    """Caller must still veto this when the question mentions a known graph entity."""
    s = get_settings()
    if not s.off_topic_enabled:
        return False
    try:
        return off_topic_score(question) >= s.off_topic_threshold
    except Exception:
        return False  # a guardrail must never break the app


# ---------- 7. output grounding ----------
_STOP = {
    "the", "a", "an", "in", "on", "at", "of", "and", "or", "but", "it", "its", "he", "she",
    "they", "this", "that", "there", "these", "those", "is", "was", "are", "were", "as", "by",
    "for", "with", "from", "to", "based", "according", "however", "also", "no", "yes", "not",
    "i", "we", "you", "if", "while", "after", "before", "since", "both", "one", "two", "all",
    "some", "most", "other", "another", "each", "his", "her", "their", "which", "who", "what",
}
_CAP = re.compile(r"\b[A-Z][\w&-]*(?:\s+[A-Z][\w&-]*)*")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def check_grounding(answer: str, context: str) -> dict:
    """Every name and number in the answer must appear in the retrieved context.
    Flags only (never blocks). Skips the canned 'not found' reply and context-free answers.
    Known blind spot: a hallucinated single capitalised word at the start of a sentence."""
    not_found = get_config()["guardrails"]["canned"]["not_found"]
    if not answer.strip() or not context.strip() or answer.strip() == not_found:
        return {"ok": True, "unsupported": [], "skipped": True}

    text = re.sub(r"\[[^\]]*\]", " ", answer)  # drop [file.txt] citations
    ctx_tokens = set(norm(context).split())
    ctx_nums = {n.replace(",", "") for n in _NUM.findall(context)}
    bad: list[str] = []

    for m in _CAP.finditer(text):
        words = m.group().split()
        stripped = False
        while words and words[0].casefold() in _STOP:
            words.pop(0)
            stripped = True
        if not words:
            continue
        prefix = text[: m.start()].rstrip()
        sentence_start = (not prefix) or prefix[-1] in ".!?:\n-*•"
        if len(words) == 1 and sentence_start and not stripped:
            continue  # "Overall, ...": an ordinary word, not a name
        name = " ".join(words)
        toks = norm(name).split()
        if toks and not all(t in ctx_tokens for t in toks):
            bad.append(name)

    for n in _NUM.findall(text):
        v = n.replace(",", "")
        if v.isdigit() and int(v) <= 10:
            continue  # counts and list numbering
        if v not in ctx_nums:
            bad.append(n)

    bad = list(dict.fromkeys(bad))[:8]
    return {"ok": not bad, "unsupported": bad, "skipped": False}
