"""Answer cache (exact + semantic) and the 'LLM calls saved' counters. SQLite, no server."""
import hashlib
import json
import re
import sqlite3
import time
from contextlib import closing

import numpy as np

from .config import get_settings
from .extract import norm
from .llm import get_embeddings

# Question words ignored when comparing the "named things" of two questions.
_QWORDS = {
    "what", "who", "whom", "which", "where", "when", "why", "how", "is", "are", "was", "were",
    "did", "does", "do", "can", "could", "tell", "list", "show", "give", "name", "explain",
    "describe", "please", "i", "we", "the", "a", "an",
}

SAVED = ("saved_blocked", "saved_smalltalk", "saved_off_topic", "saved_cache",
         "saved_rule", "saved_gate")


def _con():
    p = get_settings().cache_path
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=10)
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, route TEXT, sig TEXT,
            epoch INTEGER, ents TEXT, caps TEXT, emb TEXT, result TEXT, ts REAL);
        CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v INTEGER);
        CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, n INTEGER);
        """
    )
    return con


def _epoch(con) -> int:
    row = con.execute("SELECT v FROM meta WHERE k='epoch'").fetchone()
    return row[0] if row else 0


def bump_epoch() -> None:
    """Call after ingest/seed: documents changed, so every cached answer is stale."""
    try:
        with closing(_con()) as con:
            con.execute("INSERT INTO meta VALUES ('epoch', 1) "
                        "ON CONFLICT(k) DO UPDATE SET v = v + 1")
            con.execute("DELETE FROM qa")
            con.commit()
    except Exception:
        pass


def _sig(route: str, k) -> str:
    return f"{route}|{get_settings().model_for('answer')}|k={k or ''}"


def _caps(question: str) -> str:
    """Capitalised words except question words: 'Who founded SunArc?' vs '...NovaTech?' differ."""
    words = {norm(w) for w in re.findall(r"\b[A-Z][\w&-]*", question)}
    return json.dumps(sorted(w for w in words - _QWORDS if w))


def _key(sig: str, question: str) -> str:
    return hashlib.sha256(f"{sig}|{norm(question)}".encode("utf-8")).hexdigest()


def cache_get(question: str, route: str, k, ent_ids: list[str]) -> dict | None:
    """Exact hit, or a semantic hit that mentions the same entities and named things."""
    s = get_settings()
    if not s.cache_enabled:
        return None
    try:
        sig = _sig(route, k)
        with closing(_con()) as con:
            ep = _epoch(con)
            row = con.execute("SELECT result FROM qa WHERE key=? AND epoch=?",
                              (_key(sig, question), ep)).fetchone()
            if row:
                return {**json.loads(row[0]), "_cache": "exact"}
            ents, caps = json.dumps(sorted(ent_ids)), _caps(question)
            rows = con.execute(
                "SELECT emb, result FROM qa WHERE sig=? AND epoch=? AND ents=? AND caps=? "
                "ORDER BY id DESC LIMIT ?", (sig, ep, ents, caps, s.cache_scan_limit)).fetchall()
        if not rows:
            return None  # no candidate: skip embedding the question entirely
        q = np.array(get_embeddings().embed_query(question))
        sims = np.array([json.loads(r[0]) for r in rows]) @ q
        i = int(np.argmax(sims))
        if sims[i] >= s.cache_similarity_threshold:
            return {**json.loads(rows[i][1]), "_cache": "semantic", "_cache_score": round(float(sims[i]), 3)}
    except Exception:
        return None  # the cache must never break an answer
    return None


def cache_put(question: str, route: str, k, ent_ids: list[str], result: dict) -> None:
    if not get_settings().cache_enabled:
        return
    try:
        sig = _sig(route, k)
        emb = json.dumps([float(x) for x in get_embeddings().embed_query(question)])
        with closing(_con()) as con:
            con.execute(
                "INSERT OR REPLACE INTO qa (key, route, sig, epoch, ents, caps, emb, result, ts) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (_key(sig, question), route, sig, _epoch(con), json.dumps(sorted(ent_ids)),
                 _caps(question), emb, json.dumps(result), time.time()))
            con.execute("DELETE FROM qa WHERE id <= (SELECT MAX(id) FROM qa) - ?", (get_settings().cache_max_rows,))
            con.commit()
    except Exception:
        pass


# ---------- metrics ----------
def record(*names: str, n: int = 1) -> None:
    try:
        with closing(_con()) as con:
            for name in names:
                con.execute("INSERT INTO counters VALUES (?, ?) "
                            "ON CONFLICT(name) DO UPDATE SET n = n + ?", (name, n, n))
            con.commit()
    except Exception:
        pass


def metrics() -> dict:
    with closing(_con()) as con:
        c = dict(con.execute("SELECT name, n FROM counters").fetchall())
    saved = sum(c.get(k, 0) for k in SAVED)
    requests = c.get("requests", 0)
    return {
        "requests": requests,
        "llm_calls": c.get("llm_calls", 0),
        "llm_calls_saved": saved,
        "saved_pct": round(100 * saved / requests, 1) if requests else 0.0,
        "saved_by": {k.removeprefix("saved_"): c.get(k, 0) for k in SAVED},
        "grounding_flags": c.get("grounding_flags", 0),
        "escalated": c.get("escalated", 0),
        "fallback_used": sum(v for k, v in c.items() if k.startswith("fallback_")),
    }
