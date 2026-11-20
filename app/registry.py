"""Ingest registry: which document (by content hash) is already in the vector store and graph.
Lets the app skip unchanged files instead of embedding and extracting them again. SQLite, no server."""
import hashlib
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .config import get_settings


def _con():
    p = get_settings().cache_path.parent / "registry.sqlite"
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute(
        "CREATE TABLE IF NOT EXISTS docs (source TEXT PRIMARY KEY, sha256 TEXT NOT NULL, chunks INTEGER, "
        "entities INTEGER, relations INTEGER, graph INTEGER, ingested_at REAL)")
    return con


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(source: str) -> dict | None:
    with closing(_con()) as con:
        r = con.execute("SELECT * FROM docs WHERE source = ?", (source,)).fetchone()
    return dict(r) if r else None


def by_hash(sha: str) -> dict | None:
    with closing(_con()) as con:
        r = con.execute("SELECT * FROM docs WHERE sha256 = ? ORDER BY ingested_at LIMIT 1", (sha,)).fetchone()
    return dict(r) if r else None


def upsert(source: str, sha: str, chunks: int, graph: bool, entities: int = 0, relations: int = 0) -> None:
    with closing(_con()) as con, con:
        con.execute(
            "INSERT INTO docs VALUES (?,?,?,?,?,?,?) ON CONFLICT(source) DO UPDATE SET sha256=excluded.sha256, "
            "chunks=excluded.chunks, entities=excluded.entities, relations=excluded.relations, "
            "graph=excluded.graph, ingested_at=excluded.ingested_at",
            (source, sha, chunks, entities, relations, int(graph), time.time()))


def remove(source: str) -> None:
    with closing(_con()) as con, con:
        con.execute("DELETE FROM docs WHERE source = ?", (source,))


def all_docs() -> list[dict]:
    with closing(_con()) as con:
        return [dict(r) for r in con.execute("SELECT * FROM docs ORDER BY source")]


def duplicate_reason(source: str, sha: str, graph: bool) -> str | None:
    """Why this content need not be ingested again (None = go ahead)."""
    rec = get(source)
    if rec and rec["sha256"] == sha and (rec["graph"] or not graph):
        return "Already ingested, nothing to do"
    other = by_hash(sha)
    if other and other["source"] != source and not rec:
        return f"Same content as {other['source']}, nothing to do"
    return None
