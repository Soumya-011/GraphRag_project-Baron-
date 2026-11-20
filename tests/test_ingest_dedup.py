"""ingest_one: never ingest the same content twice. Vector store, extraction and Neo4j are replaced by fakes."""
import os
import sys
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app import ingest, registry
from app.config import get_settings


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DB_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.delenv("SKIP_ALREADY_INGESTED", raising=False)
    get_settings.cache_clear()
    calls = {"vec": 0, "graph": 0, "deleted": []}
    state = {"graph_fails": False}

    def fake_prepare(path, root=ingest.RAW_DIR):
        return ingest.source_for(path, root), [object(), object()]

    def fake_vectors(source, chunks):
        calls["vec"] += 1
        return len(chunks)

    def fake_graph(source, chunks):
        calls["graph"] += 1
        if state["graph_fails"]:
            raise RuntimeError("quota")
        return {"entities": 4, "relations": 3, "dropped": 0, "failed_chunks": 0, "llm_calls": 2}

    monkeypatch.setattr(ingest, "prepare", fake_prepare)
    monkeypatch.setattr(ingest, "_write_vectors", fake_vectors)
    monkeypatch.setattr(ingest, "ingest_graph", fake_graph)
    monkeypatch.setattr(ingest, "graph_delete_source", lambda s: calls["deleted"].append(s))
    monkeypatch.setattr(ingest, "bump_epoch", lambda: None)
    root = tmp_path / "raw"
    (root / "uploads").mkdir(parents=True)
    yield root, calls, state
    get_settings.cache_clear()


def put(root, name, text):
    p = root / "uploads" / name
    p.write_text(text, encoding="utf-8")
    return p


def test_first_ingest_registers_then_same_file_is_skipped(env):
    root, calls, _ = env
    p = put(root, "a.txt", "Acme hired Bob.")
    r = ingest.ingest_one(p, root)
    assert r["status"] == "ingested" and r["chunks"] == 2 and r["graph"]["entities"] == 4
    assert registry.get("uploads/a.txt")["graph"] == 1
    r2 = ingest.ingest_one(p, root)
    assert r2["status"] == "skipped" and "Already ingested" in r2["reason"]
    assert calls == {"vec": 1, "graph": 1, "deleted": []}                    # nothing ran twice


def test_same_content_under_another_name_is_skipped(env):
    root, calls, _ = env
    ingest.ingest_one(put(root, "a.txt", "same text"), root)
    r = ingest.ingest_one(put(root, "copy.txt", "same text"), root)
    assert r["status"] == "skipped" and "Same content as uploads/a.txt" in r["reason"]
    assert calls["vec"] == 1 and registry.get("uploads/copy.txt") is None


def test_edited_file_is_ingested_again(env):
    root, calls, _ = env
    p = put(root, "a.txt", "version one")
    ingest.ingest_one(p, root)
    p.write_text("version two", encoding="utf-8")
    r = ingest.ingest_one(p, root)
    assert r["status"] == "ingested" and calls["vec"] == 2 and calls["graph"] == 2
    assert registry.get("uploads/a.txt")["sha256"] == registry.file_sha256(p)


def test_vectors_only_then_graph_later_does_not_re_embed(env):
    root, calls, _ = env
    p = put(root, "a.txt", "text")
    ingest.ingest_one(p, root, graph=False)
    assert registry.get("uploads/a.txt")["graph"] == 0 and calls["graph"] == 0
    r = ingest.ingest_one(p, root, graph=True)
    assert r["status"] == "ingested" and calls["vec"] == 1 and calls["graph"] == 1
    assert registry.get("uploads/a.txt")["graph"] == 1


def test_graph_failure_keeps_vectors_and_retries_graph_only(env):
    root, calls, state = env
    p = put(root, "a.txt", "text")
    state["graph_fails"] = True
    r = ingest.ingest_one(p, root)
    assert r["graph_error"] and r["error"] is None and registry.get("uploads/a.txt")["graph"] == 0
    state["graph_fails"] = False
    ingest.ingest_one(p, root)
    assert calls["vec"] == 1 and calls["graph"] == 2 and registry.get("uploads/a.txt")["graph"] == 1


def test_edit_without_graph_removes_stale_graph_facts(env):
    root, calls, _ = env
    p = put(root, "a.txt", "old facts")
    ingest.ingest_one(p, root)
    p.write_text("new facts", encoding="utf-8")
    ingest.ingest_one(p, root, graph=False)
    assert calls["deleted"] == ["uploads/a.txt"] and registry.get("uploads/a.txt")["graph"] == 0


def test_force_and_switch_off(env, monkeypatch):
    root, calls, _ = env
    p = put(root, "a.txt", "text")
    ingest.ingest_one(p, root)
    assert ingest.ingest_one(p, root, force=True)["status"] == "ingested" and calls["vec"] == 2
    monkeypatch.setenv("SKIP_ALREADY_INGESTED", "false")
    get_settings.cache_clear()
    assert ingest.ingest_one(p, root)["status"] == "ingested" and calls["vec"] == 3


def test_errors_are_reported_not_raised(env, monkeypatch):
    root, calls, _ = env
    monkeypatch.setattr(ingest, "prepare", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad pdf")))
    r = ingest.ingest_one(put(root, "a.txt", "text"), root)
    assert r["error"] == "bad pdf" and registry.get("uploads/a.txt") is None


def test_ingest_paths_summary_shape(env):
    root, _, _ = env
    a, b = put(root, "a.txt", "one"), put(root, "b.txt", "two")
    out = ingest.ingest_paths([a, b], root)
    assert out["files"] == {"a.txt": 2, "b.txt": 2} and out["total_chunks"] == 4 and out["skipped"] == {}
    out2 = ingest.ingest_paths([a, b], root)
    assert set(out2["skipped"]) == {"a.txt", "b.txt"} and out2["total_chunks"] == 0
