"""Ingest registry: skipping unchanged files and duplicate content. Offline."""
import os
import sys
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app import registry
from app.config import get_settings


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DB_PATH", str(tmp_path / "cache.sqlite"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_hashes_agree(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello")
    assert registry.file_sha256(f) == registry.bytes_sha256(b"hello")


def test_upsert_get_remove_and_list():
    registry.upsert("uploads/a.txt", "h1", 3, True, 5, 4)
    r = registry.get("uploads/a.txt")
    assert (r["chunks"], r["entities"], r["relations"], r["graph"]) == (3, 5, 4, 1)
    registry.upsert("uploads/a.txt", "h2", 4, False)          # edited file replaces the row
    assert registry.get("uploads/a.txt")["sha256"] == "h2" and len(registry.all_docs()) == 1
    registry.remove("uploads/a.txt")
    assert registry.get("uploads/a.txt") is None and registry.all_docs() == []


def test_duplicate_reasons():
    assert registry.duplicate_reason("uploads/a.txt", "h1", True) is None            # never seen
    registry.upsert("uploads/a.txt", "h1", 3, True)
    assert "Already ingested" in registry.duplicate_reason("uploads/a.txt", "h1", True)
    assert "Same content as uploads/a.txt" in registry.duplicate_reason("uploads/copy.txt", "h1", True)
    assert registry.duplicate_reason("uploads/a.txt", "h-new", True) is None         # edited: re-ingest
    registry.upsert("uploads/b.txt", "h3", 2, False)                                 # stored without graph
    assert registry.duplicate_reason("uploads/b.txt", "h3", True) is None            # graph still wanted
    assert registry.duplicate_reason("uploads/b.txt", "h3", False) is not None       # vectors only: done
