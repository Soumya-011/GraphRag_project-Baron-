"""check_graph: dated (IN_YEAR) facts are reported separately and never counted as 'wrong'."""
import os
import sys
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import eval.check_graph as cg
from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def edge(s, rel, t, doc="core.txt", status="current"):
    return {"source": s, "rel": rel, "target": t, "status": status, "doc": doc}


def patch(monkeypatch, edges, dated):
    monkeypatch.setattr(cg, "graph_summary", lambda n: {"edges": edges})
    monkeypatch.setattr(cg, "_dated_rows", lambda rels: dated)
    g, o = cg.load(["gold_triples.yaml"])
    first = sorted(g)[0]
    return first


def test_year_edges_do_not_lower_precision(monkeypatch):
    monkeypatch.setenv("EXTRA_NODE_TYPES", "Year")
    get_settings.cache_clear()
    gold_first = patch(monkeypatch, [], [])
    good = edge(gold_first[0], gold_first[1], gold_first[2], status=gold_first[3])
    years = [edge("acme corp", "IN_YEAR", "2010"), edge("acme corp", "IN_YEAR", "2011")]
    monkeypatch.setattr(cg, "graph_summary", lambda n: {"edges": [good] + years})
    r = cg.compute(False)
    assert r["extracted"] == 1 and r["precision"] == 1.0 and r["wrong"] == []


def test_year_edges_counted_as_wrong_only_if_the_feature_is_off(monkeypatch):
    monkeypatch.setenv("EXTRA_NODE_TYPES", "")
    get_settings.cache_clear()
    patch(monkeypatch, [], [])
    monkeypatch.setattr(cg, "graph_summary", lambda n: {"edges": [edge("acme corp", "IN_YEAR", "2010")]})
    assert cg.compute(False)["precision"] == 0.0 and cg.compute(False)["dated"] is None


def test_dated_report_counts_quotes_that_contain_the_year(monkeypatch):
    monkeypatch.setenv("EXTRA_NODE_TYPES", "Year")
    get_settings.cache_clear()
    rows = [
        {"source": "Acme", "rel": "IN_YEAR", "target": "2010", "evidence": "Acme acquired Zed in 2010", "doc": "a.txt"},
        {"source": "Acme", "rel": "IN_YEAR", "target": "2011", "evidence": "Acme opened an office", "doc": "a.txt"},
        {"source": "Up", "rel": "IN_YEAR", "target": "2012", "evidence": "x", "doc": "uploads/u.txt"},  # ignored
    ]
    monkeypatch.setattr(cg, "_dated_rows", lambda rels: rows)
    d = cg.dated_report()
    assert d["count"] == 2 and d["year_in_quote"] == 1 and d["pct"] == 0.5
    assert d["no_year_in_quote"][0][2] == "2011"


def test_dated_report_is_none_when_off_or_database_fails(monkeypatch):
    monkeypatch.setenv("EXTRA_NODE_TYPES", "")
    get_settings.cache_clear()
    assert cg.dated_report() is None
    monkeypatch.setenv("EXTRA_NODE_TYPES", "Year")
    get_settings.cache_clear()
    monkeypatch.setattr(cg, "_dated_rows", lambda rels: (_ for _ in ()).throw(RuntimeError("db down")))
    assert cg.dated_report() is None
