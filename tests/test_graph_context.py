"""Graph context wording and graph -> hybrid escalation. No network, no Neo4j, no Gemini."""
import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ["CACHE_ENABLED"] = "false"

import pytest

from app import pipeline, rag
from app.config import get_config

NF = get_config()["guardrails"]["canned"]["not_found"]
ENTS = [{"id": "rohan kapoor", "name": "Rohan Kapoor", "label": "Person", "score": 100}]

EDGE = {"source": "Rohan Kapoor", "rel": "WORKS_AT", "target": "Meridian Capital", "status": "current",
        "evidence": "Its managing partner is Rohan Kapoor", "docs": ["meridian_capital.txt"]}


# ---------- graph context wording ----------
def test_fact_is_plain_english_with_evidence_and_source():
    line = rag._edge_line(EDGE)
    assert "WORKS_AT" not in line
    assert line.startswith("Rohan Kapoor works at Meridian Capital")
    assert '"Its managing partner is Rohan Kapoor"' in line and "[meridian_capital.txt]" in line


def test_past_flag_survives_and_empty_evidence_adds_no_quote():
    line = rag._edge_line({**EDGE, "status": "past", "evidence": ""})
    assert "(past)" in line and '"' not in line


@pytest.mark.parametrize("rel,words", [("CEO_OF", "is CEO of"), ("FOUNDED", "founded"),
                                       ("INVESTED_IN", "invested in"), ("LOCATED_IN", "is located in")])
def test_relation_wording(rel, words):
    assert words in rag._fact({**EDGE, "rel": rel})


def test_context_never_contains_raw_relation_names():
    g = {"entities": ["Priya Nair"], "edges": [{**EDGE, "rel": "CEO_OF"}],
         "paths": [{"from": "A", "to": "B", "edges": [
             {"source": "Priya Nair", "rel": "WORKS_AT", "target": "GreenGrid Energy", "status": "past"}]}]}
    ctx = rag.format_graph_context(g, 2000)
    assert "_" not in ctx.replace("meridian_capital.txt", "")
    assert "Priya Nair works at GreenGrid Energy (past)" in ctx


# ---------- escalation ----------
def _setup(monkeypatch, graph_answer, graph_gated=False, graph_calls=1):
    calls = []

    def fake_answer(question, route, k=None, ents=None):
        calls.append(route)
        if route == "graph":
            return {"route": "graph", "answer": graph_answer, "gated": graph_gated,
                    "llm_calls": graph_calls, "sources": [], "context": "x", "best_score": None}
        return {"route": "hybrid", "answer": "Joined in 2022 [meridian_capital.txt]", "gated": False,
                "llm_calls": 1, "sources": [], "context": "x", "best_score": 0.5}

    monkeypatch.setattr(pipeline.rag, "answer_question", fake_answer)
    monkeypatch.setattr(pipeline.rag, "try_rule_answer", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "decide_route", lambda q, e: {"route": "graph", "via": "cue",
                                                                "reason": "t", "llm_calls": 0})
    monkeypatch.setattr(pipeline, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "cache_put", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "record", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.gr, "check_grounding", lambda a, c: {"ok": True, "unsupported": []})
    return calls


def test_graph_not_found_escalates_to_hybrid(monkeypatch):
    calls = _setup(monkeypatch, NF)
    r = pipeline.run("When did Rohan join?", "auto", ents=ENTS)
    assert calls == ["graph", "hybrid"]
    assert r["route"] == "hybrid" and r["escalated"] == "graph->hybrid"
    assert r["llm_calls"] == 2 and "2022" in r["answer"]
    assert r["router"]["route"] == "graph"  # the router's own decision stays visible


def test_gated_graph_also_escalates(monkeypatch):
    calls = _setup(monkeypatch, NF, graph_gated=True, graph_calls=0)
    r = pipeline.run("q", "auto", ents=ENTS)
    assert calls == ["graph", "hybrid"] and r["llm_calls"] == 1


def test_good_graph_answer_does_not_escalate(monkeypatch):
    calls = _setup(monkeypatch, "Vikram Rao is the CEO of SunArc Power")
    r = pipeline.run("q", "auto", ents=ENTS)
    assert calls == ["graph"] and r["escalated"] is None and r["route"] == "graph"


def test_explicit_graph_route_is_never_escalated(monkeypatch):
    calls = _setup(monkeypatch, NF)
    r = pipeline.run("q", "graph", ents=ENTS)
    assert calls == ["graph"] and r["escalated"] is None


# ---------- entity-grouped context ----------
def test_entity_blocks_join_founder_and_investor_in_one_line():
    edges = [
        {"source": "Aarav Mehta", "rel": "FOUNDED", "target": "NovaTech Labs", "status": "current", "docs": ["a.md"], "evidence": "x"},
        {"source": "Aarav Mehta", "rel": "WORKS_AT", "target": "Orbit Systems", "status": "past", "docs": ["b.md"], "evidence": "y"},
        {"source": "Meridian Capital", "rel": "INVESTED_IN", "target": "NovaTech Labs", "status": "current", "docs": ["c.md"], "evidence": "z"},
    ]
    blocks = rag.entity_blocks(edges, ["Orbit Systems"])
    nova = next(b for b in blocks if b.startswith("NovaTech Labs:"))
    assert "founded by Aarav Mehta" in nova and "invested in by Meridian Capital" in nova
    assert any(b.startswith("Aarav Mehta:") and "Orbit Systems (past)" in b for b in blocks)
    assert blocks[0].startswith("Orbit Systems:")  # linked entity first


def test_grouped_context_has_no_raw_relation_names_and_respects_budget():
    edges = [{"source": f"P{i}", "rel": "WORKS_AT", "target": "Co", "status": "current", "docs": ["d.md"], "evidence": "q" * 50}
             for i in range(200)]
    ctx = rag.format_graph_context({"entities": ["Co"], "edges": edges, "paths": []}, 500)
    assert "_" not in ctx.replace("d.md", "") and len(ctx) <= 500 * 4 + 200
    assert "Summary by entity" in ctx
    small = rag.format_graph_context({"entities": ["Co"], "edges": edges[:3], "paths": []}, 2000)
    assert small.index("Summary by entity") < small.index("Facts with quotes")
