import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")

import pytest
from langchain_core.runnables import RunnableLambda

from app.llm import _RpmLimiter
from app.router import RouteChoice, decide_route


def E(*names):
    return [{"id": n.lower(), "name": n, "label": "Company", "score": 100} for n in names]


# (question, linked entities, expected route, expected via)
CASES = [
    ("What is the capital of France?", [], "vector", "entity"),
    ("Which companies are linked to Priya Nair?", E("Priya Nair"), "graph", "cue"),
    ("Who is the CEO of SunArc Power?", E("SunArc Power"), "graph", "cue"),
    ("Summarize NovaTech Labs", E("NovaTech Labs"), "vector", "cue"),
    ("Explain how Priya Nair is connected to SunArc Power",
     E("Priya Nair", "SunArc Power"), "hybrid", "cue"),
    ("Priya Nair and SunArc Power", E("Priya Nair", "SunArc Power"), "graph", "entity"),
]


@pytest.mark.parametrize("question,ents,route,via", CASES)
def test_rules(question, ents, route, via):
    d = decide_route(question, ents, use_llm=False)
    assert (d["route"], d["via"]) == (route, via)
    assert d["reason"] and d["llm_calls"] == 0


def test_ambiguous_without_llm_uses_default():
    d = decide_route("Vikram Rao", E("Vikram Rao"), use_llm=False)
    assert d["via"] == "default" and d["route"] == "hybrid" and d["llm_calls"] == 0


def test_ambiguous_uses_router_llm(monkeypatch):
    fake = RunnableLambda(lambda _: RouteChoice(route="graph", reason="relationship question"))
    monkeypatch.setattr("app.llm.get_structured_llm", lambda *a, **k: fake)
    d = decide_route("Vikram Rao", E("Vikram Rao"), use_llm=True)
    assert (d["route"], d["via"], d["llm_calls"]) == ("graph", "llm", 1)


def test_router_llm_failure_falls_back_to_default(monkeypatch):
    def boom(_):
        raise RuntimeError("429 quota")

    monkeypatch.setattr("app.llm.get_structured_llm", lambda *a, **k: RunnableLambda(boom))
    d = decide_route("Vikram Rao", E("Vikram Rao"), use_llm=True)
    assert d["via"] == "default" and d["llm_calls"] == 0


def test_rpm_limiter_non_blocking():
    lim = _RpmLimiter(2)
    assert lim.acquire(blocking=False) and lim.acquire(blocking=False)
    assert not lim.acquire(blocking=False)
