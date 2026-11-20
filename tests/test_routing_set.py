"""Offline check of every cue in config.yaml against the 49-question evaluation set.

No Gemini, no Neo4j: entities come from eval/gold_triples.yaml and the router runs rules only.
Questions the rules cannot decide (one entity, no cue) are skipped here: they need the router LLM
and are covered by eval/run_eval.py.
"""
import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from pathlib import Path

import pytest
import yaml

from app.extract import norm
from app.router import decide_route, link_entities

EVAL = Path(__file__).resolve().parent.parent / "eval"
QUESTIONS = [q for q in yaml.safe_load((EVAL / "questions.yaml").read_text(encoding="utf-8")) if "route" in q]
HOLDOUT = yaml.safe_load((EVAL / "holdout_questions.yaml").read_text(encoding="utf-8"))
GOLD = yaml.safe_load((EVAL / "gold_triples.yaml").read_text(encoding="utf-8"))
CORPUS_GOLD = yaml.safe_load((EVAL / "gold_triples_corpus.yaml").read_text(encoding="utf-8"))
CORPUS_QUESTIONS = [q for q in yaml.safe_load((EVAL / "questions_corpus.yaml").read_text(encoding="utf-8")) if "route" in q]
ENTITIES = [{"id": norm(n), "name": n, "label": label}
            for g in (GOLD, CORPUS_GOLD) for label, names in g["entities"].items() for n in names]


def accepted(q) -> list[str]:
    return q["route"] if isinstance(q["route"], list) else [q["route"]]


def decide(q):
    ents = link_entities(q["q"], entities=ENTITIES)
    return decide_route(q["q"], ents, use_llm=False)


ALL = QUESTIONS + HOLDOUT + CORPUS_QUESTIONS  # holdout = phrasings written after the cues were tuned


@pytest.mark.parametrize("q", ALL, ids=[q["id"] for q in ALL])
def test_rules_pick_an_acceptable_route(q):
    d = decide(q)
    if d["via"] == "default":
        pytest.skip("ambiguous: decided by the router LLM at run time")
    assert d["route"] in accepted(q), f"{q['q']!r} -> {d['route']} ({d['reason']}); wanted {accepted(q)}"


def test_rule_accuracy_summary():
    decided = [(q, decide(q)) for q in ALL]
    rules = [(q, d) for q, d in decided if d["via"] != "default"]
    ok = sum(d["route"] in accepted(q) for q, d in rules)
    strict = sum(d["route"] == accepted(q)[0] for q, d in rules)
    print(f"\nrules decided {len(rules)}/{len(decided)}; acceptable {ok}/{len(rules)}; ideal {strict}/{len(rules)}")
    assert ok == len(rules)
