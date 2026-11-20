"""Corpus integrity: documents, gold triples and questions must agree. Offline, no model needed."""
import os
import sys
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "test-key")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import yaml

from eval import build_corpus as bc
from eval import run_eval as ev

DOCS, GOLD, QUESTIONS, WORLD = bc.build()
TEXTS = WORLD["texts"]
ALL_TEXT = " ".join(TEXTS.values())


def test_generated_files_are_up_to_date():
    """If this fails: run `python -m eval.build_corpus` (someone edited the world or the files)."""
    for d in DOCS:
        assert (bc.CORPUS_DIR / d.name).read_text(encoding="utf-8").strip() == d.text
    on_disk = {p.name for p in bc.CORPUS_DIR.glob("*.txt")}
    assert on_disk == {d.name for d in DOCS}
    assert yaml.safe_load((bc.HERE / "questions_corpus.yaml").read_text(encoding="utf-8")) == yaml.safe_load(
        yaml.safe_dump(QUESTIONS))
    assert yaml.safe_load((bc.HERE / "gold_triples_corpus.yaml").read_text(encoding="utf-8")) == yaml.safe_load(
        yaml.safe_dump(GOLD))


def test_size_and_shape():
    assert len(DOCS) == 35
    assert all(len(d.text) < 480 for d in DOCS), "each document must stay a single 500-character chunk"
    assert len({d.name for d in DOCS}) == len(DOCS)
    kinds = {}
    for q in QUESTIONS:
        kinds[q["type"]] = kinds.get(q["type"], 0) + 1
    assert kinds == {"aggregation": 12, "multihop": 12, "needle": 5, "single": 5, "guard": 3}
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)


def test_every_expected_answer_is_in_the_documents():
    for q in QUESTIONS:
        for item in q.get("a", []):
            if item[0].isdigit() and len(item) < 8 and "|" in item:  # counts such as "6|six" are computed, not quoted
                continue
            assert any(alt in ALL_TEXT for alt in item.split("|")), (q["id"], item)


def test_every_gold_triple_is_stated_in_its_document():
    for d in DOCS:
        for s, _, o, _ in d.triples:
            assert s in d.text and o in d.text, (d.name, s, o)


def test_names_are_unambiguous_for_the_entity_linker():
    """A first or last name shared by two people would make a bare surname ambiguous."""
    seen = {}
    for p in WORLD["people"]:
        for tok in p.split():
            assert seen.setdefault(tok, p) == p, f"{tok!r} appears in {seen[tok]!r} and {p!r}"
    firsts = {}
    for c in WORLD["companies"]:
        first = c.split()[0]
        assert firsts.setdefault(first, c) == c, f"{first!r}: {firsts[first]!r} and {c!r}"


def test_unanswerable_questions_really_are_unanswerable():
    assert "CFO" not in ALL_TEXT and "revenue of Lumina" not in ALL_TEXT
    assert "paid" not in " ".join(t for n, t in TEXTS.items() if "hydrocore" in n)


def test_evidence_is_spread_for_the_hard_questions():
    by_id = {q["id"]: q for q in QUESTIONS}
    assert by_id["ag01"]["docs_needed"] >= 8 and by_id["ag01"]["cover_k2"] <= 40
    hard = [q for q in QUESTIONS if q["type"] in ("aggregation", "multihop") and q.get("facts", 0) >= 6]
    assert len(hard) >= 10 and all(q["cover_k2"] < 100 for q in hard)
    assert all(q["cover_k4"] <= 100 for q in QUESTIONS if "cover_k4" in q)


# ---------- runner helpers ----------
def test_recall_and_strict_scoring():
    expected = ["Neha Joshi", "Sanjay Gupta|Sanjay", "Aarav Mehta"]
    assert ev.recall("Neha Joshi and Sanjay", expected) == pytest.approx(2 / 3)
    assert not ev.keywords_ok("Neha Joshi and Sanjay", expected)
    assert ev.keywords_ok("Aarav Mehta, Neha Joshi, Sanjay Gupta", expected)
    assert ev.recall("anything", None) is None


def test_mode_parsing_and_judging():
    assert ev.split_mode("vector@2") == ("vector", 2) and ev.split_mode("graph") == ("graph", None)
    nf = "I couldn't find this in the documents."
    assert ev.judge({"not_found": True}, {"answer": nf}, nf)
    assert not ev.judge({"not_found": True}, {"answer": "The CFO is Sam."}, nf)
    assert not ev.judge({"a": ["Pune"]}, {"answer": nf}, nf)


def test_extras_flag_wrong_people_but_not_expected_ones():
    exp = ["Sanjay Gupta", "Neha Joshi"]
    q = "Which Orbit Systems alumni founded companies that Meridian Capital invested in?"
    ans = "Sanjay Gupta and Neha Joshi, and also Priya Nair founded NovaTech Labs."
    assert ev.extras(ans, exp, q) == ["Priya Nair"]
    assert ev.precision(ans, exp, q) == pytest.approx(2 / 3)
    assert ev.extras("Sanjay Gupta, Neha Joshi", exp, q) == []
    assert ev.extras("anything", None, q) == []
