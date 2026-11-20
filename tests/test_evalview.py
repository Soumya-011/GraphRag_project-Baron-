"""Eval report parsing for the developer page. Offline."""
import os
import sys
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import evalview

REPORT = """# Evaluation report (2026-11-15 19:45)

Questions: questions_corpus.yaml; cache off
Settings: top_k_vector=4, graph_max_hops=3

## Mode: graph

- Answer correctness: **81% (30/37)**
- Mean recall (share of the expected items found in the answer): **85%**
- Mean precision (expected items found / items listed; extra names lower it): **88%**; answers listing extra names: 1

| Question type | Correct | Mean recall |
|---|---|---|
| aggregation | 100% (12/12) | 100% |
| guard | 100% (3/3) | n/a |

- LLM calls used: **37** for 37 questions (plain RAG baseline: 37)

| Route taken | Questions | Correct | Avg LLM calls | Avg latency | p50 | p95 |
|---|---|---|---|---|---|---|
| graph | 37 | 81% (30/37) | 1.00 | 6751 ms | 1762 ms | 30653 ms |

## Mode: vector@2 (vector top-k forced to 2)

- Answer correctness: **41% (15/37)**
- Mean recall (share of the expected items found in the answer): **46%**
"""


def test_summary_headline_numbers():
    s = evalview.summarize(REPORT)
    assert s["settings"].startswith("top_k_vector=4") and [m["mode"] for m in s["modes"]] == ["graph", "vector@2"]
    g, v = s["modes"]
    assert (g["correct_pct"], g["correct"], g["total"], g["recall_pct"], g["precision_pct"]) == (81, 30, 37, 85, 88)
    assert (g["llm_calls"], g["avg_ms"], g["p95_ms"]) == (37, 6751, 30653)
    assert g["by_type"]["aggregation"] == {"correct_pct": 100, "n": 12, "recall_pct": 100}
    assert g["by_type"]["guard"]["recall_pct"] is None
    assert v["correct_pct"] == 41 and v["precision_pct"] is None and v["by_type"] == {}


def test_empty_and_garbage_text_do_not_crash():
    assert evalview.summarize("")["modes"] == []
    assert evalview.summarize("no modes here")["modes"] == []


def test_names_must_be_plain(monkeypatch, tmp_path):
    monkeypatch.setattr(evalview, "RESULTS", tmp_path)
    (tmp_path / "ok-1").mkdir()
    (tmp_path / "ok-1" / "report.md").write_text("x", encoding="utf-8")
    assert evalview.read_report("ok-1") == "x"
    for bad in ("", "..", "../x", "a/b", "a\\b", ".hidden", "x" * 200):
        assert evalview.read_report(bad) is None
