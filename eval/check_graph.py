"""Extraction quality: compare the Neo4j graph with the hand-labelled gold triples.

    python -m eval.check_graph        # core + the 35 corpus documents (what the graph contains)

Facts of extra node types (Year: IN_YEAR) have no hand-labelled gold, so they are NOT part of precision/recall.
They are reported separately: how many dated facts, and how many have the year inside their quoted evidence.

Run it after `python -m app.ingest` (not after `python -m app.seed`, which only copies the gold data).
Precision = extracted triples that are gold or optional / all extracted triples.
Recall    = gold triples that were extracted / all gold triples.
PARTNERED_WITH is symmetric, so either direction counts.
"""
import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app import schema  # noqa: E402
from app.extract import norm  # noqa: E402
from app.stores import get_graph, graph_summary  # noqa: E402


def triple(s, rel, t, status="current"):
    s, t = norm(s), norm(t)
    if rel == "PARTNERED_WITH" and s > t:  # symmetric relation: one canonical direction
        s, t = t, s
    return (s, rel, t, status)


def load(files):
    gold, optional = set(), set()
    for f in files:
        d = yaml.safe_load((HERE / f).read_text(encoding="utf-8"))
        gold |= {triple(*t) for t in d["gold"]}
        optional |= {triple(*t) for t in d.get("optional", [])}
    return gold, optional


def _dated_rows(rels: list[str]) -> list[dict]:
    return get_graph().query(
        "MATCH (a)-[r]->(y) WHERE type(r) IN $rels "
        "RETURN a.name AS source, type(r) AS rel, y.name AS target, r.evidence AS evidence, r.source AS doc",
        {"rels": rels})


def dated_report(exclude_prefix: str = "uploads/") -> dict | None:
    """Facts tied to a Year-like node. No gold labels exist, so the check is mechanical: the year must appear in
    the quoted evidence (extraction already guarantees it appears in the source chunk)."""
    rels = schema.extra_relation_names()
    if not rels:
        return None
    try:
        rows = [r for r in _dated_rows(rels) if not (r.get("doc") or "").startswith(exclude_prefix)]
    except Exception:
        return None
    quoted = [r for r in rows if r.get("evidence") and str(r["target"]) in r["evidence"]]
    no_year = [r for r in rows if r not in quoted]
    return {"count": len(rows), "year_in_quote": len(quoted),
            "pct": (len(quoted) / len(rows)) if rows else None,
            "no_year_in_quote": [[r["source"], r["rel"], r["target"], (r.get("evidence") or "")[:120]] for r in no_year[:30]]}


def compute(with_corpus: bool = True, exclude_prefix: str = "uploads/") -> dict:
    """Precision / recall of the graph against the gold triples. Edges from uploaded documents
    (source starting with `exclude_prefix`) are left out: they have no gold labels."""
    gold, optional = load(["gold_triples.yaml"] + (["gold_triples_corpus.yaml"] if with_corpus else []))
    all_edges = graph_summary(5000)["edges"]
    edges = [e for e in all_edges if not (e.get("doc") or "").startswith(exclude_prefix)]
    extra = set(schema.extra_relation_names())  # IN_YEAR ...: reported separately, not scored
    got = {triple(e["source"], e["rel"], e["target"], e.get("status") or "current") for e in edges
           if e["rel"] not in extra}
    if not got:
        return {"empty": True, "excluded": len(all_edges) - len(edges)}

    # a gold triple also counts if the relation is right but the past/current flag is missed
    loose = lambda s: {(a, b, c) for a, b, c, _ in s}  # noqa: E731
    hit = {g for g in gold if g in got or (g[0], g[1], g[2]) in loose(got)}
    good = {x for x in got if x in gold or x in optional or (x[0], x[1], x[2]) in loose(gold | optional)}
    return {"empty": False, "extracted": len(got), "gold": len(gold), "good": len(good), "hit": len(hit),
            "precision": len(good) / len(got), "recall": len(hit) / len(gold),
            "wrong": sorted(got - good), "missing": sorted(gold - hit), "excluded": len(all_edges) - len(edges),
            "dated": dated_report(exclude_prefix)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-corpus", action="store_true", help="(kept for old commands: the corpus is always included)")
    ap.parse_args()
    r = compute(True)
    if r["empty"]:
        sys.exit("The graph is empty. Run: python -m app.ingest")

    print(f"Extracted triples: {r['extracted']}   gold: {r['gold']}")
    print(f"Precision: {r['precision']:.0%} ({r['good']}/{r['extracted']})")
    print(f"Recall:    {r['recall']:.0%} ({r['hit']}/{r['gold']})")
    d = r.get("dated")
    if d:
        pct = f"{d['pct']:.0%}" if d["pct"] is not None else "n/a"
        print(f"Dated facts (IN_YEAR, not scored): {d['count']}   year appears in the quote: {d['year_in_quote']} ({pct})")
    if r["wrong"]:
        print("\nExtracted but not in gold (wrong, or add to `optional` if reasonable):")
        for s, rel, t, st in r["wrong"]:
            print(f"  {s} -{rel}-> {t} ({st})")
    if r["missing"]:
        print("\nMissing from the graph:")
        for s, rel, t, st in r["missing"]:
            print(f"  {s} -{rel}-> {t} ({st})")


if __name__ == "__main__":
    main()
