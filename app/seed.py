"""Load the expected sample graph into Neo4j WITHOUT any LLM.

Mirrors eval/gold_triples.yaml (plus the optional board links). Lets you build and test Phase 4-7
while extraction quota is used up. Source names equal the sample file names, so a later
`python -m app.ingest --graph-only` replaces these edges (per-document delete + rewrite).

    python -m app.seed            # write the sample graph
    python -m app.seed --clear    # wipe the whole graph first (destructive)
"""
import argparse

from .cache import bump_epoch
from .extract import norm
from .stores import get_graph, graph_delete_source, graph_write

P, C, L = "Person", "Company", "Location"
CUR, PAST = "current", "past"

SEED = {
    "novatech_profile.md": [
        ("Aarav Mehta", P, "FOUNDED", "NovaTech Labs", C, CUR),
        ("Priya Nair", P, "FOUNDED", "NovaTech Labs", C, CUR),
        ("Aarav Mehta", P, "CEO_OF", "NovaTech Labs", C, CUR),
        ("Priya Nair", P, "WORKS_AT", "NovaTech Labs", C, CUR),
        ("Meridian Capital", C, "INVESTED_IN", "NovaTech Labs", C, CUR),
        ("NovaTech Labs", C, "LOCATED_IN", "Pune", L, CUR),
        ("Kavya Iyer", P, "WORKS_AT", "NovaTech Labs", C, CUR),
        ("Kavya Iyer", P, "WORKS_AT", "Orbit Systems", C, PAST),
        ("SunArc Power", C, "PARTNERED_WITH", "NovaTech Labs", C, CUR),
    ],
    "meridian_capital.txt": [
        ("Meridian Capital", C, "LOCATED_IN", "Mumbai", L, CUR),
        ("Rohan Kapoor", P, "WORKS_AT", "Meridian Capital", C, CUR),
        ("Rohan Kapoor", P, "WORKS_AT", "NovaTech Labs", C, CUR),
        ("Meridian Capital", C, "INVESTED_IN", "NovaTech Labs", C, CUR),
        ("Meridian Capital", C, "INVESTED_IN", "GreenGrid Energy", C, CUR),
    ],
    "greengrid_news.txt": [
        ("SunArc Power", C, "ACQUIRED", "GreenGrid Energy", C, CUR),
        ("Isha Verma", P, "CEO_OF", "GreenGrid Energy", C, CUR),
        ("Isha Verma", P, "WORKS_AT", "SunArc Power", C, CUR),
        ("Meridian Capital", C, "INVESTED_IN", "GreenGrid Energy", C, CUR),
        ("Priya Nair", P, "WORKS_AT", "GreenGrid Energy", C, PAST),
        ("GreenGrid Energy", C, "LOCATED_IN", "Bengaluru", L, CUR),
    ],
    "sunarc_power.txt": [
        ("SunArc Power", C, "LOCATED_IN", "Chennai", L, CUR),
        ("Vikram Rao", P, "CEO_OF", "SunArc Power", C, CUR),
        ("SunArc Power", C, "ACQUIRED", "GreenGrid Energy", C, CUR),
        ("SunArc Power", C, "PARTNERED_WITH", "NovaTech Labs", C, CUR),
        ("Isha Verma", P, "WORKS_AT", "SunArc Power", C, CUR),
    ],
    "people_notes.md": [
        ("Aarav Mehta", P, "WORKS_AT", "Orbit Systems", C, PAST),
        ("Orbit Systems", C, "LOCATED_IN", "Hyderabad", L, CUR),
        ("Isha Verma", P, "FOUNDED", "GreenGrid Energy", C, CUR),
        ("Vikram Rao", P, "CEO_OF", "SunArc Power", C, CUR),
        ("Vikram Rao", P, "WORKS_AT", "Orbit Systems", C, PAST),
        ("Kavya Iyer", P, "WORKS_AT", "NovaTech Labs", C, CUR),
        ("Kavya Iyer", P, "WORKS_AT", "Orbit Systems", C, PAST),
        ("Priya Nair", P, "FOUNDED", "NovaTech Labs", C, CUR),
    ],
}


def build(triples: list[tuple]) -> tuple[list[dict], list[dict]]:
    ents: dict[tuple[str, str], dict] = {}
    rels = []
    for s, sl, rel, t, tl, status in triples:
        ents[(norm(s), sl)] = {"id": norm(s), "name": s, "label": sl}
        ents[(norm(t), tl)] = {"id": norm(t), "name": t, "label": tl}
        rels.append({"s": norm(s), "t": norm(t), "type": rel, "sl": sl, "tl": tl,
                     "status": status, "evidence": "seed"})
    return list(ents.values()), rels


def main():
    ap = argparse.ArgumentParser(description="Seed the sample knowledge graph (no LLM)")
    ap.add_argument("--clear", action="store_true", help="delete ALL nodes first")
    args = ap.parse_args()
    if args.clear:
        get_graph().query("MATCH (n) DETACH DELETE n")
        print("Graph cleared.")
    for source, triples in SEED.items():
        graph_delete_source(source)
        entities, rels = build(triples)
        graph_write(source, entities, rels)
        print(f"{source}: {len(entities)} entities, {len(rels)} relations")
    bump_epoch()  # cached answers are stale now
    print("Done. Check: GET /graph/summary")


if __name__ == "__main__":
    main()
