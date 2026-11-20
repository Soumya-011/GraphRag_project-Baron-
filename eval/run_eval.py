"""Evaluation runner: answer correctness, recall, route accuracy, LLM calls and latency per mode and route.

Core set (5 documents, 49 questions):
    python -m eval.run_eval                                # auto router only (about 50 Gemini calls)
    python -m eval.run_eval --modes auto,vector,graph      # normal RAG vs GraphRAG on every question

Stress corpus (35 more documents, 37 aggregation / multi-hop questions; see eval/build_corpus.py):
    python -m eval.run_eval --questions eval/questions_corpus.yaml --modes graph,vector,vector@2 \\
        --env GRAPH_MAX_EDGES=80 --env MAX_CONTEXT_TOKENS=4000

Modes: auto | vector | graph | hybrid. "vector@2" means vector retrieval with top-k = 2 (any mode takes @k).
--env KEY=VALUE overrides a setting for this run only and is printed in the report.
--ids g01,v01 runs a few questions; --resume <folder> continues after a quota error.

The answer cache is switched OFF (otherwise repeat runs would show 0 calls and fake latencies).
Free-tier note: the client-side limiter (GEMINI_RPM) paces calls, so about 40 calls take ~5 minutes.
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EARLY_GUARDS = {"blocked", "smalltalk", "off_topic"}


# ======================= scoring =======================
def _item_hit(text: str, item: str) -> bool:
    return any(alt.strip().lower() in text for alt in item.split("|") if alt.strip())


def keywords_ok(answer: str, expected: list[str]) -> bool:
    """Every item must match; an item 'x|y' matches if either alternative is in the answer."""
    text = answer.lower()
    return all(_item_hit(text, item) for item in expected)


def recall(answer: str, expected: list[str] | None) -> float | None:
    """Share of the expected items found in the answer (aggregation answers are lists)."""
    if not expected:
        return None
    text = answer.lower()
    return sum(_item_hit(text, item) for item in expected) / len(expected)


_UNIVERSE = None


def _universe() -> tuple[set[str], set[str]]:
    """(person names, company names) from the gold triples: the only names a list answer could add by mistake."""
    global _UNIVERSE
    if _UNIVERSE is None:
        people, orgs = set(), set()
        for f in ("gold_triples.yaml", "gold_triples_corpus.yaml"):
            fp = HERE / f
            if not fp.exists():
                continue
            d = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
            for sub, rel, obj, *_ in d.get("gold", []) + d.get("optional", []):
                (people if rel in ("WORKS_AT", "CEO_OF", "FOUNDED") else orgs).add(sub)
                orgs.add(obj)
        _UNIVERSE = (people, orgs - people)
    return _UNIVERSE


def extras(answer: str, expected: list[str] | None, question: str = "") -> list[str]:
    """Names of the same kind as the expected items (people or companies) that the answer lists
    but that are not expected and not in the question: likely false positives in a list answer."""
    if not expected:
        return []
    people, orgs = _universe()
    exp_text = " ".join(alt for item in expected for alt in item.split("|")).lower()
    pool = people if any(n.lower() in exp_text for n in people) else orgs
    text, q = answer.lower(), question.lower()
    return sorted(n for n in pool if n.lower() in text and n.lower() not in exp_text and n.lower() not in q)


def precision(answer: str, expected: list[str] | None, question: str = "") -> float | None:
    """Expected items found / (found + extra names). None when there is nothing to score."""
    if not expected:
        return None
    hit = sum(_item_hit(answer.lower(), item) for item in expected)
    bad = len(extras(answer, expected, question))
    return None if hit + bad == 0 else hit / (hit + bad)


def accepted_routes(item: dict) -> list[str]:
    r = item.get("route")
    return r if isinstance(r, list) else ([r] if r else [])


def split_mode(mode: str) -> tuple[str, int | None]:
    """'vector@2' -> ('vector', 2)."""
    base, _, k = mode.partition("@")
    return base, (int(k) if k else None)


def judge(item: dict, r: dict, not_found_text: str) -> bool:
    answer = (r.get("answer") or "").lower()
    canned = not_found_text.lower() in answer
    if item.get("guard"):
        return r.get("guard") == item["guard"]
    if item.get("not_found"):
        return canned
    return (not canned) and keywords_ok(answer, item["a"])


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}% ({a}/{b})" if b else "n/a"


def mean_pct(values: list[float | None]) -> str:
    vals = [v for v in values if v is not None]
    return f"{100 * statistics.mean(vals):.0f}%" if vals else "n/a"


def pctl(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))] if s else 0.0


# ======================= running =======================
def run_one(guarded_ask, item: dict, mode: str, not_found_text: str) -> dict:
    base, k = split_mode(mode)
    t0 = time.perf_counter()
    row = {"id": item["id"], "mode": mode, "type": item["type"], "q": item["q"],
           "hops": item.get("hops"), "expected_route": accepted_routes(item)}
    try:
        r = guarded_ask(item["q"], route=base, k=k)
    except Exception as e:  # quota, network, input rejected: record and keep going
        ms = int((time.perf_counter() - t0) * 1000)
        row.update(error=str(e)[:300], correct=False, acceptable=False, recall=0.0 if item.get("a") else None,
                   route=None, llm_calls=0, latency_ms=ms, net_latency_ms=ms)
        return row
    route = r.get("route")
    exp = accepted_routes(item)
    # Router accuracy is judged on the router's own decision, before any graph -> hybrid escalation.
    decided = (r.get("router") or {}).get("route") or route
    route_ok = None
    if base == "auto" and exp:
        route_ok = decided in exp and r.get("guard") not in EARLY_GUARDS
    answer = r.get("answer") or ""
    abstained = not_found_text.lower() in answer.lower()
    correct = judge(item, r, not_found_text)
    wait = r.get("wait_ms", 0) or 0
    row.update(
        answer=answer, route=route, guard=r.get("guard"), cached=r.get("cached"),
        via=(r.get("router") or {}).get("via"), reason=(r.get("router") or {}).get("reason"),
        router_route=decided if base == "auto" else None, escalated=r.get("escalated"),
        llm_calls=r.get("llm_calls", 0), latency_ms=r.get("latency_ms", 0), wait_ms=wait,
        net_latency_ms=max(0, r.get("latency_ms", 0) - wait),
        best_score=r.get("best_score"), context=(r.get("context") or "")[:24000],
        grounded=(r.get("grounding") or {}).get("ok"), fallback=bool(r.get("fallback")),
        abstained=abstained, correct=correct,
        extras=([] if abstained else extras(answer, item.get("a"), item["q"])),
        precision=(None if abstained else precision(answer, item.get("a"), item["q"])),
        recall=(None if not item.get("a") else 0.0 if abstained else recall(answer, item["a"])),
        # forced graph mode on a question the graph cannot answer: saying "not found" is also fine
        acceptable=correct or (base == "graph" and bool(exp) and "graph" not in exp and abstained),
        route_ok=route_ok, route_ideal=(decided == exp[0]) if route_ok is not None else None)
    return row


# ======================= report =======================
def summarize(rows: list[dict], mode: str) -> list[str]:
    base, k = split_mode(mode)
    rows = [r for r in rows if r["mode"] == mode]
    if not rows:
        return []
    out = [f"## Mode: {mode}" + (f" (vector top-k forced to {k})" if k else ""), ""]
    errors = [r for r in rows if r.get("error")]
    n = len(rows)
    out += [f"- Answer correctness: **{pct(sum(r['correct'] for r in rows), n)}**"
            + (f" ({len(errors)} errors counted as wrong)" if errors else ""),
            f"- Mean recall (share of the expected items found in the answer): "
            f"**{mean_pct([r.get('recall') for r in rows])}**"]

    out.append(f"- Mean precision (expected items found / items listed; extra names lower it): "
               f"**{mean_pct([r.get('precision') for r in rows])}**; "
               f"answers listing extra names: {sum(bool(r.get('extras')) for r in rows)}")

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    out += ["", "| Question type | Correct | Mean recall |", "|---|---|---|"]
    out += [f"| {t} | {pct(sum(x['correct'] for x in v), len(v))} | {mean_pct([x.get('recall') for x in v])} |"
            for t, v in sorted(by_type.items())]

    multi = [r for r in rows if (r.get("hops") or 0) >= 2]
    one = [r for r in rows if r.get("hops") == 1 and r["type"] != "guard"]
    out += ["", f"- Multi-hop questions (2+ hops): {pct(sum(r['correct'] for r in multi), len(multi))}, "
                f"recall {mean_pct([r.get('recall') for r in multi])}",
            f"- Single-fact questions: {pct(sum(r['correct'] for r in one), len(one))}"]

    if base == "auto":
        judged = [r for r in rows if r.get("route_ok") is not None]
        out += ["", f"- Router accuracy (acceptable route): **{pct(sum(r['route_ok'] for r in judged), len(judged))}**",
                f"- Router accuracy (ideal route): {pct(sum(bool(r['route_ideal']) for r in judged), len(judged))}"]
        by_via = defaultdict(list)
        for r in judged:
            by_via[r.get("via") or "guard"].append(r["route_ok"])
        out += ["", "| Decided by | Questions | Router accuracy |", "|---|---|---|"]
        out += [f"| {v} | {len(x)} | {pct(sum(x), len(x))} |" for v, x in sorted(by_via.items())]
        conf = Counter((r["expected_route"][0], r.get("router_route") or "none") for r in judged)
        out += ["", "| Ideal route -> chosen | Count |", "|---|---|"]
        out += [f"| {a} -> {b} | {c} |" for (a, b), c in sorted(conf.items())]
        misses = [r for r in judged if not r["route_ok"]]
        if misses:
            out += ["", "Router misses:", ""]
            out += [f"- {r['id']} `{r['q']}` -> {r.get('router_route')} ({r.get('via')}: {r.get('reason')}); wanted {r['expected_route']}"
                    for r in misses]

        esc = [r for r in rows if r.get("escalated")]
        if esc:
            out += ["", f"- Escalated graph -> hybrid after 'not found': {len(esc)} questions, "
                        f"{pct(sum(r['correct'] for r in esc), len(esc))} then answered correctly "
                        f"({', '.join(r['id'] for r in esc)})"]

    if base == "graph":
        gc = [r for r in rows if "graph" in r["expected_route"]]
        other = [r for r in rows if r["type"] != "guard" and "graph" not in r["expected_route"]]
        wrong = [r for r in other if not r["correct"] and not r.get("abstained")]
        out += ["", "Graph mode is only expected to answer relationship questions; for the rest the "
                    "right behaviour is to say 'not found' (the graph has no dates, amounts or descriptions).",
                f"- Graph-capable questions: **{pct(sum(r['correct'] for r in gc), len(gc))}**, "
                f"recall {mean_pct([r.get('recall') for r in gc])}",
                f"- Other questions ({len(other)}): answered correctly {sum(r['correct'] for r in other)}, "
                f"abstained {sum(bool(r.get('abstained')) and not r['correct'] for r in other)}, "
                f"confidently wrong {len(wrong)}"
                + (f" ({', '.join(r['id'] for r in wrong)})" if wrong else ""),
                f"- Acceptable (correct, or abstained where the graph cannot know): "
                f"{pct(sum(r.get('acceptable', r['correct']) for r in rows), n)}"]

    # LLM calls saved by guardrails. Baseline = plain RAG, one LLM call per question.
    guard_saved = [r for r in rows if not r.get("error") and r.get("llm_calls", 0) == 0]
    calls = sum(r.get("llm_calls", 0) for r in rows)
    kinds = Counter(r.get("guard") or "none" for r in guard_saved)
    out += ["", f"- LLM calls used: **{calls}** for {n} questions (plain RAG baseline: {n}); "
                f"questions answered with 0 LLM calls: **{pct(len(guard_saved), n)}**",
            "- Zero-call breakdown: " + (", ".join(f"{k_}: {v}" for k_, v in sorted(kinds.items())) or "none")]
    hallucination = [r for r in rows if r["type"] == "guard" and r.get("answer") is not None
                     and not r["correct"] and not r.get("guard")]
    out += [f"- Ungrounded-answer flags: {sum(r.get('grounded') is False for r in rows)}; "
            f"fallback model used on {sum(bool(r.get('fallback')) for r in rows)} questions"]
    if hallucination:
        out += ["- Unanswerable questions that got an answer instead of 'not found': "
                + ", ".join(r["id"] for r in hallucination)]

    by_route = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            by_route[r.get("route") or "none"].append(r)
    waits = [r.get("wait_ms", 0) for r in rows if not r.get("error")]
    out += ["", f"Latency excludes time spent waiting for the Gemini free-tier rate limiter "
                f"(average wait {statistics.mean(waits) if waits else 0:.0f} ms per question).", "",
            "| Route taken | Questions | Correct | Avg LLM calls | Avg latency | p50 | p95 |",
            "|---|---|---|---|---|---|---|"]
    for route, rs in sorted(by_route.items()):
        lat = [r.get("net_latency_ms", r["latency_ms"]) for r in rs]
        out.append(f"| {route} | {len(rs)} | {pct(sum(r['correct'] for r in rs), len(rs))} | "
                   f"{statistics.mean(r['llm_calls'] for r in rs):.2f} | {statistics.mean(lat):.0f} ms | "
                   f"{pctl(lat, .5):.0f} ms | {pctl(lat, .95):.0f} ms |")
    wrong = [r for r in rows if not r.get("acceptable", r["correct"])]
    if wrong:
        out += ["", "Wrong or failed answers (recall in brackets):", ""]
        for r in wrong:
            what = f"ERROR {r['error']}" if r.get("error") else (r.get("answer") or "")[:140].replace("\n", " ")
            rc = r.get("recall")
            ex = f" EXTRA: {', '.join(r['extras'])}" if r.get("extras") else ""
            out.append(f"- {r['id']} `{r['q']}` [{'-' if rc is None else f'{100 * rc:.0f}%'}]{ex} -> {what}")
    return out + [""]


def compare_report(rows: list[dict], modes: list[str]) -> list[str]:
    """Modes side by side on the same questions: correct / mean recall per question type."""
    modes = [m for m in modes if any(r["mode"] == m for r in rows)]
    if len(modes) < 2:
        return []
    by = {m: {r["id"]: r for r in rows if r["mode"] == m} for m in modes}
    ids = [i for i in by[modes[0]] if all(i in by[m] for m in modes)]
    qs = [by[modes[0]][i] for i in ids if by[modes[0]][i]["type"] != "guard"]

    def cell(m, sel):
        rs = [by[m][q["id"]] for q in sel]
        return f"{sum(r['correct'] for r in rs)}/{len(rs)} correct, recall {mean_pct([r.get('recall') for r in rs])}"

    sets = []
    for t in sorted({q["type"] for q in qs}):
        sets.append((f"type: {t}", [q for q in qs if q["type"] == t]))
    sets.append(("graph-capable questions (a graph can answer them)", [q for q in qs if "graph" in q["expected_route"]]))
    sets.append(("multi-hop (2+ hops)", [q for q in qs if (q.get("hops") or 0) >= 2]))
    out = ["## Modes side by side", "",
           "Graph mode is expected to say 'not found' on questions needing dates, amounts or descriptions, "
           "so judge it on the graph-capable rows.", "",
           "| Question set | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for name, sel in sets:
        if sel:
            out.append(f"| {name} ({len(sel)}) | " + " | ".join(cell(m, sel) for m in modes) + " |")
    return out + [""]


def evidence_report(items: list[dict], default_k: int) -> list[str]:
    """Structural facts about the questions (not model results): how spread out the evidence is."""
    spread = [q for q in items if q.get("facts")]
    if not spread:
        return []
    out = ["## Evidence spread (independent of any model)", "",
           f"How many documents hold the facts a question needs, and the MOST a perfect retriever could cover "
           f"with 2 or {default_k} documents. Where best-case coverage is far below 100%, vector retrieval "
           f"cannot answer completely however good the embeddings are.", "",
           f"| Type | Questions | Avg facts | Avg documents needed | Best-case coverage, top-2 | "
           f"Best-case coverage, top-{default_k} |", "|---|---|---|---|---|---|"]
    for t in sorted({q["type"] for q in spread}):
        g = [q for q in spread if q["type"] == t]
        out.append(f"| {t} | {len(g)} | {statistics.mean(q['facts'] for q in g):.1f} | "
                   f"{statistics.mean(q['docs_needed'] for q in g):.1f} | "
                   f"{statistics.mean(q['cover_k2'] for q in g):.0f}% | "
                   f"{statistics.mean(q['cover_k4'] for q in g):.0f}% |")
    return out + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=str(HERE / "questions.yaml"), help="one or more files, comma separated")
    ap.add_argument("--modes", default="auto", help="comma list of auto,vector,graph,hybrid; add @k for top-k (vector@2)")
    ap.add_argument("--ids", default="", help="comma list of question ids")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="override a setting for this run, e.g. --env GRAPH_MAX_EDGES=80 (repeatable)")
    ap.add_argument("--out", default="", help="output folder (default eval/results/<timestamp>)")
    ap.add_argument("--resume", default="", help="existing output folder: skip finished (mode,id) pairs")
    ap.add_argument("--use-cache", action="store_true", help="leave the answer cache on (skews calls/latency)")
    ap.add_argument("--stop-after-errors", type=int, default=5, help="abort after N errors in a row (quota)")
    args = ap.parse_args()

    for kv in args.env:
        key, sep, value = kv.partition("=")
        if not sep or not key:
            ap.error(f"--env expects KEY=VALUE, got {kv!r}")
        os.environ[key.strip()] = value.strip()
    if not args.use_cache:
        os.environ["CACHE_ENABLED"] = "false"
    sys.path.insert(0, str(HERE.parent))
    from app.config import get_config, get_settings
    from app.pipeline import guarded_ask

    not_found_text = get_config()["guardrails"]["canned"]["not_found"]
    files = [Path(f.strip()) for f in args.questions.split(",") if f.strip()]
    items = [q for f in files for q in yaml.safe_load(f.read_text(encoding="utf-8"))]
    ids = [q["id"] for q in items]
    if len(ids) != len(set(ids)):
        ap.error("question ids are not unique across the files")
    if args.ids:
        want = {i.strip() for i in args.ids.split(",")}
        items = [q for q in items if q["id"] in want]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    out_dir = Path(args.resume or args.out or HERE / "results" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "results.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    done = {(r["mode"], r["id"]) for r in rows if not r.get("error")}
    rows = [r for r in rows if (r["mode"], r["id"]) in done]  # failed rows are retried on resume

    streak = 0
    # early-exit guard questions (small talk, off-topic, injection) only make sense for the auto pipeline
    todo = [(m, q) for m in modes for q in items
            if (m, q["id"]) not in done and not (split_mode(m)[0] != "auto" and q.get("guard"))]
    print(f"{len(todo)} runs to do, results in {out_dir}", flush=True)
    with log.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        for n, (mode, q) in enumerate(todo, 1):
            row = run_one(guarded_ask, q, mode, not_found_text)
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            mark = "ERR" if row.get("error") else ("ok " if row["correct"] else "BAD")
            rc = row.get("recall")
            print(f"[{n}/{len(todo)}] {mark} {mode:9} {q['id']} -> {row.get('route')} "
                  f"{row.get('llm_calls')} calls {row['latency_ms']} ms"
                  + ("" if rc is None else f" recall {100 * rc:.0f}%"), flush=True)
            streak = streak + 1 if row.get("error") else 0
            if streak >= args.stop_after_errors:
                print(f"Stopping: {streak} errors in a row ({row['error']}). Re-run with --resume {out_dir}")
                break

    s = get_settings()
    report = [f"# Evaluation report ({datetime.now():%Y-%m-%d %H:%M})", "",
              f"Questions: {', '.join(f.name for f in files)}; cache {'on' if args.use_cache else 'off'}",
              f"Settings: top_k_vector={s.top_k_vector}, graph_max_hops={s.graph_max_hops}, "
              f"graph_max_edges={s.graph_max_edges}, max_context_tokens={s.max_context_tokens}, "
              f"min_relevance={s.min_relevance}" + (f"; overrides: {', '.join(args.env)}" if args.env else ""), ""]
    report += evidence_report(items, s.top_k_vector)
    for m in modes:
        report += summarize(rows, m)
    report += compare_report(rows, modes)
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"Saved {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
