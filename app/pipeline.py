"""Guarded query pipeline, cheapest check first:
input -> small talk -> off-topic -> cache -> rule answer -> route (auto) -> RAG (gate inside)
-> grounding -> cache."""
import time
from concurrent.futures import ThreadPoolExecutor

from . import guardrails as gr
from . import rag
from .cache import cache_get, cache_put, record
from .config import get_config, get_settings
from .llm import fallback_events, limiter_wait_ms, reset_fallback_events
from .router import decide_route, link_entities

InputRejected = gr.InputRejected


def safe_link(question: str) -> list[dict]:
    try:
        return link_entities(question)
    except Exception:  # graph down: vector-only still works
        return []


def _canned(route: str, guard: str, key: str) -> dict:
    return {
        "route": route, "answer": get_config()["guardrails"]["canned"][key],
        "gated": True, "guard": guard, "cached": False, "llm_calls": 0,
        "best_score": None, "latency_ms": 0, "sources": [], "context": "", "grounding": None,
    }


def _not_found(result: dict) -> bool:
    canned = get_config()["guardrails"]["canned"]["not_found"].strip().casefold()
    return bool(result["gated"]) or result["answer"].strip().casefold() == canned


def _early_exit(q: str, route: str, blocked: bool) -> dict | None:
    if blocked:
        record("requests", "saved_blocked")
        return _canned(route, "blocked", "blocked")
    if gr.is_smalltalk(q):
        record("requests", "saved_smalltalk")
        return _canned(route, "smalltalk", "smalltalk")
    if gr.is_off_topic(q) and not safe_link(q):  # a known entity vetoes "off-topic"
        record("requests", "saved_off_topic")
        return _canned(route, "off_topic", "off_topic")
    return None


def run(question: str, route: str, k: int | None = None, ents: list[dict] | None = None) -> dict:
    """Cache -> rule answer -> (route) -> RAG -> grounding -> cache store.
    route is 'vector' | 'graph' | 'hybrid' | 'auto'. For 'auto' the cache is keyed on 'auto', so a
    repeated question skips the router call as well."""
    t0 = time.perf_counter()
    ents = safe_link(question) if ents is None else ents
    ids = [e["id"] for e in ents]

    hit = cache_get(question, route, k, ids)
    if hit:
        kind = hit.pop("_cache")
        hit.pop("_cache_score", None)
        hit.update(guard="cache", cached=kind, fallback=None, llm_calls_original=hit.get("llm_calls", 0),
                   llm_calls=0, latency_ms=int((time.perf_counter() - t0) * 1000))
        record("requests", "saved_cache")
        return hit

    try:
        rule = rag.try_rule_answer(question, ents)
    except Exception:
        rule = None
    if rule:
        record("requests", "saved_rule")
        return {"route": route, "answer": rule, "gated": False, "guard": "rule", "cached": False,
                "llm_calls": 0, "best_score": None, "sources": [], "context": "", "grounding": None,
                "latency_ms": int((time.perf_counter() - t0) * 1000)}

    reset_fallback_events()
    decision = decide_route(question, ents) if route == "auto" else None
    chosen = decision["route"] if decision else route

    result = rag.answer_question(question, chosen, k, ents)
    escalated = None
    if route == "auto" and chosen == "graph" and _not_found(result):
        # The graph holds relationships only (no dates, amounts, descriptions). When it has nothing,
        # retry once with documents too. Costs one extra call, only on these failures.
        first_calls = result["llm_calls"]
        result = rag.answer_question(question, "hybrid", k, ents)
        result["llm_calls"] += first_calls
        chosen, escalated = "hybrid", "graph->hybrid"
        record("escalated")
    result.update(route=chosen, router=decision, guard=None, cached=False, grounding=None,
                  escalated=escalated, fallback=fallback_events() or None,
                  llm_calls=result["llm_calls"] + (decision["llm_calls"] if decision else 0),
                  latency_ms=int((time.perf_counter() - t0) * 1000))
    result["wait_ms"] = limiter_wait_ms()  # part of latency_ms spent waiting for the rate limiter
    calls = result["llm_calls"]

    if result["gated"]:
        result["guard"] = "gate"
        record("requests", *([] if calls else ["saved_gate"]))
        record("llm_calls", n=calls)
        return result

    record("requests")
    record("llm_calls", n=calls)
    if get_settings().enable_grounding_check:
        result["grounding"] = gr.check_grounding(result["answer"], result.get("context", ""))
        if not result["grounding"]["ok"]:
            record("grounding_flags")
    if result["grounding"] is None or result["grounding"]["ok"]:  # never cache a flagged answer
        cache_put(question, route, k, ids, result)
    return result


def guarded_ask(question: str, route: str = "auto", k: int | None = None) -> dict:
    q, blocked = gr.check_input(question)  # raises InputRejected
    early = _early_exit(q, route, blocked)
    return early if early else run(q, route, k)


def guarded_compare(question: str, k: int | None = None) -> dict:
    """Normal RAG and GraphRAG side by side. Input guards run once for both."""
    q, blocked = gr.check_input(question)
    early = _early_exit(q, "vector", blocked)
    if early:
        return {"question": q, "vector": early, "graph": {**early, "route": "graph"}}
    ents = safe_link(q)
    try:  # build the lazy singletons in this thread before the two workers start
        from .llm import get_embeddings
        from .stores import get_graph, get_vector_store

        get_embeddings(), get_vector_store(), get_graph()
    except Exception:
        pass
    with ThreadPoolExecutor(max_workers=2) as ex:
        fv = ex.submit(run, q, "vector", k, ents)
        fg = ex.submit(run, q, "graph", k, ents)
        return {"question": q, "vector": fv.result(), "graph": fg.result()}
