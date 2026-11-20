import itertools
import re
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from . import schema
from .config import get_config, get_settings
from .llm import get_llm
from .router import link_entities
from .stores import (
    get_vector_store,
    graph_count,
    graph_list,
    graph_neighborhood,
    graph_shortest_path,
)

def _override(name: str, default: str) -> str:
    """config.yaml -> prompts.<name> replaces the built-in text (leave it out or empty to keep the default)."""
    try:
        return (get_config().get("prompts") or {}).get(name) or default
    except Exception:
        return default


VECTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            _override("vector",
                      "Answer strictly from the provided context. Be concise. "
                      "Cite sources in square brackets like [file.txt]. "
                      "If the context does not contain the answer, reply exactly: {not_found}"),
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

_GRAPH_RULES_DEFAULT = (
    "Each fact reads 'A <relation> B', for example 'A works at B' or 'A founded B', and may be "
    "followed by a quote from the document in double quotes: the quote carries the exact role or date, "
    "so use it. 'works at' also covers executives and board members. '(past)' means the fact is no "
    "longer current. Use every relevant fact, including indirect ones reached through other entities, "
    "and say how they connect. The 'Summary by entity' lines list every fact about one entity in both directions: use them to join facts across entities. A fact applies only to the entities it names: never transfer an investment, "
    "founding or role to an acquirer or acquired company. For list questions, check every candidate "
    "against each condition and include all that qualify, none that do not. "
    "A line such as 'Acme Corp has an event in the year 2010 - \"quote\"' means the event in the quote happened "
    "in that year: for a question about a year, list the events described in those quotes. "
    "Write normal sentences. Cite documents in square brackets."
)
GRAPH_RULES = _override("graph_rules", _GRAPH_RULES_DEFAULT)

GRAPH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer strictly from the knowledge-graph facts. Be concise. " + GRAPH_RULES +
            " If the facts do not answer the question, reply exactly: {not_found}",
        ),
        ("human", "Facts:\n{context}\n\nQuestion: {question}"),
    ]
)

HYBRID_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer strictly from the knowledge-graph facts and document excerpts. Be concise. "
            + GRAPH_RULES +
            " If neither contains the answer, reply exactly: {not_found}",
        ),
        ("human", "{context}\n\nQuestion: {question}"),
    ]
)

_LABELS = {"people": "Person", "persons": "Person", "companies": "Company",
           "locations": "Location", "cities": "Location", "places": "Location"}
_COUNT_RE = re.compile(r"\bhow many\b.*?\b(people|persons|companies|locations|cities|places)\b", re.I)
_LIST_RE = re.compile(
    r"\b(?:list|show|name|give|who are|what are)\b.*?\b(?:all|every)\b.*?"
    r"\b(people|persons|companies|locations|cities|places)\b", re.I)


# ======================= shared helpers =======================
def _result(route, answer, gated, calls, t0, best=None, sources=None, **extra) -> dict:
    return {
        "route": route,
        "answer": answer,
        "gated": gated,
        "llm_calls": calls,
        "best_score": best,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "sources": sources or [],
        **extra,
    }


def _generate(prompt, context: str, question: str) -> str:
    not_found = get_config()["guardrails"]["canned"]["not_found"]
    chain = prompt | get_llm("answer") | StrOutputParser()
    return chain.invoke({"context": context, "question": question, "not_found": not_found}).strip()


def _not_found() -> str:
    return get_config()["guardrails"]["canned"]["not_found"]


# ======================= vector RAG =======================
def retrieve(question: str, k: int | None = None) -> list[dict]:
    """Top-k chunks with cosine similarity (1 = identical)."""
    k = k or get_settings().top_k_vector
    hits = get_vector_store().similarity_search_with_score(question, k=k)
    return [
        {
            "text": d.page_content,
            "source": d.metadata.get("source", "?"),
            "chunk": d.metadata.get("chunk"),
            "score": round(1 - dist, 4),
        }
        for d, dist in hits
    ]


def build_context(hits: list[dict], max_tokens: int) -> str:
    budget = max_tokens * 4  # ~4 chars per token
    parts, used = [], 0
    for h in hits:
        block = f"[{h['source']}]\n{h['text']}"
        if parts and used + len(block) > budget:
            break
        block = block[:budget]
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _vec_sources(hits: list[dict]) -> list[dict]:
    return [
        {"source": h["source"], "chunk": h["chunk"], "score": h["score"], "snippet": h["text"][:200]}
        for h in hits
    ]


def vector_rag(question: str, k: int | None = None) -> dict:
    t0 = time.perf_counter()
    s = get_settings()

    raw = retrieve(question, k)
    best = raw[0]["score"] if raw else 0.0
    hits = [h for h in raw if h["score"] >= s.min_relevance]

    if not hits:  # relevance gate: no LLM call
        return _result("vector", _not_found(), True, 0, t0, best, context="")

    context = build_context(hits, s.max_context_tokens)
    answer = _generate(VECTOR_PROMPT, context, question)
    return _result("vector", answer, False, 1, t0, best, _vec_sources(hits), context=context)


# ======================= GraphRAG =======================
def _empty_graph() -> dict:
    return {"entities": [], "edges": [], "paths": []}


def gather_graph(ents: list[dict]) -> dict:
    """Cypher templates: k-hop neighbourhood of linked entities + shortest paths between pairs.
    Year-like nodes (EXTRA_NODE_TYPES) are walked separately: a normal entity gets its own dated events
    (1 hop), a Year node gets the events tied to it (EXTRA_NODE_HOPS). Walks from normal entities never pass
    THROUGH a Year, so two unrelated companies are not joined just because both have an event in 2010."""
    s = get_settings()
    use = ents[:s.link_max_entities]
    extra = set(schema.extra_labels())
    base = [e for e in use if e["label"] not in extra]
    top = [e for e in use if e["label"] in extra]
    edges: list[dict] = []
    if base:
        ids = [e["id"] for e in base]
        edges += graph_neighborhood(ids, s.graph_max_hops, s.graph_max_edges, "base" if extra else "all")
        if extra:
            edges += graph_neighborhood(ids, 1, s.graph_max_edges, "extra")
    if top:
        edges += graph_neighborhood([e["id"] for e in top], s.extra_node_hops, s.graph_max_edges, "all")
    seen, uniq = set(), []
    for e in edges:  # the same edge can come back from two walks
        k = (e["source"], e["rel"], e["target"], e.get("evidence", ""))
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    paths = []
    for a, b in itertools.combinations(use, 2):
        kind = "all" if (a["label"] in extra or b["label"] in extra or not extra) else "base"
        p = graph_shortest_path(a["id"], b["id"], None, kind)
        if p:
            paths.append({"from": a["name"], "to": b["name"], "edges": p})
    return {"entities": [e["name"] for e in use], "edges": uniq[:s.graph_max_edges * 2], "paths": paths}


REL_TEXT = {
    "WORKS_AT": "works at", "CEO_OF": "is CEO of", "FOUNDED": "founded",
    "INVESTED_IN": "invested in", "ACQUIRED": "acquired", "PARTNERED_WITH": "partnered with",
    "LOCATED_IN": "is located in",
}


def _rel_text(rel: str) -> str:
    return REL_TEXT.get(rel) or schema.rel_texts().get(rel) or rel.lower().replace("_", " ")


def _fact(e: dict, evidence: bool = False) -> str:
    """One edge as plain English: 'Rohan Kapoor works at Meridian Capital (past) - "..."'.
    Relation names such as WORKS_AT never reach the model: it would echo them back."""
    text = _rel_text(e["rel"])
    line = f"{e['source']} {text} {e['target']}"
    if e.get("status") == "past":
        line += " (past)"
    if evidence and e.get("evidence"):
        ev = e["evidence"] if isinstance(e["evidence"], str) else " / ".join(e["evidence"])
        line += f' - "{ev}"'
    return line


def _edge_line(e: dict) -> str:
    line = _fact(e, evidence=True)
    if e.get("docs"):
        line += f" [{', '.join(e['docs'])}]"
    return line


IN_TEXT = {  # the same edge seen from the target's side
    "WORKS_AT": "people who work there", "CEO_OF": "CEO", "FOUNDED": "founded by",
    "INVESTED_IN": "invested in by", "ACQUIRED": "acquired by", "PARTNERED_WITH": "partnered with",
}


def entity_blocks(edges: list[dict], first: list[str]) -> list[str]:
    """One line per entity with ALL its facts, both directions, so a multi-hop join
    (alumnus -> founded company -> investor) sits in two short lines instead of 80 scattered ones."""
    out: dict[str, dict[str, list[str]]] = {}

    def add(ent: str, phrase: str, other: str, past: bool):
        vals = out.setdefault(ent, {}).setdefault(phrase, [])
        item = other + (" (past)" if past else "")
        if item not in vals:
            vals.append(item)

    for e in edges:
        past = e.get("status") == "past"
        add(e["source"], _rel_text(e["rel"]), e["target"], past)
        back = IN_TEXT.get(e["rel"]) or schema.in_texts().get(e["rel"])
        if back:
            add(e["target"], back, e["source"], past)
    order = sorted(out, key=lambda n: (n not in first, n))
    return [f"{n}: " + "; ".join(f"{ph} {', '.join(v)}" for ph, v in out[n].items()) for n in order]


def format_graph_context(g: dict, max_tokens: int) -> str:
    """Path lines, then one summary line per entity, then every fact with its quote
    (linked-entity edges first), all capped by the context budget."""
    if not g["edges"] and not g["paths"]:
        return ""
    budget = max_tokens * 4
    linked = set(g["entities"])
    edges = sorted(g["edges"], key=lambda e: 0 if (e["source"] in linked or e["target"] in linked) else 1)
    lines = []
    for p in g["paths"]:
        chain = "; ".join(_fact(x) for x in p["edges"])
        lines.append(f"Path {p['from']} to {p['to']}: {chain}")
    blocks = entity_blocks(edges, list(linked))
    if blocks:
        lines.append("Summary by entity (all known facts about each):")
        lines += blocks
        lines.append("Facts with quotes:")
    lines += [_edge_line(e) for e in edges]
    out, used = [], 0
    for line in lines:
        if out and used + len(line) + 1 > budget:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def _graph_docs(g: dict) -> list[dict]:
    docs = sorted({d for e in g["edges"] for d in e.get("docs", [])})
    return [{"source": d} for d in docs]


def try_rule_answer(question: str, ents: list[dict] | None = None) -> str | None:
    """Count/list questions with no entity named: answered by Cypher, no LLM (any route)."""
    m_count, m_list = _COUNT_RE.search(question), _LIST_RE.search(question)
    if not (m_count or m_list):
        return None
    if (link_entities(question) if ents is None else ents):
        return None  # "how many companies has Meridian invested in" needs real reasoning
    if m_count:
        label = _LABELS[m_count.group(1).lower()]
        return f"There are {graph_count(label)} {label.lower()} entities in the knowledge graph."
    label = _LABELS[m_list.group(1).lower()]
    names = graph_list(label)
    return ", ".join(names) if names else f"No {label.lower()} entities in the graph yet."


def graph_rag(question: str, ents: list[dict] | None = None) -> dict:
    t0 = time.perf_counter()
    s = get_settings()

    ents = link_entities(question) if ents is None else ents
    if not ents:
        return _result("graph", _not_found(), True, 0, t0, graph=_empty_graph(),
                       reason="no_entity_matched", context="")

    g = gather_graph(ents)
    context = format_graph_context(g, s.max_context_tokens)
    if not context:  # entity exists but has no relations
        return _result("graph", _not_found(), True, 0, t0, graph=g, reason="no_relations", context="")

    answer = _generate(GRAPH_PROMPT, context, question)
    return _result("graph", answer, False, 1, t0, sources=_graph_docs(g),
                   graph=g, reason="ok", context=context)


# ======================= hybrid =======================
def hybrid_rag(question: str, k: int | None = None, ents: list[dict] | None = None) -> dict:
    t0 = time.perf_counter()
    s = get_settings()

    ents = link_entities(question) if ents is None else ents
    g = gather_graph(ents) if ents else _empty_graph()
    raw = retrieve(question, k)
    best = raw[0]["score"] if raw else 0.0
    hits = [h for h in raw if h["score"] >= s.min_relevance]

    gtxt = format_graph_context(g, s.max_context_tokens if not hits else s.max_context_tokens // 2)
    vtxt = build_context(hits, s.max_context_tokens if not gtxt else s.max_context_tokens // 2)
    if not gtxt and not vtxt:
        return _result("hybrid", _not_found(), True, 0, t0, best, graph=g, reason="nothing_retrieved", context="")

    parts = []
    if gtxt:
        parts.append("Knowledge-graph facts:\n" + gtxt)
    if vtxt:
        parts.append("Document excerpts:\n" + vtxt)
    context = "\n\n".join(parts)
    answer = _generate(HYBRID_PROMPT, context, question)
    return _result("hybrid", answer, False, 1, t0, best,
                   sources=_vec_sources(hits) + _graph_docs(g), graph=g, reason="ok", context=context)


# ======================= dispatch =======================
def answer_question(question: str, route: str = "vector", k: int | None = None,
                    ents: list[dict] | None = None) -> dict:
    """Run ONE concrete route ('auto' is resolved earlier, in pipeline.run)."""
    if route == "vector":
        return vector_rag(question, k)
    if route == "graph":
        return graph_rag(question, ents)
    if route == "hybrid":
        return hybrid_rag(question, k, ents)
    raise ValueError(f"unknown route {route!r}")
