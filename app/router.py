"""Entity linker (Phase 4) and query router (Phase 6).

Router order, cheapest first, every decision returns a human-readable reason:
  1. no known entity in the question      -> vector (the graph cannot help)
  2. keyword cues from config.yaml        -> graph / vector / hybrid
  3. two or more linked entities          -> graph (a relationship question)
  4. one entity and no cue (ambiguous)    -> router LLM (one small call), else DEFAULT_ROUTE
"""
import re
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from rapidfuzz import fuzz

from . import schema
from .config import get_config, get_settings
from .extract import norm
from .stores import graph_entities

# Words that appear inside company names but are too generic to identify one on their own.
# Built-in list; config.yaml -> router.generic_words adds more (no code change).
_GENERIC_DEFAULT = {
    "capital", "labs", "lab", "power", "energy", "systems", "system", "group", "tech",
    "technologies", "ventures", "partners", "holdings", "company", "corp", "inc", "ltd",
    "limited", "solutions", "services", "global", "international",
}


def _generic() -> set[str]:
    extra = (get_config().get("router") or {}).get("generic_words") or []
    return _GENERIC_DEFAULT | {norm(w) for w in extra}


# ======================= entity linker =======================
def link_entities(question: str, entities: list[dict] | None = None,
                  threshold: int | None = None, limit: int | None = None) -> list[dict]:
    """Find graph entities mentioned in a question. No LLM.

    Match order per entity: exact phrase (100) -> fuzzy n-gram (>= threshold)
    -> a capitalised single word that identifies exactly one multi-word entity (90),
    e.g. "Priya" -> "Priya Nair". Returns [{id, name, label, score}] best first.
    """
    ents = graph_entities() if entities is None else entities
    if not ents:
        return []
    thr = threshold or get_settings().entity_fuzzy_threshold

    words = re.findall(r"\w+", question)
    toks = [norm(w) for w in words]
    caps = [w[0].isupper() for w in words]
    padded = " " + " ".join(toks) + " "

    token_owners: dict[str, set[str]] = {}
    for e in ents:
        for t in e["id"].split():
            token_owners.setdefault(t, set()).add(e["id"])

    generic = _generic()
    found = []
    for e in ents:
        eid = e["id"]
        etoks = eid.split()
        best, pos = 0, 0
        if f" {eid} " in padded:
            best, pos = 100, padded.find(f" {eid} ")
        elif len(eid) >= 4 and toks:
            n = len(etoks)
            for i in range(len(toks) - n + 1):
                score = fuzz.ratio(" ".join(toks[i:i + n]), eid)
                if score >= thr and score > best:
                    best, pos = score, i
        if best < thr and len(etoks) > 1:
            for i, t in enumerate(toks):
                if (caps[i] and len(t) >= 4 and t in etoks and t not in generic
                        and len(token_owners.get(t, ())) == 1):
                    best, pos = 90, i
                    break
        if best >= thr:
            found.append({"id": eid, "name": e["name"], "label": e["label"],
                          "score": round(best), "_pos": pos})

    found.sort(key=lambda x: (-x["score"], x["_pos"]))
    for f in found:
        f.pop("_pos")
    return found[:limit or get_settings().link_max_entities]


# ======================= router =======================
class RouteChoice(BaseModel):
    route: Literal["graph", "vector", "hybrid"]
    reason: str = ""


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Choose the retrieval route for a question about companies, people and locations.\n"
     "- graph: relationships between entities (who founded, invested in, acquired, works at, "
     "is CEO of; how X connects to Y; multi-hop chains).\n"
     "- vector: details found in document text (dates, amounts, products, descriptions, summaries).\n"
     "- hybrid: needs both, or you are unsure.\n"
     "Give a reason of at most 12 words."),
    ("human", "Question: {question}\nEntities found in it: {entities}"),
])


def _cue(padded: str, cues: list[str]) -> str | None:
    """First cue phrase found as whole words in the normalised, space-padded question."""
    for c in cues:
        n = norm(c)
        if n and f" {n} " in padded:
            return c
    return None


def _decision(route: str, via: str, reason: str, llm_calls: int = 0) -> dict:
    return {"route": route, "via": via, "reason": reason, "llm_calls": llm_calls}


def decide_route(question: str, ents: list[dict], use_llm: bool | None = None) -> dict:
    """-> {route, via: entity|cue|llm|default, reason, llm_calls}. Never raises."""
    s = get_settings()
    cfg = get_config().get("router", {})
    use_llm = s.router_use_llm_fallback if use_llm is None else use_llm

    if not ents:
        return _decision("vector", "entity", "no known entity in the question, so the graph cannot help")

    padded = " " + norm(question) + " "
    g = _cue(padded, cfg.get("graph_cues", []))
    v = _cue(padded, cfg.get("vector_cues", []))
    if g and v:
        return _decision("hybrid", "cue", f"relationship cue '{g}' and document cue '{v}'")
    if v:
        return _decision("vector", "cue", f"document cue '{v}'")
    if g:
        return _decision("graph", "cue", f"relationship cue '{g}'")
    extra = set(schema.extra_labels())
    if extra and any(e["label"] in extra for e in ents):
        names = ", ".join(e["name"] for e in ents if e["label"] in extra)
        return _decision("graph", "entity", f"{names} is a {'/'.join(sorted(extra)).lower()} node: look it up in the graph")
    if len(ents) >= 2:
        names = " and ".join(e["name"] for e in ents[:2])
        return _decision("graph", "entity", f"{names} are both in the graph: a relationship question")

    if use_llm:  # one entity, no cue: genuinely ambiguous
        try:
            from .llm import get_structured_llm

            chain = ROUTER_PROMPT | get_structured_llm(RouteChoice, "router", max_tokens=128)
            out = chain.invoke({"question": question, "entities": ", ".join(e["name"] for e in ents)})
            if isinstance(out, RouteChoice):
                return _decision(out.route, "llm", out.reason.strip() or "router model choice", 1)
        except Exception:
            pass
        return _decision(s.default_route, "default", "router model unavailable, used the default route")
    return _decision(s.default_route, "default", "ambiguous question, used the default route")
