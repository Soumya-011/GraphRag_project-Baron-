import hashlib
import json
import re
import sqlite3
import unicodedata
from contextlib import closing

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from . import schema
from .config import ROOT, get_config, get_settings
from .llm import (
    describe_llm_error,
    fallback_events,
    reset_fallback_events,
    structured_pair,
    with_fallback,
)

PROMPT_VERSION = "v3"  # bump to invalidate the extraction cache (v3: Gemini-only)
CACHE_PATH = ROOT / "data" / "extract_cache.sqlite"

TYPE_ALIASES = {
    "person": "Person", "people": "Person", "human": "Person",
    "company": "Company", "organization": "Company", "organisation": "Company",
    "org": "Company", "corporation": "Company", "firm": "Company",
    "location": "Location", "place": "Location", "city": "Location",
    "country": "Location", "gpe": "Location",
}

SYSTEM = """You extract facts from text as relation triples for a knowledge graph.
Rules:
- Use only facts explicitly stated in the text. Never invent or guess.
- Output one item in "relations" for every fact that matches an allowed relation. Be exhaustive: a text with several facts must produce several relations.
- source and target: full names exactly as written in the text. source_type and target_type: one of {types}.
- relation: exactly one of these (source -> target):
{relations}
- status: "past" if the fact ended or was before; otherwise "current". Ignore planned or future events.
- evidence: a short phrase copied from the text.
- If the text has no such facts, return an empty list.{extra_rules}
Example text: "Jane Roe, CEO of Acme Corp in Austin, said Acme acquired Zed Inc in 2020."
Example relations:
  Jane Roe (Person) CEO_OF Acme Corp (Company)
  Acme Corp (Company) LOCATED_IN Austin (Location)
  Acme Corp (Company) ACQUIRED Zed Inc (Company)"""


class Relation(BaseModel):
    source: str
    source_type: str = Field(description="entity type of the source, one of the allowed types")
    relation: str = Field(description="one of the allowed relation names")
    target: str
    target_type: str = Field(description="entity type of the target, one of the allowed types")
    status: str = Field(default="current", description="current | past")
    evidence: str = ""


class Extraction(BaseModel):
    relations: list[Relation]


class ExtractionError(RuntimeError):
    pass


# ---------- text helpers ----------
def norm(name: str) -> str:
    s = unicodedata.normalize("NFKC", name).casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^the ", "", s)


def _clean(name: str) -> str:
    return re.sub(r"\s+", " ", name or "").strip(" \t\n.,;:\"'")


def _grounded(name: str, chunk_tokens: set[str]) -> bool:
    toks = norm(name).split()
    return bool(toks) and all(t in chunk_tokens for t in toks)


def _label(raw: str) -> str:
    raw = (raw or "").strip()
    return TYPE_ALIASES.get(raw.casefold(), raw.title())


# ---------- prompt + cache ----------
def _rule_line(name: str, r: dict) -> str:
    src = " | ".join(schema.sources_of(r))
    return f"  - {name}: {src} -> {r['target']} ({r['hint']})"


def _prompt() -> ChatPromptTemplate:
    rules = schema.relation_rules()
    rels = "\n".join(_rule_line(n, r) for n, r in rules.items())
    extra = ""
    for sp in schema.extra_specs():
        extra += (f"\n- {sp['relation']}: the target is just the {sp['label']} value as written in the text "
                  f"(for example 2010 for Year). Output one {sp['relation']} relation for EACH separate fact or "
                  f"event that the text ties to a {sp['label']}, and copy the phrase describing that event "
                  f"into evidence.")
    base = (get_config().get("prompts") or {}).get("extraction") or SYSTEM
    system = (base.replace("{relations}", rels).replace("{types}", schema.type_names())
              .replace("{extra_rules}", extra))
    return ChatPromptTemplate.from_messages(
        [SystemMessage(content=system), ("human", "Text:\n{text}")]
    )


def _key(text: str) -> str:
    model = get_settings().model_for("extraction")
    rules = json.dumps(schema.relation_rules(), sort_keys=True)
    raw = f"{PROMPT_VERSION}|{model}|{rules}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_conn():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS ext (k TEXT PRIMARY KEY, v TEXT)")
    return con


def _cache_get(k: str) -> str | None:
    with closing(_cache_conn()) as con:
        row = con.execute("SELECT v FROM ext WHERE k=?", (k,)).fetchone()
    return row[0] if row else None


def _cache_put(k: str, ext: Extraction) -> None:
    with closing(_cache_conn()) as con:
        con.execute("INSERT OR REPLACE INTO ext VALUES (?, ?)", (k, ext.model_dump_json()))
        con.commit()


def _looks_factual(text: str) -> bool:
    """Two or more capitalised words that do not start a sentence: probably names and facts,
    so an EMPTY extraction is suspicious."""
    n = 0
    for sent in re.split(r"(?<=[.!?\n])\s+", text):
        n += sum(1 for w in sent.split()[1:] if w[:1].isupper())
    return n >= 2


def _plausible(out: Extraction, inp) -> bool:
    text = inp.get("text", "") if isinstance(inp, dict) else ""
    return bool(out.relations) or not _looks_factual(text)


def _extract_chunk(chain, text: str) -> tuple[Extraction, int]:
    """Returns (extraction, llm_calls_used). Cached results use 0 calls.
    Retries once, unless the fallback model was already tried (it would burn free quota)."""
    key = _key(text)
    hit = _cache_get(key)
    if hit:
        try:
            return Extraction.model_validate_json(hit), 0
        except Exception:
            pass
    last = None
    for attempt in (1, 2):
        try:
            out = chain.invoke({"text": text})
            if isinstance(out, Extraction):
                if out.relations or not _looks_factual(text):  # never cache a suspicious empty result
                    _cache_put(key, out)
                return out, attempt
            last = f"model returned {type(out).__name__}, not a structured result"
        except Exception as e:
            last = e
            if fallback_events():
                break
    raise ExtractionError(describe_llm_error(last) if last else "model returned no structured output")


# ---------- validation ----------
def validate(ext: Extraction, chunk_text: str) -> tuple[set, list[dict], int]:
    """Filter model output: whitelist, grounding in the chunk, type/direction check against the rules."""
    rules = schema.relation_rules()
    labels = set(schema.graph_labels())
    toks = set(norm(chunk_text).split())
    mentions: set = set()
    rels, seen = [], set()
    dropped = 0

    for r in ext.relations:
        rel = re.sub(r"[\s-]+", "_", r.relation.strip()).upper()
        rule = rules.get(rel)
        s, t = _clean(r.source), _clean(r.target)
        if (not rule or not s or not t or norm(s) == norm(t) or len(s) > 80 or len(t) > 80
                or not _grounded(s, toks) or not _grounded(t, toks)):
            dropped += 1
            continue
        multi = bool(rule.get("multi"))
        evidence = r.evidence.strip()
        if multi and not evidence:  # an event with no description is useless
            dropped += 1
            continue
        srcs, want_t = schema.sources_of(rule), rule["target"]
        sl, tl = _label(r.source_type), _label(r.target_type)
        if sl in labels and tl in labels:
            if sl in srcs and tl == want_t:
                want_s = sl
            elif tl in srcs and sl == want_t:  # reversed direction: fix it
                s, t = t, s
                want_s = tl
            else:
                dropped += 1
                continue
        elif len(srcs) == 1:  # unknown types: trust the rule
            want_s = srcs[0]
        elif sl in srcs:
            want_s = sl
        else:
            dropped += 1
            continue
        s, t = schema.canonical_value(want_s, s), schema.canonical_value(want_t, t)  # e.g. "March 2010" -> "2010"
        if not s or not t:
            dropped += 1
            continue
        sig = (norm(s), rel, norm(t), norm(evidence) if multi else "")
        if sig in seen:
            continue
        seen.add(sig)
        mentions.add((s, want_s))
        mentions.add((t, want_t))
        rels.append({
            "source": s, "source_label": want_s,
            "type": rel,
            "target": t, "target_label": want_t,
            "status": "past" if r.status.strip().lower() == "past" else "current",
            "evidence": evidence,
        })
    return mentions, rels, dropped


# ---------- entity resolution ----------
def _same(a: str, b: str, threshold: int) -> bool:
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if ta and tb and (ta <= tb or tb <= ta):  # "musk" within "elon musk"
        return True
    return fuzz.token_sort_ratio(a, b) >= threshold


class Resolver:
    """Maps surface names to canonical names, per label. Existing graph names win."""

    def __init__(self, threshold: int, aliases: dict, known: dict[str, list[str]]):
        self.threshold = threshold
        self.aliases = {norm(k): v for k, v in (aliases or {}).items()}
        self.canon = {label: list(v) for label, v in known.items()}
        self.exact = schema.exact_labels()  # e.g. Year: "2010" and "2011" must never merge

    def resolve(self, name: str, label: str) -> str:
        name = self.aliases.get(norm(name), name)
        n = norm(name)
        pool = self.canon.setdefault(label, [])
        for c in pool:
            if (n == norm(c)) if label in self.exact else _same(n, norm(c), self.threshold):
                return c
        pool.append(name)
        return name


def resolve_all(mentions: set, relations: list[dict], resolver: Resolver) -> tuple[list[dict], list[dict]]:
    mapping: dict[tuple[str, str], str] = {}
    # longest names first, so "Elon Musk" becomes canonical before "Musk"
    for name, label in sorted(mentions, key=lambda m: (-len(m[0]), m[0])):
        mapping.setdefault((norm(name), label), resolver.resolve(name, label))

    ents: dict[tuple[str, str], dict] = {}
    for (_, label), canon in mapping.items():
        ents[(norm(canon), label)] = {"id": norm(canon), "name": canon, "label": label}

    rels, seen = [], set()
    for r in relations:
        s = mapping[(norm(r["source"]), r["source_label"])]
        t = mapping[(norm(r["target"]), r["target_label"])]
        multi = schema.is_multi(r["type"])
        sig = (norm(s), r["type"], norm(t), norm(r["evidence"]) if multi else "")
        if norm(s) == norm(t) or sig in seen:
            continue
        seen.add(sig)
        rels.append({
            "s": norm(s), "t": norm(t), "type": r["type"],
            "sl": r["source_label"], "tl": r["target_label"],
            "status": r["status"], "evidence": r["evidence"][:300],
        })
    return list(ents.values()), rels


# ---------- document-level ----------
def extract_document(chunks: list[Document]) -> dict:
    primary, fallback = structured_pair(Extraction, "extraction")
    prompt = _prompt()
    chain = with_fallback(prompt | primary, (prompt | fallback) if fallback else None,
                          "extraction", get_settings().fallback_for("extraction") or "",
                          accept=_plausible)
    mentions: set = set()
    relations: list[dict] = []
    dropped = failed = calls = 0
    last_err = ""
    for c in chunks:
        reset_fallback_events()
        try:
            ext, n = _extract_chunk(chain, c.page_content)
        except ExtractionError as e:
            failed += 1
            calls += 1 + len(fallback_events())
            last_err = str(e)
            continue
        calls += n + len(fallback_events())
        m, r, d = validate(ext, c.page_content)
        mentions |= m
        relations += r
        dropped += d
    if chunks and failed == len(chunks):
        raise ExtractionError(f"all chunks failed: {last_err}")
    return {"mentions": mentions, "relations": relations,
            "dropped": dropped, "failed": failed, "llm_calls": calls}
