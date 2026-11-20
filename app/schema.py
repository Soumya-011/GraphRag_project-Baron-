"""Node types and relation rules in ONE place.

Base types and relation rules come from config.yaml. Extra node types (Year, Industry, ...) are switched
on in .env with EXTRA_NODE_TYPES=Year,Industry and need no code change:

* every extra type gets a relation rule generated here (default name IN_<TYPE>, source = every base type);
* config.yaml -> extra_node_types.<Type> can override the relation name, the allowed source types, the
  hint, a value pattern (e.g. a 4-digit year), the wording used in answers, and so on.

Everything that used to read config["relation_rules"] or a fixed label list now calls this module.
"""
import re

from .config import get_config, get_settings

_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9]{0,29}$")
_REL_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,39}$")


def base_labels() -> list[str]:
    return [x for x in get_config()["node_labels"] if x != "Document"]


def extra_labels() -> list[str]:
    """Labels listed in EXTRA_NODE_TYPES (validated: they go into Cypher text)."""
    base = set(base_labels())
    out: list[str] = []
    for part in re.split(r"[,;\s]+", get_settings().extra_node_types or ""):
        if not part:
            continue
        lab = part[0].upper() + part[1:]
        if not _LABEL_RE.match(lab):
            raise ValueError(f"EXTRA_NODE_TYPES: {part!r} is not a valid node label "
                             "(letters and digits, starts with a letter, at most 30 characters)")
        if lab == "Document":
            raise ValueError("EXTRA_NODE_TYPES: 'Document' is reserved")
        if lab not in base and lab not in out:
            out.append(lab)
    return out


def graph_labels() -> tuple[str, ...]:
    return tuple(base_labels() + extra_labels())


def node_filter(var: str = "n") -> str:
    """'(n:Person OR n:Company OR ...)' for Cypher WHERE clauses. Labels are validated above."""
    return "(" + " OR ".join(f"{var}:{lab}" for lab in graph_labels()) + ")"


def type_names() -> str:
    return ", ".join(graph_labels())


def extra_spec(label: str) -> dict:
    cat = (get_config().get("extra_node_types") or {}).get(label) or {}
    rel = cat.get("relation") or f"IN_{label.upper()}"
    if not _REL_RE.match(rel):
        raise ValueError(f"extra_node_types.{label}.relation {rel!r} must look like IN_YEAR")
    src = cat.get("from") or base_labels()
    src = [src] if isinstance(src, str) else list(src)
    return {
        "label": label,
        "relation": rel,
        "from": src,
        "hint": cat.get("hint") or f"the {label.lower()} this fact or event belongs to; only when the text states it",
        "pattern": cat.get("pattern") or None,          # regex; the node name becomes its first match
        "text": cat.get("text") or f"is linked to the {label.lower()}",
        "in_text": cat.get("in_text") or "linked from",
        "multi": bool(cat.get("multi", True)),          # several facts between the same two nodes
    }


def extra_specs() -> list[dict]:
    return [extra_spec(lab) for lab in extra_labels()]


def relation_rules() -> dict:
    """config.yaml rules + one generated rule per extra node type (yaml wins on a name clash)."""
    rules = {k: dict(v) for k, v in get_config()["relation_rules"].items()}
    for sp in extra_specs():
        rules.setdefault(sp["relation"], {"source": sp["from"], "target": sp["label"],
                                          "hint": sp["hint"], "multi": sp["multi"]})
    return rules


def sources_of(rule: dict) -> list[str]:
    s = rule["source"]
    return [s] if isinstance(s, str) else list(s)


def is_multi(rel: str) -> bool:
    return bool(relation_rules().get(rel, {}).get("multi"))


def rel_texts() -> dict[str, str]:
    """Plain-English wording of generated relations ('Acme Corp has an event in the year 2010')."""
    return {sp["relation"]: sp["text"] for sp in extra_specs()}


def in_texts() -> dict[str, str]:
    """The same edge seen from the extra node's side."""
    return {sp["relation"]: sp["in_text"] for sp in extra_specs()}


def canonical_value(label: str, name: str) -> str | None:
    """For types with a pattern (Year): the cleaned value, or None when the name does not fit."""
    pat = next((sp["pattern"] for sp in extra_specs() if sp["label"] == label), None)
    if not pat:
        return name
    m = re.search(pat, name)
    return m.group(0) if m else None


def exact_labels() -> set[str]:
    """Labels whose names must match exactly (no fuzzy merging): types with a pattern, e.g. Year."""
    return {sp["label"] for sp in extra_specs() if sp["pattern"]}


def extra_relation_names() -> list[str]:
    """Relations that touch an extra node (IN_YEAR ...). Walks from normal entities skip them, otherwise a
    Year node would join unrelated companies that merely share a year."""
    ex = set(extra_labels())
    return [n for n, r in relation_rules().items() if r["target"] in ex or any(x in ex for x in sources_of(r))]


def base_relation_names() -> list[str]:
    ex = set(extra_relation_names())
    return [n for n in relation_rules() if n not in ex]
