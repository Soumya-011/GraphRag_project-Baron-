import logging
import threading
from functools import lru_cache

from . import schema
from .config import get_settings


@lru_cache
def get_graph():
    """LangChain Neo4jGraph. refresh_schema=False: Community edition has no APOC."""
    from langchain_neo4j import Neo4jGraph

    s = get_settings()
    return Neo4jGraph(
        url=s.neo4j_uri,
        username=s.neo4j_user,
        password=s.neo4j_password,
        refresh_schema=False,
    )


def _collection_names(client) -> list[str]:
    # chromadb returns Collection objects (0.5) or plain names (newer versions)
    return [c if isinstance(c, str) else c.name for c in client.list_collections()]


_vector_stores: dict[str, object] = {}
_vector_lock = threading.Lock()


def reset_vector_store() -> None:
    """Forget the cached store (tests, or after the Chroma folder was replaced)."""
    with _vector_lock:
        _vector_stores.clear()


def get_vector_store():
    """LangChain Chroma store (cosine). The ONLY place that knows about Chroma:
    swap this function to move to Qdrant/Chroma Cloud later.

    Built ONCE per process under a lock. Creating a new chromadb client on every call, from several threads
    at once (two questions, Compare, an upload running in the background), raced inside chromadb and gave
    "'RustBindingsAPI' object has no attribute 'bindings'"."""
    key = str(get_settings().chroma_dir)
    with _vector_lock:
        if key not in _vector_stores:
            _vector_stores[key] = _build_vector_store()
        return _vector_stores[key]


def _build_vector_store():
    import chromadb
    from langchain_chroma import Chroma

    from .llm import get_embeddings

    s = get_settings()
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(s.chroma_dir))

    # Phase 1 created an empty L2 collection; drop it so cosine can be set.
    if s.chroma_collection in _collection_names(client):
        col = client.get_collection(s.chroma_collection)
        if col.count() == 0 and (col.metadata or {}).get("hnsw:space") != "cosine":
            client.delete_collection(s.chroma_collection)

    return Chroma(
        client=client,
        collection_name=s.chroma_collection,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def neo4j_ping() -> dict:
    s = get_settings()
    try:
        get_graph().query("RETURN 1 AS ok")
        return {"ok": True, "uri": s.neo4j_uri}
    except Exception as e:
        return {"ok": False, "uri": s.neo4j_uri, "error": str(e)[:200]}


def chroma_ping() -> dict:
    """Read-only: does not create the collection or load the embedding model."""
    s = get_settings()
    try:
        import chromadb

        s.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(s.chroma_dir))
        count = 0
        if s.chroma_collection in _collection_names(client):
            count = client.get_collection(s.chroma_collection).count()
        return {"ok": True, "collection": s.chroma_collection, "count": count}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


# ======================= graph =======================


def _ident(value: str, allowed) -> str:
    """Labels and relation types go into Cypher text, so they must be whitelisted."""
    if value not in allowed:
        raise ValueError(f"unexpected identifier {value!r}")
    return value


def _rel_union(kind: str = "all") -> str:
    """'WORKS_AT|FOUNDED|...' built from the whitelist (excludes MENTIONED_IN).
    kind: 'all', 'base' (no Year-like relations) or 'extra' (only those)."""
    allowed = list(schema.relation_rules())
    names = {"all": allowed, "base": schema.base_relation_names(), "extra": schema.extra_relation_names()}[kind]
    if not names:
        return _ident(allowed[0], allowed)  # nothing of this kind: the walk simply finds nothing useful
    return "|".join(_ident(t, allowed) for t in names)


def ensure_graph_schema() -> None:
    g = get_graph()
    for label in schema.graph_labels():
        g.query(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE")
    g.query("CREATE CONSTRAINT document_source IF NOT EXISTS "
            "FOR (n:Document) REQUIRE n.source IS UNIQUE")


def graph_existing_names() -> dict[str, list[str]]:
    rows = get_graph().query(
        f"MATCH (n) WHERE {schema.node_filter()} "
        "RETURN labels(n)[0] AS label, n.name AS name ORDER BY name"
    )
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["label"], []).append(r["name"])
    return out


def graph_delete_source(source: str) -> None:
    """Remove everything one document contributed, then orphaned entities."""
    g = get_graph()
    g.query("MATCH ()-[r {source: $source}]->() DELETE r", {"source": source})
    g.query("MATCH (d:Document {source: $source}) DETACH DELETE d", {"source": source})
    g.query(f"MATCH (n) WHERE {schema.node_filter()} AND NOT (n)--() DELETE n")


def graph_write(source: str, entities: list[dict], relations: list[dict]) -> None:
    g = get_graph()
    allowed_rel = set(schema.relation_rules())
    labels = schema.graph_labels()
    ensure_graph_schema()
    g.query("MERGE (:Document {source: $source})", {"source": source})

    for label in labels:
        rows = [{"id": e["id"], "name": e["name"]} for e in entities if e["label"] == label]
        if rows:
            g.query(
                f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) "
                "ON CREATE SET n.name = row.name "
                "WITH n MATCH (d:Document {source: $source}) "
                "MERGE (n)-[:MENTIONED_IN {source: $source}]->(d)",
                {"rows": rows, "source": source},
            )

    groups: dict[tuple, list[dict]] = {}
    for r in relations:
        groups.setdefault((r["type"], r["sl"], r["tl"]), []).append(r)
    for (rel, sl, tl), rows in groups.items():
        _ident(rel, allowed_rel)
        _ident(sl, labels)
        _ident(tl, labels)
        if schema.is_multi(rel):  # several facts between the same two nodes: one edge per quote
            key = "{source: $source, evidence: row.evidence}"
        else:
            key = "{source: $source}"
        g.query(
            f"UNWIND $rows AS row MATCH (a:{sl} {{id: row.s}}) MATCH (b:{tl} {{id: row.t}}) "
            f"MERGE (a)-[r:{rel} {key}]->(b) "
            "SET r.status = row.status, r.evidence = row.evidence",
            {"rows": rows, "source": source},
        )


def graph_summary(limit: int = 50) -> dict:
    g = get_graph()
    return {
        "nodes": g.query(
            f"MATCH (n) WHERE {schema.node_filter()} "
            "RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"),
        "relationships": g.query(
            "MATCH ()-[r]->() WHERE type(r) <> 'MENTIONED_IN' "
            "RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"),
        "edges": g.query(
            "MATCH (a)-[r]->(b) WHERE type(r) <> 'MENTIONED_IN' "
            "RETURN a.name AS source, labels(a)[0] AS source_label, type(r) AS rel, "
            "b.name AS target, labels(b)[0] AS target_label, "
            "r.status AS status, r.source AS doc LIMIT $limit", {"limit": limit}),
    }


# ======================= Phase 4: Cypher templates =======================
def graph_entities() -> list[dict]:
    """All entities (small graph: fetched per question for the entity linker)."""
    return get_graph().query(
        f"MATCH (n) WHERE {schema.node_filter()} "
        "RETURN n.id AS id, n.name AS name, labels(n)[0] AS label"
    )


def graph_count(label: str) -> int:
    _ident(label, schema.graph_labels())
    return get_graph().query(f"MATCH (n:{label}) RETURN count(n) AS c")[0]["c"]


def graph_list(label: str, limit: int | None = None) -> list[str]:
    _ident(label, schema.graph_labels())
    limit = limit or get_settings().list_limit
    rows = get_graph().query(
        f"MATCH (n:{label}) RETURN n.name AS name ORDER BY name LIMIT $limit", {"limit": limit}
    )
    return [r["name"] for r in rows]


def _evidence(values, many: bool = False) -> str:
    """Source phrases stored on an edge. 'seed' (written by app.seed) is not real evidence.
    Edges of a Year-like node (many=True) keep more phrases: each one describes a different event."""
    good = [v.strip() for v in values if v and v.strip() and v.strip() != "seed"]
    return " / ".join(good[:6 if many else 2])[:700 if many else 240]


def graph_neighborhood(ids: list[str], hops: int, limit: int, kind: str = "all") -> list[dict]:
    """Distinct directed edges within `hops` of the given entities, NEAREST FIRST.
    Provenance docs are merged; `evidence` is the phrase the extractor copied from the text
    (carries roles such as 'managing partner' or 'joined the board in 2022').
    Ordering matters: with a hub such as an investor with 9 investments, an unordered LIMIT could keep
    distant edges and drop the entity's own."""
    hops = int(hops)
    if not 1 <= hops <= 3:
        raise ValueError("hops must be 1..3")
    match = (f"MATCH p = (s)-[:{_rel_union(kind)}*1..{hops}]-(n) "
             f"WHERE s.id IN $ids AND {schema.node_filter('s')} ")
    ordered = (
        match +
        "UNWIND range(0, size(relationships(p)) - 1) AS i "
        "WITH relationships(p)[i] AS r, i "
        "WITH startNode(r) AS a, type(r) AS rel, endNode(r) AS b, min(i) AS depth, "
        "     collect(DISTINCT r.source) AS docs, collect(DISTINCT r.status) AS statuses, "
        "     collect(DISTINCT r.evidence) AS evidence "
        "RETURN a.name AS source, rel, b.name AS target, statuses, docs, evidence, depth "
        "ORDER BY depth, source, target LIMIT $limit")
    plain = (  # previous query: same edges, arbitrary order. Used only if the ordered one fails.
        match +
        "UNWIND relationships(p) AS r "
        "WITH startNode(r) AS a, type(r) AS rel, endNode(r) AS b, "
        "     collect(DISTINCT r.source) AS docs, collect(DISTINCT r.status) AS statuses, "
        "     collect(DISTINCT r.evidence) AS evidence "
        "RETURN a.name AS source, rel, b.name AS target, statuses, docs, evidence LIMIT $limit")
    params = {"ids": ids, "limit": int(limit)}
    many_rels = {rel for rel in schema.relation_rules() if schema.is_multi(rel)}
    try:
        rows = get_graph().query(ordered, params)
    except Exception as e:  # e.g. an older Neo4j without this syntax: never break answering
        logging.getLogger("graphrag.stores").warning("ordered neighborhood query failed (%s); using plain", str(e)[:120])
        rows = get_graph().query(plain, params)
    return [
        {
            "source": r["source"], "rel": r["rel"], "target": r["target"],
            "status": "current" if "current" in r["statuses"] or not r["statuses"] else "past",
            "docs": sorted(d for d in r["docs"] if d),
            "evidence": _evidence(r.get("evidence") or [], many=r["rel"] in many_rels),
        }
        for r in rows
    ]


def graph_shortest_path(a_id: str, b_id: str, max_len: int | None = None, kind: str = "all") -> list[dict] | None:
    """Shortest relation path between two entities (document nodes are never traversed)."""
    if a_id == b_id:
        return None
    max_len = int(max_len or get_settings().graph_path_max_len)
    rows = get_graph().query(
        "MATCH (a {id: $a}), (b {id: $b}) "
        f"MATCH p = shortestPath((a)-[:{_rel_union(kind)}*..{max_len}]-(b)) "
        "RETURN [r IN relationships(p) | {source: startNode(r).name, rel: type(r), "
        "target: endNode(r).name, status: r.status}] AS edges LIMIT 1",
        {"a": a_id, "b": b_id},
    )
    return rows[0]["edges"] if rows else None
