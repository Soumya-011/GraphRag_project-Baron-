from langchain_core.tools import tool

from .config import get_settings
from .rag import retrieve
from .router import link_entities
from .stores import get_graph, graph_neighborhood, graph_shortest_path


@tool
def graph_stats() -> dict:
    """Return the number of nodes and relationships in the knowledge graph."""
    g = get_graph()
    nodes = g.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
    rels = g.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    return {"nodes": nodes, "relationships": rels}


@tool
def vector_search(query: str, k: int = 5) -> list[dict]:
    """Semantic search over the ingested documents. Returns text, source and similarity score."""
    return retrieve(query, k)


@tool
def find_entity(text: str) -> list[dict]:
    """Find knowledge-graph entities (people, companies, locations) mentioned in a text."""
    return link_entities(text)


@tool
def neighbors(entity: str, hops: int = 1) -> list[dict]:
    """Relations around one entity within 1 to 3 hops. Each item is a directed edge."""
    found = link_entities(entity, limit=1)
    if not found:
        return []
    return graph_neighborhood([found[0]["id"]], hops, get_settings().graph_max_edges)


@tool
def shortest_path(entity_a: str, entity_b: str) -> list[dict]:
    """Shortest chain of relations connecting two entities (empty if none)."""
    a, b = link_entities(entity_a, limit=1), link_entities(entity_b, limit=1)
    if not a or not b:
        return []
    return graph_shortest_path(a[0]["id"], b[0]["id"]) or []
