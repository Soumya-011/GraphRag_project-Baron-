"""Loads the slow things in the background right after the server starts, so the first question does not pay for them.

Without this, the FIRST question had to import torch, load the embedding model, open Chroma, connect to Neo4j and
embed the off-topic examples: easily 20-60 seconds on a laptop. The server accepts requests immediately; the user
page shows "Starting up..." until this finishes. A step that fails is only recorded: the app still tries it again
lazily when a question needs it.
"""
import logging
import threading
import time

from .config import ROLES, get_settings

log = logging.getLogger("graphrag.warmup")
state: dict = {"state": "idle", "step": "", "seconds": None, "errors": []}
_lock = threading.Lock()


def _embeddings():
    from .llm import get_embeddings
    get_embeddings().embed_query("warm up")


def _guardrails():
    if get_settings().off_topic_enabled:
        from . import guardrails
        guardrails.off_topic_score("warm up")


def _vectors():
    from .stores import get_vector_store
    get_vector_store().similarity_search("warm up", k=1)


def _graph():
    from .stores import get_graph
    get_graph().query("RETURN 1 AS ok")


def _models():
    from .llm import get_llm
    for role in ROLES:
        get_llm(role)  # builds the clients; no network call


STEPS = [("loading the embedding model", _embeddings), ("preparing guardrails", _guardrails),
         ("opening the vector store", _vectors), ("connecting to the graph", _graph),
         ("preparing the language models", _models)]


def run() -> None:
    t0 = time.perf_counter()
    state.update(state="running", errors=[], seconds=None)
    for name, fn in STEPS:
        state["step"] = name
        try:
            fn()
        except Exception as e:  # never block the app: the step is retried lazily on first use
            state["errors"].append(f"{name}: {type(e).__name__}")
            log.warning("warm-up step failed (%s): %s", name, e)
    state.update(state="ready", step="", seconds=round(time.perf_counter() - t0, 1))
    log.info("warm-up finished in %.1fs", state["seconds"])


def start() -> None:
    """Idempotent. Returns at once; the work runs in a daemon thread."""
    with _lock:
        if state["state"] != "idle":
            return
        if not get_settings().warmup_on_start:
            state.update(state="off")
            return
        state["state"] = "running"
    threading.Thread(target=run, name="warmup", daemon=True).start()


def public() -> dict:
    """What visitors may see: ready or not, and a coarse step. No errors, no paths."""
    # idle = lifespan never ran (tests, scripts): nothing is loading, so do not show a banner forever
    return {"ready": state["state"] != "running", "step": state["step"]}


def snapshot() -> dict:
    return dict(state, errors=list(state["errors"]))
