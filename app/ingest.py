import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import registry
from .cache import bump_epoch
from .config import ROOT, get_config, get_settings
from .extract import Resolver, extract_document, resolve_all
from .stores import (
    get_vector_store,
    graph_delete_source,
    graph_existing_names,
    graph_write,
)

RAW_DIR = ROOT / "data" / "raw"
UPLOAD_DIR = RAW_DIR / "uploads"  # files added through the web UI never overwrite the core documents
SUPPORTED = {".txt", ".md", ".pdf"}


def load_file(path: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    return TextLoader(str(path), encoding="utf-8-sig").load()


def split_documents(docs: list[Document], source: str) -> list[Document]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    for i, c in enumerate(chunks):
        meta = {"source": source, "chunk": i}
        if c.metadata.get("page") is not None:  # Chroma rejects None metadata values
            meta["page"] = int(c.metadata["page"])
        c.metadata = meta
    return chunks


def source_for(path: Path, root: Path = RAW_DIR) -> str:
    path, root = Path(path).resolve(), Path(root).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name


def prepare(path: Path, root: Path = RAW_DIR) -> tuple[str, list[Document]]:
    source = source_for(path, root)
    return source, split_documents(load_file(Path(path).resolve()), source)


def _write_vectors(source: str, chunks: list[Document]) -> int:
    store = get_vector_store()
    store.delete(where={"source": source})  # idempotent re-ingest / edited files
    if chunks:
        store.add_documents(chunks, ids=[f"{source}::{c.metadata['chunk']}" for c in chunks])
    return len(chunks)


def ingest_file(path: Path, root: Path = RAW_DIR) -> int:
    """Vector store only. Returns the number of chunks."""
    source, chunks = prepare(path, root)
    return _write_vectors(source, chunks)


def ingest_graph(source: str, chunks: list[Document]) -> dict:
    """Extract entities/relations and rewrite this document's part of the graph."""
    res = extract_document(chunks)  # may raise: old graph data stays untouched
    graph_delete_source(source)
    resolver = Resolver(
        get_settings().entity_fuzzy_threshold,
        get_config().get("aliases") or {},
        graph_existing_names(),
    )
    entities, relations = resolve_all(res["mentions"], res["relations"], resolver)
    graph_write(source, entities, relations)
    return {
        "entities": len(entities),
        "relations": len(relations),
        "dropped": res["dropped"],
        "failed_chunks": res["failed"],
        "llm_calls": res["llm_calls"],
    }


def delete_document(source: str) -> None:
    """Remove one document from the vectors, the graph and the registry."""
    get_vector_store().delete(where={"source": source})
    graph_delete_source(source)
    registry.remove(source)
    bump_epoch()


def ingest_one(path: Path, root: Path = RAW_DIR, vectors: bool = True, graph: bool = True,
               force: bool = False, on_stage=None) -> dict:
    """Ingest one file unless its content is already in the store (same source + same SHA-256).
    Unchanged files are skipped, a file that was stored without a graph only gets its graph."""
    stage = on_stage or (lambda s: None)
    path = Path(path)
    source = source_for(path, root)
    out = {"source": source, "status": "ingested", "reason": None, "chunks": 0,
           "graph": None, "error": None, "graph_error": None}
    try:
        sha = registry.file_sha256(path)
        rec = registry.get(source)
        same = bool(rec) and rec["sha256"] == sha and not force
        skip_ok = get_settings().skip_already_ingested and not force
        if skip_ok and not same:
            other = registry.by_hash(sha)
            if other and other["source"] != source and not rec:
                out.update(status="skipped", reason=f"Same content as {other['source']}, nothing to do")
                return out
        need_vec = vectors and not (same and skip_ok)
        need_graph = graph and not (same and skip_ok and rec["graph"])
        if not need_vec and not need_graph:
            out.update(status="skipped", reason="Already ingested, nothing to do", chunks=rec["chunks"])
            return out

        stage("reading")
        _, chunks = prepare(path, root)
        if need_vec:
            stage("embedding")
            out["chunks"] = _write_vectors(source, chunks)
        else:
            out["chunks"] = len(chunks)

        g = None
        if need_graph and chunks:
            stage("extracting")
            try:
                g = ingest_graph(source, chunks)
            except Exception as e:  # vectors are saved; the graph can be retried later
                out["graph_error"] = str(e)[:200]
        elif not graph and rec and rec["sha256"] != sha:
            graph_delete_source(source)  # the text changed: old graph facts would be stale

        keep_old_graph = bool(rec) and same and bool(rec["graph"]) and not need_graph
        graph_ok = bool(g) or keep_old_graph
        out["graph"] = g
        if vectors or same or rec:  # graph-only re-run of a known document keeps its registry row current
            registry.upsert(
                source, sha, out["chunks"], graph_ok,
                g["entities"] if g else (rec["entities"] if keep_old_graph else 0),
                g["relations"] if g else (rec["relations"] if keep_old_graph else 0))
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


def ingest_paths(paths: list[Path], root: Path = RAW_DIR,
                 vectors: bool = True, graph: bool = True, force: bool = False) -> dict:
    out = {"files": {}, "graph": {}, "errors": {}, "graph_errors": {}, "skipped": {}}
    for p in paths:
        name = Path(p).name
        r = ingest_one(Path(p), root, vectors, graph, force)
        if r["error"]:
            out["errors"][name] = r["error"]
            continue
        if r["status"] == "skipped":
            out["skipped"][name] = r["reason"]
        out["files"][name] = r["chunks"]
        if r["graph"]:
            out["graph"][name] = r["graph"]
        if r["graph_error"]:
            out["graph_errors"][name] = r["graph_error"]
    out["total_chunks"] = sum(v for k, v in out["files"].items() if k not in out["skipped"])
    if len(out["skipped"]) < len(paths):
        bump_epoch()  # documents changed: cached answers are stale
    return out


def list_files(root: Path = RAW_DIR) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in SUPPORTED)


def ingest_directory(root: Path = RAW_DIR, **kw) -> dict:
    return ingest_paths(list_files(root), Path(root), **kw)


def main():
    ap = argparse.ArgumentParser(description="Ingest documents into the vector store and graph")
    ap.add_argument("path", nargs="?", default=str(RAW_DIR), help="file or folder")
    ap.add_argument("--no-graph", action="store_true", help="vectors only")
    ap.add_argument("--graph-only", action="store_true", help="skip embedding, rebuild graph only")
    ap.add_argument("--force", action="store_true", help="re-ingest even if the content is unchanged")
    args = ap.parse_args()
    if args.no_graph and args.graph_only:
        ap.error("--no-graph and --graph-only cannot be combined")

    target = Path(args.path)
    paths, root = ([target], target.parent) if target.is_file() else (list_files(target), target)
    print("Loading models (first run downloads ~90 MB for embeddings)...", flush=True)

    chunks_total, skipped = 0, 0
    for p in paths:
        r = ingest_paths([p], root, vectors=not args.graph_only, graph=not args.no_graph, force=args.force)
        chunks_total += r["total_chunks"]
        if p.name in r["skipped"]:
            skipped += 1
            print(f"{p.name}: skipped ({r['skipped'][p.name]})", flush=True)
            continue
        line = f"{p.name}: {r['total_chunks']} chunks"
        g = r["graph"].get(p.name)
        if g:
            line += (f" | graph: {g['entities']} entities, {g['relations']} relations, "
                     f"{g['dropped']} dropped, {g['failed_chunks']} failed chunks, "
                     f"{g['llm_calls']} LLM calls")
        for errs in (r["errors"], r["graph_errors"]):
            if p.name in errs:
                line += f" | ERROR {errs[p.name]}"
        print(line, flush=True)
    print(f"Total: {chunks_total} new chunks from {len(paths) - skipped} files ({skipped} skipped, unchanged)")


if __name__ == "__main__":
    main()
