"""Background ingest jobs: one worker (so the free-tier quota is never hit by two extractions at once)
and a polling-friendly status per file."""
import copy
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import registry
from .ingest import RAW_DIR, SUPPORTED, UPLOAD_DIR, ingest_one

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
_jobs: "OrderedDict[str, dict]" = OrderedDict()
_payloads: dict[str, list] = {}  # uploaded bytes kept out of the polled job view
_lock = threading.Lock()
MAX_JOBS_KEPT = 50
PROGRESS = {"queued": 0, "checking": 5, "reading": 15, "embedding": 40, "extracting": 75,
            "done": 100, "skipped": 100, "error": 100}


def safe_name(filename: str) -> str:
    """Keep the base name only, with plain characters, so an upload cannot escape its folder."""
    name = Path((filename or "").replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")[:100]
    return name if Path(name).suffix.lower() in SUPPORTED and len(Path(name).stem) > 0 else ""


def safe_source(source: str) -> Path | None:
    """Path of a registered source inside RAW_DIR, or None when it would point outside."""
    p = (RAW_DIR / source).resolve()
    return p if p.is_relative_to(RAW_DIR.resolve()) else None


def queued_count() -> int:
    with _lock:
        return sum(j["status"] in ("queued", "running") for j in _jobs.values())


def _set(job: dict, f: dict, **kw):
    with _lock:
        f.update(kw)
        f["progress"] = PROGRESS.get(f["stage"], f.get("progress", 0))
        if f["stage"] in ("extracting", "done", "skipped"):
            f["searchable"] = True  # vectors are stored: questions can already be asked


def _run(job_id: str):
    job = _jobs[job_id]
    with _lock:
        job["status"] = "running"
    for i, f in enumerate(job["files"]):
        t0 = time.perf_counter()
        with _lock:
            f["started"] = time.time()
        try:
            _process(job, f, _payloads[job_id][i])
            _payloads[job_id][i] = None  # free the upload
        except Exception as e:  # never kill the worker
            _set(job, f, stage="error", error=str(e)[:200])
        _set(job, f, elapsed_ms=int((time.perf_counter() - t0) * 1000))
    with _lock:
        job["status"] = "error" if all(f["stage"] == "error" for f in job["files"]) else "done"
    _payloads.pop(job_id, None)


def _process(job: dict, f: dict, data: bytes | None):
    source, force, graph = f["source"], job["force"], job["graph"]
    dest = safe_source(source)
    if dest is None:
        raise ValueError("Invalid file name")
    _set(job, f, stage="checking")
    if data is not None:
        sha = registry.bytes_sha256(data)
        reason = None if force else registry.duplicate_reason(source, sha, graph)
        if reason:
            return _set(job, f, stage="skipped", detail=reason)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    elif not dest.exists():
        raise FileNotFoundError("File is no longer on disk")
    r = ingest_one(dest, RAW_DIR, graph=graph, force=force, on_stage=lambda s: _set(job, f, stage=s))
    if r["error"]:
        return _set(job, f, stage="error", error=r["error"])
    if r["status"] == "skipped":
        return _set(job, f, stage="skipped", detail=r["reason"])
    g = r["graph"] or {}
    _set(job, f, stage="done", chunks=r["chunks"], entities=g.get("entities"), relations=g.get("relations"),
         llm_calls=g.get("llm_calls"), graph_error=r["graph_error"], detail=("Graph extraction failed, vectors are saved" if r["graph_error"] else ""))


def submit(items: list[dict], graph: bool = True, force: bool = False) -> str:
    """items: {"name", "source", "data": bytes | None}. data None = the file is already on disk."""
    job_id = uuid.uuid4().hex
    job = {"id": job_id, "status": "queued", "graph": graph, "force": force, "created": time.time(),
           "files": [{"name": i["name"], "source": i["source"], "stage": "queued",
                      "progress": 0, "detail": "", "error": None, "searchable": False, "started": None, "llm_calls": None, "chunks": None, "entities": None,
                      "relations": None, "graph_error": None, "elapsed_ms": None} for i in items]}
    with _lock:
        _jobs[job_id] = job
        _payloads[job_id] = [i.get("data") for i in items]
        while len(_jobs) > MAX_JOBS_KEPT:
            _jobs.popitem(last=False)
    _pool.submit(_run, job_id)
    return job_id


def get(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        view = copy.deepcopy({k: v for k, v in job.items() if k != "force"})
    view["server_time"] = time.time()
    return view


def upload_source(name: str) -> str:
    return f"{UPLOAD_DIR.name}/{name}"
