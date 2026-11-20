import html
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import evalview, jobs, registry, security, warmup
from .cache import metrics
from .config import ROOT, get_settings
from .ingest import RAW_DIR, SUPPORTED, UPLOAD_DIR, delete_document
from .llm import describe_llm_error, llm_health
from .pipeline import InputRejected, guarded_ask, guarded_compare
from .router import decide_route, link_entities
from .stores import chroma_ping, graph_entities, graph_summary, neo4j_ping

STATIC = ROOT / "static"    # public: user page, login page, shared upload widget
PRIVATE = ROOT / "private"  # developer page: only served after login
DEV_ASSETS = {"dev.js", "dev.css"}

_locked = bool(get_settings().dev_password)  # hide the interactive API docs when a password is set


@asynccontextmanager
async def _lifespan(_app):
    warmup.start()  # returns at once; models load in the background
    yield


app = FastAPI(title="GraphRAG", docs_url=None if _locked else "/docs", lifespan=_lifespan,
              redoc_url=None, openapi_url=None if _locked else "/openapi.json")
dev = Depends(security.require_dev)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    route: Literal["auto", "vector", "graph", "hybrid"] = "auto"
    top_k: int | None = Field(default=None, ge=1, le=20)


class LoginRequest(BaseModel):
    password: str = ""


# ======================= pages =======================
_ASSET = re.compile(r"""(["'])(?:/static/([\w.-]+\.(?:js|css))|/dev/assets/([\w.-]+\.(?:js|css)))\1""")


def _stamp(text: str) -> str:
    """Adds ?v=<file time> to script and stylesheet links. Browsers keep serving an old user.js after an update
    (no history, no start-up notice) until the link changes, so the link changes whenever the file does."""
    def one(m: re.Match) -> str:
        name, base, prefix = (m.group(2), STATIC, "/static/") if m.group(2) else (m.group(3), PRIVATE, "/dev/assets/")
        try:
            v = int((base / name).stat().st_mtime)
        except OSError:
            v = 0
        return f"{m.group(1)}{prefix}{name}?v={v}{m.group(1)}"
    return _ASSET.sub(one, text)


def _page(path: Path) -> HTMLResponse:
    """HTML with the app name and tagline from .env filled in (escaped), so there is nothing to edit in the files."""
    s = get_settings()
    text = path.read_text(encoding="utf-8")
    text = text.replace("{{APP_NAME}}", html.escape(s.app_name)).replace("{{APP_TAGLINE}}", html.escape(s.app_tagline))
    text = text.replace("{{MAX_QUESTION_CHARS}}", str(s.max_question_chars))  # the box stops typing where the server would refuse
    text = _stamp(text)
    return HTMLResponse(text, headers={"Cache-Control": "no-store"})


@app.get("/", include_in_schema=False)
def user_page():
    return _page(STATIC / "user.html")


@app.get("/login", include_in_schema=False)
def login_page():
    return _page(STATIC / "login.html")


@app.get("/dev", include_in_schema=False)
def dev_page(request: Request):
    if not security.is_dev(request):
        return RedirectResponse("/login", status_code=303)
    return _page(PRIVATE / "dev.html")


@app.get("/dev/assets/{name}", include_in_schema=False)
def dev_asset(name: str, _=dev):
    if name not in DEV_ASSETS:
        raise HTTPException(404, "Not found")
    return FileResponse(PRIVATE / name)


# ======================= auth =======================
@app.post("/auth/login")
def login(req: LoginRequest, request: Request, response: Response):
    s = get_settings()
    ip = security.client_ip(request)
    if security.login_blocked(ip):
        raise HTTPException(429, "Too many failed logins. Wait a minute.", headers={"Retry-After": "60"})
    if not security.check_password(req.password):
        security.record_login_failure(ip)
        raise HTTPException(401, "Wrong password")
    security.clear_login_failures(ip)
    response.set_cookie(security.COOKIE, security.make_token(), httponly=True, samesite="strict",
                        secure=request.url.scheme == "https", max_age=s.dev_session_hours * 3600, path="/")
    return {"ok": True}


@app.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(security.COOKIE, path="/")
    return {"ok": True}


@app.get("/auth/me")
def me(request: Request):
    return {"dev": security.is_dev(request), "protected": security.protected()}


# ======================= public: ask, upload =======================
def public_view(r: dict) -> dict:
    """What a visitor may see: no retrieved context, prompts, models, costs or graph internals."""
    router = r.get("router") or {}
    g = r.get("grounding")
    return {
        "answer": r.get("answer", ""),
        "route": r.get("route"),
        "router": {"route": router.get("route"), "via": router.get("via")} if router else None,
        "guard": r.get("guard"),
        "cached": bool(r.get("cached")),
        "escalated": bool(r.get("escalated")),
        "grounded": None if g is None else bool(g.get("ok")),
        "sources": [{"source": x.get("source"), "chunk": x.get("chunk")} for x in (r.get("sources") or [])][:8],
        "latency_ms": r.get("latency_ms"),
    }


def _ask(req: AskRequest):
    try:
        return guarded_ask(req.question, req.route, req.top_k)
    except InputRejected as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"RAG pipeline failed: {describe_llm_error(e)}")


def _compare(req: AskRequest):
    try:
        return guarded_compare(req.question, req.top_k)
    except InputRejected as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"Compare failed: {describe_llm_error(e)}")


@app.post("/ask")
def ask(req: AskRequest, request: Request):
    security.limit_or_429(request, "ask", get_settings().ask_rate_per_minute, 60)
    return public_view(_ask(req))


@app.post("/compare")
def compare_endpoint(req: AskRequest, request: Request):
    security.limit_or_429(request, "ask", get_settings().ask_rate_per_minute, 60)
    d = _compare(req)
    return {"vector": public_view(d["vector"]), "graph": public_view(d["graph"])}


@app.get("/health")
def health():
    checks = {"neo4j": neo4j_ping(), "chroma": chroma_ping()}
    return {"status": "ok" if all(c["ok"] for c in checks.values()) else "degraded"}


@app.get("/kb/status")
def kb_status():
    return {"documents": len(registry.all_docs()), **warmup.public()}


@app.get("/kb/docs")
def kb_docs():
    """Names and counts only, for the user page sidebar. Newest first."""
    docs = sorted(registry.all_docs(), key=lambda d: d["ingested_at"], reverse=True)[:200]
    return {"documents": [{"source": d["source"], "name": Path(d["source"]).name, "chunks": d["chunks"],
                           "graph": bool(d["graph"]), "ts": d["ingested_at"]} for d in docs]}


@app.get("/kb/suggest")
def kb_suggest():
    """Example questions built from the documents' own names. No LLM call."""
    docs = kb_docs()["documents"]
    if not docs:
        return {"questions": []}
    qs = ["What are the main topics in these documents?"]
    try:
        ents = graph_entities()
        companies = sorted(e["name"] for e in ents if e["label"] == "Company")
        people = sorted(e["name"] for e in ents if e["label"] == "Person")
        step = max(1, len(companies) // 2)  # spread the picks instead of always taking A and B
        qs += [f"Summarize {c}" for c in companies[::step][:2]]
        qs += [f"Which companies is {p} connected to?" for p in people[:1]]
        if companies:
            qs.append("How many companies are there?")
    except Exception:
        qs.append(f"Summarize {Path(docs[0]['name']).stem.replace('_', ' ')}")
    return {"questions": qs[:5]}


@app.post("/ingest")
def ingest(request: Request, files: list[UploadFile] = File(...), graph: bool = True):
    s = get_settings()
    security.limit_or_429(request, "upload", s.upload_rate_per_hour, 3600)
    if not files:
        raise HTTPException(400, "No files received")
    if len(files) > s.max_files_per_upload and not security.is_dev(request):
        raise HTTPException(413, f"At most {s.max_files_per_upload} files per upload")
    if jobs.queued_count() >= s.max_queued_jobs:
        raise HTTPException(429, "The server is busy ingesting other documents. Try again in a minute.")
    limit = s.max_upload_mb * 1024 * 1024
    items = []
    for f in files:
        name = jobs.safe_name(f.filename or "")
        if not name:
            raise HTTPException(415, f"Unsupported file: {Path(f.filename or '').name!r}. "
                                     f"Allowed: {', '.join(sorted(SUPPORTED))}")
        data = f.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"{name} is larger than {s.max_upload_mb} MB")
        if not data.strip():
            raise HTTPException(400, f"{name} is empty")
        items.append({"name": name, "source": jobs.upload_source(name), "data": data})
    return {"job_id": jobs.submit(items, graph=graph)}


@app.get("/ingest/jobs/{job_id}")
def ingest_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job


# ======================= developer only =======================
@app.post("/dev/ask")
def dev_ask(req: AskRequest, _=dev):
    return _ask(req)


@app.post("/dev/compare")
def dev_compare(req: AskRequest, _=dev):
    return _compare(req)


@app.get("/dev/health")
def dev_health(_=dev):
    s = get_settings()
    checks = {"llm": llm_health(), "neo4j": neo4j_ping(), "chroma": chroma_ping()}
    return {"status": "ok" if all(c["ok"] for c in checks.values()) else "degraded",
            "env": s.app_env, "checks": checks, "warmup": warmup.snapshot()}


@app.get("/dev/settings")
def dev_settings(_=dev):
    """Every setting except secrets."""
    hidden = ("key", "password", "secret")
    return {k: v for k, v in get_settings().model_dump().items() if not any(h in k for h in hidden)}


@app.get("/dev/docs")
def dev_docs(_=dev):
    docs = registry.all_docs()
    for d in docs:
        p = jobs.safe_source(d["source"])
        d["on_disk"] = bool(p and p.exists())
        d["upload"] = d["source"].startswith(f"{UPLOAD_DIR.name}/")
    return {"documents": docs}


@app.delete("/dev/docs")
def dev_delete_doc(source: str = Query(..., min_length=1), _=dev):
    rec = registry.get(source)
    if not rec:
        raise HTTPException(404, "Document not registered")
    delete_document(source)
    removed_file = False
    p = jobs.safe_source(source)
    if p and p.exists() and source.startswith(f"{UPLOAD_DIR.name}/"):
        p.unlink()  # uploads are removed; core files stay on disk and return on the next `python -m app.ingest`
        removed_file = True
    return {"deleted": source, "file_removed": removed_file}


@app.post("/dev/docs/reingest")
def dev_reingest(source: str = Query(..., min_length=1), graph: bool = True, _=dev):
    p = jobs.safe_source(source)
    if not registry.get(source) or not p or not p.exists():
        raise HTTPException(404, "Document or file not found")
    return {"job_id": jobs.submit([{"name": Path(source).name, "source": source, "data": None}],
                                  graph=graph, force=True)}


@app.get("/dev/eval/reports")
def dev_eval_reports(_=dev):
    return {"reports": evalview.list_reports()}


@app.get("/dev/eval/report")
def dev_eval_report(name: str = Query(..., min_length=1), summary: bool = False, _=dev):
    text = evalview.read_report(name)
    if text is None:
        raise HTTPException(404, "Report not found")
    return {"name": name, "summary": evalview.summarize(text)} if summary else {"name": name, "text": text}


@app.get("/dev/extraction")
def dev_extraction(corpus: bool = True, _=dev):
    """Graph precision / recall against the gold triples (uploaded documents are excluded)."""
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from eval.check_graph import compute
        return compute(corpus)
    except Exception as e:
        raise HTTPException(502, f"Extraction check failed: {str(e)[:300]}")


@app.get("/metrics")
def metrics_endpoint(_=dev):
    return metrics()


@app.get("/graph/summary")
def graph_overview(limit: int = Query(50, ge=1, le=500), _=dev):
    try:
        return graph_summary(limit)
    except Exception as e:
        raise HTTPException(502, f"Graph query failed: {str(e)[:300]}")


@app.get("/graph/link")
def graph_link(q: str = Query(..., min_length=1), _=dev):
    """Debug: which graph entities the linker finds in a question (no LLM call)."""
    try:
        return {"question": q, "entities": link_entities(q)}
    except Exception as e:
        raise HTTPException(502, f"Graph query failed: {str(e)[:300]}")


@app.get("/route")
def route_debug(q: str = Query(..., min_length=1), llm: bool = False, _=dev):
    """Debug: which route 'auto' would pick and why. Rules only unless llm=true."""
    try:
        ents = link_entities(q)
        return {"question": q, "entities": [e["name"] for e in ents],
                **decide_route(q, ents, use_llm=llm)}
    except Exception as e:
        raise HTTPException(502, f"Routing failed: {str(e)[:300]}")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
