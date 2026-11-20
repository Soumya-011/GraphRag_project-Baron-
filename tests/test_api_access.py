"""Access control, public/developer response split, upload limits and job flow. The pipeline, Neo4j and the
vector store are replaced by fakes, so no model, key or database is needed."""
import importlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("WARMUP_ON_START", "false")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

from app import registry
from app.config import get_settings

FULL = {"answer": "Rohan Kapoor leads Meridian Capital [meridian_capital.txt]", "route": "graph",
        "router": {"route": "graph", "via": "rules", "reason": "secret reasoning", "llm_calls": 0},
        "guard": None, "cached": False, "escalated": False, "grounding": {"ok": True, "unsupported": []},
        "sources": [{"source": "meridian_capital.txt", "chunk": 0, "score": 0.9, "snippet": "snippet text"}],
        "latency_ms": 1234, "wait_ms": 10, "llm_calls": 1, "context": "PRIVATE CONTEXT", "graph": {"edges": [1]},
        "fallback": ["x"], "best_score": 0.9}


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    for k, v in {"DEV_PASSWORD": "pw", "DEV_SESSION_SECRET": "t", "DEV_MAX_LOGIN_ATTEMPTS": "3",
                 "ASK_RATE_PER_MINUTE": "2", "UPLOAD_RATE_PER_HOUR": "50", "MAX_UPLOAD_MB": "1",
                 "MAX_FILES_PER_UPLOAD": "2", "CACHE_DB_PATH": str(tmp_path / "cache.sqlite")}.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    from app import api, jobs, security
    api = importlib.reload(api)
    security.clear_login_failures()
    security.limiter.reset()
    raw = tmp_path / "raw"
    monkeypatch.setattr(jobs, "RAW_DIR", raw)
    monkeypatch.setattr(api, "guarded_ask", lambda q, r, k=None: dict(FULL))
    monkeypatch.setattr(api, "guarded_compare", lambda q, k=None: {"vector": dict(FULL), "graph": dict(FULL)})
    monkeypatch.setattr(api, "neo4j_ping", lambda: {"ok": True})
    monkeypatch.setattr(api, "chroma_ping", lambda: {"ok": True})
    monkeypatch.setattr(api, "llm_health", lambda: {"ok": True, "roles": {}})
    monkeypatch.setattr(api, "metrics", lambda: {"requests": 1})
    monkeypatch.setattr(api, "graph_summary", lambda limit=50: {"edges": []})
    monkeypatch.setattr(api, "link_entities", lambda q: [])
    monkeypatch.setattr(api, "graph_entities", lambda: [
        {"id": "c1", "name": "Acme", "label": "Company"}, {"id": "c2", "name": "Zeta", "label": "Company"},
        {"id": "p1", "name": "Bob", "label": "Person"}])
    monkeypatch.setattr(api, "decide_route", lambda q, e, use_llm=False: {"route": "vector"})
    deleted = []
    monkeypatch.setattr(api, "delete_document", lambda s: (deleted.append(s), registry.remove(s)))

    def fake_ingest_one(path, root, graph=True, force=False, on_stage=None, **kw):
        on_stage and on_stage("embedding")
        src = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
        registry.upsert(src, registry.file_sha256(path), 2, graph, 3, 2)
        return {"source": src, "status": "ingested", "reason": None, "chunks": 2,
                "graph": {"entities": 3, "relations": 2, "llm_calls": 4}, "error": None, "graph_error": None}

    monkeypatch.setattr(jobs, "ingest_one", fake_ingest_one)
    c = TestClient(api.app)
    c.deleted, c.raw = deleted, raw
    yield c
    get_settings.cache_clear()


def login(c):
    assert c.post("/auth/login", json={"password": "pw"}).status_code == 200


def wait_job(c, job_id):
    for _ in range(100):
        j = c.get(f"/ingest/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


# ---------- locked endpoints ----------
LOCKED_GET = ["/metrics", "/graph/summary", "/graph/link?q=x", "/route?q=x", "/dev/health", "/dev/settings",
              "/dev/docs", "/dev/assets/dev.js"]


def test_everything_developer_is_locked_without_login(ctx):
    for path in LOCKED_GET:
        assert ctx.get(path).status_code == 401, path
    assert ctx.post("/dev/ask", json={"question": "x"}).status_code == 401
    assert ctx.post("/dev/compare", json={"question": "x"}).status_code == 401
    assert ctx.delete("/dev/docs?source=uploads/a.txt").status_code == 401
    assert ctx.post("/dev/docs/reingest?source=uploads/a.txt").status_code == 401
    r = ctx.get("/dev", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert ctx.get("/docs").status_code == 404 and ctx.get("/openapi.json").status_code == 404


def test_login_flow_wrong_password_lockout_logout(ctx):
    assert ctx.post("/auth/login", json={"password": "nope"}).status_code == 401
    assert ctx.post("/auth/login", json={"password": "nope"}).status_code == 401
    assert ctx.post("/auth/login", json={"password": "nope"}).status_code == 401
    r = ctx.post("/auth/login", json={"password": "pw"})            # locked even with the right password
    assert r.status_code == 429 and r.headers["retry-after"] == "60"


def test_login_gives_access_and_logout_removes_it(ctx):
    login(ctx)
    for path in LOCKED_GET:
        assert ctx.get(path).status_code == 200, path
    assert "Developer" in ctx.get("/dev").text
    assert ctx.get("/auth/me").json() == {"dev": True, "protected": True}
    assert ctx.post("/dev/ask", json={"question": "x"}).json()["context"] == "PRIVATE CONTEXT"
    ctx.post("/auth/logout")
    assert ctx.get("/metrics").status_code == 401 and ctx.get("/auth/me").json()["dev"] is False


def test_forged_cookie_is_rejected(ctx):
    ctx.cookies.set("graphrag_dev", "9999999999.deadbeef")
    assert ctx.get("/metrics").status_code == 401


def test_dev_assets_whitelist(ctx):
    login(ctx)
    assert ctx.get("/dev/assets/dev.css").status_code == 200
    assert ctx.get("/dev/assets/dev.html").status_code == 404
    assert ctx.get("/dev/assets/..%2Fapp%2Fconfig.py").status_code in (404, 422)


def test_no_developer_files_in_public_static():
    """Delete static/index.html and static/app.js: the old developer page must not stay public."""
    assert not (ROOT / "static" / "app.js").exists()
    assert not (ROOT / "static" / "index.html").exists()
    assert (ROOT / "private" / "dev.js").exists() and (ROOT / "private" / "dev.html").exists()


# ---------- public answers ----------
def test_public_ask_hides_internals(ctx):
    d = ctx.post("/ask", json={"question": "who leads Meridian?"}).json()
    assert d["answer"].startswith("Rohan Kapoor") and d["router"] == {"route": "graph", "via": "rules"}
    assert d["sources"] == [{"source": "meridian_capital.txt", "chunk": 0}]
    for leaked in ("context", "llm_calls", "graph", "fallback", "wait_ms", "best_score"):
        assert leaked not in d
    assert "secret reasoning" not in str(d)
    c = ctx.post("/compare", json={"question": "x"}).json()
    assert set(c) == {"vector", "graph"} and "context" not in c["graph"]


def test_public_pages_and_minimal_health(ctx):
    assert ctx.get("/").status_code == 200 and get_settings().app_name in ctx.get("/").text  # whatever APP_NAME is in .env
    assert ctx.get("/login").status_code == 200
    assert ctx.get("/health").json() == {"status": "ok"}
    assert ctx.get("/static/upload.js").status_code == 200
    assert ctx.get("/static/dev.js").status_code == 404


def test_ask_rate_limit_and_developer_bypass(ctx):
    for _ in range(2):
        assert ctx.post("/ask", json={"question": "x"}).status_code == 200
    r = ctx.post("/ask", json={"question": "x"})
    assert r.status_code == 429 and "retry-after" in r.headers
    login(ctx)
    assert ctx.post("/ask", json={"question": "x"}).status_code == 200


# ---------- upload ----------
def post_files(c, *files, graph=True):
    return c.post(f"/ingest?graph={str(graph).lower()}", files=[("files", f) for f in files])


def test_upload_validation(ctx):
    assert post_files(ctx, ("evil.exe", b"MZ", "application/octet-stream")).status_code == 415
    assert post_files(ctx, ("a.txt", b"   ", "text/plain")).status_code == 400
    assert post_files(ctx, ("big.txt", b"x" * (1024 * 1024 + 5), "text/plain")).status_code == 413
    three = [(f"{i}.txt", b"hello", "text/plain") for i in range(3)]
    assert post_files(ctx, *three).status_code == 413                       # MAX_FILES_PER_UPLOAD=2


def test_upload_job_progress_and_safe_path(ctx):
    r = post_files(ctx, ("../../evil name!.txt", b"Acme hired Bob.", "text/plain"))
    assert r.status_code == 200
    j = wait_job(ctx, r.json()["job_id"])
    f = j["files"][0]
    assert j["status"] == "done" and f["stage"] == "done" and f["progress"] == 100
    assert f["entities"] == 3 and f["source"] == "uploads/evil name_.txt"
    assert (ctx.raw / "uploads" / "evil name_.txt").read_bytes() == b"Acme hired Bob."
    assert not (ctx.raw.parent / "evil name_.txt").exists()
    assert "_data" not in str(j)


def test_same_file_uploaded_twice_is_skipped(ctx):
    first = wait_job(ctx, post_files(ctx, ("a.txt", b"same bytes", "text/plain")).json()["job_id"])
    assert first["files"][0]["stage"] == "done"
    second = wait_job(ctx, post_files(ctx, ("a.txt", b"same bytes", "text/plain")).json()["job_id"])
    assert second["files"][0]["stage"] == "skipped" and "Already ingested" in second["files"][0]["detail"]
    other = wait_job(ctx, post_files(ctx, ("renamed.txt", b"same bytes", "text/plain")).json()["job_id"])
    assert other["files"][0]["stage"] == "skipped" and "Same content" in other["files"][0]["detail"]
    assert not (ctx.raw / "uploads" / "renamed.txt").exists()               # duplicate never touches the disk
    assert ctx.get("/kb/status").json()["documents"] == 1


def test_unknown_job_is_404(ctx):
    assert ctx.get("/ingest/jobs/doesnotexist").status_code == 404


# ---------- document manager ----------
def test_docs_list_delete_and_reingest(ctx):
    wait_job(ctx, post_files(ctx, ("a.txt", b"hello world", "text/plain")).json()["job_id"])
    login(ctx)
    docs = ctx.get("/dev/docs").json()["documents"]
    assert docs[0]["source"] == "uploads/a.txt" and docs[0]["upload"] and docs[0]["on_disk"]
    j = wait_job(ctx, ctx.post("/dev/docs/reingest?source=uploads/a.txt").json()["job_id"])
    assert j["files"][0]["stage"] == "done"                                  # force: not skipped
    r = ctx.delete("/dev/docs?source=uploads/a.txt").json()
    assert r == {"deleted": "uploads/a.txt", "file_removed": True} and ctx.deleted == ["uploads/a.txt"]
    assert not (ctx.raw / "uploads" / "a.txt").exists()
    assert ctx.delete("/dev/docs?source=uploads/a.txt").status_code == 404


def test_reingest_rejects_path_traversal_and_unknown(ctx):
    login(ctx)
    assert ctx.post("/dev/docs/reingest?source=../../etc/passwd").status_code == 404
    assert ctx.post("/dev/docs/reingest?source=uploads/none.txt").status_code == 404


def test_settings_hide_secrets(ctx):
    login(ctx)
    d = ctx.get("/dev/settings").json()
    assert "gemini_model" in d and not any(k for k in d if "key" in k or "password" in k or "secret" in k)
    assert "pw" not in str(d.get("neo4j_password", ""))


# ---------- sidebar data, suggestions, progress fields ----------
def test_job_exposes_progress_fields(ctx):
    j = wait_job(ctx, post_files(ctx, ("a.txt", b"hello there", "text/plain")).json()["job_id"])
    f = j["files"][0]
    assert f["searchable"] is True and f["llm_calls"] == 4 and f["started"] and j["server_time"] >= f["started"]
    assert f["elapsed_ms"] is not None


def test_kb_docs_lists_names_and_counts_only(ctx):
    assert ctx.get("/kb/docs").json() == {"documents": []}
    wait_job(ctx, post_files(ctx, ("a.txt", b"hello there", "text/plain")).json()["job_id"])
    wait_job(ctx, post_files(ctx, ("b.txt", b"another text", "text/plain")).json()["job_id"])
    d = ctx.get("/kb/docs").json()["documents"]
    assert [x["name"] for x in d] == ["b.txt", "a.txt"]                     # newest first
    assert set(d[0]) == {"source", "name", "chunks", "graph", "ts"} and d[0]["source"] == "uploads/b.txt"


def test_kb_suggest_uses_entities_without_llm(ctx):
    assert ctx.get("/kb/suggest").json() == {"questions": []}                # nothing ingested yet
    wait_job(ctx, post_files(ctx, ("a.txt", b"hello there", "text/plain")).json()["job_id"])
    q = ctx.get("/kb/suggest").json()["questions"]
    assert "Summarize Acme" in q and "Which companies is Bob connected to?" in q and len(q) <= 5


# ---------- evaluation viewer + extraction check ----------
def test_eval_reports_are_dev_only_and_path_safe(ctx, tmp_path, monkeypatch):
    from app import evalview
    run = tmp_path / "results" / "20260101-000000"
    run.mkdir(parents=True)
    (run / "report.md").write_text("# Evaluation report\n\n## Mode: graph\n\n- Answer correctness: **50% (1/2)**\n", encoding="utf-8")
    (tmp_path / "secret.md").write_text("TOP SECRET", encoding="utf-8")
    monkeypatch.setattr(evalview, "RESULTS", tmp_path / "results")
    assert ctx.get("/dev/eval/reports").status_code == 401
    login(ctx)
    assert [r["name"] for r in ctx.get("/dev/eval/reports").json()["reports"]] == ["20260101-000000"]
    assert "Answer correctness" in ctx.get("/dev/eval/report?name=20260101-000000").json()["text"]
    m = ctx.get("/dev/eval/report?name=20260101-000000&summary=true").json()["summary"]["modes"][0]
    assert (m["mode"], m["correct_pct"], m["total"]) == ("graph", 50, 2)
    for bad in ("..", "../secret", "..%2Fsecret", "20260101-000000/../../secret", "nope"):
        assert ctx.get(f"/dev/eval/report?name={bad}").status_code in (404, 422), bad


def test_extraction_endpoint_locked_and_returns_result(ctx, monkeypatch):
    import eval.check_graph as cg
    assert ctx.get("/dev/extraction").status_code == 401
    monkeypatch.setattr(cg, "compute", lambda corpus=True, **k: {"empty": False, "precision": 0.9, "recall": 0.8})
    login(ctx)
    assert ctx.get("/dev/extraction").json()["precision"] == 0.9


def test_app_name_and_tagline_come_from_env_and_are_escaped(ctx, monkeypatch):
    monkeypatch.setenv("APP_NAME", "Acme <Docs>")
    monkeypatch.setenv("APP_TAGLINE", "Ask & learn")
    get_settings.cache_clear()
    page = ctx.get("/").text
    assert "<title>Acme &lt;Docs&gt;</title>" in page and "Ask &amp; learn" in page
    assert "{{APP_NAME}}" not in page and "{{APP_TAGLINE}}" not in page
    assert "Acme &lt;Docs&gt;" in ctx.get("/login").text
    login(ctx)
    assert "Acme &lt;Docs&gt; · Developer" in ctx.get("/dev").text


def test_script_and_style_links_change_when_the_file_changes(ctx):
    """Otherwise a browser keeps running the old user.js after an update."""
    import re as _re
    page = ctx.get("/").text
    assert _re.search(r'/static/user\.js\?v=\d+', page) and _re.search(r'/static/user\.css\?v=\d+', page)
    assert ctx.get("/static/user.js?v=123").status_code == 200


def test_question_box_limit_follows_the_env_setting(ctx):
    page = ctx.get("/").text
    assert f'maxlength="{get_settings().max_question_chars}"' in page and "{{" not in page
