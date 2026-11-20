"""Background warm-up: never blocks, never leaks errors, loads the embedding model once."""
import os
import sys
import threading
import time
import types

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ["WARMUP_ON_START"] = "false"
os.environ["CACHE_ENABLED"] = "false"

import pytest

from app import llm, warmup
from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    monkeypatch.setattr(warmup, "state", {"state": "idle", "step": "", "seconds": None, "errors": []})
    yield
    get_settings.cache_clear()


def test_run_survives_a_failing_step_and_still_ends_ready(monkeypatch):
    def boom():
        raise RuntimeError("secret path /home/x")

    monkeypatch.setattr(warmup, "STEPS", [("step one", boom), ("step two", lambda: None)])
    warmup.run()
    assert warmup.state["state"] == "ready" and warmup.state["errors"] == ["step one: RuntimeError"]
    assert warmup.public() == {"ready": True, "step": ""}
    assert "secret" not in str(warmup.public())


def test_public_view_reports_the_running_step(monkeypatch):
    seen = []
    monkeypatch.setattr(warmup, "STEPS", [("loading it", lambda: seen.append(warmup.public()))])
    warmup.run()
    assert seen == [{"ready": False, "step": "loading it"}] and warmup.public()["ready"] is True


def test_start_returns_immediately_and_runs_once(monkeypatch):
    monkeypatch.setenv("WARMUP_ON_START", "true")
    get_settings.cache_clear()
    gate, calls = threading.Event(), []
    monkeypatch.setattr(warmup, "STEPS", [("slow", lambda: (calls.append(1), gate.wait(2)))])
    t0 = time.perf_counter()
    warmup.start()
    warmup.start()                                    # second call is ignored
    assert time.perf_counter() - t0 < 0.5 and warmup.public()["ready"] is False
    gate.set()
    for _ in range(100):
        if warmup.public()["ready"]:
            break
        time.sleep(0.02)
    assert warmup.public()["ready"] is True and calls == [1]


def test_switched_off_counts_as_ready(monkeypatch):
    monkeypatch.setenv("WARMUP_ON_START", "false")
    get_settings.cache_clear()
    warmup.start()
    assert warmup.state["state"] == "off" and warmup.public()["ready"] is True


def test_real_steps_are_all_soft_when_nothing_is_installed_or_running(monkeypatch):
    for name in ("langchain_huggingface",):
        monkeypatch.setitem(sys.modules, name, None)  # import fails
    llm.get_embeddings.cache_clear()
    warmup.run()
    assert warmup.state["state"] == "ready" and any("embedding" in e for e in warmup.state["errors"])


def test_embedding_model_is_loaded_once_even_when_two_threads_ask(monkeypatch):
    made = []

    class Fake:
        def __init__(self, **kw):
            made.append(1)
            time.sleep(0.05)

    monkeypatch.setitem(sys.modules, "langchain_huggingface", types.SimpleNamespace(HuggingFaceEmbeddings=Fake))
    llm.get_embeddings.cache_clear()
    got = []
    ts = [threading.Thread(target=lambda: got.append(llm.get_embeddings())) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    llm.get_embeddings.cache_clear()
    assert len(made) == 1 and len({id(x) for x in got}) == 1
