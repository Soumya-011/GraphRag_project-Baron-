"""Developer login, signed cookie, login throttle and rate limiter. Offline, no model needed."""
import os
import sys
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app import security
from app.config import get_settings


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("DEV_PASSWORD", "s3cret")
    monkeypatch.setenv("DEV_SESSION_SECRET", "unit-test-secret")
    monkeypatch.setenv("DEV_SESSION_HOURS", "1")
    monkeypatch.setenv("DEV_MAX_LOGIN_ATTEMPTS", "3")
    get_settings.cache_clear()
    security.clear_login_failures()
    security.limiter.reset()
    yield
    get_settings.cache_clear()


def test_password_check():
    assert security.check_password("s3cret")
    assert not security.check_password("s3cre")
    assert not security.check_password("")
    assert security.protected()


def test_empty_password_means_open(monkeypatch):
    monkeypatch.setenv("DEV_PASSWORD", "")
    get_settings.cache_clear()
    assert not security.protected() and security.check_password("anything")


def test_token_roundtrip_expiry_and_tampering():
    t = security.make_token(now=1_000_000)
    assert security.valid_token(t, now=1_000_000 + 10)
    assert not security.valid_token(t, now=1_000_000 + 3601)        # 1 hour session
    exp, sig = t.split(".")
    assert not security.valid_token(f"{int(exp) + 99999}.{sig}", now=1_000_000)  # extended expiry
    assert not security.valid_token(f"{exp}.{'0' * len(sig)}", now=1_000_000)
    for bad in (None, "", "abc", "x.y", "123"):
        assert not security.valid_token(bad)


def test_token_from_another_secret_is_rejected(monkeypatch):
    t = security.make_token()
    monkeypatch.setenv("DEV_SESSION_SECRET", "different")
    get_settings.cache_clear()
    assert not security.valid_token(t)


def test_login_lockout_after_max_failures_then_expires():
    ip = "1.2.3.4"
    for i in range(3):
        assert not security.login_blocked(ip, now=100 + i)
        security.record_login_failure(ip, now=100 + i)
    assert security.login_blocked(ip, now=110)
    assert not security.login_blocked(ip, now=100 + 61 + 3)   # window passed
    security.clear_login_failures(ip)
    assert not security.login_blocked(ip, now=110)


def test_rate_limiter_window():
    rl = security.RateLimiter()
    assert all(rl.allow("a", 3, 60, now=t) for t in (0, 1, 2))
    assert not rl.allow("a", 3, 60, now=3)
    assert rl.allow("b", 3, 60, now=3)                         # other key is independent
    assert rl.allow("a", 3, 60, now=61)                        # oldest hit expired
    assert rl.allow("a", 0, 60, now=0)                         # limit 0 = off
