"""Developer-page login (signed cookie), failed-login throttle and per-IP rate limits. No state on disk."""
import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from .config import get_settings

COOKIE = "graphrag_dev"
_RUNTIME_SECRET = secrets.token_hex(32)  # used when DEV_SESSION_SECRET is empty


def protected() -> bool:
    return bool(get_settings().dev_password)


def _secret() -> bytes:
    return (get_settings().dev_session_secret or _RUNTIME_SECRET).encode()


def _sign(payload: str) -> str:
    return hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()


def check_password(given: str) -> bool:
    expected = get_settings().dev_password
    if not expected:
        return True
    a, b = hashlib.sha256((given or "").encode()).digest(), hashlib.sha256(expected.encode()).digest()
    return hmac.compare_digest(a, b)


def make_token(now: float | None = None) -> str:
    exp = int((now or time.time()) + get_settings().dev_session_hours * 3600)
    return f"{exp}.{_sign(str(exp))}"


def valid_token(token: str | None, now: float | None = None) -> bool:
    if not token or "." not in token:
        return False
    exp, sig = token.split(".", 1)
    if not exp.isdigit() or not hmac.compare_digest(sig, _sign(exp)):
        return False
    return int(exp) > (now or time.time())


def is_dev(request: Request) -> bool:
    """True when the developer area is open (no password set) or the request carries a valid login."""
    return (not protected()) or valid_token(request.cookies.get(COOKIE))


def require_dev(request: Request) -> None:
    """FastAPI dependency for every debug / admin endpoint."""
    if not is_dev(request):
        raise HTTPException(401, "Developer login required")


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Sliding window per key. `limit` and `window` are read per call so settings changes apply at once."""

    def __init__(self):
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float, now: float | None = None) -> bool:
        if limit <= 0:
            return True
        t = now if now is not None else time.time()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= t - window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(t)
            return True

    def retry_after(self, key: str, window: float, now: float | None = None) -> int:
        t = now if now is not None else time.time()
        with self._lock:
            q = self._hits.get(key)
            return max(1, int(q[0] + window - t) + 1) if q else 1

    def reset(self, key: str | None = None):
        with self._lock:
            self._hits.pop(key, None) if key else self._hits.clear()


limiter = RateLimiter()  # asks / uploads
_fail: dict[str, deque] = defaultdict(deque)
_fail_lock = threading.Lock()


def login_blocked(ip: str, now: float | None = None) -> bool:
    """True when this IP already failed DEV_MAX_LOGIN_ATTEMPTS times within the last minute."""
    limit, t = get_settings().dev_max_login_attempts, now if now is not None else time.time()
    with _fail_lock:
        q = _fail[ip]
        while q and q[0] <= t - 60:
            q.popleft()
        return limit > 0 and len(q) >= limit


def record_login_failure(ip: str, now: float | None = None) -> None:
    with _fail_lock:
        _fail[ip].append(now if now is not None else time.time())


def clear_login_failures(ip: str | None = None) -> None:
    with _fail_lock:
        _fail.pop(ip, None) if ip else _fail.clear()


def limit_or_429(request: Request, bucket: str, limit: int, window: float) -> None:
    if is_dev(request):
        return  # the developer is never throttled
    key = f"{bucket}:{client_ip(request)}"
    if not limiter.allow(key, limit, window):
        raise HTTPException(429, "Too many requests, please wait a moment.",
                            headers={"Retry-After": str(limiter.retry_after(key, window))})
