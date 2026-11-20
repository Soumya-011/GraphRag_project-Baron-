"""Check every task's model and the Gemini fallback (a few tiny calls):
    python -m tests.llm_check
"""
import time

from pydantic import BaseModel

from app.config import ROLES, get_settings
from app.llm import fallback_events, get_llm, get_structured_llm, llm_health, reset_fallback_events


class Triple(BaseModel):
    source: str
    relation: str
    target: str


def main():
    s = get_settings()
    h = llm_health()
    for role, r in h["roles"].items():
        print(f"{role:11} {r['provider']}/{r['model']}: {'ok' if r['ok'] else 'PROBLEM ' + r.get('error', '')}")
    print("fallback   ", h["fallback"], "\n")

    for role in ROLES[:2]:
        reset_fallback_events()
        t = time.perf_counter()
        out = get_llm(role).invoke("Reply with the single word OK").content
        print(f"[{role}] plain     : {out!r}  {time.perf_counter() - t:.1f}s  fallback={fallback_events() or 'no'}")

    reset_fallback_events()
    t = time.perf_counter()
    try:
        out = get_structured_llm(Triple, "extraction").invoke(
            "Elon Musk founded SpaceX. Return source, relation (FOUNDED) and target.")
        print(f"[extraction] structured: {out}  {time.perf_counter() - t:.1f}s  fallback={fallback_events() or 'no'}")
    except Exception as e:
        print("[extraction] structured FAILED:", e)


if __name__ == "__main__":
    main()
