"""Read-only access to evaluation reports (eval/results/<run>/report.md) for the developer page."""
import re
from pathlib import Path

from .config import ROOT

RESULTS = ROOT / "eval" / "results"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
MAX_BYTES = 400_000


def list_reports() -> list[dict]:
    if not RESULTS.is_dir():
        return []
    out = []
    for d in RESULTS.iterdir():
        f = d / "report.md"
        if d.is_dir() and _NAME.match(d.name) and f.is_file():
            st = f.stat()
            out.append({"name": d.name, "modified": st.st_mtime, "bytes": st.st_size})
    return sorted(out, key=lambda r: r["modified"], reverse=True)


def read_report(name: str) -> str | None:
    """Report text, or None for an unknown or unsafe name."""
    if not _NAME.match(name or ""):
        return None
    f = (RESULTS / name / "report.md").resolve()
    if not f.is_file() or not f.is_relative_to(RESULTS.resolve()):
        return None
    return f.read_text(encoding="utf-8", errors="replace")[:MAX_BYTES]


def _pct(m):
    return int(m.group(1)) if m else None


def summarize(text: str) -> dict:
    """Headline numbers per mode, taken from the markdown the runner wrote."""
    lines = text.splitlines()
    out = {"title": lines[0].lstrip("# ").strip() if lines else "",
           "settings": next((x.replace("Settings:", "").strip() for x in lines if x.startswith("Settings:")), ""),
           "modes": []}
    for block in re.split(r"(?m)^## Mode: ", text)[1:]:
        head, _, body = block.partition("\n")
        m = re.search(r"Answer correctness: \*\*(\d+)% \((\d+)/(\d+)\)", body)
        types = {}
        for row in re.finditer(r"(?m)^\| (\w[\w -]*?) \| (\d+)% \((\d+)/(\d+)\) \| (?:(\d+)%|n/a) \|$", body):
            types[row.group(1)] = {"correct_pct": int(row.group(2)), "n": int(row.group(4)),
                                   "recall_pct": int(row.group(5)) if row.group(5) else None}
        calls = re.search(r"LLM calls used: \*\*(\d+)\*\*", body)
        lat = re.search(r"^\| \w[^|]* \| \d+ \| \d+% \(\d+/\d+\) \| [\d.]+ \| (\d+) ms \| (\d+) ms \| (\d+) ms \|$", body, re.M)
        out["modes"].append({
            "mode": head.split("(")[0].strip(),
            "correct_pct": int(m.group(1)) if m else None,
            "correct": int(m.group(2)) if m else None, "total": int(m.group(3)) if m else None,
            "recall_pct": _pct(re.search(r"Mean recall[^*]*\*\*(\d+)%", body)),
            "precision_pct": _pct(re.search(r"Mean precision[^*]*\*\*(\d+)%", body)),
            "llm_calls": int(calls.group(1)) if calls else None,
            "avg_ms": int(lat.group(1)) if lat else None, "p95_ms": int(lat.group(3)) if lat else None,
            "by_type": types,
        })
    return out
