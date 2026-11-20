"""Show what the extraction model returns for one file:
    python -m tests.extract_debug data\\raw\\meridian_capital.txt
"""
import sys
from pathlib import Path

from app.config import get_settings
from app.extract import Extraction, _prompt, validate
from app.llm import get_llm, get_structured_llm


def main(path: str):
    s = get_settings()
    provider, model = s.model_for("extraction")
    print(f"extraction model: {provider} / {model}")
    text = Path(path).read_text(encoding="utf-8-sig")

    raw = get_llm("extraction", s.extraction_max_tokens).invoke(_prompt().invoke({"text": text}))
    print("RAW   :", raw.content[:800])  # free-form reply (may not be JSON for ollama/gemini)

    try:
        parsed = (_prompt() | get_structured_llm(Extraction, "extraction")).invoke({"text": text})
    except Exception as e:
        print("ERROR :", str(e)[:400])
        return
    print("PARSED:", parsed)
    _, rels, dropped = validate(parsed, text)
    print("KEPT  :", [(r["source"], r["type"], r["target"]) for r in rels], "| dropped:", dropped)


if __name__ == "__main__":
    main(sys.argv[1])
