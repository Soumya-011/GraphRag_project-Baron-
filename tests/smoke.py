from pydantic import BaseModel

from app.llm import get_llm, get_structured_llm
from app.tools import graph_stats


class Triple(BaseModel):
    source: str
    relation: str
    target: str


def main():
    print("1) plain call  :", get_llm().invoke("Reply with the single word OK").content)

    out = get_structured_llm(Triple).invoke(
        "Elon Musk founded SpaceX. Return source, relation (FOUNDED) and target."
    )
    print("2) structured  :", out)
    assert isinstance(out, Triple), "model failed structured output: pick a bigger model"

    print("3) graph tool  :", graph_stats.invoke({}))
    print("ALL OK")


if __name__ == "__main__":
    main()