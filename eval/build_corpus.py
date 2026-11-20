"""Builds the stress corpus: 35 short documents, its gold graph and 37 evaluation questions.

    python -m eval.build_corpus            # (re)write data/raw/corpus/, eval/questions_corpus.yaml,
                                           # eval/gold_triples_corpus.yaml and print the evidence-spread stats

Everything comes from ONE world definition below, so the documents, the gold triples and the expected
answers cannot disagree. The corpus extends the 5 core documents in data/raw/ (same world, new entities).
Each document is one chunk (under 500 characters) and states its facts with explicit names, no pronouns.

Facts are scattered on purpose: each investment is stated in the investee's own profile, each acquisition
in its own news item, each Orbit Systems alumnus in their own bio. A question such as "Which companies has
Meridian Capital invested in?" therefore needs 9 different documents, far more than a top-k of 2 or 4.
"""
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORPUS_DIR = ROOT / "data" / "raw" / "corpus"
CORE_DIR = ROOT / "data" / "raw"

# ======================= the world =======================
COMPANIES = {  # name: facts
    "Kestrel Robotics": dict(city="Pune", sector="warehouse robotics", product="the Kestrel picker arm", year=2020, employees=64, founders=["Neha Joshi"], ceo="Neha Joshi", investors=["Meridian Capital"]),
    "Lumina Health": dict(city="Mumbai", sector="digital health", product="the Lumina triage app", year=2019, employees=120, founders=["Tara Bose", "Dev Malhotra"], ceo="Dev Malhotra", investors=["Meridian Capital", "Orchid Ventures"]),
    "TerraNest Agritech": dict(city="Jaipur", sector="precision agriculture", product="wireless soil sensors", year=2018, employees=85, founders=["Lakshmi Narayan"], ceo="Lakshmi Narayan", investors=["Meridian Capital"]),
    "Quanta Pay": dict(city="Bengaluru", sector="payments", product="a merchant payment gateway", year=2017, employees=210, founders=["Imran Sheikh"], ceo="Farhan Qureshi", investors=["Meridian Capital"]),
    "Aerolane Drones": dict(city="Hyderabad", sector="delivery drones", product="the AL-2 cargo drone", year=2016, employees=75, founders=["Ishaan Kulkarni"], ceo="Ishaan Kulkarni", investors=["Meridian Capital"]),
    "BrightPath Learning": dict(city="Delhi", sector="education technology", product="an adaptive tutoring platform", year=2019, employees=140, founders=["Pooja Sethi"], ceo="Pooja Sethi", investors=["Orchid Ventures"]),
    "CoastLine Logistics": dict(city="Kochi", sector="freight logistics", product="a port scheduling system", year=2015, employees=310, founders=["Rahul Banerjee"], ceo="Sunita Reddy", investors=["Blue Harbor Capital"]),
    "Vega Semiconductors": dict(city="Pune", sector="chip design", product="the V1 power controller chip", year=2018, employees=95, founders=["Meera Pillai"], ceo="Meera Pillai", investors=["Blue Harbor Capital"]),
    "HydroCore Water": dict(city="Chennai", sector="water treatment", product="modular desalination units", year=2014, employees=180, founders=["Karthik Subramanian"], ceo="Karthik Subramanian", investors=["Blue Harbor Capital"]),
    "Mosaic Analytics": dict(city="Bengaluru", sector="data analytics", product="a retail demand forecasting tool", year=2016, employees=90, founders=["Divya Chauhan"], ceo="Divya Chauhan", investors=["Orchid Ventures"]),
    "Zephyr Mobility": dict(city="Pune", sector="electric scooters", product="the Zephyr Z3 scooter", year=2019, employees=230, founders=["Manish Agarwal"], ceo="Manish Agarwal", investors=["Orchid Ventures"]),
    "Saffron Foods": dict(city="Ahmedabad", sector="packaged foods", product="ready-to-cook meal kits", year=2013, employees=400, founders=["Zoya Ansari"], ceo="Zoya Ansari", investors=["Orchid Ventures"]),
    "Ironbridge Materials": dict(city="Kolkata", sector="advanced materials", product="lightweight steel alloys", year=2012, employees=260, founders=["Varun Khanna"], ceo="Varun Khanna", investors=[]),
    "Nimbus Cloud": dict(city="Chennai", sector="cloud hosting", product="a managed Kubernetes service", year=2017, employees=150, founders=["Ananya Das"], ceo="Ananya Das", investors=["Orchid Ventures"]),
    "Cobalt Batteries": dict(city="Bengaluru", sector="battery storage", product="grid-scale battery packs", year=2018, employees=170, founders=["Sanjay Gupta"], ceo="Sanjay Gupta", investors=["Meridian Capital", "Blue Harbor Capital"]),
    "Delta Wave Networks": dict(city="Mumbai", sector="telecom equipment", product="5G small cells", year=2011, employees=520, founders=["Tushar Pandey"], ceo="Tushar Pandey", investors=[]),
    "Orchard Biotech": dict(city="Hyderabad", sector="biotechnology", product="a rapid diagnostic kit", year=2017, employees=110, founders=["Rekha Shenoy"], ceo="Rekha Shenoy", investors=["Meridian Capital"]),
    "Skyforge Aerospace": dict(city="Ahmedabad", sector="aerospace components", product="satellite brackets", year=2016, employees=130, founders=["Yash Thakur"], ceo="Yash Thakur", investors=[]),
}
INVESTORS = {
    "Orchid Ventures": dict(city="Delhi", founder="Nisha Bhatt", year=2014, focus="consumer and enterprise software"),
    "Blue Harbor Capital": dict(city="Kochi", founder="Harsh Vora", year=2012, focus="logistics and industrial"),
}
ACQUISITIONS = [  # acquirer, target, year
    ("SunArc Power", "HydroCore Water", 2023), ("SunArc Power", "Mosaic Analytics", 2022),
    ("SunArc Power", "Cobalt Batteries", 2024), ("NovaTech Labs", "Vega Semiconductors", 2024),
    ("Orbit Systems", "Aerolane Drones", 2021), ("Delta Wave Networks", "Zephyr Mobility", 2023),
]
PARTNERSHIPS = [  # a, b, year
    ("Kestrel Robotics", "NovaTech Labs", 2024), ("Skyforge Aerospace", "Orbit Systems", 2022),
    ("Saffron Foods", "TerraNest Agritech", 2023), ("Ironbridge Materials", "Cobalt Batteries", 2021),
    ("Nimbus Cloud", "SunArc Power", 2024),
]
ALUMNI = [  # person, years at Orbit Systems, what they worked on, what they did next
    ("Neha Joshi", (2012, 2016), "motor control boards", ("FOUNDED", "Kestrel Robotics")),
    ("Farhan Qureshi", (2010, 2015), "payment terminals", ("CEO_OF", "Quanta Pay")),
    ("Meera Pillai", (2013, 2017), "power electronics", ("FOUNDED", "Vega Semiconductors")),
    ("Sanjay Gupta", (2011, 2016), "battery management units", ("FOUNDED", "Cobalt Batteries")),
]
CORE_CITY = {"NovaTech Labs": "Pune", "Meridian Capital": "Mumbai", "GreenGrid Energy": "Bengaluru",
             "SunArc Power": "Chennai", "Orbit Systems": "Hyderabad"}
CORE_COMPANIES = set(CORE_CITY)

CUR, PAST = "current", "past"


def join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else " and ".join(names) if len(names) == 2 else \
        ", ".join(names[:-1]) + " and " + names[-1]


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


@dataclass
class Doc:
    name: str
    text: str
    triples: list = field(default_factory=list)  # (source, relation, target, status) stated in the text


# ======================= documents =======================
def city_of(company: str) -> str:
    return COMPANIES[company]["city"] if company in COMPANIES else \
        INVESTORS[company]["city"] if company in INVESTORS else CORE_CITY[company]


def profile(i: int, c: str, d: dict) -> Doc:
    founders = join(d["founders"])
    inv = ""
    if d["investors"]:
        verb = "both invested" if len(d["investors"]) > 1 else "invested"
        inv = f" {join(d['investors'])} {verb} in {c}."
    style = i % 3
    if style == 0:
        text = (f"{c} is a {d['sector']} company based in {d['city']}. {c} was founded in {d['year']} by {founders}. "
                f"{d['ceo']} is the CEO of {c}. {c} has {d['employees']} employees and its main product is {d['product']}.{inv}")
    elif style == 1:
        text = (f"{c} is headquartered in {d['city']} and employs {d['employees']} people. {founders} founded {c} in {d['year']}. "
                f"The CEO of {c} is {d['ceo']}. {c} makes {d['product']} for the {d['sector']} market.{inv}")
    else:
        text = (f"{founders} started {c} in {d['year']}. {c} is a {d['sector']} company located in {d['city']}. "
                f"The chief executive of {c} is {d['ceo']}. Today {c} has {d['employees']} employees and sells {d['product']}.{inv}")
    triples = [(c, "LOCATED_IN", d["city"], CUR), (d["ceo"], "CEO_OF", c, CUR)]
    triples += [(f, "FOUNDED", c, CUR) for f in d["founders"]]
    triples += [(v, "INVESTED_IN", c, CUR) for v in d["investors"]]
    return Doc(f"{slug(c)}.txt", text, triples)


def investor_doc(v: str, d: dict) -> Doc:
    text = (f"{v} is a venture capital firm based in {d['city']}. {d['founder']} founded {v} in {d['year']}. "
            f"{v} focuses on {d['focus']} startups and invests at the seed and Series A stage.")
    return Doc(f"{slug(v)}.txt", text, [(v, "LOCATED_IN", d["city"], CUR), (d["founder"], "FOUNDED", v, CUR)])


def acquisition_doc(a: str, t: str, year: int) -> Doc:
    d = COMPANIES[t]
    text = (f"In {year}, {a} acquired {t}, a {d['sector']} company based in {d['city']}. The acquisition added "
            f"{d['product']} to the offerings of {a}. The deal was announced at a press event in {d['city']}.")
    return Doc(f"news_{slug(a)}_{slug(t)}_{year}.txt", text,
               [(a, "ACQUIRED", t, CUR), (t, "LOCATED_IN", d["city"], CUR)])


def partnership_doc(a: str, b: str, year: int) -> Doc:
    text = (f"{a} and {b} announced a partnership in {year}. {a} will work with {b} on joint pilot projects. "
            f"The partnership starts with a one-year trial.")
    return Doc(f"partnership_{slug(a)}_{slug(b)}_{year}.txt", text, [(a, "PARTNERED_WITH", b, CUR)])


def alumni_doc(p: str, years: tuple, topic: str, nxt: tuple) -> Doc:
    rel, company = nxt
    city = COMPANIES[company]["city"]
    later = f"founded {company} in {city}" if rel == "FOUNDED" else f"became the CEO of {company} in {city}"
    text = (f"{p} previously worked at Orbit Systems in Hyderabad, from {years[0]} to {years[1]}. At Orbit Systems, "
            f"{p} worked on {topic}. {p} later {later}.")
    return Doc(f"bio_{slug(p)}.txt", text,
               [(p, "WORKS_AT", "Orbit Systems", PAST), ("Orbit Systems", "LOCATED_IN", "Hyderabad", CUR),
                (p, rel, company, CUR), (company, "LOCATED_IN", city, CUR)])


def build_docs() -> list[Doc]:
    docs = [profile(i, c, d) for i, (c, d) in enumerate(COMPANIES.items())]
    docs += [investor_doc(v, d) for v, d in INVESTORS.items()]
    docs += [acquisition_doc(*a) for a in ACQUISITIONS]
    docs += [partnership_doc(*p) for p in PARTNERSHIPS]
    docs += [alumni_doc(*a) for a in ALUMNI]
    return docs


# ======================= world graph =======================
def core_gold() -> list[tuple]:
    g = yaml.safe_load((HERE / "gold_triples.yaml").read_text(encoding="utf-8"))
    return [tuple(t) for t in g["gold"]], g["entities"]


def world(docs: list[Doc]):
    core, core_entities = core_gold()
    corpus = []
    for d in docs:
        for t in d.triples:
            if t not in corpus:
                corpus.append(t)
    people = set(core_entities["Person"])
    companies = set(core_entities["Company"])
    places = set(core_entities["Location"])
    for t in corpus:
        s, rel, o, _ = t
        (people if rel in ("FOUNDED", "CEO_OF", "WORKS_AT") else companies).add(s)
        (places if rel == "LOCATED_IN" else companies).add(o)
    return core, corpus, people, companies, places


# ======================= questions =======================
class G:
    """Tiny query helper over (source, relation, target, status) triples."""

    def __init__(self, triples):
        self.t = {(s, r, o) for s, r, o, _ in triples}

    def out(self, s, rel):
        return sorted({o for a, r, o in self.t if a == s and r == rel})

    def into(self, o, rel):
        return sorted({a for a, r, b in self.t if b == o and r == rel})

    def partners(self, c):
        return sorted({b for a, r, b in self.t if a == c and r == "PARTNERED_WITH"} |
                      {a for a, r, b in self.t if b == c and r == "PARTNERED_WITH"})

    def facts(self, pairs):  # support triples for a derived answer
        return sorted({x for x in pairs if x in self.t})


def questions(all_triples, companies, people):
    g = G(all_triples)
    Q = []

    def add(qid, text, typ, answers, support=(), hops=1, route=("graph", "hybrid"), **extra):
        def alt(name):  # companies may be answered by their first word ("NovaTech", "Meridian")
            return f"{name}|{name.split()[0]}" if name in companies and " " in name else name
        item = dict(id=qid, q=text, type=typ, route=list(route), hops=hops, a=[alt(x) for x in answers])
        item.update(extra)
        Q.append((item, list(support)))

    inv = lambda v: g.out(v, "INVESTED_IN")
    founded = lambda c: g.into(c, "FOUNDED")
    in_city = lambda city: g.into(city, "LOCATED_IN")
    alumni = g.into("Orbit Systems", "WORKS_AT")
    sup_inv = lambda v, cs: [(v, "INVESTED_IN", c) for c in cs]
    sup_loc = lambda cs: [(c, "LOCATED_IN", g.out(c, "LOCATED_IN")[0]) for c in cs]
    sup_fnd = lambda cs: [(p, "FOUNDED", c) for c in cs for p in founded(c)]

    # ---- aggregation: one relation, many documents ----
    mer = inv("Meridian Capital")
    add("ag01", "Which companies has Meridian Capital invested in?", "aggregation", mer, sup_inv("Meridian Capital", mer))
    for qid, city in [("ag02", "Pune"), ("ag03", "Bengaluru"), ("ag04", "Chennai")]:
        cs = in_city(city)
        add(qid, f"Which companies are located in {city}?", "aggregation", cs, sup_loc(cs))
    add("ag05", "List every person who has worked at Orbit Systems.", "aggregation", alumni,
        [(p, "WORKS_AT", "Orbit Systems") for p in alumni])
    sun = g.out("SunArc Power", "ACQUIRED")
    add("ag06", "Which companies has SunArc Power acquired?", "aggregation", sun,
        [("SunArc Power", "ACQUIRED", c) for c in sun])
    orc = inv("Orchid Ventures")
    add("ag07", "Which companies has Orchid Ventures invested in?", "aggregation", orc, sup_inv("Orchid Ventures", orc))
    add("ag08", "How many companies has Orchid Ventures invested in?", "aggregation", [f"{len(orc)}|{['zero','one','two','three','four','five','six','seven','eight','nine'][len(orc)]}"],
        sup_inv("Orchid Ventures", orc))
    add("ag09", "How many companies has SunArc Power acquired?", "aggregation",
        [f"{len(sun)}|{['zero','one','two','three','four','five','six','seven'][len(sun)]}"],
        [("SunArc Power", "ACQUIRED", c) for c in sun])
    add("ag10", "Who are the investors in Cobalt Batteries?", "aggregation", g.into("Cobalt Batteries", "INVESTED_IN"),
        [(v, "INVESTED_IN", "Cobalt Batteries") for v in g.into("Cobalt Batteries", "INVESTED_IN")])
    add("ag11", "Which investors have invested in Lumina Health?", "aggregation", g.into("Lumina Health", "INVESTED_IN"),
        [(v, "INVESTED_IN", "Lumina Health") for v in g.into("Lumina Health", "INVESTED_IN")])
    blue = inv("Blue Harbor Capital")
    add("ag12", "Which companies has Blue Harbor Capital invested in?", "aggregation", blue, sup_inv("Blue Harbor Capital", blue))

    # ---- multi-hop: two or three relations chained across documents ----
    fm = sorted({p for c in mer for p in founded(c)})
    add("mh01", "Which people founded companies that Meridian Capital invested in?", "multihop", fm,
        sup_inv("Meridian Capital", mer) + sup_fnd(mer), hops=2, route=("graph",))
    am = sorted(set(alumni) & set(fm))
    add("mh02", "Which Orbit Systems alumni founded companies that Meridian Capital invested in?", "multihop", am,
        [(p, "WORKS_AT", "Orbit Systems") for p in am] +
        [(p, "FOUNDED", c) for p in am for c in mer if p in founded(c)] +
        [("Meridian Capital", "INVESTED_IN", c) for p in am for c in mer if p in founded(c)], hops=3, route=("graph",))
    fs = sorted({p for c in sun for p in founded(c)})
    add("mh03", "Which people founded companies that SunArc Power acquired?", "multihop", fs,
        [("SunArc Power", "ACQUIRED", c) for c in sun] + sup_fnd(sun), hops=2, route=("graph",))
    cities = sorted({x for c in mer for x in g.out(c, "LOCATED_IN")})
    add("mh04", "In which cities are the companies that Meridian Capital invested in located?", "multihop", cities,
        sup_inv("Meridian Capital", mer) + sup_loc(mer), hops=2, route=("graph",))
    pune_partners = sorted(set(in_city("Pune")) & set(g.partners("NovaTech Labs")))
    add("mh05", "Which companies are located in Pune and partnered with NovaTech Labs?", "multihop", pune_partners,
        [(c, "LOCATED_IN", "Pune") for c in pune_partners] +
        [x for c in pune_partners for x in [(c, "PARTNERED_WITH", "NovaTech Labs"), ("NovaTech Labs", "PARTNERED_WITH", c)]],
        hops=2, route=("graph",))
    acq = g.into("Cobalt Batteries", "ACQUIRED")[0]
    add("mh06", "Who is the CEO of the company that acquired Cobalt Batteries?", "multihop", g.into(acq, "CEO_OF"),
        [(acq, "ACQUIRED", "Cobalt Batteries")] + [(p, "CEO_OF", acq) for p in g.into(acq, "CEO_OF")], hops=2, route=("graph",))
    invs = sorted({v for c in sun for v in g.into(c, "INVESTED_IN")})
    add("mh07", "Which investors have invested in companies that SunArc Power acquired?", "multihop", invs,
        [("SunArc Power", "ACQUIRED", c) for c in sun] + [(v, "INVESTED_IN", c) for c in sun for v in g.into(c, "INVESTED_IN")],
        hops=2, route=("graph",))
    alum_cos = sorted({c for p in alumni for c in g.out(p, "FOUNDED")})
    backers = sorted({v for c in alum_cos for v in g.into(c, "INVESTED_IN")})
    add("mh08", "Which investors have backed companies founded by Orbit Systems alumni?", "multihop", backers,
        [(p, "WORKS_AT", "Orbit Systems") for p in alumni if g.out(p, "FOUNDED")] +
        [(p, "FOUNDED", c) for p in alumni for c in g.out(p, "FOUNDED")] +
        [(v, "INVESTED_IN", c) for c in alum_cos for v in g.into(c, "INVESTED_IN")], hops=3, route=("graph",))
    ceo_alumni = sorted({p for c in mer for p in g.into(c, "CEO_OF") if p in alumni})
    add("mh09", "Which Orbit Systems alumni are now the CEO of a company that Meridian Capital invested in?", "multihop",
        ceo_alumni, [(p, "WORKS_AT", "Orbit Systems") for p in ceo_alumni] +
        [(p, "CEO_OF", c) for p in ceo_alumni for c in mer if p in g.into(c, "CEO_OF")] +
        [("Meridian Capital", "INVESTED_IN", c) for p in ceo_alumni for c in mer if p in g.into(c, "CEO_OF")],
        hops=3, route=("graph",))
    shared = sorted({c for v in g.into("Quanta Pay", "INVESTED_IN") for c in inv(v)} - {"Quanta Pay"})
    add("mh10", "Which companies share an investor with Quanta Pay?", "multihop", shared,
        [(v, "INVESTED_IN", "Quanta Pay") for v in g.into("Quanta Pay", "INVESTED_IN")] +
        [(v, "INVESTED_IN", c) for v in g.into("Quanta Pay", "INVESTED_IN") for c in shared if c in inv(v)],
        hops=2, route=("graph",))
    sun_blr = sorted(set(sun) & set(in_city("Bengaluru")))
    add("mh11", "Which companies acquired by SunArc Power are located in Bengaluru?", "multihop", sun_blr,
        [("SunArc Power", "ACQUIRED", c) for c in sun_blr] + [(c, "LOCATED_IN", "Bengaluru") for c in sun_blr],
        hops=2, route=("graph",))
    orb = g.out("Orbit Systems", "ACQUIRED")[0]
    add("mh12", "Who is the CEO of the company that Orbit Systems acquired?", "multihop", g.into(orb, "CEO_OF"),
        [("Orbit Systems", "ACQUIRED", orb)] + [(p, "CEO_OF", orb) for p in g.into(orb, "CEO_OF")], hops=2, route=("graph",))

    # ---- single facts hidden in a larger corpus (should stay easy) ----
    nd = [("nd01", "How many employees does Kestrel Robotics have?", "64", ("vector",)),
          ("nd02", "What is the main product of Nimbus Cloud?", "Kubernetes", ("vector",)),
          ("nd03", "In what year was Saffron Foods founded?", "2013", ("vector", "hybrid")),
          ("nd04", "What does HydroCore Water make?", "desalination", ("vector",)),
          ("nd05", "How many employees does Delta Wave Networks have?", "520", ("vector",))]
    for qid, text, ans, route in nd:
        add(qid, text, "needle", [ans], hops=1, route=route)
    for qid, text, ans, sup in [
        ("sh01", "Who founded Kestrel Robotics?", "Neha Joshi", [("Neha Joshi", "FOUNDED", "Kestrel Robotics")]),
        ("sh02", "Where is TerraNest Agritech located?", "Jaipur", [("TerraNest Agritech", "LOCATED_IN", "Jaipur")]),
        ("sh03", "Who is the CEO of Quanta Pay?", "Farhan Qureshi", [("Farhan Qureshi", "CEO_OF", "Quanta Pay")]),
        ("sh04", "Who acquired Vega Semiconductors?", "NovaTech Labs", [("NovaTech Labs", "ACQUIRED", "Vega Semiconductors")]),
        ("sh05", "Which company did Meera Pillai found?", "Vega Semiconductors", [("Meera Pillai", "FOUNDED", "Vega Semiconductors")])]:
        add(qid, text, "single", [ans], sup)

    # ---- unanswerable: the documents do not say ----
    for qid, text in [("un01", "Who is the CFO of Kestrel Robotics?"),
                      ("un02", "How much did SunArc Power pay for HydroCore Water?"),
                      ("un03", "What is the revenue of Lumina Health?")]:
        Q.append((dict(id=qid, q=text, type="guard", not_found=True), []))
    return Q


# ======================= evidence spread (structural, model-independent) =======================
def mentions(text: str, name: str, kind: str) -> bool:
    if name in text:
        return True
    parts = name.split()
    if kind == "person":
        return len(parts[-1]) >= 4 and parts[-1] in text
    if kind == "company":
        return len(parts) > 1 and len(parts[0]) >= 4 and parts[0] in text
    return False


def evidence_spread(Q, texts: dict, people, companies):
    kind = lambda n: "person" if n in people else "company" if n in companies else "place"
    ments = {d: {n for n in people | companies | {"Pune", "Mumbai", "Chennai", "Bengaluru", "Hyderabad", "Delhi",
                                               "Kochi", "Jaipur", "Ahmedabad", "Kolkata"} if mentions(t, n, kind(n))}
             for d, t in texts.items()}

    def covers(d, tr):
        return tr[0] in ments[d] and tr[2] in ments[d]

    for item, support in Q:
        if not support:
            continue
        support = list(dict.fromkeys(support))
        cand = [d for d in texts if any(covers(d, t) for t in support)]
        cand = sorted(cand, key=lambda d: -sum(covers(d, t) for t in support))[:22]

        def best(k):
            top = 0
            for combo in itertools.combinations(cand, min(k, len(cand))):
                top = max(top, sum(any(covers(d, t) for d in combo) for t in support))
            return round(100 * top / len(support))

        left, docs = set(support), 0  # greedy cover: about how many documents hold all the facts
        while left:
            d = max(cand, key=lambda d: sum(covers(d, t) for t in left))
            left -= {t for t in left if covers(d, t)}
            docs += 1
            if docs > len(cand):
                break
        item.update(facts=len(support), docs_needed=docs, cover_k2=best(2), cover_k4=best(4))


# ======================= build + write =======================
def core_texts() -> dict:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(CORE_DIR.iterdir())
            if p.is_file() and p.suffix in (".txt", ".md")}


def build():
    docs = build_docs()
    core, corpus, people, companies, places = world(docs)
    Q = questions(core + corpus, companies, people)
    texts = {**core_texts(), **{f"corpus/{d.name}": d.text for d in docs}}
    evidence_spread(Q, texts, people, companies)
    gold = {"entities": {"Person": sorted(people - set(core_gold()[1]["Person"])),
                         "Company": sorted(companies - set(core_gold()[1]["Company"])),
                         "Location": sorted(places - set(core_gold()[1]["Location"]))},
            "gold": [list(t) for t in corpus],
            "optional": [[p, "WORKS_AT", c, CUR] for p, rel, c, _ in corpus if rel in ("CEO_OF", "FOUNDED")]}
    return docs, gold, [q for q, _ in Q], dict(people=people, companies=companies, places=places, texts=texts)


def write(docs, gold, questions_):
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for old in CORPUS_DIR.glob("*.txt"):
        old.unlink()
    for d in docs:
        (CORPUS_DIR / d.name).write_text(d.text + "\n", encoding="utf-8")
    (HERE / "gold_triples_corpus.yaml").write_text(
        "# Generated by eval/build_corpus.py - do not edit by hand.\n"
        + yaml.safe_dump(gold, sort_keys=False, allow_unicode=True, width=200), encoding="utf-8")
    (HERE / "questions_corpus.yaml").write_text(
        "# Generated by eval/build_corpus.py - do not edit by hand.\n"
        "# facts / docs_needed / cover_k2 / cover_k4: how many facts the answer needs, in about how many documents they\n"
        "# are spread, and the MOST a perfect retriever could cover with 2 or 4 documents (percent of the facts).\n"
        + yaml.safe_dump(questions_, sort_keys=False, allow_unicode=True, width=200), encoding="utf-8")


def main():
    docs, gold, qs, _ = build()
    write(docs, gold, qs)
    print(f"{len(docs)} documents -> {CORPUS_DIR}")
    print(f"{len(qs)} questions -> {HERE / 'questions_corpus.yaml'}")
    print(f"{len(gold['gold'])} gold triples -> {HERE / 'gold_triples_corpus.yaml'}\n")
    print(f"{'id':5} {'facts':>5} {'docs':>5} {'best@2':>7} {'best@4':>7}  question")
    for q in qs:
        if "facts" in q:
            print(f"{q['id']:5} {q['facts']:5} {q['docs_needed']:5} {q['cover_k2']:6}% {q['cover_k4']:6}%  {q['q'][:70]}")


if __name__ == "__main__":
    sys.exit(main())
