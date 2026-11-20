"""Extra node types (Year), generated relation rules, per-task API keys, settings that left the code.
No network, no Neo4j, no Gemini."""
import os
import sys
import types

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ["CACHE_ENABLED"] = "false"

import pytest

from app import extract, llm, rag, router, schema, stores
from app.config import ROLES, Settings, get_config, get_settings


@pytest.fixture
def env(monkeypatch):
    """Settings for the test come from the variables set here and the code defaults, NEVER from your .env
    (blank values in .env are ignored, so a real EXTRA_NODE_TYPES=Year there would leak into 'off' tests)."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()

    def setenv(**kw):
        for k, v in kw.items():
            monkeypatch.setenv(k, v)
        get_settings.cache_clear()
    yield setenv
    get_settings.cache_clear()


def ext(*rels):
    return extract.Extraction(relations=[extract.Relation(**r) for r in rels])


def year_rel(src="Acme Corp", year="2010", evidence="Acme acquired Zed Inc in 2010", **kw):
    return {"source": src, "source_type": "Company", "relation": "IN_YEAR", "target": year,
            "target_type": "Year", "evidence": evidence, **kw}


# ======================= node types and generated rules =======================
def test_off_by_default_nothing_changes(env):
    env(EXTRA_NODE_TYPES="")
    assert schema.extra_labels() == []
    assert schema.relation_rules() == get_config()["relation_rules"]
    assert schema.graph_labels() == ("Person", "Company", "Location")


def test_year_rule_is_generated_from_env_and_yaml(env):
    env(EXTRA_NODE_TYPES="year")           # lower case is accepted
    assert schema.extra_labels() == ["Year"]
    rule = schema.relation_rules()["IN_YEAR"]
    assert rule["target"] == "Year" and schema.sources_of(rule) == ["Person", "Company", "Location"]
    assert rule["multi"] is True
    assert "WORKS_AT" in schema.relation_rules()  # the yaml rules are still there
    assert schema.node_filter("n") == "(n:Person OR n:Company OR n:Location OR n:Year)"


def test_unknown_type_gets_default_rule(env):
    env(EXTRA_NODE_TYPES="Year, Industry")
    rule = schema.relation_rules()["IN_INDUSTRY"]
    assert rule["target"] == "Industry" and "industry" in rule["hint"]
    assert schema.extra_labels() == ["Year", "Industry"]


@pytest.mark.parametrize("bad", ["Year-DROP", "9lives", "Doc!", "Document"])
def test_bad_type_names_are_refused(env, bad):
    env(EXTRA_NODE_TYPES=bad)
    with pytest.raises(ValueError):
        schema.extra_labels()


def test_base_type_in_the_list_is_ignored(env):
    env(EXTRA_NODE_TYPES="Person,Year")
    assert schema.extra_labels() == ["Year"]


def test_extraction_prompt_lists_generated_rule(env):
    env(EXTRA_NODE_TYPES="Year")
    text = extract._prompt().format_messages(text="x")[0].content
    assert "IN_YEAR: Person | Company | Location -> Year" in text
    assert "Person, Company, Location, Year" in text and "{types}" not in text


def test_extraction_cache_key_changes_with_the_rules(env):
    env(EXTRA_NODE_TYPES="")
    a = extract._key("same text")
    env(EXTRA_NODE_TYPES="Year")
    assert extract._key("same text") != a


# ======================= validation of Year facts =======================
TEXT = "In March 2010 Acme Corp acquired Zed Inc. Acme Corp also opened an office in 2010."


def test_year_fact_is_kept_and_value_is_cleaned(env):
    env(EXTRA_NODE_TYPES="Year")
    m, rels, dropped = extract.validate(ext(year_rel(src="Acme", year="March 2010")), "Acme acquired Zed Inc in March 2010")
    assert dropped == 0
    assert rels[0]["type"] == "IN_YEAR" and rels[0]["target"] == "2010" and rels[0]["target_label"] == "Year"
    assert ("2010", "Year") in m and ("Acme", "Company") in m


def test_year_fact_without_a_quote_is_dropped(env):
    env(EXTRA_NODE_TYPES="Year")
    _, rels, dropped = extract.validate(ext(year_rel(evidence="")), TEXT)
    assert rels == [] and dropped == 1


def test_year_must_appear_in_the_text(env):
    env(EXTRA_NODE_TYPES="Year")
    _, rels, dropped = extract.validate(ext(year_rel(year="1999")), TEXT)
    assert rels == [] and dropped == 1


def test_value_that_is_not_a_year_is_dropped(env):
    env(EXTRA_NODE_TYPES="Year")
    t = "Acme Corp grew a lot in the spring."
    _, rels, dropped = extract.validate(ext(year_rel(year="spring", evidence="grew a lot in the spring")), t)
    assert rels == [] and dropped == 1


def test_reversed_year_fact_is_flipped(env):
    env(EXTRA_NODE_TYPES="Year")
    r = {"source": "2010", "source_type": "Year", "relation": "IN_YEAR", "target": "Acme Corp",
         "target_type": "Company", "evidence": "Acme acquired Zed Inc in 2010"}
    _, rels, _ = extract.validate(ext(r), TEXT)
    assert rels[0]["source"] == "Acme Corp" and rels[0]["target"] == "2010"


def test_year_relation_unknown_when_feature_is_off(env):
    env(EXTRA_NODE_TYPES="")
    _, rels, dropped = extract.validate(ext(year_rel()), TEXT)
    assert rels == [] and dropped == 1


def test_two_events_of_one_entity_in_one_year_both_survive(env):
    env(EXTRA_NODE_TYPES="Year")
    a = year_rel(evidence="Acme Corp acquired Zed Inc in March 2010")
    b = year_rel(evidence="Acme Corp also opened an office in 2010")
    mentions, rels, _ = extract.validate(ext(a, b), TEXT)
    assert len(rels) == 2
    res = extract.Resolver(88, {}, {})
    ents, out = extract.resolve_all(mentions, rels, res)
    assert len(out) == 2 and sorted(e["id"] for e in ents if e["label"] == "Year") == ["2010"]


def test_neighbouring_years_never_merge(env):
    env(EXTRA_NODE_TYPES="Year")
    res = extract.Resolver(50, {}, {})            # even with a very loose fuzzy threshold
    assert res.resolve("2010", "Year") != res.resolve("2011", "Year")


def test_normal_relations_are_unaffected(env):
    env(EXTRA_NODE_TYPES="Year")
    r = {"source": "Jane Roe", "source_type": "Person", "relation": "CEO_OF", "target": "Acme Corp",
         "target_type": "Company", "evidence": "Jane Roe is CEO of Acme Corp"}
    _, rels, dropped = extract.validate(ext(r), "Jane Roe is CEO of Acme Corp")
    assert dropped == 0 and rels[0]["type"] == "CEO_OF"


# ======================= graph writing and walking =======================
class FakeGraph:
    def __init__(self):
        self.queries = []

    def query(self, q, params=None):
        self.queries.append(q)
        return []


def test_graph_write_keeps_one_edge_per_quote_for_year_links(env, monkeypatch):
    env(EXTRA_NODE_TYPES="Year")
    g = FakeGraph()
    monkeypatch.setattr(stores, "get_graph", lambda: g)
    ents = [{"id": "acme corp", "name": "Acme Corp", "label": "Company"}, {"id": "2010", "name": "2010", "label": "Year"}]
    rels = [{"s": "acme corp", "t": "2010", "type": "IN_YEAR", "sl": "Company", "tl": "Year",
             "status": "current", "evidence": "x"}]
    stores.graph_write("doc.txt", ents, rels)
    joined = "\n".join(g.queries)
    assert "MERGE (n:Year {id: row.id})" in joined
    assert "[r:IN_YEAR {source: $source, evidence: row.evidence}]" in joined
    assert "CREATE CONSTRAINT year_id" in joined


def test_graph_write_normal_edge_merge_is_unchanged(env, monkeypatch):
    env(EXTRA_NODE_TYPES="")
    g = FakeGraph()
    monkeypatch.setattr(stores, "get_graph", lambda: g)
    rels = [{"s": "a", "t": "b", "type": "FOUNDED", "sl": "Person", "tl": "Company", "status": "current", "evidence": ""}]
    stores.graph_write("d.txt", [], rels)
    assert any("[r:FOUNDED {source: $source}]" in q for q in g.queries)


def test_relation_union_splits_base_and_extra(env):
    env(EXTRA_NODE_TYPES="Year")
    assert "IN_YEAR" not in stores._rel_union("base").split("|")
    assert stores._rel_union("extra") == "IN_YEAR"
    assert "IN_YEAR" in stores._rel_union("all").split("|")


def test_unknown_relation_cannot_enter_cypher(env):
    env(EXTRA_NODE_TYPES="Year")
    with pytest.raises(ValueError):
        stores._ident("IN_YEAR) DETACH DELETE (n", list(schema.relation_rules()))


def test_gather_graph_walks_normal_entities_without_crossing_years(env, monkeypatch):
    env(EXTRA_NODE_TYPES="Year")
    calls = []
    monkeypatch.setattr(rag, "graph_neighborhood", lambda ids, hops, limit, kind="all": calls.append((ids, hops, kind)) or [])
    monkeypatch.setattr(rag, "graph_shortest_path", lambda a, b, n=None, kind="all": calls.append(("path", kind)) or None)
    rag.gather_graph([{"id": "acme corp", "name": "Acme Corp", "label": "Company"},
                      {"id": "zed inc", "name": "Zed Inc", "label": "Company"}])
    assert (["acme corp", "zed inc"], 3, "base") in calls      # normal walk skips IN_YEAR
    assert (["acme corp", "zed inc"], 1, "extra") in calls     # plus their own dated events
    assert ("path", "base") in calls


def test_gather_graph_for_a_year_node(env, monkeypatch):
    env(EXTRA_NODE_TYPES="Year", EXTRA_NODE_HOPS="1")
    calls = []
    monkeypatch.setattr(rag, "graph_neighborhood", lambda ids, hops, limit, kind="all": calls.append((ids, hops, kind)) or [])
    monkeypatch.setattr(rag, "graph_shortest_path", lambda *a, **k: None)
    rag.gather_graph([{"id": "2010", "name": "2010", "label": "Year"}])
    assert calls == [(["2010"], 1, "all")]


def test_gather_graph_without_extras_is_the_old_behaviour(env, monkeypatch):
    env(EXTRA_NODE_TYPES="")
    calls = []
    monkeypatch.setattr(rag, "graph_neighborhood", lambda ids, hops, limit, kind="all": calls.append((hops, kind)) or [])
    monkeypatch.setattr(rag, "graph_shortest_path", lambda *a, **k: None)
    rag.gather_graph([{"id": "a", "name": "A", "label": "Company"}])
    assert calls == [(3, "all")]


def test_year_facts_read_as_plain_english(env):
    env(EXTRA_NODE_TYPES="Year")
    e = {"source": "Acme Corp", "rel": "IN_YEAR", "target": "2010", "status": "current",
         "evidence": "Acme acquired Zed Inc in 2010", "docs": ["news.txt"]}
    line = rag._edge_line(e)
    assert line.startswith("Acme Corp has an event in the year 2010") and "IN_YEAR" not in line
    assert '"Acme acquired Zed Inc in 2010"' in line
    block = rag.entity_blocks([e], ["2010"])
    assert any(b.startswith("2010: events involving Acme Corp") for b in block)


# ======================= routing a year question =======================
def test_year_question_goes_to_the_graph(env):
    env(EXTRA_NODE_TYPES="Year")
    ents = [{"id": "2010", "name": "2010", "label": "Year", "score": 100}]
    d = router.decide_route("What events happened in 2010?", ents, use_llm=False)
    assert d["route"] == "graph" and "year" in d["reason"]


def test_year_entity_is_linked_from_the_question(env):
    env(EXTRA_NODE_TYPES="Year")
    known = [{"id": "2010", "name": "2010", "label": "Year"}, {"id": "2011", "name": "2011", "label": "Year"}]
    found = router.link_entities("What events happened in 2010?", known)
    assert [f["name"] for f in found] == ["2010"]


def test_year_route_is_not_chosen_when_feature_is_off(env):
    env(EXTRA_NODE_TYPES="")
    ents = [{"id": "x", "name": "X", "label": "Company", "score": 100}]
    assert router.decide_route("X", ents, use_llm=False)["via"] == "default"


# ======================= several Gemini accounts =======================
def mk(**kw):
    base = dict(GEMINI_API_KEY="main", _env_file=None)
    base.update(kw)
    return Settings(**base)


def test_every_task_uses_the_main_key_by_default():
    s = mk()
    assert all(s.api_key_for(r) == "main" for r in ROLES)
    assert s.fallback_key_for("answer") == "main"


def test_each_task_can_have_its_own_key():
    s = mk(EXTRACTION_API_KEY="acct-b", ANSWER_API_KEY="acct-c")
    assert s.api_key_for("extraction") == "acct-b"
    assert s.api_key_for("answer") == "acct-c"
    assert s.api_key_for("router") == "main"


def test_fallback_key_alone_retries_the_same_model_on_the_other_account():
    s = mk(FALLBACK_MODEL="", FALLBACK_API_KEY="acct-b", ANSWER_MODEL="gemini-2.5-flash")
    assert s.fallback_for("answer") == "gemini-2.5-flash" and s.fallback_key_for("answer") == "acct-b"


def test_fallback_with_same_model_and_same_key_is_off():
    s = mk(FALLBACK_MODEL="gemini-2.5-flash", ANSWER_MODEL="gemini-2.5-flash")
    assert s.fallback_for("answer") is None


def test_fallback_model_on_the_same_account_still_works():
    s = mk(FALLBACK_MODEL="gemini-2.5-flash-lite", ANSWER_MODEL="gemini-2.5-flash")
    assert s.fallback_for("answer") == "gemini-2.5-flash-lite"


def test_fallback_off_when_nothing_is_set():
    assert mk(FALLBACK_MODEL="", FALLBACK_API_KEY="").fallback_for("answer") is None


def test_main_key_is_optional_when_every_task_has_its_own():
    s = mk(GEMINI_API_KEY="", ANSWER_API_KEY="a", EXTRACTION_API_KEY="b", ROUTER_API_KEY="c")
    assert s.api_key_for("router") == "c"


def test_missing_key_for_a_task_is_refused():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        mk(GEMINI_API_KEY="", ANSWER_API_KEY="a")


@pytest.fixture
def fake_gemini(monkeypatch):
    made = []

    class Fake:
        def __init__(self, **kw):
            made.append(kw)

        def with_structured_output(self, schema):
            return self

    monkeypatch.setitem(sys.modules, "langchain_google_genai", types.SimpleNamespace(ChatGoogleGenerativeAI=Fake))
    llm.build_model.cache_clear()
    llm._limiter.cache_clear()
    llm.get_llm.cache_clear()
    yield made
    llm.build_model.cache_clear()
    llm._limiter.cache_clear()
    llm.get_llm.cache_clear()


def test_models_are_built_with_the_task_key(env, fake_gemini):
    env(GEMINI_API_KEY="main", EXTRACTION_API_KEY="acct-b", FALLBACK_API_KEY="acct-c",
        GEMINI_MODEL="gemini-2.5-flash", ANSWER_MODEL="gemini-2.5-flash",
        EXTRACTION_MODEL="gemini-2.5-flash", FALLBACK_MODEL="gemini-2.5-flash")
    llm.structured_pair(extract.Extraction, "extraction")
    llm.get_llm("answer")
    keys = [(m["model"], m["google_api_key"]) for m in fake_gemini]
    assert ("gemini-2.5-flash", "acct-b") in keys     # extraction primary
    assert ("gemini-2.5-flash", "acct-c") in keys     # extraction fallback: same model, other account
    assert ("gemini-2.5-flash", "main") in keys       # answer primary


def test_rate_limiter_is_per_account_and_model(env, fake_gemini):
    env(GEMINI_RPM="5")
    assert llm._limiter("m", "a") is not llm._limiter("m", "b")
    assert llm._limiter("m", "a") is llm._limiter("m", "a")


def test_token_and_retry_settings_come_from_env(env, fake_gemini):
    env(LLM_TOKEN_MULTIPLIER="2", LLM_MIN_OUTPUT_TOKENS="100", LLM_MAX_RETRIES="4")
    llm.build_model("m", 500, "k")
    assert fake_gemini[-1]["max_output_tokens"] == 1000 and fake_gemini[-1]["max_retries"] == 4


def test_health_checks_each_key_once_and_never_shows_it(env, monkeypatch):
    env(GEMINI_API_KEY="main-key-1234", EXTRACTION_API_KEY="other-key-9999", FALLBACK_API_KEY="other-key-9999",
        FALLBACK_MODEL="gemini-2.5-flash-lite")
    seen = []

    def fake_get(url, params=None, headers=None, timeout=0):
        seen.append(headers["x-goog-api-key"])
        return types.SimpleNamespace(status_code=200 if headers["x-goog-api-key"] != "other-key-9999" else 403)

    monkeypatch.setattr(llm.httpx, "get", fake_get)
    h = llm.llm_health()
    assert sorted(set(seen)) == ["main-key-1234", "other-key-9999"] and len(seen) == 2
    assert h["ok"] is False and h["roles"]["extraction"]["ok"] is False and h["roles"]["answer"]["ok"] is True
    assert h["roles"]["answer"]["key"] == "...1234"
    assert "main-key-1234" not in str(h) and "other-key-9999" not in str(h)


def test_settings_page_hides_every_key(env):
    env(ANSWER_API_KEY="secret-a", FALLBACK_API_KEY="secret-f")
    shown = {k: v for k, v in get_settings().model_dump().items() if not any(h in k for h in ("key", "password", "secret"))}
    assert "secret-a" not in str(shown) and "secret-f" not in str(shown)


# ======================= settings and prompts that left the code =======================
def test_new_limits_have_the_old_defaults():
    s = mk()
    assert (s.link_max_entities, s.graph_path_max_len, s.list_limit, s.cache_max_rows, s.max_queued_jobs) == (3, 5, 100, 2000, 5)
    assert (s.llm_token_multiplier, s.llm_min_output_tokens, s.llm_max_retries) == (3, 1536, 1)


def test_extra_hops_are_validated():
    with pytest.raises(ValueError, match="EXTRA_NODE_HOPS"):
        mk(EXTRA_NODE_HOPS="4")


def test_link_limit_comes_from_env(env):
    env(LINK_MAX_ENTITIES="1")
    ents = [{"id": "alpha corp", "name": "Alpha Corp", "label": "Company"},
            {"id": "beta corp", "name": "Beta Corp", "label": "Company"}]
    assert len(router.link_entities("Alpha Corp and Beta Corp", ents)) == 1


def test_generic_words_can_be_added_in_yaml(monkeypatch):
    cfg = dict(get_config())
    cfg["router"] = {**cfg["router"], "generic_words": ["Dynamics"]}
    monkeypatch.setattr(router, "get_config", lambda: cfg)
    assert "dynamics" in router._generic() and "capital" in router._generic()


def test_prompt_override_from_yaml(monkeypatch):
    monkeypatch.setattr(rag, "get_config", lambda: {"prompts": {"graph_rules": "CUSTOM RULES"}})
    assert rag._override("graph_rules", "default") == "CUSTOM RULES"
    assert rag._override("vector", "default") == "default"
    monkeypatch.setattr(rag, "get_config", lambda: {})
    assert rag._override("vector", "default") == "default"


def test_extraction_prompt_override_from_yaml(env, monkeypatch):
    env(EXTRA_NODE_TYPES="")
    cfg = {**get_config(), "prompts": {"extraction": "MY PROMPT {types}\n{relations}{extra_rules}"}}
    monkeypatch.setattr(extract, "get_config", lambda: cfg)
    text = extract._prompt().format_messages(text="x")[0].content
    assert text.startswith("MY PROMPT Person, Company, Location") and "WORKS_AT" in text


# ======================= Chroma client is built once, even from many threads =======================
def test_vector_store_is_built_once_under_concurrency(env, monkeypatch):
    import threading
    import time

    built = []

    def slow_build():
        built.append(1)
        time.sleep(0.05)
        return object()

    monkeypatch.setattr(stores, "_build_vector_store", slow_build)
    stores.reset_vector_store()
    got = []
    threads = [threading.Thread(target=lambda: got.append(stores.get_vector_store())) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    stores.reset_vector_store()
    assert len(built) == 1 and len({id(x) for x in got}) == 1
