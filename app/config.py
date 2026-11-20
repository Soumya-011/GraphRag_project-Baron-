from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

ROLES = ("answer", "extraction", "router")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore", env_ignore_empty=True)

    app_name: str = "Threadline"
    app_tagline: str = "Follow the thread across all your documents."
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # ---- Gemini (free tier) is the only LLM provider ----
    gemini_api_key: str = ""  # default key for every task
    # Optional keys from OTHER Google accounts/projects (free-tier quota is per project). Blank = GEMINI_API_KEY.
    answer_api_key: str = ""
    extraction_api_key: str = ""
    router_api_key: str = ""
    fallback_api_key: str = ""  # key used by the fallback model (any role)
    gemini_model: str = "gemini-3.5-flash-lite"  # default model for every task
    # Per-task overrides (empty = gemini_model). Free-tier quotas are PER MODEL, so putting the
    # small router task on flash-lite keeps the flash quota for answers and extraction.
    answer_model: str = ""
    extraction_model: str = ""
    router_model: str = "gemini-3.1-flash-lite"
    # Used when the primary model errors (e.g. 429 quota) or returns nothing. Empty = off.
    fallback_model: str = "gemini-3.1-flash-lite"
    # Client-side cap on calls per minute, per model. Free tier is about 10 (flash) / 15 (lite);
    # keep it a little below. 0 = no limiter.
    gemini_rpm: int = 9

    llm_temperature: float = 0.0
    llm_max_tokens: int = 384  # Gemini "thinking" tokens share the cap, so llm.py multiplies it
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 1
    warmup_on_start: bool = True     # load models in the background right after the server starts
    llm_token_multiplier: int = 3   # Gemini "thinking" tokens share the output cap
    llm_min_output_tokens: int = 1536
    extraction_max_tokens: int = 768

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "documents"

    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_vector: int = 4
    graph_max_hops: int = 3
    graph_max_edges: int = 80
    max_context_tokens: int = 4000

    max_question_chars: int = 500
    min_relevance: float = 0.25
    off_topic_enabled: bool = True
    off_topic_threshold: float = 0.55
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.92
    cache_scan_limit: int = 300
    cache_db_path: str = "./data/cache.sqlite"
    enable_grounding_check: bool = True
    entity_fuzzy_threshold: int = 88
    link_max_entities: int = 3       # entities taken from one question
    graph_path_max_len: int = 5      # longest shortest-path searched between two entities
    list_limit: int = 100            # "list all companies" answers
    cache_max_rows: int = 2000
    max_queued_jobs: int = 5         # uploads waiting for the single ingest worker
    # Extra node types, comma separated (e.g. Year or Year,Industry). Relation rules are generated.
    extra_node_types: str = ""
    extra_node_hops: int = 1         # hops walked from a Year-like node (1 = only its own events)

    # ---- developer page + public limits ----
    dev_password: str = ""           # empty = developer page open (laptop only)
    dev_session_secret: str = ""     # empty = random per start (logins end on restart)
    dev_session_hours: int = 8
    dev_max_login_attempts: int = 5  # failed logins per minute per IP
    max_upload_mb: int = 5
    max_files_per_upload: int = 5
    upload_rate_per_hour: int = 10
    ask_rate_per_minute: int = 20
    skip_already_ingested: bool = True

    router_use_llm_fallback: bool = True  # ask the router model only when rules cannot decide
    default_route: Literal["graph", "vector", "hybrid"] = "hybrid"  # used if that call fails

    def model_for(self, role: str) -> str:
        """Gemini model used by a task: 'answer', 'extraction' or 'router'."""
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        return getattr(self, f"{role}_model") or self.gemini_model

    def api_key_for(self, role: str) -> str:
        """API key used by a task: its own *_API_KEY, else GEMINI_API_KEY."""
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        return getattr(self, f"{role}_api_key") or self.gemini_api_key

    def fallback_key_for(self, role: str) -> str:
        return self.fallback_api_key or self.api_key_for(role)

    def fallback_for(self, role: str) -> str | None:
        """Fallback model for a task, or None when off or identical (same model AND same key) to the primary.
        FALLBACK_API_KEY alone (no FALLBACK_MODEL) retries the same model on the other account."""
        primary = self.model_for(role)
        fb = self.fallback_model or (primary if self.fallback_api_key else "")
        if not fb:
            return None
        same = fb == primary and self.fallback_key_for(role) == self.api_key_for(role)
        return None if same else fb

    @model_validator(mode="after")
    def _check(self):
        if any(not self.api_key_for(r) for r in ROLES):
            raise ValueError("GEMINI_API_KEY is required (free key: https://aistudio.google.com/apikey), "
                             "unless ANSWER_API_KEY, EXTRACTION_API_KEY and ROUTER_API_KEY are all set")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not 1 <= self.graph_max_hops <= 3:
            raise ValueError("GRAPH_MAX_HOPS must be between 1 and 3")
        if not 1 <= self.extra_node_hops <= 3:
            raise ValueError("EXTRA_NODE_HOPS must be between 1 and 3")
        return self

    @property
    def chroma_dir(self) -> Path:
        return (ROOT / self.chroma_persist_dir).resolve()

    @property
    def cache_path(self) -> Path:
        return (ROOT / self.cache_db_path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
