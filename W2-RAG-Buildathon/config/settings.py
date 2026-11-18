"""Central application settings loaded from environment variables (.env).

Both the documented variable names (LLM_BASE_URL, LLM_MODEL) and the legacy
names already present in some local .env files (WATSONX_BASE_URL,
WATSONX_MODEL_NAME) are accepted. The token is never logged or printed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_LLM_BASE_URL = "https://maas.apps.wdc04-prod-ocp-01.cl.sec.ibm.com/models/gpt-oss-120b/v1"
DEFAULT_LLM_MODEL = "gpt-oss-120b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    # LLM (IBM watsonx.ai MaaS, OpenAI-compatible endpoint)
    watsonx_token: str = field(repr=False)
    llm_base_url: str
    llm_model: str
    llm_temperature: float
    llm_timeout: int

    # Embeddings / chunking / retrieval
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    history_turns: int

    # Storage
    documents_dir: Path
    vectorstore_dir: Path
    faiss_index_dir: Path
    registry_path: Path
    max_upload_mb: int

    # Frontend -> backend
    backend_url: str

    @property
    def llm_configured(self) -> bool:
        return bool(self.watsonx_token and self.llm_base_url and self.llm_model)

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems (empty = OK)."""
        problems: list[str] = []
        if not self.watsonx_token:
            problems.append("WATSONX_TOKEN is not set. Add it to .env (see .env.example).")
        if not self.llm_base_url.startswith(("http://", "https://")):
            problems.append(f"LLM_BASE_URL is not a valid URL: {self.llm_base_url!r}")
        if not self.llm_model:
            problems.append("LLM_MODEL is not set.")
        if self.chunk_size <= 0:
            problems.append("CHUNK_SIZE must be > 0.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            problems.append("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")
        if self.top_k <= 0:
            problems.append("TOP_K must be > 0.")
        return problems

    def ensure_dirs(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


def build_settings() -> Settings:
    documents_dir = Path(_env("DOCUMENTS_DIR", default=str(PROJECT_ROOT / "documents"))).resolve()
    vectorstore_dir = Path(_env("VECTORSTORE_DIR", default=str(PROJECT_ROOT / "vectorstore"))).resolve()
    return Settings(
        watsonx_token=_env("WATSONX_TOKEN"),
        llm_base_url=_env("LLM_BASE_URL", "WATSONX_BASE_URL", default=DEFAULT_LLM_BASE_URL).rstrip("/"),
        llm_model=_env("LLM_MODEL", "WATSONX_MODEL_NAME", default=DEFAULT_LLM_MODEL),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.1),
        llm_timeout=_env_int("LLM_TIMEOUT", 120),
        embedding_model=_env("EMBEDDING_MODEL", default=DEFAULT_EMBEDDING_MODEL),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
        top_k=_env_int("TOP_K", 5),
        history_turns=_env_int("HISTORY_TURNS", 3),
        documents_dir=documents_dir,
        vectorstore_dir=vectorstore_dir,
        faiss_index_dir=vectorstore_dir / "faiss_index",
        registry_path=vectorstore_dir / "document_registry.json",
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 50),
        backend_url=_env("BACKEND_URL", default="http://127.0.0.1:8000").rstrip("/"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return build_settings()
