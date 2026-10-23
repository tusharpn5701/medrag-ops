"""Project paths, params.yaml loading and runtime settings (environment variables)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(os.environ.get("MEDRAG_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FEEDBACK_FILE = ROOT / "data" / "feedback" / "eval_production.jsonl"


def load_params(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    """Runtime configuration for the API. Every field can be set as MEDRAG_<NAME>."""

    model_config = SettingsConfigDict(env_prefix="MEDRAG_", env_file=".env", extra="ignore")

    bundle_dir: Path = ARTIFACTS / "bundle"
    top_k: int = 5
    max_query_chars: int = 1000

    # generation (OpenAI-compatible endpoint; Groq by default)
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-20b"
    llm_timeout_s: float = 20.0
    prompt_path: Path = ROOT / "prompts" / "answer_v1.yaml"

    # request logging
    log_dir: Path = ROOT / "logs"
    log_bucket: str | None = None          # S3-compatible bucket for durable logs
    log_prefix: str = "request-logs"
    s3_endpoint_url: str | None = None     # e.g. DagsHub storage or Cloudflare R2
    log_flush_records: int = 25
    log_flush_seconds: float = 60.0
    admin_token: str | None = None         # protects /admin endpoints


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    # Accept the provider's usual variable name as well
    if s.llm_api_key is None and os.environ.get("GROQ_API_KEY"):
        s.llm_api_key = os.environ["GROQ_API_KEY"]
    return s
