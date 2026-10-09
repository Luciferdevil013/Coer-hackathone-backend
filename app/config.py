"""Central configuration. Every tunable threshold lives here so detectors stay pure."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DG_", env_file=BACKEND_DIR / ".env", extra="ignore")

    data_dir: Path = BACKEND_DIR / "data"
    demo_dir: Path = BACKEND_DIR / "demo"
    models_dir: Path = BACKEND_DIR / "models"

    # --- Local ML models ---
    semantic_enabled: bool = True          # all-MiniLM-L6-v2 (falls back to string matching if absent)
    semantic_doc_threshold: float = 0.85   # cosine for "same document, reworded" (unrelated policies ~0.45)
    semantic_column_threshold: float = 0.50   # header matching also needs mutual-best + margin + same type
    anomaly_z_threshold: float = 3.5       # robust z a value must reach (on top of Isolation Forest)

    # --- LLM (optional; every AI feature has a deterministic fallback) ---
    llm_enabled: bool = True              # only used when an API key is present
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "low"               # planner + explanations are short, well-specified tasks
    llm_timeout_s: float = 45.0

    # --- Cost model (INR). Defaults: AWS S3 Standard vs Glacier Deep Archive, us-east-1 ---
    usd_to_inr: float = 88.0
    hot_usd_per_gb_month: float = 0.023
    archive_usd_per_gb_month: float = 0.00099

    # --- Detector thresholds ---
    stale_days: int = 365                 # untouched for a year => stale
    large_file_bytes: int = 25 * 1024**2  # "large" for content files on disk
    large_object_bytes: int = 5 * 1024**3 # "large" for cloud-inventory objects
    retention_days: int = 3 * 365         # DPDP Third Schedule default (3 years since last use)
    exposure_users: int = 10              # more users than this => over-exposed
    near_dup_threshold: float = 0.85
    row_dup_threshold: float = 0.80
    pii_column_match_ratio: float = 0.6
    max_rows_per_dataset: int = 250_000
    max_content_bytes: int = 300 * 1024**2
    pii_sample_values: int = 5_000
    usage_window_days: int = 365

    @property
    def workspaces_dir(self) -> Path:
        return self.data_dir / "workspaces"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.data_dir / 'dataguardian.db'}"

    @property
    def inr_hot_per_gb_year(self) -> float:
        return self.hot_usd_per_gb_month * 12 * self.usd_to_inr

    @property
    def inr_archive_per_gb_year(self) -> float:
        return self.archive_usd_per_gb_month * 12 * self.usd_to_inr

    @property
    def anthropic_key_present(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.workspaces_dir.mkdir(parents=True, exist_ok=True)
    return s
