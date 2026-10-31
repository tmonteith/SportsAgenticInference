"""Application configuration backed by environment variables and .env files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for SportsAgenticInference."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # FastAPI Gateway (exposed to SportsAgenticAgent)
    host: str = Field(
        default="0.0.0.0",
        description="Bind host for the FastAPI inference gateway",
    )
    port: int = Field(
        default=8081,
        ge=1,
        le=65535,
        description="Port for the FastAPI inference gateway (SportsAgenticAgent expects 8081)",
    )
    log_level: str = Field(
        default="INFO",
        description="Application logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    # Model Configuration (microsoft/Phi-4-mini-instruct)
    model_id: str = Field(
        default="microsoft/Phi-4-mini-instruct",
        description="Canonical Hugging Face model identifier",
    )
    model_alias: str = Field(
        default="local-model",
        description="Model alias accepted from SportsAgenticAgent (default: local-model)",
    )
    hf_gguf_repo: str = Field(
        default="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        description="Hugging Face repository hosting GGUF weights for microsoft/Phi-4-mini-instruct",
    )
    hf_gguf_filename: str = Field(
        default="microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        description="GGUF filename to download and load into llama-server",
    )
    model_cache_dir: str = Field(
        default="models",
        description="Local directory where GGUF model artifacts are stored",
    )
    auto_download_model: bool = Field(
        default=True,
        description="Whether to download the GGUF model automatically on startup if missing",
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="Optional Hugging Face API token for model downloads",
    )

    # Internal llama-server Subprocess & Proxy Configuration
    llama_server_bin: str = Field(
        default="llama-server",
        description="Path or binary name for the llama.cpp llama-server executable",
    )
    llama_internal_host: str = Field(
        default="127.0.0.1",
        description="Internal host where the managed llama-server subprocess listens",
    )
    llama_internal_port: int = Field(
        default=8082,
        ge=1,
        le=65535,
        description="Internal port where the managed llama-server subprocess listens",
    )
    llama_ctx_size: int = Field(
        default=4096,
        ge=512,
        description="Context window size (--ctx-size) passed to llama-server",
    )
    llama_threads: int = Field(
        default=4,
        ge=1,
        description="CPU thread count (--threads) passed to llama-server",
    )
    llama_n_gpu_layers: int = Field(
        default=0,
        ge=0,
        description="Number of GPU layers (--n-gpu-layers) offloaded by llama-server",
    )
    llama_startup_timeout_seconds: float = Field(
        default=300.0,
        gt=0.0,
        description="Maximum time in seconds to wait for llama-server to pass health checks on startup",
    )
    llama_request_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Timeout in seconds for proxied inference requests to llama-server",
    )
    auto_start_llama_server: bool = Field(
        default=True,
        description="Whether the FastAPI lifespan should automatically spawn llama-server",
    )
    external_llama_url: Optional[str] = Field(
        default=None,
        description="Optional external llama-server base URL; when set, subprocess management is skipped",
    )

    @property
    def resolved_model_path(self) -> Path:
        """Full filesystem path to the configured GGUF model file."""
        return Path(self.model_cache_dir) / self.hf_gguf_filename

    @property
    def resolved_llama_base_url(self) -> str:
        """Base URL of the upstream llama-server instance."""
        if self.external_llama_url:
            return self.external_llama_url.rstrip("/")
        return f"http://{self.llama_internal_host}:{self.llama_internal_port}"


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
