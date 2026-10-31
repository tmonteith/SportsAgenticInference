"""Model manager for acquiring and verifying microsoft/Phi-4-mini-instruct GGUF weights."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, List, Optional

from huggingface_hub import hf_hub_download

from app.logging_config import get_logger
from app.schemas import ModelCard

logger = get_logger("model")


class ModelManagerError(Exception):
    """Raised when the GGUF model cannot be found or downloaded."""


class ModelManager:
    """Manages local caching and Hugging Face downloading of Phi-4-mini-instruct GGUF weights."""

    def __init__(
        self,
        model_id: str = "microsoft/Phi-4-mini-instruct",
        model_alias: str = "local-model",
        hf_gguf_repo: str = "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        hf_gguf_filename: str = "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        cache_dir: str | Path = "models",
        auto_download: bool = True,
        hf_token: Optional[str] = None,
        downloader: Optional[Callable[..., str]] = None,
    ) -> None:
        self.model_id = model_id
        self.model_alias = model_alias
        self.hf_gguf_repo = hf_gguf_repo
        self.hf_gguf_filename = hf_gguf_filename
        self.cache_dir = Path(cache_dir)
        self.auto_download = auto_download
        self.hf_token = hf_token
        self._downloader = downloader or hf_hub_download

    @property
    def model_path(self) -> Path:
        """Target path of the GGUF model file on disk."""
        return self.cache_dir / self.hf_gguf_filename

    def is_model_present(self) -> bool:
        """Return True if the GGUF file exists and is non-empty."""
        path = self.model_path
        return path.is_file() and path.stat().st_size > 0

    def ensure_model_sync(self) -> Path:
        """Ensure the GGUF file exists locally, downloading from Hugging Face if allowed."""
        if self.is_model_present():
            logger.info(
                "Model artifact already present at %s (%d bytes)",
                self.model_path,
                self.model_path.stat().st_size,
            )
            return self.model_path

        if not self.auto_download:
            msg = (
                f"Model file not found at {self.model_path} and AUTO_DOWNLOAD_MODEL is False"
            )
            logger.error(msg)
            raise ModelManagerError(msg)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading %s (%s) from Hugging Face repo '%s' into %s ...",
            self.model_id,
            self.hf_gguf_filename,
            self.hf_gguf_repo,
            self.cache_dir,
        )
        try:
            downloaded = self._downloader(
                repo_id=self.hf_gguf_repo,
                filename=self.hf_gguf_filename,
                local_dir=str(self.cache_dir),
                token=self.hf_token or None,
            )
            resolved = Path(downloaded)
            logger.info("Successfully downloaded model to %s", resolved)
            return resolved
        except Exception as exc:
            msg = (
                f"Failed to download '{self.hf_gguf_filename}' from '{self.hf_gguf_repo}': {exc}"
            )
            logger.error(msg)
            raise ModelManagerError(msg) from exc

    async def ensure_model(self) -> Path:
        """Asynchronously ensure the GGUF model file exists locally."""
        return await asyncio.to_thread(self.ensure_model_sync)

    def resolve_requested_model(self, requested_model: Optional[str]) -> str:
        """Normalize incoming model names (e.g. 'local-model') to the canonical model_id."""
        if not requested_model:
            return self.model_id
        cleaned = requested_model.strip()
        if cleaned in {
            self.model_alias,
            self.model_id,
            "phi-4-mini-instruct",
            "Phi-4-mini-instruct",
            self.hf_gguf_filename,
        }:
            return self.model_id
        return self.model_id

    def get_model_cards(self) -> List[ModelCard]:
        """Return OpenAI-compatible model cards for /v1/models."""
        cards = [
            ModelCard(
                id=self.model_id,
                owned_by="microsoft",
                root=self.hf_gguf_filename,
            ),
        ]
        if self.model_alias and self.model_alias != self.model_id:
            cards.append(
                ModelCard(
                    id=self.model_alias,
                    owned_by="sportsagentic",
                    root=self.model_id,
                )
            )
        return cards
