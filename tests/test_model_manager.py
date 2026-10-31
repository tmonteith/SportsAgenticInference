"""Unit tests for ModelManager GGUF caching, Hugging Face download, and model alias resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import pytest

from app.model_manager import ModelManager, ModelManagerError


def test_model_present_cache_hit(dummy_model_dir: Path) -> None:
    """When the GGUF file already exists, ensure_model_sync returns it without calling downloader."""
    called = False

    def fail_downloader(**kwargs: Any) -> str:
        nonlocal called
        called = True
        raise AssertionError("Downloader should not be invoked when GGUF file is cached")

    mgr = ModelManager(
        cache_dir=dummy_model_dir,
        auto_download=True,
        downloader=fail_downloader,
    )
    assert mgr.is_model_present() is True
    path = mgr.ensure_model_sync()
    assert path == dummy_model_dir / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
    assert called is False


async def test_ensure_model_downloads_when_missing(tmp_path: Path) -> None:
    """When the GGUF file is missing and auto_download is True, downloader is called."""
    cache_dir = tmp_path / "empty_models"
    captured_kwargs: Dict[str, Any] = {}

    def fake_downloader(**kwargs: Any) -> str:
        captured_kwargs.update(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(b"GGUF-downloaded-phi4")
        return str(target)

    mgr = ModelManager(
        cache_dir=cache_dir,
        auto_download=True,
        hf_token="hf_test_token",
        downloader=fake_downloader,
    )
    assert mgr.is_model_present() is False

    resolved = await mgr.ensure_model()
    assert resolved.exists()
    assert mgr.is_model_present() is True
    assert captured_kwargs["repo_id"] == "bartowski/microsoft_Phi-4-mini-instruct-GGUF"
    assert captured_kwargs["filename"] == "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
    assert captured_kwargs["token"] == "hf_test_token"


def test_ensure_model_raises_when_auto_download_disabled(tmp_path: Path) -> None:
    """When GGUF is missing and auto_download is False, ModelManagerError is raised."""
    mgr = ModelManager(
        cache_dir=tmp_path / "missing_dir",
        auto_download=False,
    )
    with pytest.raises(ModelManagerError, match="AUTO_DOWNLOAD_MODEL is False"):
        mgr.ensure_model_sync()


def test_ensure_model_raises_on_download_failure(tmp_path: Path) -> None:
    """When Hugging Face download raises an exception, ModelManagerError wraps it."""
    def broken_downloader(**kwargs: Any) -> str:
        raise RuntimeError("HF network failure")

    mgr = ModelManager(
        cache_dir=tmp_path / "fail_dir",
        auto_download=True,
        downloader=broken_downloader,
    )
    with pytest.raises(ModelManagerError, match="HF network failure"):
        mgr.ensure_model_sync()


def test_resolve_requested_model_and_cards(dummy_model_dir: Path) -> None:
    """Verify model alias resolution ('local-model' -> 'microsoft/Phi-4-mini-instruct') and model cards."""
    mgr = ModelManager(cache_dir=dummy_model_dir)
    assert mgr.resolve_requested_model("local-model") == "microsoft/Phi-4-mini-instruct"
    assert mgr.resolve_requested_model("microsoft/Phi-4-mini-instruct") == "microsoft/Phi-4-mini-instruct"
    assert mgr.resolve_requested_model("phi-4-mini-instruct") == "microsoft/Phi-4-mini-instruct"
    assert mgr.resolve_requested_model(None) == "microsoft/Phi-4-mini-instruct"
    assert mgr.resolve_requested_model("any-other-model") == "microsoft/Phi-4-mini-instruct"

    cards = mgr.get_model_cards()
    ids = [c.id for c in cards]
    assert "microsoft/Phi-4-mini-instruct" in ids
    assert "local-model" in ids
