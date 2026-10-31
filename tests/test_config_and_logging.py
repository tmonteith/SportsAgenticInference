"""Unit tests for configuration and centralized logging."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.logging_config import get_logger, setup_logging


def test_default_settings() -> None:
    """Verify default Settings match SportsAgenticAgent and Phi-4-mini-instruct requirements."""
    settings = Settings(_env_file=None)
    assert settings.port == 8081
    assert settings.model_id == "microsoft/Phi-4-mini-instruct"
    assert settings.model_alias == "local-model"
    assert settings.hf_gguf_repo == "bartowski/microsoft_Phi-4-mini-instruct-GGUF"
    assert settings.hf_gguf_filename == "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
    assert settings.llama_internal_host == "127.0.0.1"
    assert settings.llama_internal_port == 8082
    assert settings.resolved_llama_base_url == "http://127.0.0.1:8082"
    assert settings.resolved_model_path == Path("models/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf")


def test_external_llama_url_override() -> None:
    """Verify external_llama_url overrides the internal host/port URL."""
    settings = Settings(
        _env_file=None,
        external_llama_url="http://10.0.0.5:9090/",
    )
    assert settings.resolved_llama_base_url == "http://10.0.0.5:9090"


def test_get_settings_cached() -> None:
    """Verify get_settings returns a cached Settings instance."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_logging_setup_and_namespace() -> None:
    """Verify setup_logging and get_logger configure sportsagentic.inference loggers."""
    root_logger = setup_logging("DEBUG")
    assert root_logger.name == "sportsagentic.inference"
    assert root_logger.level == logging.DEBUG
    assert len(root_logger.handlers) >= 1

    # Calling setup_logging again updates existing handlers without duplicating them
    handler_count = len(root_logger.handlers)
    setup_logging("INFO")
    assert len(root_logger.handlers) == handler_count
    assert root_logger.level == logging.INFO

    child = get_logger("proxy")
    assert child.name == "sportsagentic.inference.proxy"

    already_prefixed = get_logger("sportsagentic.inference.model")
    assert already_prefixed.name == "sportsagentic.inference.model"

    parent_prefixed = get_logger("sportsagentic.custom")
    assert parent_prefixed.name == "sportsagentic.custom"
