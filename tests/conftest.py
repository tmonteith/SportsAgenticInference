"""Shared pytest fixtures for SportsAgenticInference tests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import pytest

from app.config import Settings


@pytest.fixture
def sample_mcp_openai_tools() -> List[Dict[str, Any]]:
    """OpenAI-formatted MCP tools identical to those sent by SportsAgenticAgent."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_cfb_scores",
                "description": "Retrieve College Football scores by team, week, year, or date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "team": {"type": "string"},
                        "year": {"type": "integer"},
                        "week": {"type": "integer"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_team_schedule",
                "description": "Retrieve full season schedule for a college football team",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "team": {"type": "string"},
                        "year": {"type": "integer"},
                    },
                    "required": ["team"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_cfb_rankings",
                "description": "Retrieve AP Top 25 or CFP national rankings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {"type": "integer"},
                        "week": {"type": "integer"},
                    },
                },
            },
        },
    ]


@pytest.fixture
def dummy_model_dir(tmp_path: Path) -> Path:
    """Create a temporary directory containing a dummy GGUF file."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    gguf_file = model_dir / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
    gguf_file.write_bytes(b"GGUF\x00\x00\x00\x03dummy-phi-4-mini-weights")
    return model_dir


@pytest.fixture
def test_settings(dummy_model_dir: Path) -> Settings:
    """Return test Settings configured with a dummy model cache and auto-start disabled."""
    return Settings(
        host="127.0.0.1",
        port=8081,
        log_level="DEBUG",
        model_id="microsoft/Phi-4-mini-instruct",
        model_alias="local-model",
        hf_gguf_repo="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        hf_gguf_filename="microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        model_cache_dir=str(dummy_model_dir),
        auto_download_model=False,
        llama_internal_host="127.0.0.1",
        llama_internal_port=8082,
        auto_start_llama_server=False,
    )
