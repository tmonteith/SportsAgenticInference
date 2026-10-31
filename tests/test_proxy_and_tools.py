"""Unit tests for LlamaInferenceProxy and Phi-4-mini-instruct tool-call normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import httpx
import pytest
import respx

from app.model_manager import ModelManager
from app.proxy import (
    InferenceProxyError,
    InferenceUpstreamTimeoutError,
    InferenceUpstreamUnavailableError,
    LlamaInferenceProxy,
    extract_known_tool_names,
    normalize_chat_completion_payload,
    parse_phi_inline_tool_calls,
)
from app.schemas import ChatCompletionRequest, ChatMessage


def test_extract_known_tool_names(sample_mcp_openai_tools: List[Dict[str, Any]]) -> None:
    """Verify tool names are extracted from OpenAI function-calling schema."""
    names = extract_known_tool_names(sample_mcp_openai_tools)
    assert names == ["get_cfb_scores", "get_team_schedule", "get_cfb_rankings"]


def test_parse_phi_special_tag_tool_calls() -> None:
    """Verify Phi-4-mini <|tool_call|> and <|tool|> tags are parsed into OpenAI tool_calls."""
    content = (
        '<|tool_call|>[{"name": "get_cfb_scores", "arguments": {"team": "Ole Miss", "week": 7}}]<|/tool_call|>'
    )
    calls = parse_phi_inline_tool_calls(content, known_tool_names=["get_cfb_scores"])
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_cfb_scores"
    assert json.loads(calls[0]["function"]["arguments"]) == {"team": "Ole Miss", "week": 7}


def parse_phi_fenced_and_raw_json_tool_calls() -> None:
    """Verify fenced JSON, raw JSON, and Python-style function call tool invocations are extracted."""
    fenced = '```json\n{"name": "get_team_schedule", "arguments": {"team": "Alabama"}}\n```'
    calls = parse_phi_inline_tool_calls(fenced, known_tool_names=["get_team_schedule"])
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_team_schedule"
    assert json.loads(calls[0]["function"]["arguments"]) == {"team": "Alabama"}

    # Python-style get_cfb_scores(year=2026, week=8, date="2026-10-31", team="CSU")
    py_style = (
        "To get the score of the CSU game from last week (Week 8), I will use get_cfb_scores.\n"
        '```bash\nget_cfb_scores(year=2026, week=8, date="2026-10-31", team="CSU")\n```'
    )
    py_calls = parse_phi_inline_tool_calls(py_style, known_tool_names=["get_cfb_scores"])
    assert len(py_calls) == 1
    assert py_calls[0]["function"]["name"] == "get_cfb_scores"
    args = json.loads(py_calls[0]["function"]["arguments"])
    assert args == {"year": 2026, "week": 8, "date": "2026-10-31", "team": "CSU"}

    # Unknown tool name is ignored when known_tool_names is provided
    ignored = parse_phi_inline_tool_calls(fenced, known_tool_names=["get_cfb_scores"])
    assert ignored == []



def test_normalize_chat_completion_payload_converts_inline_tool_call() -> None:
    """Verify normalize_chat_completion_payload promotes inline Phi-4-mini tool calls to message.tool_calls."""
    raw_payload = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '<|tool|>{"name": "get_cfb_rankings", "parameters": {"week": 9}}<|/tool|>',
                },
                "finish_reason": "stop",
            }
        ],
    }
    normalized = normalize_chat_completion_payload(
        raw_payload,
        canonical_model_id="microsoft/Phi-4-mini-instruct",
        known_tool_names=["get_cfb_rankings"],
    )
    assert normalized["model"] == "microsoft/Phi-4-mini-instruct"
    msg = normalized["choices"][0]["message"]
    assert msg["content"] == ""
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "get_cfb_rankings"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"week": 9}
    assert normalized["choices"][0]["finish_reason"] == "tool_calls"


@respx.mock
async def test_proxy_complete_chat_rewrites_model_and_returns_response(
    dummy_model_dir: Path,
    sample_mcp_openai_tools: List[Dict[str, Any]],
) -> None:
    """Verify LlamaInferenceProxy translates 'local-model' to 'microsoft/Phi-4-mini-instruct'."""
    route = respx.post("http://127.0.0.1:8082/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-phi4",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "microsoft/Phi-4-mini-instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_cfb_scores",
                                        "arguments": {"team": "Ole Miss"},
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 12, "total_tokens": 54},
            },
        )
    )

    model_mgr = ModelManager(cache_dir=dummy_model_dir)
    proxy = LlamaInferenceProxy(
        base_url="http://127.0.0.1:8082",
        model_manager=model_mgr,
        timeout_seconds=10.0,
    )

    req = ChatCompletionRequest(
        model="local-model",
        messages=[ChatMessage(role="user", content="What was the Ole Miss score?")],
        tools=sample_mcp_openai_tools,
    )
    resp = await proxy.complete_chat(req)
    assert route.called
    sent_json = json.loads(route.calls[0].request.content)
    assert sent_json["model"] == "microsoft/Phi-4-mini-instruct"

    tc = resp["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_cfb_scores"
    assert json.loads(tc["function"]["arguments"]) == {"team": "Ole Miss"}


@respx.mock
async def test_proxy_error_handling(dummy_model_dir: Path) -> None:
    """Verify timeout, 5xx, and 4xx error mappings in LlamaInferenceProxy."""
    model_mgr = ModelManager(cache_dir=dummy_model_dir)
    proxy = LlamaInferenceProxy(
        base_url="http://127.0.0.1:8082",
        model_manager=model_mgr,
        timeout_seconds=1.0,
    )
    req = ChatCompletionRequest(
        model="local-model",
        messages=[ChatMessage(role="user", content="Hello")],
    )

    # 1. Timeout
    respx.post("http://127.0.0.1:8082/v1/chat/completions").mock(
        side_effect=httpx.ReadTimeout("timed out")
    )
    with pytest.raises(InferenceUpstreamTimeoutError):
        await proxy.complete_chat(req)

    # 2. HTTP 503 Server Error
    respx.post("http://127.0.0.1:8082/v1/chat/completions").mock(
        return_value=httpx.Response(503, text="Loading model")
    )
    with pytest.raises(InferenceUpstreamUnavailableError):
        await proxy.complete_chat(req)

    # 3. HTTP 400 Client Error
    respx.post("http://127.0.0.1:8082/v1/chat/completions").mock(
        return_value=httpx.Response(400, text="Invalid context length")
    )
    with pytest.raises(InferenceProxyError):
        await proxy.complete_chat(req)
