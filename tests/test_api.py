"""API and SportsAgenticAgent integration contract tests for SportsAgenticInference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import httpx
import respx

from app.config import Settings
from app.main import create_app


@respx.mock
async def test_health_endpoint_ok_and_degraded(dummy_model_dir: Path) -> None:
    """Verify GET /health returns 200 when upstream llama-server is healthy and 503 when unhealthy."""
    settings = Settings(
        model_cache_dir=str(dummy_model_dir),
        external_llama_url="http://127.0.0.1:8082",
        auto_start_llama_server=False,
    )
    application = create_app(settings)

    # 1. Healthy upstream
    respx.get("http://127.0.0.1:8082/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8081") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_id"] == "microsoft/Phi-4-mini-instruct"
        assert body["model_alias"] == "local-model"
        assert body["model_file_present"] is True
        assert body["llama_server_healthy"] is True

    # 2. Unhealthy upstream (returns 503 so SportsAgenticAgent detects degraded state)
    respx.get("http://127.0.0.1:8082/health").mock(
        return_value=httpx.Response(503, json={"status": "loading model"})
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8081") as client:
        resp = await client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["llama_server_healthy"] is False


async def test_list_models_endpoint(test_settings: Settings) -> None:
    """Verify GET /v1/models lists both microsoft/Phi-4-mini-instruct and local-model."""
    application = create_app(test_settings)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8081") as client:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [item["id"] for item in data["data"]]
        assert "microsoft/Phi-4-mini-instruct" in model_ids
        assert "local-model" in model_ids


@respx.mock
async def test_sportsagentic_agent_multi_turn_tool_calling_contract(
    dummy_model_dir: Path,
    sample_mcp_openai_tools: List[Dict[str, Any]],
) -> None:
    """Verify end-to-end compatibility with SportsAgenticAgent's LlamaServerClient workflow."""
    settings = Settings(
        model_cache_dir=str(dummy_model_dir),
        external_llama_url="http://127.0.0.1:8082",
        auto_start_llama_server=False,
    )
    application = create_app(settings)

    # Turn 1: Phi-4-mini-instruct emits a tool call for get_cfb_scores
    # Turn 2: After SportsAgenticAgent executes the MCP tool and sends role='tool', Phi-4-mini synthesizes the answer
    call_count = 0

    def mock_upstream_completion(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        payload = json.loads(request.content)
        assert payload["model"] == "microsoft/Phi-4-mini-instruct"

        if call_count == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-turn1",
                    "object": "chat.completion",
                    "created": 1700000001,
                    "model": "microsoft/Phi-4-mini-instruct",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": '<|tool_call|>[{"name": "get_cfb_scores", "arguments": {"team": "Ole Miss"}}]<|/tool_call|>',
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        # Verify the second request contains the assistant tool_calls and role='tool' result
        roles = [m["role"] for m in payload["messages"]]
        assert "tool" in roles
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-turn2",
                "object": "chat.completion",
                "created": 1700000002,
                "model": "microsoft/Phi-4-mini-instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Ole Miss defeated Oklahoma 26-14 in Oxford.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    respx.post("http://127.0.0.1:8082/v1/chat/completions").mock(
        side_effect=mock_upstream_completion
    )

    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8081") as client:
        # Turn 1 from SportsAgenticAgent
        turn1_resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": "You are a college football assistant."},
                    {"role": "user", "content": "What was the score of the Ole Miss game?"},
                ],
                "tools": sample_mcp_openai_tools,
            },
        )
        assert turn1_resp.status_code == 200
        turn1_data = turn1_resp.json()
        assistant_msg = turn1_data["choices"][0]["message"]
        assert assistant_msg["content"] == ""
        assert len(assistant_msg["tool_calls"]) == 1
        tc = assistant_msg["tool_calls"][0]
        assert tc["function"]["name"] == "get_cfb_scores"
        assert json.loads(tc["function"]["arguments"]) == {"team": "Ole Miss"}

        # Turn 2 from SportsAgenticAgent with MCP tool output
        turn2_resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": "You are a college football assistant."},
                    {"role": "user", "content": "What was the score of the Ole Miss game?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [tc],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": "get_cfb_scores",
                        "content": json.dumps({"home": "Ole Miss 26", "away": "Oklahoma 14"}),
                    },
                ],
                "tools": sample_mcp_openai_tools,
            },
        )
        assert turn2_resp.status_code == 200
        turn2_data = turn2_resp.json()
        assert (
            turn2_data["choices"][0]["message"]["content"]
            == "Ole Miss defeated Oklahoma 26-14 in Oxford."
        )


@respx.mock
async def test_chat_completions_error_status_codes(test_settings: Settings) -> None:
    """Verify upstream timeout (504), unavailable (503), and bad gateway (502) HTTP status codes."""
    application = create_app(test_settings)
    transport = httpx.ASGITransport(app=application)
    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": "Hi"}],
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8081") as client:
        # 504 Gateway Timeout
        respx.post("http://127.0.0.1:8082/v1/chat/completions").mock(
            side_effect=httpx.ReadTimeout("timeout")
        )
        r_timeout = await client.post("/v1/chat/completions", json=payload)
        assert r_timeout.status_code == 504

        # 503 Service Unavailable
        respx.post("http://127.0.0.1:8082/v1/chat/completions").mock(
            return_value=httpx.Response(503, text="slot unavailable")
        )
        r_unavail = await client.post("/v1/chat/completions", json=payload)
        assert r_unavail.status_code == 503

        # 502 Bad Gateway
        respx.post("http://127.0.0.1:8082/v1/chat/completions").mock(
            return_value=httpx.Response(400, text="bad request upstream")
        )
        r_bad = await client.post("/v1/chat/completions", json=payload)
        assert r_bad.status_code == 502
