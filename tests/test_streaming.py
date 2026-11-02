"""Tests for POST /v1/chat/completions with stream=true."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
import socket
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple
import httpx
import pytest
import uvicorn

from app.config import Settings
from app.main import ChatCompletionStreamResponse, create_app
from app.model_manager import ModelManager
from app.proxy import (
    InferenceProxyError,
    InferenceUpstreamTimeoutError,
    InferenceUpstreamUnavailableError,
    InlineToolCallFilter,
    LlamaInferenceProxy,
    normalize_chat_completion_payload,
)
from app.schemas import ChatCompletionRequest, ChatMessage

UPSTREAM_URL = "http://upstream.test/v1/chat/completions"
TOOL_NAMES = ["get_cfb_scores", "get_team_schedule"]
OLE_MISS_ARGS = {"team": "Ole Miss", "week": 7}
JSON_CALL = json.dumps({"name": "get_cfb_scores", "arguments": OLE_MISS_ARGS})

INLINE_TOOL_CALL_REPLIES = [
    f"<|tool_call|>[{JSON_CALL}]<|/tool_call|>",
    f"<|tool_calls|>[{JSON_CALL}]<|/tool_calls|>",
    f"<|tool|>{JSON_CALL}<|/tool|>",
    f"<|tool_call|>{JSON_CALL}<|end|>",
    f"<|tool_call|>{JSON_CALL}",
    JSON_CALL,
    f"[{JSON_CALL}]",
    f"  \n{JSON_CALL}\n",
    f"```json\n{JSON_CALL}\n```",
    f"```\n[{JSON_CALL}]\n```",
    'get_cfb_scores(team="Ole Miss", week=7)',
    'get_cfb_scores (team="Ole Miss", week=7)',
    '```bash\nget_cfb_scores(team="Ole Miss", week=7)\n```',
    '```get_cfb_scores(team="Ole Miss", week=7)```',
]

NARRATED_TOOL_CALL_REPLIES = [
    ("Let me check. ", f"<|tool_call|>[{JSON_CALL}]<|/tool_call|> Done."),
    ("Let me check.\n", f"```json\n{JSON_CALL}\n```"),
    ("I will use get_cfb_scores.\n", '```bash\nget_cfb_scores(team="Ole Miss", week=7)\n```'),
    ("Calling it: ", 'get_cfb_scores(team="Ole Miss", week=7) now'),
]

PLAIN_TEXT_REPLIES = [
    "Ole Miss beat Oklahoma 26-14.",
    "[1] Alabama, [2] Georgia",
    "{curly} text",
    "Is 3 < 4? Yes, and `code` is fine.",
    "Run ```ls``` first.",
    "```python\nprint('hi')\n```\nThat prints hi.",
    '```json\n{"team": "Ole Miss"}\n```\nThat is the team.',
    "The get_cfb_scores tool and get_cfb are not calls.",
    'budget_cfb_scores(team="Ole Miss") is another function.',
    '<|tool_call|>{"name": "not_a_tool"}<|/tool_call|>',
    "get_team",
    "``",
    "<|tool",
    "Café   line separator",
]


def sse(payload: Any) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def content_chunk(text: str, model: str = "gguf-name") -> Dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }


def role_chunk() -> Dict[str, Any]:
    chunk = content_chunk("")
    chunk["choices"][0]["delta"] = {"role": "assistant", "content": None}
    return chunk


def finish_chunk(reason: str = "stop") -> Dict[str, Any]:
    chunk = content_chunk("")
    chunk["choices"][0].update(delta={}, finish_reason=reason)
    return chunk


def reply_bytes(pieces: List[str], reason: str = "stop") -> List[bytes]:
    """Build the SSE body llama-server sends for a reply split into the given pieces."""
    return [
        sse(role_chunk()),
        *[sse(content_chunk(piece)) for piece in pieces],
        sse(finish_chunk(reason)),
        b"data: [DONE]\n\n",
    ]


def body_stream(frames: List[bytes], error: Optional[Exception] = None) -> AsyncIterator[bytes]:
    async def generate() -> AsyncIterator[bytes]:
        for frame in frames:
            yield frame
        if error is not None:
            raise error

    return generate()


def make_proxy(dummy_model_dir: Path, handler: Callable[[httpx.Request], Any]) -> LlamaInferenceProxy:
    return LlamaInferenceProxy(
        base_url="http://upstream.test",
        model_manager=ModelManager(cache_dir=dummy_model_dir),
        timeout_seconds=1.0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def make_request(tools: Optional[List[Dict[str, Any]]] = None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="local-model",
        messages=[ChatMessage(role="user", content="What was the Ole Miss score?")],
        tools=tools,
        stream=True,
    )


def parse_frames(body: bytes) -> Tuple[List[Dict[str, Any]], List[str], bool]:
    """Split a relayed body into data chunks, error lines, and whether [DONE] arrived."""
    chunks: List[Dict[str, Any]] = []
    errors: List[str] = []
    done = False
    for frame in body.decode().split("\n\n"):
        if frame.startswith("error:"):
            errors.append(frame)
        elif frame == "data: [DONE]":
            done = True
        elif frame.startswith("data:"):
            chunks.append(json.loads(frame[5:]))
    return chunks, errors, done


def merged_reply(chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """Read the chunks the way a streaming client does."""
    text = ""
    calls: Dict[int, Dict[str, Any]] = {}
    finish_reason = None
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice["delta"]
            text += delta.get("content") or ""
            for call in delta.get("tool_calls") or []:
                calls[call["index"]] = call
            finish_reason = choice.get("finish_reason") or finish_reason
    return text, list(calls.values()), finish_reason


async def relay(
    dummy_model_dir: Path,
    frames: List[bytes],
    tools: Optional[List[Dict[str, Any]]] = None,
    error: Optional[Exception] = None,
) -> Tuple[List[Dict[str, Any]], List[str], bool]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body_stream(frames, error))

    proxy = make_proxy(dummy_model_dir, handler)
    stream = await proxy.open_chat_stream(make_request(tools))
    body = b""
    async for frame in stream:
        body += frame
    await stream.aclose()
    return parse_frames(body)


def splits(text: str) -> List[List[str]]:
    """Return one-character pieces plus every way of cutting the text in two."""
    return [list(text), *[[text[:cut], text[cut:]] for cut in range(1, len(text))]]


def run_filter(pieces: List[str], known: Optional[List[str]] = TOOL_NAMES) -> Tuple[str, List[Any]]:
    inline = InlineToolCallFilter(known)
    text = "".join(inline.feed(piece) for piece in pieces) + inline.finish()
    return text, inline.take_calls()


def non_streamed_calls(content: str, known: Optional[List[str]] = TOOL_NAMES) -> List[Dict[str, Any]]:
    normalized = normalize_chat_completion_payload(
        {"choices": [{"message": {"content": content}}]},
        canonical_model_id="model",
        known_tool_names=known,
    )
    return normalized["choices"][0]["message"].get("tool_calls", [])


@pytest.mark.parametrize("reply", INLINE_TOOL_CALL_REPLIES)
def test_filter_turns_split_inline_calls_into_tool_calls(reply: str) -> None:
    """Verify every inline form gives one tool call and no text, wherever chunks split it."""
    for pieces in splits(reply):
        text, calls = run_filter(pieces)
        assert text == "", pieces
        assert calls == non_streamed_calls(reply), pieces
        assert calls[0]["function"]["name"] == "get_cfb_scores"
        assert json.loads(calls[0]["function"]["arguments"]) == OLE_MISS_ARGS


@pytest.mark.parametrize("narration,call", NARRATED_TOOL_CALL_REPLIES)
def test_filter_releases_narration_and_drops_everything_from_the_call_on(
    narration: str, call: str
) -> None:
    """Verify text before a call reaches the client while the call and what follows do not."""
    reply = narration + call
    for pieces in splits(reply):
        text, calls = run_filter(pieces)
        assert text == narration, pieces
        assert calls == non_streamed_calls(reply), pieces


@pytest.mark.parametrize("reply", PLAIN_TEXT_REPLIES)
def test_filter_keeps_plain_text_intact(reply: str) -> None:
    """Verify text that only looks like a call comes through unchanged, wherever it is split."""
    for pieces in splits(reply):
        assert run_filter(pieces) == (reply, []), pieces


def test_filter_releases_ordinary_text_without_waiting_for_the_end() -> None:
    """Verify holding back is limited to text that could still become a call."""
    inline = InlineToolCallFilter(TOOL_NAMES)
    assert inline.feed("Ole ") == "Ole "
    assert inline.feed("Miss won") == "Miss won"
    assert inline.feed(" [1] and `x") == " [1] and `x"
    assert InlineToolCallFilter(TOOL_NAMES).feed("[1] Alabama") == "[1] Alabama"
    assert InlineToolCallFilter(TOOL_NAMES).feed("{curly}") == "{curly}"


def test_filter_without_offered_tools_matches_non_streaming() -> None:
    """Verify a JSON call is still recognised with no tools, and tool_name() text is not."""
    assert run_filter(list(JSON_CALL), known=None)[1] == non_streamed_calls(JSON_CALL, known=None)
    python_style = 'get_cfb_scores(team="Ole Miss")'
    assert run_filter(list(python_style), known=None) == (python_style, [])


async def test_stream_relays_chunks_and_rewrites_model(dummy_model_dir: Path) -> None:
    """Verify chunks pass through in order, with the same model resolution as complete_chat."""
    sent: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, content=body_stream(reply_bytes(["Ole ", "Miss"])))

    proxy = make_proxy(dummy_model_dir, handler)
    stream = await proxy.open_chat_stream(make_request())
    body = b"".join([frame async for frame in stream])
    await stream.aclose()

    assert sent[0]["model"] == proxy.model_manager.model_id
    assert sent[0]["stream"] is True

    chunks, errors, done = parse_frames(body)
    assert errors == []
    assert done is True
    assert [chunk["model"] for chunk in chunks] == [proxy.model_manager.model_id] * 4
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}
    assert [c["choices"][0]["delta"].get("content") for c in chunks[1:3]] == ["Ole ", "Miss"]
    assert merged_reply(chunks) == ("Ole Miss", [], "stop")


async def test_stream_reassembles_lines_split_across_network_reads(dummy_model_dir: Path) -> None:
    """Verify frames and multi-byte characters survive being cut at any byte."""
    body = b"".join(reply_bytes(["Café "]))
    raw = "".join(frame.decode() for frame in reply_bytes(["x"])).replace('"x"', '"Café  "')
    body = raw.encode("utf-8")
    chunks, errors, done = await relay(dummy_model_dir, [body[i : i + 1] for i in range(len(body))])
    assert errors == []
    assert done is True
    assert merged_reply(chunks) == ("Café  ", [], "stop")


async def test_stream_passes_native_tool_call_deltas_through(
    dummy_model_dir: Path,
    sample_mcp_openai_tools: List[Dict[str, Any]],
) -> None:
    """Verify tool_calls llama-server produced itself reach the client untouched."""
    native = content_chunk("")
    native["choices"][0]["delta"] = {
        "tool_calls": [
            {
                "index": 0,
                "id": "abc",
                "type": "function",
                "function": {"name": "get_cfb_scores", "arguments": '{"team"'},
            }
        ]
    }
    fragment = content_chunk("")
    fragment["choices"][0]["delta"] = {
        "tool_calls": [{"index": 0, "function": {"arguments": ': "Ole Miss"}'}}]
    }
    chunks, errors, done = await relay(
        dummy_model_dir,
        [sse(role_chunk()), sse(native), sse(fragment), sse(finish_chunk("tool_calls")), b"data: [DONE]\n\n"],
        tools=sample_mcp_openai_tools,
    )
    assert errors == []
    assert done is True
    assert chunks[1]["choices"][0]["delta"] == native["choices"][0]["delta"]
    assert chunks[2]["choices"][0]["delta"] == fragment["choices"][0]["delta"]
    assert chunks[3]["choices"][0]["finish_reason"] == "tool_calls"


async def test_stream_emits_inline_call_as_tool_calls_delta(
    dummy_model_dir: Path,
    sample_mcp_openai_tools: List[Dict[str, Any]],
) -> None:
    """Verify a tag split across chunks becomes a tool_calls delta with finish_reason tool_calls."""
    pieces = ["<|to", "ol_call|>[", JSON_CALL[:10], JSON_CALL[10:], "]<|/tool", "_call|>"]
    chunks, errors, done = await relay(
        dummy_model_dir,
        reply_bytes(pieces),
        tools=sample_mcp_openai_tools,
    )
    assert errors == []
    assert done is True
    text, calls, finish_reason = merged_reply(chunks)
    assert text == ""
    assert finish_reason == "tool_calls"
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_cfb_scores"
    assert json.loads(calls[0]["function"]["arguments"]) == OLE_MISS_ARGS


async def test_stream_keeps_narration_separate_from_the_tool_call_chunk(
    dummy_model_dir: Path,
    sample_mcp_openai_tools: List[Dict[str, Any]],
) -> None:
    """Verify text and a call found in one upstream chunk are sent as two chunks, text first."""
    chunks, _, _ = await relay(
        dummy_model_dir,
        reply_bytes([f"Checking. <|tool|>{JSON_CALL}<|/tool|>"]),
        tools=sample_mcp_openai_tools,
    )
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    assert deltas[1] == {"content": "Checking. "}
    assert deltas[2]["tool_calls"][0]["function"]["name"] == "get_cfb_scores"
    assert "content" not in deltas[2]


async def test_stream_resolves_held_text_when_upstream_sends_no_finish_reason(
    dummy_model_dir: Path,
    sample_mcp_openai_tools: List[Dict[str, Any]],
) -> None:
    """Verify [DONE] still flushes held text, so a bare JSON reply is not lost."""
    frames = [sse(content_chunk("{curly")), sse(content_chunk('"} text')), b"data: [DONE]\n\n"]
    chunks, errors, done = await relay(dummy_model_dir, frames, tools=sample_mcp_openai_tools)
    assert errors == []
    assert done is True
    assert merged_reply(chunks) == ('{curly"} text', [], "stop")


@pytest.mark.parametrize(
    "side_effect,expected",
    [
        (httpx.ReadTimeout("slow"), InferenceUpstreamTimeoutError),
        (httpx.ConnectTimeout("slow"), InferenceUpstreamTimeoutError),
        (httpx.ConnectError("refused"), InferenceUpstreamUnavailableError),
        (httpx.Response(503, text="Loading model"), InferenceUpstreamUnavailableError),
        (httpx.Response(400, text="Bad context"), InferenceProxyError),
    ],
)
async def test_open_maps_failures_before_the_stream_starts(
    dummy_model_dir: Path,
    side_effect: Any,
    expected: type,
) -> None:
    """Verify connection and status failures raise the same errors as complete_chat."""

    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(side_effect, Exception):
            raise side_effect
        return side_effect

    proxy = make_proxy(dummy_model_dir, handler)
    with pytest.raises(expected) as raised:
        await proxy.open_chat_stream(make_request())
    assert raised.type is expected


@pytest.mark.parametrize(
    "frames,error,expected",
    [
        ([], None, InferenceUpstreamUnavailableError),
        ([b": ping\n\n"], None, InferenceUpstreamUnavailableError),
        ([], httpx.ReadTimeout("slow"), InferenceUpstreamTimeoutError),
        ([b": ping\n\n"], httpx.ReadTimeout("slow"), InferenceUpstreamTimeoutError),
        ([], httpx.ReadError("reset"), InferenceUpstreamUnavailableError),
        ([b'error: {"code":500,"message":"boom"}\n\n'], None, InferenceUpstreamUnavailableError),
        ([b'error: {"code":400,"message":"too long"}\n\n'], None, InferenceProxyError),
        ([b"error: plain words\n\n"], None, InferenceProxyError),
        ([b'data: {"error":{"code":500,"message":"boom"}}\n\n'], None, InferenceUpstreamUnavailableError),
        ([b"data: not json\n\n"], None, InferenceProxyError),
    ],
)
async def test_open_maps_failures_before_the_first_event(
    dummy_model_dir: Path,
    frames: List[bytes],
    error: Optional[Exception],
    expected: type,
) -> None:
    """Verify a 200 whose body fails before any event still counts as a pre-stream failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body_stream(frames, error))

    proxy = make_proxy(dummy_model_dir, handler)
    with pytest.raises(expected) as raised:
        await proxy.open_chat_stream(make_request())
    assert raised.type is expected


@pytest.mark.parametrize(
    "side_effect,status",
    [
        (httpx.ReadTimeout("slow"), 504),
        (httpx.ConnectError("refused"), 503),
        (httpx.Response(503, text="Loading model"), 503),
        (httpx.Response(400, text="Bad context"), 502),
        (httpx.Response(200, content=b'error: {"code":400,"message":"too long"}\n\n'), 502),
        (httpx.Response(200, content=b'error: {"code":500,"message":"boom"}\n\n'), 503),
    ],
)
async def test_route_returns_status_codes_for_stream_failures_before_the_first_event(
    test_settings: Settings,
    side_effect: Any,
    status: int,
) -> None:
    """Verify the Agent gets the 502/503/504 it fails over on, not a 200 with an empty body."""

    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(side_effect, Exception):
            raise side_effect
        return side_effect

    application = create_app(test_settings)
    application.state.inference_proxy._external_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8081") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )
    assert resp.status_code == status


async def test_route_streams_with_sse_headers(test_settings: Settings) -> None:
    """Verify the response headers, and that non-streaming requests still return JSON."""

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=body_stream(reply_bytes(["Hello"])))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Hello"}}]},
        )

    application = create_app(test_settings)
    application.state.inference_proxy._external_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    transport = httpx.ASGITransport(app=application)
    messages = [{"role": "user", "content": "Hi"}]
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8081") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": messages, "stream": True},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"
        assert parse_frames(resp.content)[2] is True

        plain = await client.post("/v1/chat/completions", json={"messages": messages})
        assert plain.headers["content-type"].startswith("application/json")
        assert plain.json()["choices"][0]["message"]["content"] == "Hello"


@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("slow"), httpx.ReadError("reset"), httpx.RemoteProtocolError("cut off")],
)
async def test_failure_mid_stream_reports_an_error_and_no_done(
    dummy_model_dir: Path,
    error: Exception,
) -> None:
    """Verify an upstream failure after the first event cannot look like a clean end."""
    chunks, errors, done = await relay(
        dummy_model_dir,
        [sse(role_chunk()), sse(content_chunk("Ole "))],
        error=error,
    )
    assert merged_reply(chunks) == ("Ole ", [], None)
    assert len(errors) == 1
    assert done is False


async def test_upstream_error_line_is_passed_on_verbatim(dummy_model_dir: Path) -> None:
    """Verify llama-server's own error: line reaches the client and ends the stream."""
    line = b'error: {"code":500,"message":"slot died","type":"server_error"}\n\n'
    chunks, errors, done = await relay(
        dummy_model_dir,
        [sse(role_chunk()), sse(content_chunk("Ole ")), line, sse(content_chunk("ignored"))],
    )
    assert merged_reply(chunks)[0] == "Ole "
    assert errors == [line.decode().strip()]
    assert done is False


async def test_upstream_error_chunk_ends_the_stream(dummy_model_dir: Path) -> None:
    """Verify an error sent as a data: payload also ends the stream without [DONE]."""
    chunks, errors, done = await relay(
        dummy_model_dir,
        [sse(role_chunk()), sse({"error": {"code": 500, "message": "slot died"}}), b"data: [DONE]\n\n"],
    )
    assert len(chunks) == 1
    assert "slot died" in errors[0]
    assert done is False


@pytest.mark.parametrize(
    "tail",
    [[], [b'data: {"choices":[{"delta":{"content":"cut']],
)
async def test_stream_closed_without_finish_is_reported(
    dummy_model_dir: Path,
    tail: List[bytes],
) -> None:
    """Verify a stream that just stops gets an error line rather than a synthetic ending."""
    chunks, errors, done = await relay(
        dummy_model_dir,
        [sse(role_chunk()), sse(content_chunk("Ole ")), *tail],
    )
    assert merged_reply(chunks) == ("Ole ", [], None)
    assert len(errors) == 1
    assert done is False


async def test_failure_mid_stream_never_releases_a_held_tool_call(
    dummy_model_dir: Path,
    sample_mcp_openai_tools: List[Dict[str, Any]],
) -> None:
    """Verify half a tool call stays hidden when the stream dies around it."""
    chunks, errors, done = await relay(
        dummy_model_dir,
        [sse(role_chunk()), sse(content_chunk('<|tool_call|>[{"name": "get_cfb'))],
        tools=sample_mcp_openai_tools,
        error=httpx.ReadError("reset"),
    )
    assert merged_reply(chunks) == ("", [], None)
    assert len(errors) == 1
    assert done is False


async def test_stream_that_ends_after_finish_reason_is_not_an_error(dummy_model_dir: Path) -> None:
    """Verify [DONE] is optional once a finish_reason arrived, and is not invented."""
    chunks, errors, done = await relay(
        dummy_model_dir,
        [sse(content_chunk("Hi")), sse(finish_chunk())],
    )
    assert merged_reply(chunks) == ("Hi", [], "stop")
    assert errors == []
    assert done is False


async def test_aclose_closes_upstream_body_that_was_never_iterated(dummy_model_dir: Path) -> None:
    """Verify an unread stream still releases upstream, as when a client leaves straight away."""
    closed = asyncio.Event()

    class TrackedBody(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield sse(role_chunk())

        async def aclose(self) -> None:
            closed.set()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=TrackedBody())

    proxy = make_proxy(dummy_model_dir, handler)
    stream = await proxy.open_chat_stream(make_request())
    assert not closed.is_set()
    await stream.aclose()
    await stream.aclose()
    assert closed.is_set()


async def test_response_closes_upstream_when_client_leaves_before_the_body_starts(
    dummy_model_dir: Path,
) -> None:
    """Verify the response closes upstream itself, since the body iterator may never run."""
    closed = asyncio.Event()

    class TrackedBody(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield sse(role_chunk())
            await asyncio.sleep(60)

        async def aclose(self) -> None:
            closed.set()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=TrackedBody())

    proxy = make_proxy(dummy_model_dir, handler)
    stream = await proxy.open_chat_stream(make_request())

    async def receive() -> Dict[str, Any]:
        return {"type": "http.disconnect"}

    async def send(message: Dict[str, Any]) -> None:
        await asyncio.sleep(60)

    assert not closed.is_set()
    await ChatCompletionStreamResponse(stream)({"type": "http"}, receive, send)
    assert closed.is_set()


class StubLlamaServer:
    """A socket-level SSE upstream, so timing and disconnects behave as they do in production."""

    def __init__(self, respond: Callable[[asyncio.StreamWriter], Awaitable[None]]) -> None:
        self.respond = respond
        self.disconnected = asyncio.Event()
        self.server: Optional[asyncio.AbstractServer] = None

    @property
    def url(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        headers = (await reader.readuntil(b"\r\n\r\n")).lower()
        length = int(headers.split(b"content-length:")[1].split(b"\r\n")[0])
        await reader.readexactly(length)
        replying = asyncio.ensure_future(self.respond(writer))
        # The request has been read, so EOF here means the proxy hung up.
        await reader.read()
        self.disconnected.set()
        replying.cancel()
        writer.close()

    async def __aenter__(self) -> "StubLlamaServer":
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        assert self.server is not None
        self.server.close()


async def send_headers(writer: asyncio.StreamWriter) -> None:
    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
    )
    await writer.drain()


async def send_chunk(writer: asyncio.StreamWriter, data: bytes) -> None:
    writer.write(b"%x\r\n%s\r\n" % (len(data), data))
    await writer.drain()


@contextlib.asynccontextmanager
async def running_gateway(upstream_url: str, dummy_model_dir: Path) -> AsyncIterator[str]:
    """Serve the real app with uvicorn, which is what delivers client disconnects."""
    application = create_app(
        Settings(
            model_cache_dir=str(dummy_model_dir),
            external_llama_url=upstream_url,
            auto_start_llama_server=False,
            llama_request_timeout_seconds=30.0,
        )
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(application, log_config=None, lifespan="off"))
    serving = asyncio.ensure_future(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}/v1/chat/completions"
    finally:
        server.should_exit = True
        server.force_exit = True
        await serving
        sock.close()


async def test_client_disconnect_mid_stream_closes_upstream(dummy_model_dir: Path) -> None:
    """Verify a client hanging up mid-reply drops the upstream connection, freeing the slot."""

    async def respond(writer: asyncio.StreamWriter) -> None:
        await send_headers(writer)
        await send_chunk(writer, sse(role_chunk()))
        await send_chunk(writer, sse(content_chunk("first")))
        await asyncio.sleep(60)

    async with StubLlamaServer(respond) as upstream:
        async with running_gateway(upstream.url, dummy_model_dir) as url:
            async with httpx.AsyncClient(timeout=5.0) as client:
                async with client.stream(
                    "POST",
                    url,
                    json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
                ) as resp:
                    assert resp.status_code == 200
                    async for chunk in resp.aiter_bytes():
                        if b"first" in chunk:
                            break
                assert not upstream.disconnected.is_set()
            await asyncio.wait_for(upstream.disconnected.wait(), timeout=3.0)


async def test_client_disconnect_before_first_token_closes_upstream(dummy_model_dir: Path) -> None:
    """Verify a client hanging up during prompt processing also drops the upstream connection."""

    async def respond(writer: asyncio.StreamWriter) -> None:
        await send_headers(writer)
        await asyncio.sleep(60)

    async with StubLlamaServer(respond) as upstream:
        async with running_gateway(upstream.url, dummy_model_dir) as url:
            async with httpx.AsyncClient() as client:
                with pytest.raises(httpx.ReadTimeout):
                    await client.post(
                        url,
                        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
                        timeout=0.3,
                    )
            await asyncio.wait_for(upstream.disconnected.wait(), timeout=3.0)


async def test_read_timeout_applies_to_gaps_not_the_whole_reply(dummy_model_dir: Path) -> None:
    """Verify a reply may outlast the timeout, as long as no single gap exceeds it."""
    pause = 0.08

    async def respond(writer: asyncio.StreamWriter) -> None:
        await send_headers(writer)
        for frame in reply_bytes(["a", "b", "c", "d", "e", "f"]):
            await send_chunk(writer, frame)
            await asyncio.sleep(pause)
        await send_chunk(writer, b"")

    async with StubLlamaServer(respond) as upstream:
        proxy = LlamaInferenceProxy(
            base_url=upstream.url,
            model_manager=ModelManager(cache_dir=dummy_model_dir),
            timeout_seconds=0.25,
        )
        started = time.monotonic()
        stream = await proxy.open_chat_stream(make_request())
        body = b"".join([frame async for frame in stream])
        await stream.aclose()

    assert time.monotonic() - started > 0.25
    chunks, errors, done = parse_frames(body)
    assert errors == []
    assert done is True
    assert merged_reply(chunks)[0] == "abcdef"


async def test_read_timeout_on_a_gap_fails_the_stream(dummy_model_dir: Path) -> None:
    """Verify a stalled upstream ends the stream with an error, not a silent hang."""

    async def respond(writer: asyncio.StreamWriter) -> None:
        await send_headers(writer)
        await send_chunk(writer, sse(content_chunk("Ole ")))
        await asyncio.sleep(60)

    async with StubLlamaServer(respond) as upstream:
        proxy = LlamaInferenceProxy(
            base_url=upstream.url,
            model_manager=ModelManager(cache_dir=dummy_model_dir),
            timeout_seconds=0.2,
        )
        stream = await proxy.open_chat_stream(make_request())
        body = b"".join([frame async for frame in stream])
        await stream.aclose()
        await asyncio.wait_for(upstream.disconnected.wait(), timeout=3.0)

    chunks, errors, done = parse_frames(body)
    assert merged_reply(chunks)[0] == "Ole "
    assert "ReadTimeout" in errors[0]
    assert done is False


async def test_chunks_reach_the_client_as_they_arrive(dummy_model_dir: Path) -> None:
    """Verify nothing between upstream and client buffers the reply."""
    release = asyncio.Event()

    async def respond(writer: asyncio.StreamWriter) -> None:
        await send_headers(writer)
        await send_chunk(writer, sse(role_chunk()))
        await send_chunk(writer, sse(content_chunk("early")))
        await release.wait()
        for frame in [sse(content_chunk("late")), sse(finish_chunk()), b"data: [DONE]\n\n"]:
            await send_chunk(writer, frame)
        await send_chunk(writer, b"")

    async with StubLlamaServer(respond) as upstream:
        async with running_gateway(upstream.url, dummy_model_dir) as url:
            async with httpx.AsyncClient(timeout=5.0) as client:
                async with client.stream(
                    "POST",
                    url,
                    json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
                ) as resp:
                    received = b""
                    async for chunk in resp.aiter_bytes():
                        received += chunk
                        # Only sent once the client has seen the first text.
                        if b"early" in received:
                            release.set()
    assert parse_frames(received)[2] is True
    assert merged_reply(parse_frames(received)[0])[0] == "earlylate"
