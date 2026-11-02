"""Async proxy and Phi-4-mini-instruct tool-call normalizer for llama-server."""

from __future__ import annotations

import json
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import anyio
import httpx

from app.logging_config import get_logger
from app.model_manager import ModelManager
from app.schemas import ChatCompletionRequest

logger = get_logger("proxy")


class InferenceProxyError(Exception):
    """Base error raised when proxying an inference request to llama-server fails."""


class InferenceUpstreamTimeoutError(InferenceProxyError):
    """Raised when the upstream llama-server request times out."""


class InferenceUpstreamUnavailableError(InferenceProxyError):
    """Raised when the upstream llama-server cannot be reached or returns HTTP 5xx."""


def extract_known_tool_names(tools: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Extract function names from an OpenAI tools array."""
    if not tools:
        return []
    names: List[str] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        func = item.get("function")
        if isinstance(func, dict) and isinstance(func.get("name"), str):
            names.append(func["name"])
        elif isinstance(item.get("name"), str):
            names.append(item["name"])
    return names


def normalize_tool_call_entry(
    raw_entry: Dict[str, Any],
    index: int,
    known_tool_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a single tool call dict into OpenAI's {id, type, function: {name, arguments}} schema."""
    func_block = raw_entry.get("function")
    if isinstance(func_block, dict):
        name = func_block.get("name")
        raw_args = func_block.get("arguments", "{}")
    else:
        name = raw_entry.get("name") or raw_entry.get("tool")
        raw_args = (
            raw_entry.get("arguments")
            if "arguments" in raw_entry
            else raw_entry.get("parameters", raw_entry.get("params", "{}"))
        )

    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()

    if known_tool_names and name not in known_tool_names:
        return None

    if isinstance(raw_args, str):
        stripped_args = raw_args.strip()
        if not stripped_args:
            serialized_args = "{}"
        else:
            try:
                parsed_args = json.loads(stripped_args)
                serialized_args = (
                    json.dumps(parsed_args) if isinstance(parsed_args, dict) else "{}"
                )
            except json.JSONDecodeError:
                serialized_args = "{}"
    elif isinstance(raw_args, dict):
        serialized_args = json.dumps(raw_args)
    else:
        serialized_args = "{}"

    normalized: Dict[str, Any] = {
        "id": str(raw_entry.get("id") or f"call_{index}"),
        "type": "function",
        "function": {
            "name": name,
            "arguments": serialized_args,
        },
    }
    if isinstance(raw_entry.get("extra_content"), dict):
        normalized["extra_content"] = raw_entry["extra_content"]
    return normalized


def parse_python_style_tool_call(
    content: str,
    known_tool_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Parse a tool call emitted as Python/bash syntax like get_cfb_scores(year=2026, team="CSU")."""
    if not known_tool_names:
        return None

    names_pattern = "|".join(re.escape(name) for name in known_tool_names)
    pattern = rf"\b({names_pattern})\s*\((.*?)\)"
    match = re.search(pattern, content, flags=re.DOTALL)
    if not match:
        return None

    func_name = match.group(1)
    args_str = match.group(2).strip()

    # Parse key=value arguments (e.g. year=2026, week=8, team="CSU", date='2026-10-31')
    arg_matches = re.findall(
        r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s,)]+))',
        args_str,
    )
    arguments: Dict[str, Any] = {}
    for key, val_dq, val_sq, val_raw in arg_matches:
        raw = val_dq if val_dq != "" else (val_sq if val_sq != "" else val_raw)
        if raw == "True":
            parsed_val: Any = True
        elif raw == "False":
            parsed_val = False
        elif raw == "None":
            parsed_val = None
        else:
            try:
                parsed_val = int(raw)
            except ValueError:
                try:
                    parsed_val = float(raw)
                except ValueError:
                    parsed_val = raw
        arguments[key] = parsed_val

    return normalize_tool_call_entry(
        {"name": func_name, "arguments": arguments},
        index=1,
        known_tool_names=known_tool_names,
    )


def parse_phi_inline_tool_calls(
    content: str,
    known_tool_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract tool calls emitted inline by microsoft/Phi-4-mini-instruct.

    Supports:
    - <|tool_call|>[...]<|/tool_call|> or <|tool|>[...]<|/tool|> tags
    - ```json ... ``` fenced blocks containing a tool call object or array
    - Raw JSON object or array matching a tool invocation schema
    - Python/code style tool_name(key=val) function calls
    """
    if not content or not content.strip():
        return []

    stripped = content.strip()
    candidates: List[str] = []

    # 1. Phi-4-mini special tool tags (<|tool_call|>, <|tool_calls|>, <|tool|>)
    tag_matches = re.findall(
        r"<\|tool(?:_calls?)?\|>\s*(.*?)\s*(?:<\|/tool(?:_calls?)?\|>|<\|end\|>|$)",
        stripped,
        flags=re.DOTALL,
    )
    candidates.extend(tag_matches)

    # 2. Markdown JSON code fences
    fence_matches = re.findall(
        r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
        stripped,
        flags=re.DOTALL,
    )
    candidates.extend(fence_matches)

    # 3. Direct JSON object or array in content
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        candidates.append(stripped)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue

        items: List[Any] = parsed if isinstance(parsed, list) else [parsed]
        extracted: List[Dict[str, Any]] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            norm = normalize_tool_call_entry(item, index=idx, known_tool_names=known_tool_names)
            if norm is not None:
                extracted.append(norm)

        if extracted:
            return extracted

    # 4. Fallback for Python/code-style function call syntax e.g. get_cfb_scores(team="CSU", week=8)
    py_call = parse_python_style_tool_call(stripped, known_tool_names=known_tool_names)
    if py_call is not None:
        return [py_call]

    return []



def normalize_chat_completion_payload(
    payload: Dict[str, Any],
    canonical_model_id: str,
    known_tool_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Normalize an upstream llama-server chat completion response for SportsAgenticAgent."""
    normalized = dict(payload)
    normalized["model"] = canonical_model_id
    normalized.setdefault("id", "chatcmpl-phi4-mini")
    normalized.setdefault("object", "chat.completion")
    normalized.setdefault("created", int(time.time()))

    choices = normalized.get("choices")
    if not isinstance(choices, list) or not choices:
        return normalized

    updated_choices: List[Dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        choice_copy = dict(choice)
        message = dict(choice_copy.get("message") or {})
        message.setdefault("role", "assistant")

        raw_tool_calls = message.get("tool_calls") or []
        clean_tool_calls: List[Dict[str, Any]] = []
        if isinstance(raw_tool_calls, list):
            for idx, tc in enumerate(raw_tool_calls, start=1):
                if isinstance(tc, dict):
                    norm_tc = normalize_tool_call_entry(tc, index=idx)
                    if norm_tc is not None:
                        clean_tool_calls.append(norm_tc)

        content = message.get("content") or ""
        if not clean_tool_calls and isinstance(content, str) and content.strip():
            inline_calls = parse_phi_inline_tool_calls(
                content,
                known_tool_names=known_tool_names,
            )
            if inline_calls:
                clean_tool_calls = inline_calls
                content = ""

        message["content"] = content
        if clean_tool_calls:
            message["tool_calls"] = clean_tool_calls
            choice_copy["finish_reason"] = "tool_calls"
        else:
            choice_copy.setdefault("finish_reason", "stop")

        choice_copy["message"] = message
        updated_choices.append(choice_copy)

    normalized["choices"] = updated_choices
    return normalized


# These mirror the patterns in parse_phi_inline_tool_calls.
_TOOL_TAG_OPEN = r"<\|tool(?:_calls?)?\|>"
_TOOL_TAG_CLOSE = re.compile(r"<\|/tool(?:_calls?)?\|>|<\|end\|>")
_JSON_FENCE_BLOCK = re.compile(r"```(?:json)?\s*(?:\{.*?\}|\[.*?\])\s*```", flags=re.DOTALL)
_FENCE_OPENER = re.compile(r"```([A-Za-z]*)\s*")
_FENCE_MARKER = "```"
# Markers that a chunk boundary can cut in two.
_PARTIAL_MARKERS = ("<|tool_call|>", "<|tool_calls|>", "<|tool|>", _FENCE_MARKER)
# The opening of a JSON tool call that fills the whole reply. A reply such as "[1] Alabama" is not one.
_JSON_CALL_START = re.compile(r'\s*(?:\Z|\{\s*(?:\Z|")|\[\s*(?:\Z|\{\s*(?:\Z|")))')


class InlineToolCallFilter:
    """Separates streamed content into text that is safe to show and inline tool calls.

    Text that could still turn into a call parse_phi_inline_tool_calls accepts stays held
    until it is complete, then goes through that parser, so streamed and non-streamed
    replies agree. Narration before a call is released, since a stream cannot take it back.
    """

    def __init__(self, known_tool_names: Optional[List[str]] = None) -> None:
        self._known_tool_names = known_tool_names or None
        self._names = list(known_tool_names or [])
        names_pattern = "|".join(re.escape(name) for name in self._names)
        call_start = rf"\b(?:{names_pattern})\s*\("
        if self._names:
            self._call_start: Optional[re.Pattern[str]] = re.compile(call_start)
            self._name_at_end: Optional[re.Pattern[str]] = re.compile(
                rf"\b(?:{names_pattern})\s*\Z"
            )
            self._candidate = re.compile(rf"({_TOOL_TAG_OPEN})|(```)|({call_start})")
        else:
            self._call_start = None
            self._name_at_end = None
            self._candidate = re.compile(rf"({_TOOL_TAG_OPEN})|(```)")
        self._content = ""
        self._sent = 0
        self._accepting = True
        self._pending_calls: List[Dict[str, Any]] = []
        self.found_calls = False
        self.finished = False

    def feed(self, text: str) -> str:
        """Add streamed text and return the part that is now safe to show."""
        if not self._accepting:
            return ""
        self._content += text
        start = self._sent
        self._scan()
        return self._content[start : self._sent]

    def finish(self) -> str:
        """Resolve held text at the end of the reply and return any that is plain text."""
        self.finished = True
        if not self._accepting:
            return ""
        held = self._content[self._sent :]
        self._sent = len(self._content)
        self._record_calls(parse_phi_inline_tool_calls(held, self._known_tool_names))
        return "" if self.found_calls else held

    def stop_text(self) -> None:
        """Drop held and later text, because the reply has switched to native tool_calls."""
        self._accepting = False

    def take_calls(self) -> List[Dict[str, Any]]:
        """Return tool calls found since the last call."""
        calls, self._pending_calls = self._pending_calls, []
        return calls

    def _record_calls(self, calls: List[Dict[str, Any]]) -> None:
        if calls:
            self._pending_calls = calls
            self.found_calls = True
            self._accepting = False

    def _scan(self) -> None:
        # A bare JSON call fills the whole reply, so it is only ruled out at the end.
        if self._sent == 0 and _JSON_CALL_START.match(self._content):
            return

        while self._accepting:
            match = self._candidate.search(self._content, self._sent)
            if match is None:
                self._sent = self._partial_marker_start()
                return

            self._sent = match.start()
            if match.group(1):
                end = self._resolve_tag(match.start(), match.end())
            elif match.group(2):
                end = self._resolve_fence(match.start())
            else:
                end = self._resolve_call(match.start(), match.end())

            if end is None or not self._accepting:
                return
            self._sent = end

    def _resolve_tag(self, start: int, body: int) -> Optional[int]:
        close = _TOOL_TAG_CLOSE.search(self._content, body)
        return None if close is None else self._parse_span(start, close.end())

    def _resolve_fence(self, start: int) -> Optional[int]:
        """Return where the text after a code fence is safe from, or None if still undecided."""
        opener = _FENCE_OPENER.match(self._content, start)
        assert opener is not None
        if opener.end() == len(self._content):
            return None

        if opener.group(1) in ("", "json") and self._content[opener.end()] in "{[":
            block = _JSON_FENCE_BLOCK.match(self._content, start)
            return None if block is None else self._parse_span(start, block.end())

        # A Python-style call can sit in a fence of any language, or straight after the backticks.
        for body in (start + len(_FENCE_MARKER), opener.end()):
            call = self._call_start.match(self._content, body) if self._call_start else None
            if call is not None:
                return self._resolve_call(start, call.end())
            if self._could_become_call(body):
                return None
        return opener.end()

    def _resolve_call(self, start: int, arguments: int) -> Optional[int]:
        close = self._content.find(")", arguments)
        return None if close < 0 else self._parse_span(start, close + 1)

    def _parse_span(self, start: int, end: int) -> int:
        self._record_calls(
            parse_phi_inline_tool_calls(self._content[start:end], self._known_tool_names)
        )
        return end

    def _could_become_call(self, start: int) -> bool:
        """Return whether the text from start to the end could still grow into name(."""
        if self._name_at_end is not None and self._name_at_end.match(self._content, start):
            return True
        tail = self._content[start:]
        return any(name.startswith(tail) for name in self._names)

    def _partial_marker_start(self) -> int:
        """Return where a marker cut off by the end of the text begins."""
        content = self._content
        if self._name_at_end is not None:
            name = self._name_at_end.search(content, self._sent)
            if name is not None:
                return name.start()

        longest = max([len(item) for item in (*_PARTIAL_MARKERS, *self._names)])
        for start in range(max(self._sent, len(content) - longest), len(content)):
            tail = content[start:]
            if any(marker.startswith(tail) for marker in _PARTIAL_MARKERS):
                return start
            if not _is_word_boundary(content, start):
                continue
            if any(name.startswith(tail) for name in self._names):
                return start
        return len(content)


def _is_word_boundary(text: str, index: int) -> bool:
    return index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_")


def _sse_data_frame(chunk: Dict[str, Any]) -> bytes:
    # Non-ASCII stays escaped, so no line reader can split a frame on a Unicode line separator.
    return b"data: " + json.dumps(chunk).encode("ascii") + b"\n\n"


def _sse_error_frame(message: str) -> bytes:
    payload = {"message": message, "type": "upstream_error"}
    return b"error: " + json.dumps(payload).encode("ascii") + b"\n\n"


def _stream_error(line: bytes) -> Optional[InferenceProxyError]:
    """Return the failure an upstream SSE line reports, or None for an ordinary line."""
    field, _, value = line.partition(b":")
    value = value.strip()
    if field not in (b"data", b"error") or (field == b"data" and value == b"[DONE]"):
        return None

    try:
        parsed = json.loads(value)
    except ValueError:
        parsed = None

    if field == b"data":
        if not isinstance(parsed, dict):
            return InferenceProxyError(f"Invalid JSON from upstream llama-server: {value!r}")
        if not parsed.get("error"):
            return None
        parsed = parsed["error"]

    detail = value.decode("utf-8", errors="replace")
    code = None
    if isinstance(parsed, dict):
        detail = str(parsed.get("message") or detail)
        code = parsed.get("code")
    if isinstance(code, int) and code >= 500:
        return InferenceUpstreamUnavailableError(f"llama-server stream error: {detail}")
    return InferenceProxyError(f"llama-server stream error: {detail}")


async def _iter_sse_lines(response: httpx.Response) -> AsyncIterator[bytes]:
    """Yield upstream SSE lines as soon as each one is complete.

    Lines split on newlines alone. httpx's aiter_lines also splits on characters such as
    U+2028, which llama-server can send unescaped inside a JSON string.
    """
    pending = b""
    async for chunk in response.aiter_bytes():
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        for line in lines:
            yield line.rstrip(b"\r")
    if pending:
        yield pending


async def _close_upstream(
    response: Optional[httpx.Response],
    client: Optional[httpx.AsyncClient],
) -> None:
    # Shielded because this usually runs while a client disconnect is cancelling the request.
    with anyio.CancelScope(shield=True):
        if response is not None:
            await response.aclose()
        if client is not None:
            await client.aclose()


class ChatCompletionStream:
    """An open upstream chat completion, relayed as server-sent event frames.

    Iterating yields frames ready to send. aclose() must always be called, because it is
    what drops the upstream connection, and that is how llama-server learns to stop.
    """

    def __init__(
        self,
        response: httpx.Response,
        lines: AsyncIterator[bytes],
        first_line: bytes,
        model: str,
        known_tool_names: Optional[List[str]],
        owned_client: Optional[httpx.AsyncClient],
    ) -> None:
        self._response = response
        self._lines = lines
        self._first_line = first_line
        self._model = model
        self._known_tool_names = known_tool_names
        self._owned_client = owned_client
        self._filters: Dict[Any, InlineToolCallFilter] = {}
        self._last_chunk: Dict[str, Any] = {}
        self._ended = False
        self._closed = False
        self._start = time.monotonic()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for line in self._upstream_lines():
                for frame in self._relay_line(line):
                    yield frame
                if self._ended:
                    break
            if not self._ended and not self._reply_finished():
                logger.warning("llama-server stream ended before the reply was complete")
                yield _sse_error_frame("llama-server stream ended before the reply was complete")
        except httpx.HTTPError as exc:
            # Held text is dropped: it may be half a tool call, and the reply has failed anyway.
            logger.warning("llama-server stream failed after it started: %r", exc)
            yield _sse_error_frame(f"llama-server stream failed: {exc!r}")

    async def aclose(self) -> None:
        """Release the upstream connection. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        await _close_upstream(self._response, self._owned_client)
        logger.info(
            "llama-server stream closed after %.1fms (tool_calls=%s)",
            (time.monotonic() - self._start) * 1000.0,
            any(item.found_calls for item in self._filters.values()),
        )

    async def _upstream_lines(self) -> AsyncIterator[bytes]:
        yield self._first_line
        async for line in self._lines:
            yield line

    def _reply_finished(self) -> bool:
        return bool(self._filters) and all(item.finished for item in self._filters.values())

    def _relay_line(self, line: bytes) -> List[bytes]:
        error = _stream_error(line)
        if error is not None:
            logger.warning("llama-server stream failed after it started: %s", error)
            self._ended = True
            if line.startswith(b"error:"):
                return [line + b"\n\n"]
            return [_sse_error_frame(str(error))]

        if not line.startswith(b"data:"):
            return [line + b"\n\n"] if line else []

        if line[5:].strip() == b"[DONE]":
            self._ended = True
            return [*self._finish_open_choices(), b"data: [DONE]\n\n"]

        chunk = json.loads(line[5:])
        chunk["model"] = self._model
        self._last_chunk = chunk
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return [_sse_data_frame(chunk)]

        frames: List[bytes] = []
        for choice in choices:
            if isinstance(choice, dict):
                frames.extend(self._relay_choice(chunk, choice))
        return frames

    def _finish_open_choices(self) -> List[bytes]:
        """Flush held text for a stream that reached [DONE] without a finish_reason."""
        frames: List[bytes] = []
        for index, item in self._filters.items():
            if not item.finished:
                closing = {"index": index, "delta": {}, "finish_reason": "stop"}
                frames.extend(self._relay_choice(self._last_chunk, closing))
        return frames

    def _relay_choice(self, chunk: Dict[str, Any], choice: Dict[str, Any]) -> List[bytes]:
        index = choice.get("index", 0)
        if index not in self._filters:
            self._filters[index] = InlineToolCallFilter(self._known_tool_names)
        inline = self._filters[index]

        delta = dict(choice.get("delta") or {})
        if delta.get("tool_calls"):
            inline.stop_text()

        visible = ""
        if isinstance(delta.get("content"), str):
            visible = inline.feed(delta.pop("content"))
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            visible += inline.finish()
        if visible:
            delta["content"] = visible

        # The call goes in a chunk of its own so that text never arrives alongside it.
        deltas = [delta] if delta else []
        calls = inline.take_calls()
        if calls:
            deltas.append(
                {"tool_calls": [{"index": position, **call} for position, call in enumerate(calls)]}
            )
        if finish_reason and not deltas:
            deltas.append({})
        if finish_reason and inline.found_calls:
            finish_reason = "tool_calls"

        frames: List[bytes] = []
        for position, item in enumerate(deltas, start=1):
            relayed = {**choice, "delta": item, "finish_reason": None}
            if position == len(deltas):
                relayed["finish_reason"] = finish_reason
            frames.append(_sse_data_frame({**chunk, "choices": [relayed]}))
        return frames


class LlamaInferenceProxy:
    """Forwards OpenAI chat completion requests to llama-server and normalizes responses."""

    def __init__(
        self,
        base_url: str,
        model_manager: ModelManager,
        timeout_seconds: float = 60.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_manager = model_manager
        self.timeout_seconds = timeout_seconds
        self._external_client = http_client

    @property
    def chat_completions_url(self) -> str:
        """Full URL to upstream llama-server /v1/chat/completions."""
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    async def complete_chat(
        self,
        request: ChatCompletionRequest,
    ) -> Dict[str, Any]:
        """Forward a ChatCompletionRequest to llama-server and return a normalized response."""
        resolved_model = self.model_manager.resolve_requested_model(request.model)
        upstream_payload = request.model_dump(exclude_none=True)
        upstream_payload["model"] = resolved_model

        known_tools = extract_known_tool_names(request.tools)
        start = time.monotonic()
        logger.info(
            "Forwarding chat completion to llama-server (%s, model='%s' -> '%s', tools=%d)",
            self.chat_completions_url,
            request.model,
            resolved_model,
            len(known_tools),
        )

        if self._external_client is not None:
            raw_data = await self._send_post(self._external_client, upstream_payload)
        else:
            timeout = httpx.Timeout(self.timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                raw_data = await self._send_post(client, upstream_payload)

        elapsed_ms = (time.monotonic() - start) * 1000.0
        normalized = normalize_chat_completion_payload(
            raw_data,
            canonical_model_id=resolved_model,
            known_tool_names=known_tools or None,
        )

        choices = normalized.get("choices") or []
        tool_call_count = 0
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            tool_call_count = len(msg.get("tool_calls") or [])

        logger.info(
            "llama-server completed request in %.1fms (tool_calls=%d)",
            elapsed_ms,
            tool_call_count,
        )
        return normalized

    async def open_chat_stream(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionStream:
        """Start a streamed completion, returning once llama-server has sent its first event.

        Failures up to that point raise the same errors as complete_chat, so the caller can
        still answer with a real status code. Later failures appear in the stream itself.
        """
        resolved_model = self.model_manager.resolve_requested_model(request.model)
        upstream_payload = request.model_dump(exclude_none=True)
        upstream_payload["model"] = resolved_model

        known_tools = extract_known_tool_names(request.tools)
        logger.info(
            "Streaming chat completion from llama-server (%s, model='%s' -> '%s', tools=%d)",
            self.chat_completions_url,
            request.model,
            resolved_model,
            len(known_tools),
        )

        # httpx read timeouts apply to each read, so timeout_seconds limits the wait for the
        # first token and every gap between chunks, not the length of the whole reply.
        client = self._external_client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds)
        )
        owned_client = None if self._external_client is not None else client
        response: Optional[httpx.Response] = None
        opened = False
        try:
            upstream_request = client.build_request(
                "POST",
                self.chat_completions_url,
                json=upstream_payload,
                headers={"Content-Type": "application/json"},
            )
            response = await client.send(upstream_request, stream=True)
            if not response.is_success:
                await response.aread()
            response.raise_for_status()

            lines = _iter_sse_lines(response)
            first_line = None
            async for line in lines:
                if line and not line.startswith(b":"):
                    first_line = line
                    break
            if first_line is None:
                raise InferenceUpstreamUnavailableError(
                    "llama-server closed the stream before sending any data"
                )
            error = _stream_error(first_line)
            if error is not None:
                logger.warning("llama-server stream failed before it started: %s", error)
                raise error

            opened = True
            return ChatCompletionStream(
                response=response,
                lines=lines,
                first_line=first_line,
                model=resolved_model,
                known_tool_names=known_tools or None,
                owned_client=owned_client,
            )
        except httpx.TimeoutException as exc:
            logger.warning(
                "Upstream llama-server timed out after %.1fs: %s",
                self.timeout_seconds,
                exc,
            )
            raise InferenceUpstreamTimeoutError(
                f"llama-server timed out after {self.timeout_seconds}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text
            logger.warning("Upstream llama-server returned HTTP %d: %s", status, body)
            if status >= 500:
                raise InferenceUpstreamUnavailableError(
                    f"llama-server returned server error HTTP {status}: {body}"
                ) from exc
            raise InferenceProxyError(
                f"llama-server returned HTTP {status}: {body}"
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("Failed to connect to upstream llama-server: %s", exc)
            raise InferenceUpstreamUnavailableError(
                f"Could not connect to upstream llama-server: {exc}"
            ) from exc
        finally:
            if not opened:
                await _close_upstream(response, owned_client)

    async def _send_post(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            response = await client.post(
                self.chat_completions_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise InferenceProxyError("Upstream llama-server returned non-object JSON")
            return data
        except httpx.TimeoutException as exc:
            logger.warning(
                "Upstream llama-server timed out after %.1fs: %s",
                self.timeout_seconds,
                exc,
            )
            raise InferenceUpstreamTimeoutError(
                f"llama-server timed out after {self.timeout_seconds}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text
            logger.warning("Upstream llama-server returned HTTP %d: %s", status, body)
            if status >= 500:
                raise InferenceUpstreamUnavailableError(
                    f"llama-server returned server error HTTP {status}: {body}"
                ) from exc
            raise InferenceProxyError(
                f"llama-server returned HTTP {status}: {body}"
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("Failed to connect to upstream llama-server: %s", exc)
            raise InferenceUpstreamUnavailableError(
                f"Could not connect to upstream llama-server: {exc}"
            ) from exc
        except ValueError as exc:
            logger.error("Invalid JSON from upstream llama-server: %s", exc)
            raise InferenceProxyError(
                f"Invalid JSON from upstream llama-server: {exc}"
            ) from exc
