"""Async proxy and Phi-4-mini-instruct tool-call normalizer for llama-server."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

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
