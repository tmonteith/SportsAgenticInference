"""Pydantic schemas for OpenAI-compatible chat completions, models, and health status."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """Single message in an OpenAI-compatible chat conversation."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions request payload."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(
        default="local-model",
        description="Requested model identifier (accepts 'local-model' or 'microsoft/Phi-4-mini-instruct')",
    )
    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Conversation history and user prompt",
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="OpenAI function-calling tool definitions converted from MCP tools",
    )
    tool_choice: Optional[Any] = Field(
        default=None,
        description="Tool selection policy",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream completion chunks",
    )


class ToolFunctionCall(BaseModel):
    """Function name and serialized JSON arguments inside a tool call."""

    name: str
    arguments: str


class ToolCallObject(BaseModel):
    """OpenAI-compatible tool call object returned in assistant messages."""

    id: str
    type: Literal["function"] = "function"
    function: ToolFunctionCall
    extra_content: Optional[Dict[str, Any]] = None


class AssistantResponseMessage(BaseModel):
    """Assistant message inside a chat completion choice."""

    model_config = ConfigDict(extra="allow")

    role: str = "assistant"
    content: Optional[str] = ""
    tool_calls: Optional[List[ToolCallObject]] = None


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: AssistantResponseMessage
    finish_reason: Optional[str] = "stop"


class CompletionUsage(BaseModel):
    """Token usage statistics."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible /v1/chat/completions response payload."""

    model_config = ConfigDict(extra="allow")

    id: str = "chatcmpl-phi4-mini"
    object: str = "chat.completion"
    created: int = 0
    model: str = "microsoft/Phi-4-mini-instruct"
    choices: List[ChatCompletionChoice]
    usage: Optional[CompletionUsage] = None


class ModelCard(BaseModel):
    """OpenAI-compatible model metadata entry."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "microsoft"
    root: Optional[str] = None


class ModelListResponse(BaseModel):
    """OpenAI-compatible /v1/models response."""

    object: str = "list"
    data: List[ModelCard]


class HealthResponse(BaseModel):
    """Health check payload returned by GET /health."""

    status: Literal["ok", "degraded", "error"]
    model_id: str
    model_alias: str
    model_file_present: bool
    model_path: str
    llama_server_running: bool
    llama_server_healthy: bool
    detail: str
