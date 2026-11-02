"""FastAPI application entry point for SportsAgenticInference."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import time
from typing import Any, AsyncIterator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from app.config import Settings, get_settings
from app.logging_config import get_logger, setup_logging
from app.model_manager import ModelManager, ModelManagerError
from app.process_manager import LlamaProcessError, LlamaServerProcessManager
from app.proxy import (
    ChatCompletionStream,
    InferenceProxyError,
    InferenceUpstreamTimeoutError,
    InferenceUpstreamUnavailableError,
    LlamaInferenceProxy,
)
from app.schemas import (
    ChatCompletionRequest,
    HealthResponse,
    ModelListResponse,
)

logger = get_logger("api")

class ChatCompletionStreamResponse(StreamingResponse):
    """Sends an open upstream stream as server-sent events, closing it however the response ends."""

    def __init__(self, stream: ChatCompletionStream) -> None:
        super().__init__(
            stream,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self.stream = stream

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Starlette can cancel the response before the body starts, so closing cannot
        # be left to the body iterator.
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.stream.aclose()


async def wait_for_disconnect(request: Request) -> None:
    """Return once the client has gone.

    Request.is_disconnected() cannot be used here: under BaseHTTPMiddleware, which the
    request logging below relies on, it always reports False.
    """
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def open_stream_while_connected(
    proxy: LlamaInferenceProxy,
    payload: ChatCompletionRequest,
    request: Request,
) -> Optional[ChatCompletionStream]:
    """Open an upstream stream, returning None if the client leaves before the first token."""
    opening = asyncio.ensure_future(proxy.open_chat_stream(payload))
    disconnect = asyncio.ensure_future(wait_for_disconnect(request))
    try:
        await asyncio.wait({opening, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        if opening.done():
            return opening.result()
        logger.info("Client disconnected before the first token")
        return None
    finally:
        opening.cancel()
        disconnect.cancel()
        await asyncio.wait({opening, disconnect})


def build_components(
    settings: Settings,
) -> tuple[ModelManager, LlamaServerProcessManager, LlamaInferenceProxy]:
    """Instantiate core inference services from Settings."""
    model_manager = ModelManager(
        model_id=settings.model_id,
        model_alias=settings.model_alias,
        hf_gguf_repo=settings.hf_gguf_repo,
        hf_gguf_filename=settings.hf_gguf_filename,
        cache_dir=settings.model_cache_dir,
        auto_download=settings.auto_download_model,
        hf_token=settings.hf_token,
    )
    process_manager = LlamaServerProcessManager(
        model_manager=model_manager,
        llama_server_bin=settings.llama_server_bin,
        host=settings.llama_internal_host,
        port=settings.llama_internal_port,
        ctx_size=settings.llama_ctx_size,
        threads=settings.llama_threads,
        n_gpu_layers=settings.llama_n_gpu_layers,
        startup_timeout_seconds=settings.llama_startup_timeout_seconds,
        external_llama_url=settings.external_llama_url,
    )
    inference_proxy = LlamaInferenceProxy(
        base_url=settings.resolved_llama_base_url,
        model_manager=model_manager,
        timeout_seconds=settings.llama_request_timeout_seconds,
    )
    return model_manager, process_manager, inference_proxy


def create_app(custom_settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = custom_settings or get_settings()
    setup_logging(settings.log_level)

    model_manager, process_manager, inference_proxy = build_components(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Starting SportsAgenticInference (model=%s, alias=%s, port=%d, upstream=%s)",
            settings.model_id,
            settings.model_alias,
            settings.port,
            settings.resolved_llama_base_url,
        )
        if settings.auto_start_llama_server:
            try:
                await application.state.process_manager.start(wait_until_healthy=True)
            except (ModelManagerError, LlamaProcessError) as exc:
                logger.warning(
                    "llama-server did not start during lifespan initialization: %s",
                    exc,
                )
        yield
        logger.info("Shutting down SportsAgenticInference")
        await application.state.process_manager.stop()

    application = FastAPI(
        title="SportsAgenticInference",
        description="OpenAI-compatible inference server running llama-server and microsoft/Phi-4-mini-instruct for SportsAgenticAgent",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.state.settings = settings
    application.state.model_manager = model_manager
    application.state.process_manager = process_manager
    application.state.inference_proxy = inference_proxy

    @application.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    def get_model_manager(request: Request) -> ModelManager:
        return request.app.state.model_manager

    def get_process_manager(request: Request) -> LlamaServerProcessManager:
        return request.app.state.process_manager

    def get_inference_proxy(request: Request) -> LlamaInferenceProxy:
        return request.app.state.inference_proxy

    @application.get("/health", response_model=HealthResponse)
    async def health_endpoint(
        model_mgr: ModelManager = Depends(get_model_manager),
        proc_mgr: LlamaServerProcessManager = Depends(get_process_manager),
    ) -> JSONResponse:
        """Report health of the GGUF model artifact and upstream llama-server process."""
        model_present = model_mgr.is_model_present()
        running = proc_mgr.is_running
        upstream_ok, detail = await proc_mgr.check_upstream_health()

        is_healthy = upstream_ok and (model_present or bool(proc_mgr.external_llama_url))
        status_label = "ok" if is_healthy else "degraded"
        http_status = 200 if is_healthy else 503

        payload = HealthResponse(
            status=status_label,
            model_id=model_mgr.model_id,
            model_alias=model_mgr.model_alias,
            model_file_present=model_present,
            model_path=str(model_mgr.model_path),
            llama_server_running=running,
            llama_server_healthy=upstream_ok,
            detail=detail,
        )
        return JSONResponse(status_code=http_status, content=payload.model_dump())

    @application.get("/v1/models", response_model=ModelListResponse)
    async def list_models_endpoint(
        model_mgr: ModelManager = Depends(get_model_manager),
    ) -> ModelListResponse:
        """Return OpenAI-compatible model list including microsoft/Phi-4-mini-instruct and local-model."""
        return ModelListResponse(data=model_mgr.get_model_cards())

    @application.post("/v1/chat/completions")
    async def chat_completions_endpoint(
        payload: ChatCompletionRequest,
        request: Request,
        proxy: LlamaInferenceProxy = Depends(get_inference_proxy),
    ) -> Any:
        """Execute an OpenAI-compatible chat completion via llama-server."""
        try:
            if payload.stream:
                stream = await open_stream_while_connected(proxy, payload, request)
                if stream is None:
                    return Response(status_code=499)
                return ChatCompletionStreamResponse(stream)
            return await proxy.complete_chat(payload)
        except InferenceUpstreamTimeoutError as exc:
            logger.error("Chat completion timed out: %s", exc)
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except InferenceUpstreamUnavailableError as exc:
            logger.error("Upstream llama-server unavailable: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except InferenceProxyError as exc:
            logger.error("Chat completion failed: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return application


app = create_app(
    Settings(auto_start_llama_server=False)
    if "pytest" in __import__("sys").modules
    else None
)
