"""Async subprocess supervisor for the llama.cpp llama-server binary."""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import time
from typing import List, Optional

import httpx

from app.logging_config import get_logger
from app.model_manager import ModelManager

logger = get_logger("process")


class LlamaProcessError(Exception):
    """Raised when llama-server cannot be started or exits unexpectedly."""


class LlamaServerProcessManager:
    """Supervises the local llama-server subprocess and probes its health."""

    def __init__(
        self,
        model_manager: ModelManager,
        llama_server_bin: str = "llama-server",
        host: str = "127.0.0.1",
        port: int = 8082,
        ctx_size: int = 4096,
        threads: int = 4,
        n_gpu_layers: int = 0,
        startup_timeout_seconds: float = 60.0,
        external_llama_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.model_manager = model_manager
        self.llama_server_bin = llama_server_bin
        self.host = host
        self.port = port
        self.ctx_size = ctx_size
        self.threads = threads
        self.n_gpu_layers = n_gpu_layers
        self.startup_timeout_seconds = startup_timeout_seconds
        self.external_llama_url = external_llama_url.rstrip("/") if external_llama_url else None
        self._external_http_client = http_client

        self._process: Optional[asyncio.subprocess.Process] = None
        self._log_task: Optional[asyncio.Task[None]] = None

    @property
    def base_url(self) -> str:
        """Return the base URL of the managed or external llama-server."""
        if self.external_llama_url:
            return self.external_llama_url
        return f"http://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        """Return the /health endpoint of the upstream llama-server."""
        return f"{self.base_url}/health"

    @property
    def is_running(self) -> bool:
        """Return True if an external URL is configured or the subprocess is alive."""
        if self.external_llama_url:
            return True
        return self._process is not None and self._process.returncode is None

    def build_command(self, model_path: Path) -> List[str]:
        """Construct the CLI arguments for launching llama-server."""
        return [
            self.llama_server_bin,
            "--model",
            str(model_path),
            "--alias",
            self.model_manager.model_id,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.ctx_size),
            "--threads",
            str(self.threads),
            "--n-gpu-layers",
            str(self.n_gpu_layers),
            "--jinja",
        ]

    async def start(self, wait_until_healthy: bool = True) -> None:
        """Ensure the model exists and start the llama-server subprocess."""
        if self.external_llama_url:
            logger.info("Using external llama-server at %s; skipping subprocess launch", self.base_url)
            return

        if self.is_running:
            logger.debug("llama-server subprocess is already running")
            return

        model_path = await self.model_manager.ensure_model()

        resolved_bin = shutil.which(self.llama_server_bin) or self.llama_server_bin
        if not Path(resolved_bin).exists() and shutil.which(self.llama_server_bin) is None:
            msg = f"llama-server executable '{self.llama_server_bin}' not found in PATH"
            logger.error(msg)
            raise LlamaProcessError(msg)

        cmd = self.build_command(model_path)
        logger.info("Starting llama-server subprocess: %s", " ".join(cmd))

        # Ensure Metal bfloat16 compilation workaround is active on macOS
        proc_env = dict(__import__("os").environ)
        proc_env.setdefault("GGML_METAL_BF16_DISABLE", "1")

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=proc_env,
            )
        except OSError as exc:
            msg = f"Failed to spawn llama-server ('{self.llama_server_bin}'): {exc}"
            logger.error(msg)
            raise LlamaProcessError(msg) from exc

        self._log_task = asyncio.create_task(self._stream_process_logs())

        if wait_until_healthy:
            await self.wait_until_ready()

    async def _stream_process_logs(self) -> None:
        """Forward llama-server stdout/stderr lines to the application logger."""
        if self._process is None or self._process.stdout is None:
            return
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    logger.debug("[llama-server] %s", decoded)
        except asyncio.CancelledError:
            return

    async def wait_until_ready(self, poll_interval: float = 0.25) -> None:
        """Poll the upstream /health endpoint until llama-server reports ready."""
        deadline = time.monotonic() + self.startup_timeout_seconds
        last_detail = "Waiting for startup"

        while time.monotonic() < deadline:
            if not self.external_llama_url and not self.is_running:
                code = self._process.returncode if self._process else "unknown"
                msg = f"llama-server exited prematurely with code {code}"
                logger.error(msg)
                raise LlamaProcessError(msg)

            ok, detail = await self.check_upstream_health()
            if ok:
                logger.info("llama-server is ready at %s (%s)", self.base_url, detail)
                return
            last_detail = detail
            await asyncio.sleep(poll_interval)

        msg = (
            f"llama-server did not become ready within {self.startup_timeout_seconds}s "
            f"(last status: {last_detail})"
        )
        logger.error(msg)
        await self.stop()
        raise LlamaProcessError(msg)

    async def check_upstream_health(self) -> tuple[bool, str]:
        """Probe the upstream llama-server /health endpoint."""
        if not self.is_running:
            return False, "llama-server subprocess is not running"

        try:
            if self._external_http_client is not None:
                resp = await self._external_http_client.get(self.health_url)
            else:
                async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
                    resp = await client.get(self.health_url)
            resp.raise_for_status()
            return True, f"Upstream llama-server healthy at {self.health_url}"
        except httpx.HTTPStatusError as exc:
            return (
                False,
                f"Upstream llama-server returned HTTP {exc.response.status_code}",
            )
        except httpx.HTTPError as exc:
            return False, f"Upstream llama-server unreachable ({exc})"

    async def stop(self, grace_period_seconds: float = 5.0) -> None:
        """Gracefully stop the managed llama-server subprocess."""
        if self._log_task is not None:
            self._log_task.cancel()
            try:
                await self._log_task
            except asyncio.CancelledError:
                pass
            self._log_task = None

        if self._process is None or self._process.returncode is not None:
            self._process = None
            return

        logger.info("Stopping llama-server subprocess (pid=%s)", self._process.pid)
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=grace_period_seconds)
            logger.info("llama-server subprocess stopped cleanly")
        except asyncio.TimeoutError:
            logger.warning("llama-server did not exit within %.1fs; sending SIGKILL", grace_period_seconds)
            self._process.kill()
            await self._process.wait()
        finally:
            self._process = None
