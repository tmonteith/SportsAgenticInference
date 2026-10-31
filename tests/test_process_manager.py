"""Unit tests for LlamaServerProcessManager command construction, lifecycle, and health probes."""

from __future__ import annotations

from pathlib import Path
import sys
import httpx
import pytest
import respx

from app.model_manager import ModelManager
from app.process_manager import LlamaProcessError, LlamaServerProcessManager


def test_build_command_includes_required_flags(dummy_model_dir: Path) -> None:
    """Verify build_command includes --model, --alias, --host, --port, --ctx-size, --threads, and --jinja."""
    model_mgr = ModelManager(cache_dir=dummy_model_dir)
    proc_mgr = LlamaServerProcessManager(
        model_manager=model_mgr,
        llama_server_bin="/usr/local/bin/llama-server",
        host="127.0.0.1",
        port=8082,
        ctx_size=4096,
        threads=4,
        n_gpu_layers=0,
    )
    cmd = proc_mgr.build_command(model_mgr.model_path)
    assert cmd == [
        "/usr/local/bin/llama-server",
        "--model",
        str(model_mgr.model_path),
        "--alias",
        "microsoft/Phi-4-mini-instruct",
        "--host",
        "127.0.0.1",
        "--port",
        "8082",
        "--ctx-size",
        "4096",
        "--threads",
        "4",
        "--n-gpu-layers",
        "0",
        "--jinja",
    ]


@respx.mock
async def test_external_llama_url_skips_subprocess_and_checks_health(
    dummy_model_dir: Path,
) -> None:
    """When external_llama_url is configured, start() does not spawn a process and health probes external URL."""
    respx.get("http://external-llama:8082/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    model_mgr = ModelManager(cache_dir=dummy_model_dir)
    proc_mgr = LlamaServerProcessManager(
        model_manager=model_mgr,
        external_llama_url="http://external-llama:8082",
    )
    assert proc_mgr.is_running is True
    await proc_mgr.start()
    ok, detail = await proc_mgr.check_upstream_health()
    assert ok is True
    assert "healthy" in detail.lower()
    await proc_mgr.stop()


async def test_start_raises_when_binary_missing(dummy_model_dir: Path) -> None:
    """When the llama-server binary does not exist, LlamaProcessError is raised."""
    model_mgr = ModelManager(cache_dir=dummy_model_dir)
    proc_mgr = LlamaServerProcessManager(
        model_manager=model_mgr,
        llama_server_bin="/nonexistent/bin/llama-server-missing",
    )
    with pytest.raises(LlamaProcessError, match="not found"):
        await proc_mgr.start(wait_until_healthy=False)


@respx.mock
async def test_subprocess_start_health_wait_and_stop(
    dummy_model_dir: Path,
    tmp_path: Path,
) -> None:
    """Spawn a dummy Python script acting as llama-server, wait for health, and stop cleanly."""
    fake_script = tmp_path / "fake_llama_server.py"
    fake_script.write_text(
        "import time, sys\n"
        "print('fake llama-server starting', flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    fake_script.chmod(0o755)

    respx.get("http://127.0.0.1:18082/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )

    model_mgr = ModelManager(cache_dir=dummy_model_dir)
    proc_mgr = LlamaServerProcessManager(
        model_manager=model_mgr,
        llama_server_bin=sys.executable,
        host="127.0.0.1",
        port=18082,
        startup_timeout_seconds=5.0,
    )
    # Override build_command to run our lightweight script
    proc_mgr.build_command = lambda model_path: [sys.executable, str(fake_script)]  # type: ignore[method-assign]

    await proc_mgr.start(wait_until_healthy=True)
    assert proc_mgr.is_running is True

    ok, detail = await proc_mgr.check_upstream_health()
    assert ok is True
    assert "healthy" in detail.lower()

    await proc_mgr.stop(grace_period_seconds=2.0)
    assert proc_mgr.is_running is False


async def test_subprocess_premature_exit_raises(
    dummy_model_dir: Path,
    tmp_path: Path,
) -> None:
    """If the subprocess exits immediately with a non-zero code, wait_until_ready raises LlamaProcessError."""
    crash_script = tmp_path / "crash_llama.py"
    crash_script.write_text("import sys\nsys.exit(2)\n", encoding="utf-8")

    model_mgr = ModelManager(cache_dir=dummy_model_dir)
    proc_mgr = LlamaServerProcessManager(
        model_manager=model_mgr,
        llama_server_bin=sys.executable,
        host="127.0.0.1",
        port=18083,
        startup_timeout_seconds=2.0,
    )
    proc_mgr.build_command = lambda model_path: [sys.executable, str(crash_script)]  # type: ignore[method-assign]

    with pytest.raises(LlamaProcessError, match="exited prematurely"):
        await proc_mgr.start(wait_until_healthy=True)
