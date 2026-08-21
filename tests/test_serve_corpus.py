from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tools.bench.run_serve_corpus import (
    CampaignError,
    require_server_log_identity,
    summary_row,
)


def test_request_log_v9_identity_is_accepted() -> None:
    current = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 9,
        "event": "server_start",
    }
    require_server_log_identity(current, "server_start")

    stale = dict(current, schema_version=8)
    with pytest.raises(CampaignError):
        require_server_log_identity(stale, "server_start")


def test_summary_retains_one_canonical_weights_id() -> None:
    records = [{"weights_id": "nvfp4", "metrics": {}}]
    row = summary_row(
        "context_profile",
        "qwen3_6_27b",
        "fixture",
        "fixture",
        "mtp0",
        "greedy",
        records,
    )
    assert row["weights_id"] == "nvfp4"

    with pytest.raises(CampaignError):
        summary_row(
            "context_profile",
            "qwen3_6_27b",
            "fixture",
            "fixture",
            "mtp0",
            "greedy",
            [*records, {"weights_id": "groupwise-int", "metrics": {}}],
        )


def _required_readiness_fixture(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        pytest.skip(f"{name} is required for the real server readiness test")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"{name} is not a regular file: {path}")
    return path


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _get_json(url: str, *, api_key: str | None = None) -> tuple[int, dict[str, object]]:
    headers = {} if api_key is None else {"Authorization": f"Bearer {api_key}"}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=1.0) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_ready_endpoint_is_live_only_after_warmup_and_bypasses_api_auth(
    tmp_path: Path,
) -> None:
    binary = _required_readiness_fixture("NINFER_SERVE_BINARY")
    artifact = _required_readiness_fixture("NINFER_TEST_ARTIFACT")
    projection = _required_readiness_fixture("NINFER_TEST_PROJECTION")
    api_key = "readiness-contract-test"
    port = _free_loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "ninfer-serve.log"
    command = [
        str(binary),
        str(artifact),
        "--refusal-projection",
        str(projection),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-id",
        "readiness-contract-model",
        "--max-context",
        "2048",
        "--kv-capacity",
        "2048",
        "--max-concurrency",
        "1",
        "--kv-dtype",
        "int8",
        "--prefill-chunk",
        "1024",
        "--default-max-tokens",
        "16",
        "--api-key",
        api_key,
    ]

    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    saw_not_serving = False
    try:
        deadline = time.monotonic() + float(
            os.environ.get("NINFER_READY_TIMEOUT_SECONDS", "300")
        )
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(
                    f"ninfer-serve exited {process.returncode} before readiness:\n"
                    f"{log_path.read_text(encoding='utf-8', errors='replace')}"
                )
            try:
                health_status, health_body = _get_json(f"{base_url}/health")
            except (OSError, urllib.error.URLError):
                saw_not_serving = True
                time.sleep(0.25)
                continue
            if health_status == 200:
                break
            time.sleep(0.25)
        else:
            pytest.fail(
                "ninfer-serve did not become available before the readiness deadline:\n"
                + log_path.read_text(encoding="utf-8", errors="replace")
            )

        assert saw_not_serving, "the readiness check unexpectedly succeeded before server warmup"
        assert health_body == {"status": "ok"}
        ready_status, ready_body = _get_json(f"{base_url}/health/ready")
        assert ready_status == 200
        assert ready_body == {"status": "ready"}

        unauthenticated_status, _ = _get_json(f"{base_url}/v1/models")
        authenticated_status, models = _get_json(f"{base_url}/v1/models", api_key=api_key)
        assert unauthenticated_status == 401
        assert authenticated_status == 200
        assert models["data"][0]["id"] == "readiness-contract-model"
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
