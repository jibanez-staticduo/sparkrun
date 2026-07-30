"""Shared pytest fixtures for sparkrun tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sparkrun.core.bootstrap import init_sparkrun


@pytest.fixture(autouse=True)
def isolate_stateful(tmp_path: Path, monkeypatch):
    """Redirect SAF stateful root to temp dir for test isolation.

    Prevents tests from writing to the real ~/.config/sparkrun/.
    Also resets the bootstrap singleton between tests.
    """
    monkeypatch.setenv("STATEFUL_ROOT", str(tmp_path / "stateful"))
    monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "1")
    # Hard-disable external-plugin auto-loading during tests. The feature flag
    # alone is not enough: pytest reads the developer's REAL ~/.config/sparkrun
    # (the SAF stateful root isn't "ready"), so a developer who enabled
    # core.external_plugins would otherwise load their real plugins mid-suite.
    # Loader tests pass explicit paths (which bypass this) or delenv it.
    monkeypatch.setenv("SPARKRUN_NO_EXTERNAL_PLUGINS", "1")
    # The experimental local/k8s executors gate themselves off by default on
    # the stable channel (via is_multi_extension). Most of the suite exercises
    # their behavior directly and predates the feature-flag gating, so enable
    # them here to preserve that contract. tests/test_features.py exercises the
    # gating itself in clean subprocesses that strip these env overrides.
    monkeypatch.setenv("SPARKRUN_FEATURE_EXECUTOR_LOCAL", "1")
    monkeypatch.setenv("SPARKRUN_FEATURE_EXECUTOR_K8S", "1")
    # The `setup k8s` command group is likewise gated off by default; enable it
    # so the CLI tests that exercise it keep passing (the gate itself is tested
    # explicitly in test_k8s_setup with the env override cleared).
    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_K8S", "1")
    # Point the user config dir at the sandbox too. STATEFUL_ROOT alone does
    # not cover it: DEFAULT_CONFIG_DIR is computed from Path.home() at import
    # time, so without this a test silently reads the developer's real
    # ~/.config/sparkrun -- its default cluster, its saved clusters, its
    # registries. Such a test passes on the machine that wrote that config and
    # fails everywhere else, which is the most expensive way to find out.
    import sparkrun.core.config as _config_module

    monkeypatch.setattr(_config_module, "DEFAULT_CONFIG_DIR", tmp_path / "config", raising=False)

    import sparkrun.core.bootstrap

    sparkrun.core.bootstrap._variables = None
    yield
    sparkrun.core.bootstrap._variables = None


@pytest.fixture
def cluster_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for cluster definitions."""
    d = tmp_path / "clusters"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def hosts_file(tmp_path: Path) -> Path:
    """Create a temporary hosts file with sample hosts."""
    f = tmp_path / "hosts.txt"
    f.write_text("10.0.0.1\n10.0.0.2\n10.0.0.3\n")
    return f


@pytest.fixture
def tmp_recipe_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample YAML recipe files.

    Creates both v1 (eugr-style) and v2 format recipes for testing.

    Returns:
        Path to temporary directory containing recipe files.
    """
    recipe_dir = tmp_path / "recipes"
    recipe_dir.mkdir()

    # v2 vllm recipe
    v2_vllm = {
        "sparkrun_version": "2",
        "name": "Test vLLM Recipe",
        "description": "A test recipe for vLLM",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "mode": "auto",
        "container": "scitrera/dgx-spark-vllm:latest",
        "defaults": {
            "port": 8000,
            "host": "0.0.0.0",
            "tensor_parallel": 1,
            "gpu_memory_utilization": 0.9,
        },
        "env": {
            "VLLM_BATCH_INVARIANT": "1",
        },
        "command": "vllm serve {model} --port {port} --host {host}",
    }
    with open(recipe_dir / "test-vllm.yaml", "w") as f:
        yaml.dump(v2_vllm, f)

    # v2 sglang recipe
    v2_sglang = {
        "sparkrun_version": "2",
        "name": "Test SGLang Recipe",
        "description": "A test recipe for SGLang",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
        "mode": "cluster",
        "min_nodes": 2,
        "container": "scitrera/dgx-spark-sglang:latest",
        "defaults": {
            "port": 30000,
            "host": "0.0.0.0",
            "tensor_parallel": 2,
        },
    }
    with open(recipe_dir / "test-sglang.yaml", "w") as f:
        yaml.dump(v2_sglang, f)

    # v1 recipe with mods (should auto-set eugr builder).
    # v1 default values are strings (template-substituted via {port}); the
    # migration's `.replace("{{", "{")` step assumes str values.
    v1_eugr = {
        "recipe_version": "1",
        "name": "Test EUGR Recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "build_args": ["ARG1=value1"],
        "mods": ["mod1.patch"],
        "defaults": {
            "port": "8000",
        },
    }
    with open(recipe_dir / "test-eugr.yaml", "w") as f:
        yaml.dump(v1_eugr, f)

    # v1 recipe without mods (should auto-set eugr builder, resolve to vllm-distributed)
    v1_plain = {
        "recipe_version": "1",
        "name": "Test Plain v1 Recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "defaults": {
            "port": "8000",
        },
    }
    with open(recipe_dir / "test-plain-v1.yaml", "w") as f:
        yaml.dump(v1_plain, f)

    return recipe_dir


@pytest.fixture
def v(tmp_path: Path) -> Any:
    """Initialize sparkrun and return the Variables instance.

    Uses WARNING log level to reduce test output noise.
    Resets the global _variables singleton for test isolation.

    Returns:
        Initialized Variables instance.
    """
    # Reset global singleton to ensure test isolation
    import sparkrun.core.bootstrap

    sparkrun.core.bootstrap._variables = None

    return init_sparkrun(log_level="WARNING")


@pytest.fixture
def sample_v2_recipe_data() -> dict[str, Any]:
    """Return a dict for a v2 vllm recipe.

    Returns:
        Dictionary containing a valid v2 recipe.
    """
    return {
        "sparkrun_version": "2",
        "name": "Sample vLLM Recipe",
        "description": "A sample vLLM recipe for testing",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "mode": "auto",
        "min_nodes": 1,
        "max_nodes": 4,
        "container": "scitrera/dgx-spark-vllm:0.16.0",
        "defaults": {
            "port": 8000,
            "host": "0.0.0.0",
            "tensor_parallel": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "VLLM_BATCH_INVARIANT": "1",
        },
        "command": "vllm serve {model} --port {port} -tp {tensor_parallel}",
    }


@pytest.fixture
def sample_v1_recipe_data() -> dict[str, Any]:
    """Return a dict for a v1 eugr-style recipe with mods and build_args.

    Returns:
        Dictionary containing a valid v1 recipe that should auto-set eugr builder.
    """
    return {
        "recipe_version": "1",
        "name": "Sample EUGR Recipe",
        "description": "A v1 recipe with custom build",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "cluster_only": True,
        "build_args": [
            "VLLM_VERSION=0.5.0",
            "CUSTOM_FLAG=true",
        ],
        "mods": [
            "custom_attention.patch",
            "performance_tweaks.patch",
        ],
        "defaults": {
            # v1 defaults are template-substituted as strings ({port} → "8000")
            "port": "8000",
            "tensor_parallel": "2",
        },
        "env": {
            "NCCL_DEBUG": "INFO",
        },
        "command": "python -m vllm.entrypoints.openai.api_server --model {model}",
    }


@pytest.fixture
def sample_sglang_recipe_data() -> dict[str, Any]:
    """Return a dict for a v2 sglang recipe.

    Returns:
        Dictionary containing a valid v2 SGLang recipe.
    """
    return {
        "sparkrun_version": "2",
        "name": "Sample SGLang Recipe",
        "description": "A sample SGLang recipe for testing",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
        "mode": "cluster",
        "min_nodes": 2,
        "max_nodes": 8,
        "container": "scitrera/dgx-spark-sglang:0.5.8",
        "defaults": {
            "port": 30000,
            "host": "0.0.0.0",
            "tensor_parallel": 2,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 32768,
        },
        "env": {
            "NCCL_CUMEM_ENABLE": "0",
        },
        "command": "python3 -m sglang.launch_server --model-path {model} --port {port}",
    }


@pytest.fixture
def log_sources_spy(monkeypatch):
    """Capture what ``RuntimePlugin.follow_logs`` hands to the log reader.

    ``follow_logs`` is a printing shim over
    :meth:`~sparkrun.runtimes.base.RuntimePlugin.log_sources` +
    :func:`~sparkrun.orchestration.logs.print_log_sources`, so intercepting
    the latter records the resolved sources — host, container name, and
    file-vs-stdout mode.  That is the observable behaviour the older
    ``stream_container_file_logs`` / ``stream_remote_logs`` mocks stood in
    for, asserted directly instead of via which helper got called (and
    without spawning a reader subprocess).

    Yields a list of captured calls, each with ``.sources`` plus the
    ``follow`` / ``tail`` / ``dry_run`` / ``ssh_kwargs`` keywords.
    """
    from types import SimpleNamespace

    calls: list[Any] = []

    def _capture(executor, sources, **kwargs):
        calls.append(SimpleNamespace(executor=executor, sources=list(sources), **kwargs))

    monkeypatch.setattr("sparkrun.orchestration.logs.print_log_sources", _capture)
    return calls
