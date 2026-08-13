"""Tests for sparkrun.tuning module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.tuning.sglang import (
    build_tuning_command,
    get_sglang_tuning_dir,
    get_sglang_tuning_env,
    get_sglang_tuning_volumes,
    SglangTuner,
    SGLANG_CLONE_DIR,
    TUNE_CONTAINER_NAME,
    TUNING_CONTAINER_OUTPUT_PATH,
    TUNING_CONTAINER_PATH,
    TUNING_ENV_PATH,
    DEFAULT_TP_SIZES,
    _format_duration,
)
from sparkrun.tuning.vllm import (
    build_vllm_tune_export,
    build_vllm_tune_invocation,
    get_vllm_tuning_dir,
    get_vllm_tuning_env,
    get_vllm_tuning_volumes,
    VllmTuner,
    VLLM_TUNING_CACHE_SUBDIR,
    VLLM_TUNING_CONTAINER_PATH,
    DEFAULT_TP_SIZES as VLLM_DEFAULT_TP_SIZES,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestGetSglangTuningDir:
    def test_returns_path_under_cache(self):
        d = get_sglang_tuning_dir()
        assert isinstance(d, Path)
        assert str(d).endswith("sparkrun/tuning/sglang")

    def test_is_under_the_cache_root(self):
        """The tuning dir hangs off DEFAULT_CACHE_DIR, wherever that points.

        Was ``startswith(Path.home())`` — which only held because the cache dir
        leaked out to the real ``~/.cache/sparkrun`` during tests. Asserting
        against the configured root tests the actual contract.
        """
        import sparkrun.tuning._common as tuning_common

        d = get_sglang_tuning_dir()
        assert str(d).startswith(str(tuning_common.DEFAULT_CACHE_DIR))


class TestGetSglangTuningVolumes:
    def test_returns_none_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning._common.DEFAULT_CACHE_DIR",
            tmp_path / "nonexistent_cache",
        )
        assert get_sglang_tuning_volumes() is None

    def test_returns_none_when_dir_empty(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "sparkrun" / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_dir",
            lambda: tuning_dir,
        )
        assert get_sglang_tuning_volumes() is None

    def test_returns_mapping_when_json_exists(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_dir",
            lambda: tuning_dir,
        )
        result = get_sglang_tuning_volumes()
        assert result is not None
        assert result[str(tuning_dir)] == TUNING_CONTAINER_PATH

    def test_returns_mapping_for_nested_json(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "sglang"
        nested = tuning_dir / "configs" / "triton_3_2_0"
        nested.mkdir(parents=True)
        (nested / "E=128_N=256.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_dir",
            lambda: tuning_dir,
        )
        result = get_sglang_tuning_volumes()
        assert result is not None


class TestGetSglangTuningEnv:
    def test_returns_none_when_no_configs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_volumes",
            lambda: None,
        )
        assert get_sglang_tuning_env() is None

    def test_returns_env_when_configs_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_volumes",
            lambda: {"/some/path": TUNING_CONTAINER_PATH},
        )
        result = get_sglang_tuning_env()
        assert result is not None
        assert "SGLANG_MOE_CONFIG_DIR" in result
        assert result["SGLANG_MOE_CONFIG_DIR"] == TUNING_ENV_PATH


# ---------------------------------------------------------------------------
# build_tuning_command
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_under_60_seconds(self):
        assert _format_duration(5.3) == "5.3s"
        assert _format_duration(0.0) == "0.0s"
        assert _format_duration(59.9) == "59.9s"

    def test_minutes(self):
        assert _format_duration(60) == "1m 0s"
        assert _format_duration(90) == "1m 30s"
        assert _format_duration(754) == "12m 34s"

    def test_hours(self):
        assert _format_duration(3600) == "1h 0m 0s"
        assert _format_duration(3661) == "1h 1m 1s"
        assert _format_duration(7384) == "2h 3m 4s"


class TestBuildTuningCommand:
    def test_contains_model(self):
        cmd = build_tuning_command("Qwen/Qwen3-MoE", 4)
        assert "Qwen/Qwen3-MoE" in cmd

    def test_contains_tp_size(self):
        cmd = build_tuning_command("test-model", 8)
        assert "--tp-size 8" in cmd

    def test_contains_tune_flag(self):
        cmd = build_tuning_command("test-model", 1)
        assert "--tune" in cmd

    def test_contains_config_dir(self):
        cmd = build_tuning_command("test-model", 2)
        assert "SGLANG_MOE_CONFIG_DIR=" in cmd
        assert TUNING_CONTAINER_OUTPUT_PATH in cmd

    def test_config_dir_is_base_path(self):
        """SGLANG_MOE_CONFIG_DIR should be the base output path — SGLang
        internally appends configs/triton_<ver>/ via get_config_file_name()."""
        cmd = build_tuning_command("test-model", 2, triton_version="3.6.0")
        assert "SGLANG_MOE_CONFIG_DIR=%s " % TUNING_CONTAINER_OUTPUT_PATH in cmd
        # Should NOT double-nest the triton version in the env var
        assert "SGLANG_MOE_CONFIG_DIR=%s/triton_" % TUNING_CONTAINER_OUTPUT_PATH not in cmd

    def test_versioned_mkdir(self):
        """When triton_version is provided, pre-create the versioned subdir."""
        cmd = build_tuning_command("test-model", 2, triton_version="3.6.0")
        assert "mkdir -p %s/triton_3_6_0" % TUNING_CONTAINER_OUTPUT_PATH in cmd

    def test_no_version_uses_base_dir(self):
        """Without triton_version, output_subdir is the base path itself."""
        cmd = build_tuning_command("test-model", 2)
        assert "mkdir -p %s " % TUNING_CONTAINER_OUTPUT_PATH in cmd

    def test_unknown_version_uses_base_dir(self):
        cmd = build_tuning_command("test-model", 2, triton_version="unknown")
        assert "mkdir -p %s " % TUNING_CONTAINER_OUTPUT_PATH in cmd
        assert "triton_unknown" not in cmd

    def test_cwd_is_output_dir(self):
        """CWD must be the output subdir so save_configs() writes there."""
        cmd = build_tuning_command("test-model", 1)
        assert "cd %s " % TUNING_CONTAINER_OUTPUT_PATH in cmd

    def test_script_uses_full_clone_path(self):
        """Script must be invoked via full path since CWD is the output dir."""
        cmd = build_tuning_command("test-model", 1)
        assert "%s/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py" % SGLANG_CLONE_DIR in cmd

    def test_various_tp_sizes(self):
        for tp in DEFAULT_TP_SIZES:
            cmd = build_tuning_command("model", tp)
            assert "--tp-size %d" % tp in cmd

    def test_model_with_single_quote_is_shell_safe(self):
        """Model names with single quotes must not break shell quoting."""
        import shlex

        cmd = build_tuning_command("org/model'name", 1)
        assert shlex.quote("org/model'name") in cmd

    def test_model_with_spaces_is_shell_safe(self):
        """Model names with spaces must be shell-quoted."""
        import shlex

        cmd = build_tuning_command("org/model name", 1)
        assert shlex.quote("org/model name") in cmd


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


class TestTuneSglangCLI:
    def test_help(self):
        from sparkrun.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["tune", "sglang", "--help"])
        assert result.exit_code == 0
        assert "Tune SGLang fused MoE Triton kernels" in result.output
        assert "--tp" in result.output
        assert "--skip-clone" in result.output

    def test_tune_group_help(self):
        from sparkrun.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["tune", "--help"])
        assert result.exit_code == 0
        assert "sglang" in result.output

    def test_rejects_non_sglang_recipe(self, tmp_path, monkeypatch):
        """tune sglang should reject recipes with non-sglang runtime."""
        from sparkrun.cli import main

        # Create a vllm recipe
        recipe_file = tmp_path / "test-vllm.yaml"
        recipe_file.write_text(
            yaml.dump(
                {
                    "sparkrun_version": "2",
                    "name": "Test vLLM",
                    "model": "test/model",
                    "runtime": "vllm",
                    "container": "test:latest",
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "tune",
                "sglang",
                str(recipe_file),
                "-H",
                "10.0.0.1",
                "-n",
            ],
        )
        assert result.exit_code != 0
        assert "requires an SGLang recipe" in result.output


# ---------------------------------------------------------------------------
# SglangTuner dry-run
# ---------------------------------------------------------------------------


class TestSglangTunerDryRun:
    @pytest.fixture
    def sglang_recipe_file(self, tmp_path):
        recipe_file = tmp_path / "test-sglang.yaml"
        recipe_file.write_text(
            yaml.dump(
                {
                    "sparkrun_version": "2",
                    "name": "Test SGLang",
                    "model": "Qwen/Qwen3-MoE",
                    "runtime": "sglang",
                    "container": "scitrera/dgx-spark-sglang:latest",
                    "defaults": {"tensor_parallel": 2},
                }
            )
        )
        return recipe_file

    def test_tuner_dry_run_returns_zero(self):
        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="Qwen/Qwen3-MoE",
            dry_run=True,
        )
        rc = tuner.run_tuning(tp_sizes=(1,))
        assert rc == 0

    def test_tuner_dry_run_all_default_tp_sizes(self):
        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="Qwen/Qwen3-MoE",
            dry_run=True,
        )
        rc = tuner.run_tuning()
        assert rc == 0

    def test_tuner_custom_output_dir(self, tmp_path):
        custom_dir = str(tmp_path / "custom_tuning")
        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="test-model",
            output_dir=custom_dir,
            dry_run=True,
        )
        assert tuner.output_dir == custom_dir

    def test_tuner_skip_clone(self):
        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="test-model",
            skip_clone=True,
            dry_run=True,
        )
        assert tuner.skip_clone is True
        rc = tuner.run_tuning(tp_sizes=(1,))
        assert rc == 0

    def test_default_tp_sizes_constant(self):
        assert DEFAULT_TP_SIZES == (1, 2, 4, 8)

    def test_tuner_dry_run_parallel(self):
        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="Qwen/Qwen3-MoE",
            dry_run=True,
        )
        rc = tuner.run_tuning(tp_sizes=(1, 2, 4), parallel=2)
        assert rc == 0

    def test_container_name_constant(self):
        assert TUNE_CONTAINER_NAME == "sparkrun_tune"


# ---------------------------------------------------------------------------
# Auto-mount integration
# ---------------------------------------------------------------------------


class TestSglangRuntimeAutoMount:
    def test_get_extra_volumes_empty_when_no_configs(self, v, monkeypatch):
        """SglangRuntime.get_extra_volumes returns {} when no tuning configs."""
        from sparkrun.core.bootstrap import get_runtime

        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_volumes",
            lambda: None,
        )
        runtime = get_runtime("sglang", v)
        assert runtime.get_extra_volumes() == {}

    def test_get_extra_env_empty_when_no_configs(self, v, monkeypatch):
        """SglangRuntime.get_extra_env returns {} when no tuning configs."""
        from sparkrun.core.bootstrap import get_runtime

        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_env",
            lambda: None,
        )
        runtime = get_runtime("sglang", v)
        assert runtime.get_extra_env() == {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}

    def test_get_extra_volumes_returns_mapping(self, v, monkeypatch):
        """SglangRuntime.get_extra_volumes returns mapping when configs exist."""
        from sparkrun.core.bootstrap import get_runtime

        expected = {"/cache/tuning/sglang": TUNING_CONTAINER_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_volumes",
            lambda: expected,
        )
        runtime = get_runtime("sglang", v)
        assert runtime.get_extra_volumes() == expected

    def test_get_extra_env_returns_env(self, v, monkeypatch):
        """SglangRuntime.get_extra_env returns env when configs exist."""
        from sparkrun.core.bootstrap import get_runtime

        expected = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub", "SGLANG_MOE_CONFIG_DIR": TUNING_ENV_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_env",
            lambda: {"SGLANG_MOE_CONFIG_DIR": TUNING_ENV_PATH},
        )
        runtime = get_runtime("sglang", v)
        assert runtime.get_extra_env() == expected


# ---------------------------------------------------------------------------
# Base runtime hooks
# ---------------------------------------------------------------------------


class TestBaseRuntimeHooks:
    def test_default_get_extra_volumes(self, v):
        """Base RuntimePlugin.get_extra_volumes returns empty dict."""
        from sparkrun.core.bootstrap import get_runtime

        # llama-cpp doesn't override get_extra_volumes
        runtime = get_runtime("llama-cpp", v)
        assert runtime.get_extra_volumes() == {}

    def test_default_get_extra_env(self, v):
        """Base RuntimePlugin.get_extra_env returns HF_HOME."""
        from sparkrun.core.bootstrap import get_runtime

        runtime = get_runtime("llama-cpp", v)
        assert runtime.get_extra_env() == {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}


# ===========================================================================
# vLLM tuning tests
# ===========================================================================

# ---------------------------------------------------------------------------
# vLLM path helpers
# ---------------------------------------------------------------------------


class TestGetVllmTuningDir:
    def test_returns_path_under_cache(self):
        d = get_vllm_tuning_dir()
        assert isinstance(d, Path)
        assert str(d).endswith("sparkrun/tuning/vllm")

    def test_is_under_the_cache_root(self):
        """See :meth:`TestGetSglangTuningDir.test_is_under_the_cache_root`."""
        import sparkrun.tuning._common as tuning_common

        d = get_vllm_tuning_dir()
        assert str(d).startswith(str(tuning_common.DEFAULT_CACHE_DIR))


class TestGetVllmTuningVolumes:
    def test_returns_none_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning._common.DEFAULT_CACHE_DIR",
            tmp_path / "nonexistent_cache",
        )
        assert get_vllm_tuning_volumes() is None

    def test_returns_none_when_dir_empty(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "sparkrun" / "tuning" / "vllm"
        tuning_dir.mkdir(parents=True)
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_dir",
            lambda: tuning_dir,
        )
        assert get_vllm_tuning_volumes() is None

    def test_returns_mapping_when_json_exists(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "vllm"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_dir",
            lambda: tuning_dir,
        )
        result = get_vllm_tuning_volumes()
        assert result is not None
        assert result[str(tuning_dir)] == VLLM_TUNING_CONTAINER_PATH

    def test_returns_mapping_for_nested_json(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "vllm"
        nested = tuning_dir / "configs" / "triton_3_2_0"
        nested.mkdir(parents=True)
        (nested / "E=128_N=256.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_dir",
            lambda: tuning_dir,
        )
        result = get_vllm_tuning_volumes()
        assert result is not None


class TestGetVllmTuningEnv:
    def test_returns_none_when_no_configs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: None,
        )
        assert get_vllm_tuning_env() is None

    def test_returns_env_when_configs_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: {"/some/path": VLLM_TUNING_CONTAINER_PATH},
        )
        result = get_vllm_tuning_env()
        assert result is not None
        assert "VLLM_TUNED_CONFIG_FOLDER" in result
        assert result["VLLM_TUNED_CONFIG_FOLDER"] == VLLM_TUNING_CONTAINER_PATH


# ---------------------------------------------------------------------------
# vllm-tune invocation builders
# ---------------------------------------------------------------------------


class TestBuildVllmTuneInvocation:
    def _invoke(self, **overrides):
        kwargs = {
            "install_path": "/home/user/.cache/sparkrun/vllm-tune/main/vllm-tune.sh",
            "model": "Qwen/Qwen3-MoE",
            "tp_size": 2,
            "mode": "all",
            "image": "vllm/vllm-openai:latest",
            "sparkrun_dir": "/home/user/.cache/sparkrun/tuning/vllm",
        }
        kwargs.update(overrides)
        return build_vllm_tune_invocation(**kwargs)

    def test_calls_vllm_tune_via_bash(self):
        cmd = self._invoke()
        assert "bash " in cmd
        assert "vllm-tune.sh" in cmd

    def test_contains_model(self):
        cmd = self._invoke(model="Qwen/Qwen3-MoE")
        assert "Qwen/Qwen3-MoE" in cmd

    def test_contains_tp_size(self):
        cmd = self._invoke(tp_size=8)
        assert "--tp 8" in cmd

    def test_contains_mode(self):
        cmd = self._invoke(mode="fp8")
        assert "--mode fp8" in cmd

    def test_runs_standalone_foreground(self):
        cmd = self._invoke()
        assert "--standalone" in cmd
        assert "--foreground" in cmd

    def test_passes_image_through(self):
        cmd = self._invoke(image="my-registry.example.com/vllm:custom")
        assert "my-registry.example.com/vllm:custom" in cmd

    def test_exports_sparkrun_tuning_dir_env(self):
        cmd = self._invoke(sparkrun_dir="/opt/cache/sparkrun/tuning/vllm")
        assert "SPARKRUN_TUNING_DIR=" in cmd
        assert "/opt/cache/sparkrun/tuning/vllm" in cmd

    def test_various_tp_sizes(self):
        for tp in VLLM_DEFAULT_TP_SIZES:
            cmd = self._invoke(tp_size=tp)
            assert "--tp %d" % tp in cmd

    def test_model_with_single_quote_is_shell_safe(self):
        import shlex

        cmd = self._invoke(model="org/model'name")
        assert shlex.quote("org/model'name") in cmd

    def test_image_with_special_chars_is_shell_safe(self):
        import shlex

        cmd = self._invoke(image="registry/img:tag$with$dollars")
        assert shlex.quote("registry/img:tag$with$dollars") in cmd


class TestBuildVllmTuneExport:
    def test_uses_export_sparkrun_flag(self):
        cmd = build_vllm_tune_export(
            install_path="/p/vllm-tune.sh",
            model="m",
            tp_size=2,
            mode="all",
            sparkrun_dir="/cache/tuning/vllm",
        )
        assert "--export-sparkrun" in cmd
        assert "--sparkrun-dir " in cmd

    def test_carries_tp_and_mode(self):
        cmd = build_vllm_tune_export(
            install_path="/p/vllm-tune.sh",
            model="m",
            tp_size=4,
            mode="moe",
            sparkrun_dir="/cache/tuning/vllm",
        )
        assert "--tp 4" in cmd
        assert "--mode moe" in cmd


# ---------------------------------------------------------------------------
# vllm-tune install script
# ---------------------------------------------------------------------------


class TestVllmTuneInstallScript:
    def test_install_script_is_readable(self):
        from sparkrun.scripts import read_script

        body = read_script("vllm_tune_install.sh")
        assert body.startswith("#!/bin/bash")

    def test_install_script_uses_required_env_vars(self):
        from sparkrun.scripts import read_script

        body = read_script("vllm_tune_install.sh")
        for var in ("VLLM_TUNE_REPO", "VLLM_TUNE_REF", "VLLM_TUNE_DEST"):
            assert var in body

    def test_install_script_does_git_fetch_or_clone(self):
        from sparkrun.scripts import read_script

        body = read_script("vllm_tune_install.sh")
        assert "git clone" in body
        assert "git fetch" in body

    def test_install_prelude_keeps_home_shell_expandable(self, monkeypatch):
        """Regression: VLLM_TUNE_DEST must let the remote shell expand `$HOME`.

        If the whole dest is single-quoted, the clone lands in a directory
        literally named ``$HOME`` while build_vllm_tune_invocation (which
        interpolates the path raw) looks at the expanded path — yielding an
        ``rc=127 No such file or directory`` mismatch at tune time.
        """
        import sparkrun.orchestration.primitives as primitives
        from sparkrun.tuning.vllm import VllmTuner

        captured = {}

        class _Result:
            success = True
            stdout = "/home/u/.cache/sparkrun/vllm-tune/main/vllm-tune.sh"
            stderr = ""

        def _fake_run_script_on_host(host, body, **kwargs):
            captured["body"] = body
            return _Result()

        monkeypatch.setattr(primitives, "run_script_on_host", _fake_run_script_on_host)

        tuner = VllmTuner(host="10.0.0.1", image="img:latest", model="m", vllm_tune_ref="main")
        tuner._install_vllm_tune()

        body = captured["body"]
        # `$HOME` is present and NOT trapped inside single quotes (which would
        # suppress expansion on the remote shell).
        assert "VLLM_TUNE_DEST=$HOME/.cache/sparkrun/vllm-tune/main" in body
        assert "'$HOME" not in body


# ---------------------------------------------------------------------------
# Config pin resolution
# ---------------------------------------------------------------------------


class TestVllmTunePinResolution:
    def test_default_pin_falls_back_to_built_ins(self):
        from sparkrun.core.config import DEFAULT_VLLM_TUNE_REF, DEFAULT_VLLM_TUNE_REPO
        from sparkrun.tuning.vllm import _resolve_vllm_tune_pin

        repo, ref = _resolve_vllm_tune_pin(None, None)
        assert repo == DEFAULT_VLLM_TUNE_REPO
        assert ref == DEFAULT_VLLM_TUNE_REF

    def test_cli_override_wins(self):
        from sparkrun.tuning.vllm import _resolve_vllm_tune_pin

        repo, ref = _resolve_vllm_tune_pin(None, "v1.2.3")
        assert ref == "v1.2.3"

    def test_config_pin_used_when_no_cli_override(self, tmp_path):
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.tuning.vllm import _resolve_vllm_tune_pin

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "tuning": {
                        "vllm_tune_repo": "https://example.com/fork.git",
                        "vllm_tune_ref": "custom-branch",
                    }
                }
            )
        )
        config = SparkrunConfig(config_path=cfg_path)
        repo, ref = _resolve_vllm_tune_pin(config, None)
        assert repo == "https://example.com/fork.git"
        assert ref == "custom-branch"

    def test_rejects_unsafe_git_url(self, tmp_path):
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.tuning.vllm import _resolve_vllm_tune_pin

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"tuning": {"vllm_tune_repo": "-malicious-flag"}}))
        config = SparkrunConfig(config_path=cfg_path)
        with pytest.raises(ValueError):
            _resolve_vllm_tune_pin(config, None)


# ---------------------------------------------------------------------------
# vLLM CLI smoke tests
# ---------------------------------------------------------------------------


class TestTuneVllmCLI:
    def test_help(self):
        from sparkrun.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["tune", "vllm", "--help"])
        assert result.exit_code == 0
        assert "vllm-tune" in result.output
        assert "--tp" in result.output
        assert "--mode" in result.output
        assert "--vllm-tune-ref" in result.output
        # --skip-clone was retired with the vllm-tune integration.
        assert "--skip-clone" not in result.output

    def test_tune_group_lists_vllm(self):
        from sparkrun.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["tune", "--help"])
        assert result.exit_code == 0
        assert "vllm" in result.output

    def test_rejects_non_vllm_recipe(self, tmp_path, monkeypatch):
        """tune vllm should reject recipes with non-vllm runtime."""
        from sparkrun.cli import main

        # Create an sglang recipe
        recipe_file = tmp_path / "test-sglang.yaml"
        recipe_file.write_text(
            yaml.dump(
                {
                    "sparkrun_version": "2",
                    "name": "Test SGLang",
                    "model": "test/model",
                    "runtime": "sglang",
                    "container": "test:latest",
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "tune",
                "vllm",
                str(recipe_file),
                "-H",
                "10.0.0.1",
                "-n",
            ],
        )
        assert result.exit_code != 0
        assert "requires a vLLM recipe" in result.output

    def test_accepts_vllm_ray_recipe(self, tmp_path, monkeypatch):
        """tune vllm should accept vllm-ray runtime recipes in dry-run."""
        from sparkrun.cli import main

        recipe_file = tmp_path / "test-vllm-ray.yaml"
        recipe_file.write_text(
            yaml.dump(
                {
                    "sparkrun_version": "2",
                    "name": "Test vLLM Ray",
                    "model": "test/model",
                    "runtime": "vllm-ray",
                    "container": "test:latest",
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "tune",
                "vllm",
                str(recipe_file),
                "-H",
                "10.0.0.1",
                "-n",
            ],
        )
        # Should not fail with runtime validation error
        assert "requires a vLLM recipe" not in (result.output or "")


# ---------------------------------------------------------------------------
# VllmTuner dry-run
# ---------------------------------------------------------------------------


class TestVllmTunerDryRun:
    def test_tuner_dry_run_returns_zero(self):
        tuner = VllmTuner(
            host="10.0.0.1",
            image="test:latest",
            model="Qwen/Qwen3-MoE",
            dry_run=True,
        )
        rc = tuner.run_tuning(tp_sizes=(1,))
        assert rc == 0

    def test_tuner_dry_run_all_default_tp_sizes(self):
        tuner = VllmTuner(
            host="10.0.0.1",
            image="test:latest",
            model="Qwen/Qwen3-MoE",
            dry_run=True,
        )
        rc = tuner.run_tuning()
        assert rc == 0

    def test_tuner_custom_output_dir(self, tmp_path):
        custom_dir = str(tmp_path / "custom_tuning")
        tuner = VllmTuner(
            host="10.0.0.1",
            image="test:latest",
            model="test-model",
            output_dir=custom_dir,
            dry_run=True,
        )
        assert tuner.output_dir == custom_dir

    def test_default_tp_sizes_constant(self):
        assert VLLM_DEFAULT_TP_SIZES == (1, 2, 4, 8)

    def test_tuner_dry_run_parallel(self):
        tuner = VllmTuner(
            host="10.0.0.1",
            image="test:latest",
            model="Qwen/Qwen3-MoE",
            dry_run=True,
        )
        rc = tuner.run_tuning(tp_sizes=(1, 2, 4), parallel=2)
        assert rc == 0

    def test_default_mode_is_all(self):
        tuner = VllmTuner(host="10.0.0.1", image="test:latest", model="m", dry_run=True)
        assert tuner.mode == "all"

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            VllmTuner(
                host="10.0.0.1",
                image="test:latest",
                model="m",
                mode="bogus",
                dry_run=True,
            )

    def test_vllm_tune_ref_override(self):
        tuner = VllmTuner(
            host="10.0.0.1",
            image="test:latest",
            model="m",
            vllm_tune_ref="my-pr-branch",
            dry_run=True,
        )
        assert tuner.vllm_tune_ref == "my-pr-branch"

    def test_cache_subdir_constant(self):
        assert VLLM_TUNING_CACHE_SUBDIR == "tuning/vllm"


# ---------------------------------------------------------------------------
# vLLM runtime auto-mount integration
# ---------------------------------------------------------------------------


class TestVllmRayRuntimeAutoMount:
    def test_get_extra_volumes_empty_when_no_configs(self, v, monkeypatch):
        """VllmRayRuntime.get_extra_volumes returns {} when no tuning configs."""
        from sparkrun.core.bootstrap import get_runtime

        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: None,
        )
        runtime = get_runtime("vllm-ray", v)
        assert runtime.get_extra_volumes() == {}

    def test_get_extra_env_empty_when_no_configs(self, v, monkeypatch):
        """VllmRayRuntime.get_extra_env returns base env when no tuning configs."""
        from sparkrun.core.bootstrap import get_runtime

        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_env",
            lambda: None,
        )
        runtime = get_runtime("vllm-ray", v)
        assert runtime.get_extra_env() == {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}

    def test_get_extra_volumes_returns_mapping(self, v, monkeypatch):
        """VllmRayRuntime.get_extra_volumes returns mapping when configs exist."""
        from sparkrun.core.bootstrap import get_runtime

        expected = {"/cache/tuning/vllm": VLLM_TUNING_CONTAINER_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: expected,
        )
        runtime = get_runtime("vllm-ray", v)
        assert runtime.get_extra_volumes() == expected

    def test_get_extra_env_returns_env(self, v, monkeypatch):
        """VllmRayRuntime.get_extra_env returns env when configs exist."""
        from sparkrun.core.bootstrap import get_runtime

        expected = {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH,
        }
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_env",
            lambda: {"VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH},
        )
        runtime = get_runtime("vllm-ray", v)
        assert runtime.get_extra_env() == expected


class TestVllmDistributedAutoMount:
    def test_get_extra_volumes_empty_when_no_configs(self, v, monkeypatch):
        """VllmDistributedRuntime.get_extra_volumes returns {} when no configs."""
        from sparkrun.core.bootstrap import get_runtime

        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: None,
        )
        runtime = get_runtime("vllm-distributed", v)
        assert runtime.get_extra_volumes() == {}

    def test_get_extra_env_empty_when_no_configs(self, v, monkeypatch):
        """VllmDistributedRuntime.get_extra_env returns base env when no configs."""
        from sparkrun.core.bootstrap import get_runtime

        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_env",
            lambda: None,
        )
        runtime = get_runtime("vllm-distributed", v)
        assert runtime.get_extra_env() == {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}

    def test_get_extra_volumes_returns_mapping(self, v, monkeypatch):
        """VllmDistributedRuntime.get_extra_volumes returns mapping when configs exist."""
        from sparkrun.core.bootstrap import get_runtime

        expected = {"/cache/tuning/vllm": VLLM_TUNING_CONTAINER_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: expected,
        )
        runtime = get_runtime("vllm-distributed", v)
        assert runtime.get_extra_volumes() == expected

    def test_get_extra_env_returns_env(self, v, monkeypatch):
        """VllmDistributedRuntime.get_extra_env returns env when configs exist."""
        from sparkrun.core.bootstrap import get_runtime

        expected = {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH,
        }
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_env",
            lambda: {"VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH},
        )
        runtime = get_runtime("vllm-distributed", v)
        assert runtime.get_extra_env() == expected


class TestEugrVllmAutoMount:
    def test_inherits_vllm_ray_auto_mount(self, v, monkeypatch):
        """EugrVllmRayRuntime inherits get_extra_volumes from VllmRayRuntime."""
        from sparkrun.core.bootstrap import get_runtime

        expected = {"/cache/tuning/vllm": VLLM_TUNING_CONTAINER_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: expected,
        )
        runtime = get_runtime("eugr-vllm", v)
        assert runtime.get_extra_volumes() == expected

    def test_inherits_vllm_ray_auto_env(self, v, monkeypatch):
        """EugrVllmRayRuntime inherits get_extra_env from VllmRayRuntime."""
        from sparkrun.core.bootstrap import get_runtime

        expected = {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH,
        }
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_env",
            lambda: {"VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH},
        )
        runtime = get_runtime("eugr-vllm", v)
        assert runtime.get_extra_env() == expected


# ===========================================================================
# Pre-check tests
# ===========================================================================


class TestPreCheckTp:
    """Test that _pre_check_tp is called and can skip tuning."""

    def test_base_pre_check_returns_false_in_dry_run(self):
        """BaseTuner._pre_check_tp returns False in dry-run mode."""
        from sparkrun.tuning._common import BaseTuner

        tuner = BaseTuner.__new__(BaseTuner)
        tuner.dry_run = True
        assert tuner._pre_check_tp(1, "3.6.0") is False

    def test_sglang_pre_check_returns_false_in_dry_run(self):
        """SglangTuner._pre_check_tp returns False in dry-run mode."""
        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="test-model",
            dry_run=True,
        )
        assert tuner._pre_check_tp(1, "3.6.0") is False

    def test_sglang_pre_check_returns_false_on_error(self):
        """SglangTuner._pre_check_tp returns False on any exception."""
        from unittest.mock import patch

        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="test-model",
            dry_run=False,
        )
        with patch("sparkrun.orchestration.primitives.run_command_on_host", side_effect=RuntimeError("connection refused")):
            assert tuner._pre_check_tp(1, "3.6.0") is False

    def test_sglang_pre_check_returns_true_on_success(self):
        """SglangTuner._pre_check_tp returns True when command succeeds."""
        from unittest.mock import patch
        from sparkrun.orchestration.ssh import RemoteResult

        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="test-model",
            dry_run=False,
        )
        mock_result = RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr="")
        with patch("sparkrun.orchestration.primitives.run_command_on_host", return_value=mock_result):
            assert tuner._pre_check_tp(1, "3.6.0") is True

    def test_sglang_pre_check_returns_false_on_failure(self):
        """SglangTuner._pre_check_tp returns False when command fails."""
        from unittest.mock import patch
        from sparkrun.orchestration.ssh import RemoteResult

        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="test-model",
            dry_run=False,
        )
        mock_result = RemoteResult(host="10.0.0.1", returncode=1, stdout="", stderr="")
        with patch("sparkrun.orchestration.primitives.run_command_on_host", return_value=mock_result):
            assert tuner._pre_check_tp(1, "3.6.0") is False


# ===========================================================================
# Sync path tests
# ===========================================================================


class TestSyncGetLocalTuningDir:
    """Test that _get_local_tuning_dir returns correct paths."""

    def test_sglang_path(self):
        from sparkrun.tuning.sync import _get_local_tuning_dir

        d = _get_local_tuning_dir("sglang")
        assert str(d).endswith("sparkrun/tuning/sglang")

    def test_vllm_path(self):
        from sparkrun.tuning.sync import _get_local_tuning_dir

        d = _get_local_tuning_dir("vllm")
        assert str(d).endswith("sparkrun/tuning/vllm")

    def test_vllm_ray_maps_to_vllm(self):
        from sparkrun.tuning.sync import _get_local_tuning_dir

        d = _get_local_tuning_dir("vllm-ray")
        assert str(d).endswith("sparkrun/tuning/vllm")

    def test_vllm_distributed_maps_to_vllm(self):
        from sparkrun.tuning.sync import _get_local_tuning_dir

        d = _get_local_tuning_dir("vllm-distributed")
        assert str(d).endswith("sparkrun/tuning/vllm")

    def test_eugr_vllm_maps_to_vllm(self):
        from sparkrun.tuning.sync import _get_local_tuning_dir

        d = _get_local_tuning_dir("eugr-vllm")
        assert str(d).endswith("sparkrun/tuning/vllm")

    def test_other_runtime_uses_tuning_prefix(self):
        from sparkrun.tuning.sync import _get_local_tuning_dir

        d = _get_local_tuning_dir("llama-cpp")
        assert str(d).endswith("sparkrun/tuning/llama-cpp")


# ===========================================================================
# Mount-point constant tests
# ===========================================================================


class TestMountPointConstants:
    """Verify mount-point layout is correct for SGLang config lookup."""

    def test_sglang_container_path_ends_with_configs(self):
        """TUNING_CONTAINER_PATH must end with /configs so SGLang's internal
        $SGLANG_MOE_CONFIG_DIR/configs/triton_X_Y_Z/ resolves correctly."""
        assert TUNING_CONTAINER_PATH.endswith("/configs")

    def test_sglang_env_path_is_parent_of_container_path(self):
        """TUNING_ENV_PATH must be the parent of TUNING_CONTAINER_PATH."""
        assert TUNING_CONTAINER_PATH.startswith(TUNING_ENV_PATH)
        assert TUNING_CONTAINER_PATH == TUNING_ENV_PATH + "/configs"


# ===========================================================================
# Long-run plumbing: streaming, logging, timeouts, parallel safety
# ===========================================================================


class TestStreamedTuneScript:
    """The wrapper that makes a multi-hour sweep observable and recoverable."""

    def _script(self, cmd="vllm-tune.sh model --tp 2", log="/home/u/.cache/sparkrun/tuning/logs/t.log"):
        from sparkrun.tuning._common import build_streamed_tune_script

        return build_streamed_tune_script(cmd, log)

    def test_forces_unbuffered_python(self):
        """Without this, 'streaming' delivers 8 KB at a time and looks hung."""
        assert "PYTHONUNBUFFERED=1" in self._script()

    def test_tees_to_the_host_side_log(self):
        script = self._script()
        assert 'tee -a "/home/u/.cache/sparkrun/tuning/logs/t.log"' in script

    def test_creates_the_log_directory(self):
        assert 'mkdir -p "/home/u/.cache/sparkrun/tuning/logs"' in self._script()

    def test_propagates_the_tuning_exit_code_not_tees(self):
        """With tee in the pipeline, $? is tee's — which is 0 for any failure."""
        script = self._script()
        assert "rc=${PIPESTATUS[0]}" in script
        assert "exit $rc" in script

    def test_redirects_stderr_into_the_stream(self):
        assert "2>&1 | tee" in self._script()

    def test_compound_command_is_fully_captured_and_its_rc_propagates(self, tmp_path):
        """Executed for real, because the failure mode is shell precedence.

        Tuning commands are compound (``mkdir … && cd … && python3 …``).  With
        a bare ``cmd 2>&1 | tee``, the redirection and pipe bind to the *last*
        simple command only: the earlier stages' output bypasses the log and
        ``PIPESTATUS[0]`` reports the wrong command's status.  Substring
        assertions can't see that — running it can.
        """
        import subprocess

        log = tmp_path / "logs" / "t.log"
        script = self._script(
            cmd="printf 'first\\n' && printf 'second\\n' >&2 && exit 42",
            log=str(log),
        )
        proc = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True)

        assert proc.returncode == 42, "tee's exit status masked the tuning failure"
        contents = log.read_text()
        assert "first" in contents and "second" in contents
        assert "rc=42" in contents


class TestTuneLogPath:
    def test_log_dir_is_a_sibling_of_the_config_cache(self):
        """Logs must not land in the dir that is bind-mounted and rsynced."""
        from sparkrun.tuning._common import remote_tune_log_dir

        assert remote_tune_log_dir("/home/u/.cache/sparkrun/tuning/vllm") == "/home/u/.cache/sparkrun/tuning/logs"

    def test_log_name_carries_backend_model_and_tp(self):
        from sparkrun.tuning._common import remote_tune_log_path

        path = remote_tune_log_path(
            "/home/u/.cache/sparkrun/tuning/vllm",
            "vllm-tune",
            "Qwen/Qwen3-MoE",
            4,
            "20260813-101500",
        )
        assert path.startswith("/home/u/.cache/sparkrun/tuning/logs/")
        assert "vllm-tune" in path
        assert "Qwen-Qwen3-MoE" in path
        assert "tp4" in path
        assert path.endswith("-20260813-101500.log")

    def test_model_slug_has_no_path_separators(self):
        """A '/' in the model id would otherwise redirect the log elsewhere."""
        from sparkrun.tuning._common import remote_tune_log_path

        path = remote_tune_log_path("/c/tuning/vllm", "vllm-tune", "org/model", 1, "ts")
        assert path.count("/") == "/c/tuning/logs/".count("/")


class TestResolveTuningTimeout:
    def test_default_is_24_hours(self):
        from sparkrun.tuning._common import resolve_tuning_timeout

        assert resolve_tuning_timeout(None, None) == 24 * 3600

    def test_cli_override_wins_over_config(self, tmp_path):
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.tuning._common import resolve_tuning_timeout

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"tuning": {"timeout_hours": 10}}))
        config = SparkrunConfig(config_path=cfg_file)

        assert resolve_tuning_timeout(36, config) == 36 * 3600

    def test_config_used_when_no_cli_override(self, tmp_path):
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.tuning._common import resolve_tuning_timeout

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"tuning": {"timeout_hours": 10}}))
        config = SparkrunConfig(config_path=cfg_file)

        assert resolve_tuning_timeout(None, config) == 10 * 3600

    def test_zero_means_no_cap(self):
        from sparkrun.tuning._common import resolve_tuning_timeout

        assert resolve_tuning_timeout(0, None) is None

    def test_fractional_hours_supported(self):
        from sparkrun.tuning._common import resolve_tuning_timeout

        assert resolve_tuning_timeout(0.5, None) == 1800

    def test_describe_timeout_renders_none_as_none(self):
        from sparkrun.tuning._common import describe_timeout

        assert describe_timeout(None) == "none"


class TestResolveTuningParallel:
    """Concurrent jobs on one GPU contend, so the measured latencies are wrong."""

    def test_downgrades_to_sequential_by_default(self):
        from sparkrun.tuning._common import resolve_tuning_parallel

        assert resolve_tuning_parallel(4, n_jobs=4) == 1

    def test_force_parallel_honours_the_request(self):
        from sparkrun.tuning._common import resolve_tuning_parallel

        assert resolve_tuning_parallel(4, n_jobs=4, force=True) == 4

    def test_single_job_is_never_downgraded_or_warned(self):
        from sparkrun.tuning._common import resolve_tuning_parallel

        assert resolve_tuning_parallel(4, n_jobs=1) == 4
        assert resolve_tuning_parallel(1, n_jobs=4) == 1


class TestTailText:
    def test_short_text_passes_through(self):
        from sparkrun.tuning._common import tail_text

        assert tail_text("a\nb\nc") == "a\nb\nc"

    def test_long_text_is_tailed_and_says_so(self):
        from sparkrun.tuning._common import tail_text

        out = tail_text("\n".join(str(i) for i in range(500)), max_lines=10)
        assert out.splitlines()[0].startswith("... (490 earlier lines omitted)")
        assert out.endswith("499")
        assert len(out.splitlines()) == 11


class TestVllmTunerStreaming:
    """`--foreground` exists so output streams; the call site must not swallow it."""

    def _tuner(self, **kw):
        return VllmTuner(host="10.0.0.1", image="test:latest", model="Qwen/Qwen3-MoE", **kw)

    def _ok(self):
        from sparkrun.orchestration.ssh import RemoteResult

        return RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr="")

    def _fail(self, rc=1, stdout="boom"):
        from sparkrun.orchestration.ssh import RemoteResult

        return RemoteResult(host="10.0.0.1", returncode=rc, stdout=stdout, stderr="err")

    def test_tune_runs_through_the_streaming_dispatcher(self):
        from unittest.mock import patch

        tuner = self._tuner()
        with (
            patch("sparkrun.orchestration.primitives.run_script_on_host_streaming", return_value=self._ok()) as stream,
            patch("sparkrun.orchestration.primitives.run_command_on_host", return_value=self._ok()),
        ):
            assert tuner._run_one_tp("/opt/vllm-tune/vllm-tune.sh", 2) == 0

        assert stream.call_count == 1
        kwargs = stream.call_args.kwargs
        # The GPU is held for hours: a killed sparkrun must not orphan it, and
        # a dead link must not masquerade as a slow sweep.
        assert kwargs["session_guard"] is True
        assert kwargs["keepalive"] is True
        assert kwargs["timeout"] == tuner.timeout_sec
        assert kwargs["quiet"] is False
        # ...and the payload is the tee-wrapped script, not the bare command.
        script = stream.call_args.args[1]
        assert "PYTHONUNBUFFERED=1" in script
        assert "tee -a" in script
        assert "--foreground" in script

    def test_parallel_path_captures_instead_of_interleaving(self):
        from unittest.mock import patch

        tuner = self._tuner(force_parallel=True)
        with (
            patch("sparkrun.orchestration.primitives.run_script_on_host_streaming", return_value=self._ok()) as stream,
            patch("sparkrun.orchestration.primitives.run_command_on_host", return_value=self._ok()),
        ):
            tuner._run_tp_parallel("/opt/vllm-tune.sh", (1, 2), 2, [])

        assert all(call.kwargs["quiet"] is True for call in stream.call_args_list)

    def test_export_is_attempted_even_when_tuning_fails(self):
        """A sweep that died late may still have written usable configs."""
        from unittest.mock import patch

        tuner = self._tuner()
        with (
            patch("sparkrun.orchestration.primitives.run_script_on_host_streaming", return_value=self._fail()),
            patch("sparkrun.orchestration.primitives.run_command_on_host", return_value=self._ok()) as export,
        ):
            rc = tuner._run_one_tp("/opt/vllm-tune.sh", 2)

        assert rc == 1
        assert export.call_count == 1
        assert "--export-sparkrun" in export.call_args.args[1]

    def test_timeout_is_reported_as_a_timeout_with_the_remedy(self, caplog):
        import logging
        from unittest.mock import patch
        from sparkrun.orchestration.ssh import TIMEOUT_RETURNCODE

        tuner = self._tuner(timeout_hours=2)
        with (
            patch(
                "sparkrun.orchestration.primitives.run_script_on_host_streaming",
                return_value=self._fail(rc=TIMEOUT_RETURNCODE, stdout=""),
            ),
            patch("sparkrun.orchestration.primitives.run_command_on_host", return_value=self._ok()),
            caplog.at_level(logging.ERROR, logger="sparkrun.tuning._common"),
        ):
            rc = tuner._run_one_tp("/opt/vllm-tune.sh", 2)

        assert rc == TIMEOUT_RETURNCODE
        text = caplog.text
        assert "cap" in text and "--timeout" in text
        # ...and it names the host-side log, which survived the kill.
        assert "/tuning/logs/" in text

    def test_failure_output_is_tailed_not_dumped(self, caplog):
        """Hours of tqdm output buried the actual error (issue #206)."""
        import logging
        from unittest.mock import patch

        tuner = self._tuner()
        firehose = "\n".join("progress %d" % i for i in range(5000)) + "\nRealError: boom"
        with (
            patch("sparkrun.orchestration.primitives.run_script_on_host_streaming", return_value=self._fail(stdout=firehose)),
            patch("sparkrun.orchestration.primitives.run_command_on_host", return_value=self._ok()),
            caplog.at_level(logging.ERROR, logger="sparkrun.tuning._common"),
        ):
            tuner._run_one_tp("/opt/vllm-tune.sh", 2)

        assert "RealError: boom" in caplog.text
        assert "earlier lines omitted" in caplog.text
        assert "progress 0\n" not in caplog.text


class TestTuningPartialResultsSurvive:
    """Earlier TP successes must reach the control node when a later one fails."""

    def test_vllm_syncs_back_after_a_failed_tp(self):
        from unittest.mock import patch

        tuner = VllmTuner(host="10.0.0.1", image="i", model="m")
        with (
            patch.object(VllmTuner, "_install_vllm_tune", return_value="/opt/vllm-tune.sh"),
            patch.object(VllmTuner, "_preflight", return_value=0),
            patch.object(VllmTuner, "_ensure_remote_output_dir", return_value=0),
            patch.object(VllmTuner, "_run_one_tp", side_effect=[0, 1]),
            patch.object(VllmTuner, "_sync_back_configs") as sync,
        ):
            rc = tuner.run_tuning(tp_sizes=(1, 2))

        assert rc == 1
        assert sync.call_count == 1

    def test_sglang_syncs_back_after_a_failed_tp(self):
        from unittest.mock import patch

        tuner = SglangTuner(host="10.0.0.1", image="i", model="m", skip_clone=True)
        with (
            patch.object(SglangTuner, "_launch_container", return_value=0),
            patch.object(SglangTuner, "_detect_triton_version", return_value="3.6.0"),
            patch.object(SglangTuner, "_pre_check_tp", return_value=False),
            patch.object(SglangTuner, "_run_tune_for_tp", side_effect=[0, 1]),
            patch.object(SglangTuner, "_sync_back_configs") as sync,
            patch.object(SglangTuner, "_cleanup_container"),
        ):
            rc = tuner.run_tuning(tp_sizes=(1, 2))

        assert rc == 1
        assert sync.call_count == 1


class TestSglangTunerStreaming:
    def test_tune_runs_through_the_streaming_dispatcher(self):
        from unittest.mock import patch
        from sparkrun.orchestration.ssh import RemoteResult

        tuner = SglangTuner(host="10.0.0.1", image="i", model="m")
        ok = RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr="")
        with patch("sparkrun.orchestration.primitives.run_script_on_host_streaming", return_value=ok) as stream:
            assert tuner._run_tune_for_tp(2, "3.6.0") == 0

        kwargs = stream.call_args.kwargs
        assert kwargs["session_guard"] is True
        assert kwargs["keepalive"] is True
        assert kwargs["timeout"] == tuner.timeout_sec
        assert "PYTHONUNBUFFERED=1" in stream.call_args.args[1]


class TestTuneCliOptions:
    def test_vllm_help_documents_timeout_and_force_parallel(self):
        from sparkrun.cli import main

        result = CliRunner().invoke(main, ["tune", "vllm", "--help"])
        assert result.exit_code == 0
        assert "--timeout" in result.output
        assert "--force-parallel" in result.output

    def test_sglang_help_documents_timeout(self):
        from sparkrun.cli import main

        result = CliRunner().invoke(main, ["tune", "sglang", "--help"])
        assert result.exit_code == 0
        assert "--timeout" in result.output


class TestTunerOutputVisibleAtDefaultVerbosity:
    """A multi-hour job that prints nothing is the bug being fixed.

    The CLI's default tier is PROGRESS (25); the tuners emitted their banner
    and every step line at INFO (20), so `sparkrun tune` printed *nothing at
    all* unless the user happened to pass -v.
    """

    def _records_at_progress(self, caplog, run):
        from sparkrun.core.progress import PROGRESS

        with caplog.at_level(PROGRESS):
            run()
        return [r for r in caplog.records if r.levelno >= PROGRESS]

    def test_vllm_banner_and_steps_reach_default_verbosity(self, caplog):
        tuner = VllmTuner(host="10.0.0.1", image="i", model="m", dry_run=True)
        records = self._records_at_progress(caplog, lambda: tuner.run_tuning(tp_sizes=(1,)))
        text = "\n".join(r.getMessage() for r in records)

        assert "Kernel Tuner" in text
        assert "Step 4/5" in text
        assert "Timeout:" in text

    def test_sglang_banner_and_steps_reach_default_verbosity(self, caplog):
        tuner = SglangTuner(host="10.0.0.1", image="i", model="m", dry_run=True)
        records = self._records_at_progress(caplog, lambda: tuner.run_tuning(tp_sizes=(1,)))
        text = "\n".join(r.getMessage() for r in records)

        assert "Kernel Tuner" in text
        assert "Step 4/5" in text
