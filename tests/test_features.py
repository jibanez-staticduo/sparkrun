"""Tests for channel-aware feature flags and plugin gating.

Covers:

- :func:`sparkrun.core.features.is_feature_enabled` precedence
  (env > config > channel-default > baseline > fail-closed).
- Per-channel defaults on the built-in executor flags.
- Plugin self-gate: a gated-off executor hides itself from the SAF registry
  via ``is_multi_extension``; channel / config / env enabling brings it back.
- ``resolve_executor`` fails loudly for an explicitly-requested but gated
  executor.
- A guard that every plugin ``required_feature_flag`` maps to a registered flag.
"""

from __future__ import annotations

import yaml

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.features import (
    FeatureFlag,
    all_features,
    get_feature,
    is_feature_enabled,
)


def _run_gated_snippet(tmp_path, config_data: dict, snippet: str, env_extra: dict | None = None) -> str:
    """Run *snippet* in a clean subprocess with an isolated sparkrun config.

    Bootstrap gating is observed end-to-end here because SAF registers
    plugins in a process-global registry — once an in-process test enables
    ``local``/``k8s`` they never un-register. A fresh subprocess with
    ``HOME`` redirected exercises the real production path
    (reading ``$HOME/.config/sparkrun/config.yaml``) with a clean registry.
    """
    import os
    import subprocess
    import sys

    home = tmp_path / "home"
    cfg_dir = home / ".config" / "sparkrun"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(config_data))

    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARKRUN_FEATURE_")}
    env.pop("STATEFUL_ROOT", None)
    env["HOME"] = str(home)
    env["SPARKRUN_NO_TELEMETRY"] = "1"
    if env_extra:
        env.update(env_extra)

    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, "subprocess failed: %s\n%s" % (proc.stdout, proc.stderr)
    return proc.stdout.strip()


def _executor_names(tmp_path, config_data: dict, env_extra: dict | None = None) -> list[str]:
    """Return the SAF-registered executor names for a gated config."""
    import json

    snippet = (
        "import json\n"
        "from sparkrun.core.bootstrap import init_sparkrun\n"
        "from sparkrun.orchestration.executor import list_executors\n"
        "v = init_sparkrun()\n"
        "print(json.dumps(list_executors(v)))\n"
    )
    out = _run_gated_snippet(tmp_path, config_data, snippet, env_extra)
    return json.loads(out.splitlines()[-1])


# ---------------------------------------------------------------------------
# FeatureFlag.default_for_channel
# ---------------------------------------------------------------------------


class TestFeatureFlagDefaults:
    def test_default_for_listed_channel(self):
        flag = FeatureFlag("x", "d", channel_defaults={"alpha": True}, default=False)
        assert flag.default_for_channel("alpha") is True
        assert flag.default_for_channel("stable") is False

    def test_unlisted_channel_uses_baseline(self):
        flag = FeatureFlag("x", "d", channel_defaults={"alpha": True}, default=True)
        assert flag.default_for_channel("beta") is True

    def test_channel_normalized(self):
        flag = FeatureFlag("x", "d", channel_defaults={"alpha": True})
        # "yolo" normalizes to alpha.
        assert flag.default_for_channel("yolo") is True


# ---------------------------------------------------------------------------
# cli.setup.features — visibility-only flag (group always functional)
# ---------------------------------------------------------------------------


class TestCliSetupFeaturesFlag:
    def test_channel_defaults(self):
        # On for beta/alpha, off for stable — no env/config overrides.
        assert is_feature_enabled("cli.setup.features", channel="stable", env={}) is False
        assert is_feature_enabled("cli.setup.features", channel="beta", env={}) is True
        assert is_feature_enabled("cli.setup.features", channel="alpha", env={}) is True

    def test_hidden_on_stable_but_still_functional(self, tmp_path):
        # Clean subprocess (the group's `hidden=` freezes at import from config).
        snippet = (
            "from click.testing import CliRunner\n"
            "from sparkrun.cli import main\n"
            "h = CliRunner().invoke(main, ['setup', '--help']).output\n"
            "listed = any(l.strip().startswith('features') for l in h.splitlines())\n"
            "r = CliRunner().invoke(main, ['setup', 'features', 'list'])\n"
            "print('LISTED', listed)\n"
            "print('LISTEXIT', r.exit_code)\n"
        )
        out = _run_gated_snippet(tmp_path, {"self_update": {"channel": "stable"}}, snippet)
        assert "LISTED False" in out  # hidden from --help on stable
        assert "LISTEXIT 0" in out  # ...but the command still works

    def test_visible_on_beta(self, tmp_path):
        snippet = (
            "from click.testing import CliRunner\n"
            "from sparkrun.cli import main\n"
            "h = CliRunner().invoke(main, ['setup', '--help']).output\n"
            "print('LISTED', any(l.strip().startswith('features') for l in h.splitlines()))\n"
        )
        out = _run_gated_snippet(tmp_path, {"self_update": {"channel": "beta"}}, snippet)
        assert "LISTED True" in out


# ---------------------------------------------------------------------------
# Resolution precedence
# ---------------------------------------------------------------------------


class TestResolutionPrecedence:
    def test_unknown_flag_fails_closed(self):
        assert is_feature_enabled("does.not.exist", env={}) is False

    def test_builtin_executor_flags_off_on_all_channels(self):
        # The experimental executors ship off by default everywhere; only an
        # explicit config/env opt-in enables them.
        for channel in ("stable", "beta", "alpha"):
            assert is_feature_enabled("executor.k8s", channel=channel, env={}) is False
            assert is_feature_enabled("executor.local", channel=channel, env={}) is False

    def test_env_overrides_channel(self):
        assert is_feature_enabled("executor.k8s", channel="stable", env={"SPARKRUN_FEATURE_EXECUTOR_K8S": "1"}) is True
        assert is_feature_enabled("executor.k8s", channel="alpha", env={"SPARKRUN_FEATURE_EXECUTOR_K8S": "0"}) is False

    def test_config_override_beats_channel(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"features": {"channel": "stable", "executor.k8s": True}}))
        config = SparkrunConfig(cfg_path)
        # env empty so config wins over the stable channel default.
        assert is_feature_enabled("executor.k8s", config=config, env={}) is True

    def test_env_beats_config(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"features": {"executor.k8s": True}}))
        config = SparkrunConfig(cfg_path)
        assert is_feature_enabled("executor.k8s", config=config, env={"SPARKRUN_FEATURE_EXECUTOR_K8S": "0"}) is False


# ---------------------------------------------------------------------------
# SparkrunConfig surface
# ---------------------------------------------------------------------------


class TestConfigSurface:
    def test_feature_channel_defaults_to_self_update(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"self_update": {"channel": "beta"}}))
        config = SparkrunConfig(cfg_path)
        assert config.feature_channel == "beta"

    def test_feature_channel_explicit_override(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"self_update": {"channel": "stable"}, "features": {"channel": "alpha"}}))
        config = SparkrunConfig(cfg_path)
        assert config.feature_channel == "alpha"

    def test_feature_override_unset_is_none(self, tmp_path):
        config = SparkrunConfig(tmp_path / "missing.yaml")
        assert config.feature_override("executor.k8s") is None

    def test_channel_key_not_treated_as_flag(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"features": {"channel": "alpha"}}))
        config = SparkrunConfig(cfg_path)
        assert config.feature_override("channel") is None


# ---------------------------------------------------------------------------
# Plugin self-gate — is_multi_extension hides gated-off executors
# ---------------------------------------------------------------------------


class TestGatingDecision:
    """Unit-level tests of the plugin self-gate (``is_multi_extension``).

    The gate resolves via env first, so these are deterministic regardless
    of the developer's real config file.
    """

    def test_hidden_when_flag_disabled(self, monkeypatch):
        from sparkrun.orchestration.executors.k8s import K8sExecutor

        monkeypatch.setenv("SPARKRUN_FEATURE_EXECUTOR_K8S", "0")
        assert K8sExecutor().is_multi_extension(None) is False

    def test_exposed_when_flag_enabled(self, monkeypatch):
        from sparkrun.orchestration.executors.k8s import K8sExecutor

        monkeypatch.setenv("SPARKRUN_FEATURE_EXECUTOR_K8S", "1")
        assert K8sExecutor().is_multi_extension(None) is True

    def test_local_hidden_when_flag_disabled(self, monkeypatch):
        from sparkrun.orchestration.executors.local import LocalExecutor

        monkeypatch.setenv("SPARKRUN_FEATURE_EXECUTOR_LOCAL", "0")
        assert LocalExecutor().is_multi_extension(None) is False

    def test_plugin_without_required_feature_always_exposed(self):
        from sparkrun.orchestration.executors.docker import DockerExecutor

        assert DockerExecutor().is_multi_extension(None) is True


class TestBootstrapGatingEndToEnd:
    """End-to-end gating in a clean subprocess (real production config path)."""

    def test_stable_excludes_experimental_executors(self, tmp_path):
        names = _executor_names(tmp_path, {"features": {"channel": "stable"}})
        assert "docker" in names
        assert "local" not in names
        assert "k8s" not in names

    def test_alpha_still_excludes_experimental_executors(self, tmp_path):
        # The experimental executors are off on every channel, including alpha.
        names = _executor_names(tmp_path, {"features": {"channel": "alpha"}})
        assert "local" not in names
        assert "k8s" not in names

    def test_config_override_includes_experimental_executors(self, tmp_path):
        names = _executor_names(tmp_path, {"features": {"executor.local": True, "executor.k8s": True}})
        assert {"docker", "local", "k8s"} <= set(names)

    def test_explicit_override_on_stable(self, tmp_path):
        names = _executor_names(tmp_path, {"features": {"channel": "stable", "executor.k8s": True}})
        assert "k8s" in names
        assert "local" not in names

    def test_env_override_on_stable(self, tmp_path):
        names = _executor_names(
            tmp_path,
            {"features": {"channel": "stable"}},
            env_extra={"SPARKRUN_FEATURE_EXECUTOR_LOCAL": "1"},
        )
        assert "local" in names
        assert "k8s" not in names


# ---------------------------------------------------------------------------
# resolve_executor fails loudly for a gated explicit request
# ---------------------------------------------------------------------------


class TestGatedExplicitRequestFails:
    def test_gated_executor_request_raises_with_guidance(self, tmp_path):
        snippet = (
            "from sparkrun.core.bootstrap import init_sparkrun\n"
            "from sparkrun.core.config import SparkrunConfig, get_config_root\n"
            "from sparkrun.orchestration.executor import ExecutorUnavailableError, resolve_executor\n"
            "v = init_sparkrun()\n"
            "config = SparkrunConfig(get_config_root(v) / 'config.yaml')\n"
            "try:\n"
            "    resolve_executor(cli_overrides={'executor': 'k8s'}, config=config, v=v)\n"
            "    print('NO_RAISE')\n"
            "except ExecutorUnavailableError as e:\n"
            "    print('RAISED:' + str(e))\n"
        )
        out = _run_gated_snippet(tmp_path, {"features": {"channel": "stable"}}, snippet)
        line = out.splitlines()[-1]
        assert line.startswith("RAISED:"), out
        assert "executor.k8s" in line
        assert "setup features enable" in line


# ---------------------------------------------------------------------------
# Guard: every plugin required_feature_flag maps to a registered flag
# ---------------------------------------------------------------------------


class TestRequiredFeatureGuard:
    def test_executor_required_features_are_registered(self):
        from scitrera_app_framework.util import find_types_in_modules
        from sparkrun.orchestration.executors._base import Executor

        for cls in find_types_in_modules("sparkrun.orchestration.executors", Executor):
            feature = getattr(cls, "required_feature_flag", None)
            if feature:
                assert get_feature(feature) is not None, "%s gates on unregistered flag %r" % (cls.__name__, feature)

    def test_builtin_executor_flags_present(self):
        names = {f.name for f in all_features()}
        assert {"executor.local", "executor.k8s"} <= names
