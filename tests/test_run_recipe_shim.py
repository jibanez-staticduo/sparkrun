"""Argv-mapping tests for the ``run-recipe.sh`` spark-vllm-docker shim.

The shim exposes a hidden ``RUN_RECIPE_DEBUG=1`` hook that prints the assembled
``sparkrun`` argv to stderr and exits 0 *without* invoking sparkrun, which is
what lets us assert the legacy-option -> native-option mapping here without a
cluster (or even a working sparkrun install).

Assertions are ``endswith``-based because the runner prefix is environment
dependent (``<repo>/.venv/bin/sparkrun`` in a dev checkout, plain ``sparkrun``
on PATH, or ``uv tool run sparkrun`` as the last resort).

The upstream tool these map onto is ``run-recipe.py`` in spark-vllm-docker; a
pinned copy lives under ``reference_materials/spark-vllm-docker/``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SHIM = Path(__file__).resolve().parent.parent / "run-recipe.sh"

RECIPE = "qwen3-1.7b-vllm"


@pytest.fixture(scope="module")
def stub_path(tmp_path_factory):
    """A PATH entry holding a no-op ``sparkrun``.

    Only used when the repo has no ``.venv`` (e.g. CI): it keeps
    ``resolve_runner`` from falling through to its ``pip install uv``
    bootstrap, which would hit the network.
    """
    bindir = tmp_path_factory.mktemp("stubbin")
    stub = bindir / "sparkrun"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    return bindir


def shim(*args, env=None):
    """Run the shim under the debug hook; return (returncode, stderr)."""
    e = dict(os.environ)
    e["RUN_RECIPE_DEBUG"] = "1"
    if env:
        e.update(env)
    p = subprocess.run(
        [str(SHIM), *args],
        capture_output=True,
        text=True,
        env=e,
    )
    return p.returncode, p.stderr


def argv(*args, **kwargs):
    """Run the shim and return the single assembled ``DEBUG argv:`` line."""
    rc, err = shim(*args, **kwargs)
    assert rc == 0, "shim exited %d:\n%s" % (rc, err)
    lines = [ln for ln in err.splitlines() if ln.startswith("DEBUG argv:")]
    assert len(lines) == 1, "expected exactly one DEBUG argv line, got:\n%s" % err
    return lines[0][len("DEBUG argv:") :].strip()


@pytest.fixture(autouse=True)
def _path_with_stub(stub_path, monkeypatch):
    monkeypatch.setenv("PATH", "%s%s%s" % (os.environ.get("PATH", ""), os.pathsep, stub_path))


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_solo_defaults_tp1_and_foreground():
    assert argv(RECIPE, "--solo").endswith("run %s --solo --tp 1 --foreground" % RECIPE)


def test_explicit_tp_suppresses_the_solo_tp1_default():
    """--solo only injects --tp 1 when the user did not pick a tp themselves."""
    assert argv(RECIPE, "--solo", "--tp", "2").endswith("run %s --solo --tp 2 --foreground" % RECIPE)


def test_daemon_drops_foreground():
    assert argv(RECIPE, "--solo", "-d").endswith("run %s --solo --tp 1" % RECIPE)


@pytest.mark.parametrize(
    "legacy,native",
    [
        (["--port", "9000"], "--port 9000"),
        (["--host", "0.0.0.0"], "-o host=0.0.0.0"),
        (["--tp", "4"], "--tp 4"),
        (["--tensor-parallel", "4"], "--tp 4"),
        (["--gpu-mem", "0.8"], "--gpu-mem 0.8"),
        (["--gpu-memory-utilization", "0.8"], "--gpu-mem 0.8"),
        (["--max-model-len", "4096"], "--max-model-len 4096"),
        (["-n", "h1,h2"], "--hosts h1,h2"),
        (["--nodes", "h1,h2"], "--hosts h1,h2"),
        (["-t", "img:tag"], "--image img:tag"),
        (["--container", "img:tag"], "--image img:tag"),
        (["--name", "vllm_node"], "--container-name vllm_node"),
        (["--master-port", "29501"], "--init-port 29501"),
        (["--head-port", "29501"], "--init-port 29501"),
        (["-e", "HF_TOKEN=x"], "--executor-args -e HF_TOKEN=x"),
        (["--nccl-debug", "INFO"], "--executor-args -e NCCL_DEBUG=INFO"),
        (["--mem-limit-gb", "32"], "--memory-limit 32G"),
        (["--shm-size-gb", "64"], "-o shm_size=64g"),
        (["--mem-swap-limit-gb", "48"], "--executor-args --memory-swap 48g"),
        (["--pids-limit", "4096"], "--executor-args --pids-limit 4096"),
        (["--non-privileged"], "-o privileged=false"),
        (["--dry-run"], "--dry-run"),
    ],
)
def test_value_option_mapping(legacy, native):
    assert argv(RECIPE, *legacy).endswith("run %s %s --foreground" % (RECIPE, native))


def test_equals_form_is_accepted():
    assert argv(RECIPE, "--port=9000").endswith("run %s --port 9000 --foreground" % RECIPE)


# ---------------------------------------------------------------------------
# -v/--volume  (upstream e987390)
# ---------------------------------------------------------------------------


def test_volume_maps_to_executor_args():
    assert argv(RECIPE, "--solo", "-v", "/local/models:/models").endswith(
        "run %s --solo --executor-args -v /local/models:/models --tp 1 --foreground" % RECIPE
    )


def test_volume_is_repeatable():
    line = argv(RECIPE, "--solo", "-v", "/a:/b", "-v", "/c:/d")
    assert "--executor-args -v /a:/b --executor-args -v /c:/d" in line


def test_volume_long_and_equals_forms():
    assert "--executor-args -v /a:/b" in argv(RECIPE, "--solo", "--volume", "/a:/b")
    assert "--executor-args -v /a:/b" in argv(RECIPE, "--solo", "--volume=/a:/b")


def test_volume_allowed_in_cluster_mode_unlike_publish():
    """Upstream emits -v in both the solo and cluster branches; -p is solo-only."""
    assert "--executor-args -v /a:/b" in argv(RECIPE, "-n", "h1,h2", "-v", "/a:/b")
    rc, err = shim(RECIPE, "-n", "h1,h2", "-p", "8000:8000")
    assert rc == 1
    assert "only supported in solo mode" in err


# ---------------------------------------------------------------------------
# Ray selection — upstream: use_ray = args.ray and not is_solo
# ---------------------------------------------------------------------------


def test_no_ray_is_the_multi_node_default():
    """Neither flag -> no backend override; sparkrun defaults to vllm-distributed."""
    assert argv(RECIPE, "-n", "h1,h2").endswith("run %s --hosts h1,h2 --foreground" % RECIPE)


def test_ray_opts_into_the_ray_backend():
    assert "-o distributed_executor_backend=ray" in argv(RECIPE, "-n", "h1,h2", "--ray")


def test_no_ray_is_accepted_for_compatibility():
    assert "-o distributed_executor_backend=mp" in argv(RECIPE, "-n", "h1,h2", "--no-ray")


@pytest.mark.parametrize("flag", ["--ray", "--no-ray"])
@pytest.mark.parametrize("order", ["flag_first", "solo_first"])
def test_solo_ignores_ray_flags_regardless_of_order(flag, order):
    """Upstream silently drops the backend choice in solo mode rather than erroring.

    Order matters as a regression guard: the flags are recorded during parsing
    and resolved afterwards, precisely so a trailing --solo still suppresses
    them.
    """
    args = [flag, "--solo"] if order == "flag_first" else ["--solo", flag]
    rc, err = shim(RECIPE, *args)
    assert rc == 0, err
    assert "distributed_executor_backend" not in err
    assert "ignored in solo mode" in err


def test_ray_and_no_ray_are_mutually_exclusive():
    rc, err = shim(RECIPE, "-n", "h1,h2", "--ray", "--no-ray")
    assert rc == 1
    assert "mutually exclusive" in err


# ---------------------------------------------------------------------------
# Unsupported / unknown options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opt",
    [
        "--apply-mod",
        "--eth-if",
        "--ib-if",
        "--discover",
        "--show-env",
        "--build-only",
        "--download-only",
        "--force-build",
        "--force-download",
        "-j",
        "--keep-entrypoint",
        "--no-cache-dirs",
        "--earlyoom",
        "--earlyoom-args",
    ],
)
def test_unsupported_options_exit_2_with_a_native_pointer(opt):
    rc, err = shim(RECIPE, opt, "x")
    assert rc == 2, err
    assert "is not supported by" in err


def test_unknown_option_is_rejected():
    rc, err = shim(RECIPE, "--definitely-not-an-option")
    assert rc == 1
    assert "unknown option" in err


def test_setup_is_a_noop_and_still_runs():
    rc, err = shim(RECIPE, "--solo", "--setup")
    assert rc == 0, err
    assert "--setup is a no-op" in err


def test_second_positional_is_rejected():
    rc, err = shim(RECIPE, "extra-recipe")
    assert rc == 1
    assert "unexpected extra argument" in err


def test_missing_recipe_is_rejected():
    rc, err = shim("--solo")
    assert rc == 1
    assert "no recipe specified" in err


def test_no_args_prints_usage():
    rc, err = shim()
    assert rc == 1
    assert "Usage:" in err


def test_help_exits_zero():
    rc, err = shim("--help")
    assert rc == 0
    assert "Usage:" in err


# ---------------------------------------------------------------------------
# --list and --config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--list", "-l"])
def test_list_short_circuits_to_recipe_list(flag):
    rc, err = shim(flag)
    assert rc == 0, err
    assert err.strip().endswith("recipe list")


def test_list_wins_over_a_recipe_and_other_options():
    rc, err = shim(RECIPE, "--solo", "--list")
    assert rc == 0, err
    assert err.strip().endswith("recipe list")


def test_config_imports_a_cluster_and_retargets_the_run(tmp_path):
    envf = tmp_path / "cluster.env"
    envf.write_text("CLUSTER_NODES=10.0.0.1,10.0.0.2\n")
    rc, err = shim(RECIPE, "--solo", "--config", str(envf))
    assert rc == 0, err
    assert "cluster import --from-spark-vllm-docker-env %s" % envf in err
    # --cluster is prepended so it precedes the mapped run options.
    assert "run %s --cluster <from-env> --solo --tp 1 --foreground" % RECIPE in err


# ---------------------------------------------------------------------------
# Engine passthrough after `--`
# ---------------------------------------------------------------------------


def test_passthrough_key_value_becomes_an_option_override():
    line = argv(RECIPE, "--solo", "--", "--max-num-seqs", "64")
    assert "-o max_num_seqs=64" in line


def test_passthrough_equals_form():
    line = argv(RECIPE, "--solo", "--", "--max-num-seqs=64")
    assert "-o max_num_seqs=64" in line


def test_passthrough_bare_flag_becomes_true():
    line = argv(RECIPE, "--solo", "--", "--enforce-eager")
    assert "-o enforce_eager=true" in line


def test_passthrough_flag_followed_by_flag_does_not_consume_it():
    line = argv(RECIPE, "--solo", "--", "--enforce-eager", "--trust-remote-code")
    assert "-o enforce_eager=true" in line
    assert "-o trust_remote_code=true" in line


def test_passthrough_bare_token_is_warned_and_dropped():
    rc, err = shim(RECIPE, "--solo", "--", "orphan")
    assert rc == 0, err
    assert "ignoring bare passthrough token 'orphan'" in err


def test_passthrough_does_not_swallow_shim_options():
    """Options before `--` are still mapped; only the tail is passthrough."""
    line = argv(RECIPE, "--solo", "--port", "9000", "--", "--max-num-seqs", "64")
    assert "--port 9000" in line
    assert "-o max_num_seqs=64" in line
