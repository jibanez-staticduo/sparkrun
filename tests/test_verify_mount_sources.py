"""Tests for the ``Executor.verify_mount_sources`` preflight seam.

Covers the shared host-substrate probe (``ssh.verify_host_paths``), the base
no-op default, and the docker/local overrides that delegate to the shared
helper.  The launcher-level integration (pre-placed model preflight) lives in
``test_recipe_cluster_config.py``.
"""

from __future__ import annotations

from sparkrun.orchestration.ssh import RemoteResult, verify_host_paths


def _patch_parallel(monkeypatch, by_host):
    """Stub run_remote_scripts_parallel to return canned per-host results.

    *by_host*: {host: (returncode, stdout)}.
    """
    captured = {}

    def _fake(hosts, script, **kw):
        captured["script"] = script
        captured["kw"] = kw
        return [RemoteResult(host=h, returncode=rc, stdout=out, stderr="") for h, (rc, out) in by_host.items()]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", _fake)
    return captured


# ---------------------------------------------------------------------------
# verify_host_paths (shared host-substrate probe)
# ---------------------------------------------------------------------------


def test_verify_host_paths_reports_only_missing(monkeypatch):
    # host-a is missing the path (echoes it back); host-b has it (empty stdout).
    _patch_parallel(monkeypatch, {"host-a": (0, "/nfs/models/qwen3\n"), "host-b": (0, "")})
    missing = verify_host_paths(["host-a", "host-b"], ["/nfs/models/qwen3"])
    assert missing == {"host-a": ["/nfs/models/qwen3"]}


def test_verify_host_paths_all_present_is_empty(monkeypatch):
    _patch_parallel(monkeypatch, {"host-a": (0, ""), "host-b": (0, "")})
    assert verify_host_paths(["host-a", "host-b"], ["/m"]) == {}


def test_verify_host_paths_unreachable_host_is_skipped_not_blocked(monkeypatch):
    # A non-zero probe means "couldn't verify" → the host is omitted (tolerant),
    # never reported as missing.
    _patch_parallel(monkeypatch, {"host-a": (255, ""), "host-b": (0, "")})
    assert verify_host_paths(["host-a", "host-b"], ["/m"]) == {}


def test_verify_host_paths_multi_path_partial_missing_preserves_order(monkeypatch):
    _patch_parallel(monkeypatch, {"host-a": (0, "/b\n/a\n")})  # both missing, out-of-order echo
    missing = verify_host_paths(["host-a"], ["/a", "/b", "/c"])
    assert missing == {"host-a": ["/a", "/b"]}  # requested order, /c present


def test_verify_host_paths_ignores_unrequested_noise(monkeypatch):
    _patch_parallel(monkeypatch, {"host-a": (0, "warning: something\n/a\n")})
    assert verify_host_paths(["host-a"], ["/a"]) == {"host-a": ["/a"]}


def test_verify_host_paths_empty_inputs_short_circuit(monkeypatch):
    # No SSH at all for empty host or path lists.
    called = {"n": 0}
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or [],
    )
    assert verify_host_paths([], ["/a"]) == {}
    assert verify_host_paths(["h"], []) == {}
    assert called["n"] == 0


def test_verify_host_paths_forwards_ssh_kwargs(monkeypatch):
    captured = _patch_parallel(monkeypatch, {"h": (0, "")})
    verify_host_paths(["h"], ["/a"], {"ssh_user": "u", "ssh_key": "/k", "ssh_options": ["-x"], "timeout": 30})
    assert captured["kw"]["ssh_user"] == "u"
    assert captured["kw"]["ssh_key"] == "/k"
    assert captured["kw"]["ssh_options"] == ["-x"]
    assert captured["kw"]["timeout"] == 30


# ---------------------------------------------------------------------------
# Executor.verify_mount_sources — base no-op + host-substrate overrides
# ---------------------------------------------------------------------------


def _min_executor(cls):
    """Instantiate an executor subclass with the abstract command-gens stubbed."""
    from sparkrun.orchestration.executors._base import Executor

    class _E(cls):
        executor_name = "test"

        def run_cmd(self, *a, **k):
            return ""

        exec_cmd = stop_cmd = logs_cmd = status_cmd = inspect_exists_cmd = pull_cmd = lambda self, *a, **k: ""

    # Only concrete executors (docker/local) are passed here; the base is tested
    # via a direct minimal subclass.
    assert issubclass(cls, Executor)
    return _E()


def test_base_executor_verify_mount_sources_is_noop():
    from sparkrun.orchestration.executors._base import Executor

    e = _min_executor(Executor)
    assert e.verify_mount_sources(["/a", "/b"], ["h1", "h2"]) == {}


def test_docker_executor_delegates_to_host_probe(monkeypatch):
    from sparkrun.orchestration.executors.docker import DockerExecutor

    calls = {}

    def _fake(hosts, paths, ssh_kwargs=None):
        calls["args"] = (hosts, paths, ssh_kwargs)
        return {"h1": ["/nfs/m"]}

    monkeypatch.setattr("sparkrun.orchestration.ssh.verify_host_paths", _fake)
    out = DockerExecutor().verify_mount_sources(["/nfs/m"], ["h1"], ssh_kwargs={"ssh_user": "u"})
    assert out == {"h1": ["/nfs/m"]}
    assert calls["args"] == (["h1"], ["/nfs/m"], {"ssh_user": "u"})


def test_local_executor_delegates_to_host_probe(monkeypatch):
    from sparkrun.orchestration.executors.local import LocalExecutor

    monkeypatch.setattr("sparkrun.orchestration.ssh.verify_host_paths", lambda h, p, kw=None: {"h1": p})
    out = LocalExecutor().verify_mount_sources(["/m"], ["h1"])
    assert out == {"h1": ["/m"]}
