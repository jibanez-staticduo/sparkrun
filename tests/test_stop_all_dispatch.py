"""Tests for teardown dispatch and result handling ("stop must not lie").

Three related contracts, all broken by the same root cause — localhost
work wrapped in SSH, with the result discarded:

1. **Discovery** must not SSH to a host that *is* this machine (on a host
   without self-SSH configured that hides every local workload).
2. **Teardown** must dispatch the same way, verify the containers are
   actually gone, and report a truthful count.
3. **A failed teardown is never success**: job metadata stays, the count
   excludes it, and the CLI exits non-zero.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.core.cluster_manager import ClusterGroup, ClusterSoloEntry, ClusterStatusResult
from sparkrun.orchestration.ssh import RemoteResult

SOLO_CID = "sparkrun_aaaabbbbccccdddd_eeeeffff0000"
GROUP_CID = "sparkrun_1111222233334444_55556666"


def _solo_report(hosts=None, **kwargs):
    """Discovery result with one running solo container on localhost."""
    entry = ClusterSoloEntry(
        cluster_id=SOLO_CID,
        host="localhost",
        name=SOLO_CID + "_solo",
        status="Up 5 minutes",
        image="img",
        meta={},
    )
    return ClusterStatusResult(
        groups={},
        solo_entries=[entry],
        errors={},
        idle_hosts=[],
        pending_ops=[],
        total_containers=1,
        host_count=1,
    )


def _two_host_group_report(hosts=None, **kwargs):
    """A two-host cluster: head on localhost, worker on a remote host."""
    group = ClusterGroup(
        cluster_id=GROUP_CID,
        members=[("localhost", "head", "Up 5 minutes", "img"), ("10.0.0.2", "worker0", "Up 5 minutes", "img")],
        meta={"recipe": "test-recipe"},
    )
    return ClusterStatusResult(
        groups={GROUP_CID: group},
        solo_entries=[],
        errors={},
        idle_hosts=[],
        pending_ops=[],
        total_containers=2,
        host_count=2,
    )


@pytest.fixture
def track_metadata(monkeypatch):
    """Record every remove_job_metadata call."""
    removed: list[str] = []
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.remove_job_metadata",
        lambda cid, cache_dir=None, **kw: removed.append(cid),
    )
    return removed


def _mock_teardown(monkeypatch, outcome):
    """Patch the teardown dispatch; *outcome* maps host → RemoteResult."""
    dispatched: list[str] = []

    def mock_run_command_on_host(host, command, ssh_kwargs=None, timeout=None, dry_run=False, quiet=False):
        dispatched.append(host)
        return outcome(host)

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_command_on_host", mock_run_command_on_host)
    return dispatched


# ---------------------------------------------------------------------------
# stop --all: result handling
# ---------------------------------------------------------------------------


def test_stop_all_failure_exits_nonzero_and_preserves_metadata(monkeypatch, track_metadata):
    """A failed stop: exit 1, per-host error, metadata NOT removed, truthful count."""
    monkeypatch.setattr("sparkrun.api.status_report", _solo_report)
    dispatched = _mock_teardown(
        monkeypatch,
        lambda host: RemoteResult(host=host, returncode=255, stdout="", stderr="Host key verification failed."),
    )

    result = CliRunner().invoke(main, ["stop", "--all", "--hosts", "localhost"], catch_exceptions=False)

    assert result.exit_code == 1
    assert dispatched == ["localhost"]
    assert track_metadata == []
    assert "Stopped 0 job(s), 0 container(s) across 0 host(s)." in result.output
    assert "failed to stop containers on localhost" in result.output
    assert "Host key verification failed." in result.output


def test_stop_all_success_cleans_metadata_and_reports(monkeypatch, track_metadata):
    """A successful stop: exit 0, metadata removed, count read from the teardown."""
    monkeypatch.setattr("sparkrun.api.status_report", _solo_report)
    _mock_teardown(
        monkeypatch,
        lambda host: RemoteResult(host=host, returncode=0, stdout="sparkrun_removed=1\n", stderr=""),
    )

    result = CliRunner().invoke(main, ["stop", "--all", "--hosts", "localhost"], catch_exceptions=False)

    assert result.exit_code == 0
    assert track_metadata == [SOLO_CID]
    assert "Stopped 1 job(s), 1 container(s) across 1 host(s)." in result.output


def test_stop_all_partial_failure_keeps_metadata_for_the_whole_job(monkeypatch, track_metadata):
    """One member host fails → the job's metadata stays: it's still running."""
    monkeypatch.setattr("sparkrun.api.status_report", _two_host_group_report)
    _mock_teardown(
        monkeypatch,
        lambda host: (
            RemoteResult(host=host, returncode=0, stdout="sparkrun_removed=1\n", stderr="")
            if host == "localhost"
            else RemoteResult(host=host, returncode=255, stdout="", stderr="unreachable")
        ),
    )

    result = CliRunner().invoke(main, ["stop", "--all", "--hosts", "localhost,10.0.0.2"], catch_exceptions=False)

    assert result.exit_code == 1
    # The head's container was removed, but the job is not "stopped".
    assert track_metadata == []
    assert "Stopped 0 job(s), 1 container(s) across 1 host(s)." in result.output
    assert "failed to stop containers on 10.0.0.2" in result.output


def test_stop_all_discovery_error_exits_nonzero(monkeypatch):
    """A host that errors during discovery must not read as 'nothing to stop'."""

    def erroring_report(hosts=None, **kwargs):
        return ClusterStatusResult(
            groups={},
            solo_entries=[],
            errors={"localhost": "Host key verification failed."},
            idle_hosts=[],
            pending_ops=[],
            total_containers=0,
            host_count=1,
        )

    monkeypatch.setattr("sparkrun.api.status_report", erroring_report)

    result = CliRunner().invoke(main, ["stop", "--all", "--hosts", "localhost"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "could not query localhost" in result.output
    assert "Host key verification failed." in result.output
    assert "No sparkrun containers running." not in result.output


def test_stop_all_dry_run_stops_nothing_and_succeeds(monkeypatch, track_metadata):
    """--dry-run reports the discovered shape and touches no metadata."""
    monkeypatch.setattr("sparkrun.api.status_report", _solo_report)
    _mock_teardown(monkeypatch, lambda host: RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr=""))

    result = CliRunner().invoke(main, ["stop", "--all", "--hosts", "localhost", "--dry-run"], catch_exceptions=False)

    assert result.exit_code == 0
    assert track_metadata == []
    assert "Stopped 1 job(s), 1 container(s) across 1 host(s)." in result.output


# ---------------------------------------------------------------------------
# api.stop_all: the console-free contract library/GUI callers get
# ---------------------------------------------------------------------------


def test_api_stop_all_reports_failures_without_printing(monkeypatch, track_metadata):
    """api.stop_all returns structured results; no console, no sys.exit."""
    import sparkrun.api as api

    _mock_teardown(
        monkeypatch,
        lambda host: RemoteResult(host=host, returncode=1, stdout="", stderr="containers still present: x"),
    )

    result = api.stop_all(["localhost"], discovered=_solo_report())

    assert result.success is False
    assert result.hosts_failed == {"localhost": "containers still present: x"}
    assert result.jobs_stopped == 0
    assert result.containers_removed == 0
    assert track_metadata == []


def test_api_stop_all_nothing_running_is_success(monkeypatch):
    """No containers and no discovery errors → success, no teardown attempted."""
    import sparkrun.api as api

    dispatched = _mock_teardown(monkeypatch, lambda host: RemoteResult(host=host, returncode=0, stdout="", stderr=""))
    empty = ClusterStatusResult(
        groups={},
        solo_entries=[],
        errors={},
        idle_hosts=["localhost"],
        pending_ops=[],
        total_containers=0,
        host_count=1,
    )

    result = api.stop_all(["localhost"], discovered=empty)

    assert result.success is True
    assert result.jobs_stopped == 0
    assert dispatched == []


# ---------------------------------------------------------------------------
# Discovery dispatch
# ---------------------------------------------------------------------------


def test_query_status_dispatches_locally_for_local_host(monkeypatch):
    """Status discovery must not SSH to a local host (no self-SSH required)."""
    from sparkrun.orchestration.executors.docker import DockerExecutor

    def mock_local_script(script, dry_run=False, timeout=None):
        assert "docker ps" in script
        return RemoteResult(host="localhost", returncode=0, stdout="", stderr="")

    def fail_if_sshed(*args, **kwargs):
        raise AssertionError("query_status attempted SSH for a local host")

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_local_script", mock_local_script)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", fail_if_sshed)

    status = DockerExecutor().query_status(["localhost"])
    occ = status.for_host("localhost")
    assert occ is not None, "local host missing from status (dispatch failed)"
    assert occ.workloads == ()


def test_query_status_mixed_local_and_remote_hosts(monkeypatch):
    """A local head + remote worker: each dispatched its own way, both re-keyed."""
    from sparkrun.orchestration.executors.docker import DockerExecutor

    sshed: list[str] = []

    def mock_local_script(script, dry_run=False, timeout=None):
        return RemoteResult(host="localhost", returncode=0, stdout="", stderr="")

    def mock_remote_script(host, script, **kwargs):
        sshed.append(host)
        return RemoteResult(host=host, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_local_script", mock_local_script)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", mock_remote_script)

    status = DockerExecutor().query_status(["localhost", "10.0.0.2"])

    assert sshed == ["10.0.0.2"], "only the remote host should be SSHed"
    assert {h.host for h in status.hosts} == {"localhost", "10.0.0.2"}
    assert status.errors == {}


def test_query_status_local_failure_is_recorded_as_an_error(monkeypatch):
    """A failed *local* dispatch lands in status.errors like a failed SSH."""
    from sparkrun.orchestration.executors.docker import DockerExecutor

    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_local_script",
        lambda script, dry_run=False, timeout=None: RemoteResult(host="localhost", returncode=127, stdout="", stderr="bash: not found"),
    )

    status = DockerExecutor().query_status(["localhost"])

    assert status.for_host("localhost") is None
    assert status.errors == {"localhost": "bash: not found"}


def test_parallel_local_dispatch_is_opt_in(monkeypatch):
    """Without allow_local, a local host still goes over SSH.

    ``api.setup`` probes and meshes SSH credentials — for it SSH-to-self
    is the point, so the local partition must never be implicit.
    """
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    sshed: list[str] = []

    def mock_remote_script(host, script, **kwargs):
        sshed.append(host)
        return RemoteResult(host=host, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", mock_remote_script)
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_local_script",
        lambda *a, **kw: pytest.fail("local dispatch must be opt-in"),
    )

    run_remote_scripts_parallel(["localhost"], "echo hi\n")
    assert sshed == ["localhost"]


def test_run_local_script_honours_timeout():
    """The local branch must not be able to hang a caller that set a timeout."""
    from sparkrun.orchestration.ssh import run_local_script

    result = run_local_script("sleep 5\n", timeout=1)

    assert result.returncode == 124
    assert "timed out" in result.stderr


# ---------------------------------------------------------------------------
# Teardown verification
# ---------------------------------------------------------------------------


def _run_script(script: str, docker_stub: str = "") -> "tuple[int, str, str]":
    import subprocess

    proc = subprocess.run(["bash", "-c", docker_stub + script], capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def test_teardown_script_fails_when_a_container_survives():
    """``docker rm -f ... || true`` always exits 0 — the teardown must not."""
    from sparkrun.orchestration.docker import docker_teardown_script

    # A stub 'docker' that accepts the rm and then still lists the container.
    stub = "docker() { if [ \"$1\" = 'ps' ]; then echo sparkrun_zzz_solo; else return 0; fi; }\n"
    rc, _out, err = _run_script(docker_teardown_script(["sparkrun_zzz_solo"]), stub)

    assert rc == 1
    assert "still present" in err


def test_teardown_script_fails_when_docker_is_unusable():
    """A dead/absent docker must fail the teardown, not read as success."""
    from sparkrun.orchestration.docker import docker_teardown_script

    script = docker_teardown_script(["sparkrun_zzz_solo"]).replace("docker ", "docker-does-not-exist ")
    rc, _out, _err = _run_script(script)

    assert rc == 1


def test_teardown_script_reports_removed_count():
    """The removed count comes from the teardown, not from len(hosts)."""
    from sparkrun.orchestration.docker import docker_teardown_script, parse_teardown_removed

    # Real docker semantics: only 'a' exists, and ``rm -f`` exits 0 for both
    # (see test_teardown_script_does_not_count_containers_that_never_existed).
    # The census must be what distinguishes them.  State goes through a file
    # because the census runs inside ``$(...)`` — a subshell, where a variable
    # assignment would not survive.
    stub = (
        "_state=$(mktemp)\n"
        "docker() {\n"
        '  if [ "$1" = "ps" ]; then [ -s "$_state" ] || echo a; return 0; fi\n'
        '  if [ "$1" = "rm" ]; then echo removed > "$_state"; return 0; fi\n'
        "  return 0\n"
        "}\n"
    )
    rc, out, _err = _run_script(docker_teardown_script(["a", "b"]), stub)

    assert rc == 0
    assert parse_teardown_removed(out) == 1


def test_teardown_script_does_not_count_containers_that_never_existed():
    """``docker rm -f <missing>`` exits 0, so removals must not be the count.

    Regression: ``stop`` feeds every candidate name (solo / head / worker /
    node_N) to every host and counted each successful ``rm``, so a
    single-container job reported four containers removed — and a host holding
    nothing at all reported removals.
    """
    from sparkrun.orchestration.docker import docker_teardown_script, parse_teardown_removed

    # An empty but perfectly healthy docker: ps succeeds listing nothing, and
    # rm -f exits 0 for names that do not exist (verified against Docker 28).
    stub = "docker() { return 0; }\n"
    rc, out, _err = _run_script(docker_teardown_script(["a", "b", "c"]), stub)

    assert rc == 0
    assert parse_teardown_removed(out) == 0


def test_teardown_script_tolerates_containers_that_are_already_gone():
    """Teardown is idempotent: nothing to remove is success, not failure."""
    from sparkrun.orchestration.docker import docker_teardown_script, parse_teardown_removed

    stub = "docker() { if [ \"$1\" = 'ps' ]; then return 0; else return 1; fi; }\n"
    rc, out, _err = _run_script(docker_teardown_script(["gone"]), stub)

    assert rc == 0
    assert parse_teardown_removed(out) == 0


def test_parse_teardown_removed_defaults_to_zero():
    """An absent/garbled marker reports nothing removed, never a guess."""
    from sparkrun.orchestration.docker import parse_teardown_removed

    assert parse_teardown_removed("") == 0
    assert parse_teardown_removed("unrelated output\n") == 0
    assert parse_teardown_removed("sparkrun_removed=notanumber\n") == 0
    assert parse_teardown_removed("noise\nsparkrun_removed=3\n") == 3
