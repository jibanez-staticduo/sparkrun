"""Tests for ``stop --all`` teardown dispatch and result handling.

The stop leg must route through ``run_command_on_host`` (local-vs-SSH
dispatch — a bare ``run_remote_command`` tries to SSH to localhost, which
fails on hosts without self-SSH configured), and a failed stop must never be
reported as success: metadata stays, the count is truthful, and the exit
code is non-zero.
"""

from __future__ import annotations

from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.core.cluster_manager import ClusterSoloEntry, ClusterStatusResult
from sparkrun.orchestration.ssh import RemoteResult


def _one_solo_report(hosts, *, executor=None, cluster=None, ssh_kwargs=None, sctx=None):
    """Discovery result with one running solo container on localhost."""
    entry = ClusterSoloEntry(
        cluster_id="sparkrun_aaaabbbbccccdddd_eeeeffff0000",
        host="localhost",
        name="sparkrun_aaaabbbbccccdddd_eeeeffff0000_solo",
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


def test_stop_all_failure_exits_nonzero_and_preserves_metadata(monkeypatch):
    """A failed stop: exit 1, per-host error, metadata NOT removed, truthful count."""
    removed = []
    dispatched = []

    def mock_run_command_on_host(host, command, ssh_kwargs=None, timeout=None, dry_run=False, quiet=False):
        dispatched.append(host)
        return RemoteResult(host=host, returncode=255, stdout="", stderr="Host key verification failed.")

    monkeypatch.setattr("sparkrun.api.status_report", _one_solo_report)
    monkeypatch.setattr("sparkrun.orchestration.primitives.run_command_on_host", mock_run_command_on_host)
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.remove_job_metadata",
        lambda cid, cache_dir=None: removed.append(cid),
    )

    runner = CliRunner()
    result = runner.invoke(main, ["stop", "--all", "--hosts", "localhost"], catch_exceptions=False)

    assert result.exit_code == 1
    assert dispatched == ["localhost"]
    assert removed == []
    assert "Stopped 0 job(s), 0 container(s) across 0 host(s)." in result.output
    assert "failed to stop containers on localhost" in result.output
    assert "Host key verification failed." in result.output


def test_stop_all_success_cleans_metadata_and_reports(monkeypatch):
    """A successful stop: exit 0, metadata removed, truthful count."""
    removed = []

    def mock_run_command_on_host(host, command, ssh_kwargs=None, timeout=None, dry_run=False, quiet=False):
        return RemoteResult(host=host, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("sparkrun.api.status_report", _one_solo_report)
    monkeypatch.setattr("sparkrun.orchestration.primitives.run_command_on_host", mock_run_command_on_host)
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.remove_job_metadata",
        lambda cid, cache_dir=None: removed.append(cid),
    )

    runner = CliRunner()
    result = runner.invoke(main, ["stop", "--all", "--hosts", "localhost"], catch_exceptions=False)

    assert result.exit_code == 0
    assert removed == ["sparkrun_aaaabbbbccccdddd_eeeeffff0000"]
    assert "Stopped 1 job(s), 1 container(s) across 1 host(s)." in result.output
