"""Unit tests for sparkrun.orchestration.health module."""

import shutil
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from sparkrun.orchestration.health import (
    listen_probe_cmd,
    proc_listen_probe_cmd,
    wait_for_port,
)
from sparkrun.orchestration.ssh import RemoteResult


def _result(success: bool) -> RemoteResult:
    return RemoteResult(host="testhost", returncode=0 if success else 1, stdout="", stderr="")


def test_wait_for_port_uses_listen_state_check_not_nc():
    """Port probe must not open a TCP connection.

    `nc -z localhost <port>` opens a real connection, which consumes
    one-shot rendezvous accepts (Atlas NCCL bootstrap hands its unique
    NCCL ID to the probe instead of the real worker). The readiness
    probe must check LISTEN state via `ss` and be side-effect free.
    """
    mock_run = MagicMock(return_value=_result(True))
    with patch("sparkrun.orchestration.primitives.run_command_on_host", mock_run):
        assert wait_for_port("testhost", 25000, max_retries=1, retry_interval=0) is True

    cmd = mock_run.call_args.args[1]
    assert "nc -z" not in cmd
    assert "ss -tln" in cmd
    assert "25000" in cmd


def test_wait_for_port_returns_true_when_listening():
    """A successful ss check reports the port ready."""
    mock_run = MagicMock(return_value=_result(True))
    with patch("sparkrun.orchestration.primitives.run_command_on_host", mock_run):
        assert wait_for_port("testhost", 8000, max_retries=3, retry_interval=0) is True
    assert mock_run.call_count == 1


def test_wait_for_port_returns_false_after_retries():
    """A port that never listens reports not ready after max_retries."""
    mock_run = MagicMock(return_value=_result(False))
    with patch("sparkrun.orchestration.primitives.run_command_on_host", mock_run):
        assert wait_for_port("testhost", 8000, max_retries=3, retry_interval=0) is False
    assert mock_run.call_count == 3


def test_wait_for_port_aborts_when_container_exits():
    """If the container stops while polling, abort early instead of retrying."""
    mock_run = MagicMock(return_value=_result(False))
    with (
        patch("sparkrun.orchestration.primitives.run_command_on_host", mock_run),
        patch("sparkrun.orchestration.health.is_container_running", return_value=False),
    ):
        assert wait_for_port("testhost", 8000, max_retries=5, retry_interval=0, container_name="srv") is False
    # Attempt 1 probes the port; attempt 2 aborts on the dead container before probing.
    assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# Probe command construction
# ---------------------------------------------------------------------------


def test_listen_probe_matches_port_exactly():
    """`ss` does the port matching, so a probe can't hit a longer port."""
    cmd = listen_probe_cmd(8000)
    assert 'ss -tln "sport = :8000"' in cmd
    assert "18000" not in cmd


def test_proc_probe_reads_both_socket_tables():
    """IPv6 sockets must be covered, and via `cat` so a missing tcp6 is benign."""
    cmd = proc_listen_probe_cmd(8000)
    assert "/proc/net/tcp /proc/net/tcp6" in cmd
    # As awk operands, an absent /proc/net/tcp6 is a fatal error that skips END.
    assert cmd.startswith("cat ")


# ---------------------------------------------------------------------------
# /proc fallback executed against fixture socket tables
#
# The `ss` half is exercised by the live cluster; the fallback is where a
# hand-written match is easy to get subtly wrong, so run it for real.
# ---------------------------------------------------------------------------

_PROC_HEADER = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
_TAIL = "00000000:00000000 00:00000000 00000000     0        0 4022183 1 0000000000000000 100 0 0 10 0"

# Port 8000 == 0x1F40, 25000 == 0x61A8.
_LISTEN_8000 = "   0: 00000000:1F40 00000000:0000 0A " + _TAIL
_LISTEN_8000_V6 = "   0: 00000000000000000000000000000000:1F40 00000000000000000000000000000000:0000 0A " + _TAIL
_TIME_WAIT_LOCAL_8000 = "  20: 0B0B180A:1F40 2F201268:C5E2 06 " + _TAIL
_ESTABLISHED_REMOTE_8000 = "  21: 0B0B180A:859E 2F201268:1F40 01 " + _TAIL


def _run_proc_probe(port: int, rows: list[str], tmp_path) -> int:
    if not shutil.which("awk"):
        pytest.skip("awk not available")
    table = tmp_path / "proc_net_tcp"
    table.write_text(_PROC_HEADER + "".join(r + "\n" for r in rows))
    cmd = proc_listen_probe_cmd(port, proc_paths=str(table))
    return subprocess.run(["bash", "-c", cmd], capture_output=True).returncode


def test_proc_probe_matches_listening_socket(tmp_path):
    assert _run_proc_probe(8000, [_LISTEN_8000], tmp_path) == 0


def test_proc_probe_matches_ipv6_listening_socket(tmp_path):
    assert _run_proc_probe(8000, [_LISTEN_8000_V6], tmp_path) == 0


def test_proc_probe_ignores_time_wait_from_a_stopped_workload(tmp_path):
    """A relaunch inside the TIME_WAIT window must not report ready.

    Stopping a workload leaves ~60s of TIME_WAIT rows on the serve port.
    Matching those makes `wait_for_port` return before the new server has
    bound, and `wait_for_healthy` then reads the still-refused connections
    as "the server died" and fails a launch that was merely loading.
    """
    assert _run_proc_probe(8000, [_TIME_WAIT_LOCAL_8000], tmp_path) != 0


def test_proc_probe_ignores_outbound_connection_to_same_port(tmp_path):
    """`rem_address` is someone else's listener, not ours."""
    assert _run_proc_probe(8000, [_ESTABLISHED_REMOTE_8000], tmp_path) != 0


def test_proc_probe_ignores_unrelated_ports(tmp_path):
    assert _run_proc_probe(25000, [_LISTEN_8000, _TIME_WAIT_LOCAL_8000], tmp_path) != 0


def test_proc_probe_reports_not_listening_on_empty_table(tmp_path):
    assert _run_proc_probe(8000, [], tmp_path) != 0


def test_proc_probe_reports_not_listening_when_table_is_absent(tmp_path):
    """A host without /proc/net/tcp answers "no", not "yes"."""
    if not shutil.which("awk"):
        pytest.skip("awk not available")
    cmd = proc_listen_probe_cmd(8000, proc_paths=str(tmp_path / "nope"))
    assert subprocess.run(["bash", "-c", cmd], capture_output=True).returncode != 0


def test_full_probe_falls_through_to_proc_when_ss_missing(tmp_path):
    """With `ss` absent the composed command still resolves via /proc."""
    if not shutil.which("awk"):
        pytest.skip("awk not available")
    table = tmp_path / "proc_net_tcp"
    table.write_text(_PROC_HEADER + _LISTEN_8000 + "\n")
    cmd = listen_probe_cmd(8000, proc_paths=str(table))
    # Empty PATH additions can't hide `ss` portably; shadow it with a stub
    # that always fails, which is how a missing/too-old `ss` behaves here.
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "ss").write_text("#!/bin/sh\nexit 127\n")
    (stub / "ss").chmod(0o755)
    env = {"PATH": "%s:/usr/bin:/bin" % stub}
    assert subprocess.run(["bash", "-c", cmd], capture_output=True, env=env).returncode == 0
