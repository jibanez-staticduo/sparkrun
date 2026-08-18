"""Unit tests for sparkrun.orchestration.health module."""

from unittest.mock import MagicMock, patch

from sparkrun.orchestration.health import wait_for_port
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
        assert wait_for_port(
            "testhost", 8000, max_retries=5, retry_interval=0, container_name="srv"
        ) is False
    # Attempt 1 probes the port; attempt 2 aborts on the dead container before probing.
    assert mock_run.call_count == 1
