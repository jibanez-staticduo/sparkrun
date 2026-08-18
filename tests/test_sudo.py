"""Tests for sudo_user validation in run_indirect_sudo_script."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from sparkrun.orchestration.sudo import run_indirect_sudo_script, run_sudo_script_on_host
from sparkrun.orchestration.ssh import RemoteResult


@patch("sparkrun.orchestration.ssh._run_subprocess")
def test_valid_username_passes(mock_run):
    """A well-formed Unix username proceeds to SSH execution."""
    mock_run.return_value = RemoteResult(host="1.2.3.4", returncode=0, stdout="ok", stderr="")
    res = run_indirect_sudo_script(
        host="1.2.3.4",
        script="echo hello",
        sudo_user="drew",
        sudo_password="secret",
        ssh_kwargs={"ssh_user": "bob"},
    )
    assert res.success
    mock_run.assert_called_once()


def test_malicious_username_raises():
    """A username with injection payload is rejected before any SSH call."""
    with pytest.raises(ValueError, match="Invalid username"):
        run_indirect_sudo_script(
            host="1.2.3.4",
            script="echo hello",
            sudo_user="evil';os.system('rm -rf ~');('",
            sudo_password="secret",
        )


def test_username_with_semicolon_raises():
    """A username containing a semicolon is rejected."""
    with pytest.raises(ValueError, match="Invalid username"):
        run_indirect_sudo_script(
            host="1.2.3.4",
            script="true",
            sudo_user="user;bad",
            sudo_password="pw",
        )


def test_empty_username_raises():
    """An empty username is rejected."""
    with pytest.raises(ValueError, match="Invalid username"):
        run_indirect_sudo_script(
            host="1.2.3.4",
            script="true",
            sudo_user="",
            sudo_password="pw",
        )


@patch("sparkrun.orchestration.ssh._run_subprocess")
def test_dry_run_skips_ssh_but_still_validates(mock_run):
    """dry_run=True skips SSH execution but validation still runs first."""
    # Valid username: no SSH, returns dry-run result
    res = run_indirect_sudo_script(
        host="1.2.3.4",
        script="echo hello",
        sudo_user="admin",
        sudo_password="pw",
        dry_run=True,
    )
    assert res.returncode == 0
    assert "[dry-run]" in res.stdout
    mock_run.assert_not_called()

    # Invalid username: should still raise even in dry_run mode
    with pytest.raises(ValueError, match="Invalid username"):
        run_indirect_sudo_script(
            host="1.2.3.4",
            script="true",
            sudo_user="evil';bad",
            sudo_password="pw",
            dry_run=True,
        )


# ---------------------------------------------------------------------------
# NOPASSWD (password=None) handling
# ---------------------------------------------------------------------------


@patch("sparkrun.orchestration.ssh._run_subprocess")
def test_remote_host_accepts_none_password(mock_run):
    """A remote host with password=None runs non-interactively instead of crashing.

    ``ensure_sudo_password`` returns None once every host answers ``sudo -n
    true``, so None reaches this path on a fully NOPASSWD cluster.  The local
    dispatch has always handled it (``sudo -n``); the remote one used to
    concatenate it onto the script and raise TypeError.
    """
    mock_run.return_value = RemoteResult(host="192.168.1.42", returncode=0, stdout="ok", stderr="")

    res = run_sudo_script_on_host(
        "192.168.1.42",
        "apt update",
        None,
        ssh_kwargs={"ssh_user": "bob"},
    )

    assert res.success
    assert res.host == "192.168.1.42"
    cmd = mock_run.call_args[0][0]
    assert cmd[-4:] == ["sudo", "-n", "bash", "-s"]


@patch("sparkrun.orchestration.sudo.subprocess.run")
def test_local_host_accepts_none_password(mock_run):
    """The local dispatch keeps using ``sudo -n`` for password=None."""
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "ok"
    mock_run.return_value.stderr = ""

    res = run_sudo_script_on_host("localhost", "apt update", None)

    assert res.success
    assert res.host == "localhost"
    assert mock_run.call_args[0][0] == ["sudo", "-n", "bash", "-s"]
    assert mock_run.call_args[1]["input"] == "apt update"
