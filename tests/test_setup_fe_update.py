"""CLI tests for `sparkrun setup fe-system-update`.

Sudo/SSH are mocked.  The assertions are about *scheduling* — that hosts run a
step concurrently, that steps stay barriered, and that the control node (this
machine, however it is spelled in the host list) is updated and rebooted last,
since its reboot takes the CLI down with it.
"""

from __future__ import annotations

import threading

from unittest import mock

from click.testing import CliRunner

from sparkrun.cli._setup._fe_update import _FE_UPDATE_STEPS, setup_fe_system_update
from sparkrun.orchestration.ssh import RemoteResult

_STEP_CMDS = [cmd for _desc, cmd in _FE_UPDATE_STEPS]
_REBOOT = "reboot"


def _kind(script: str) -> str:
    """Label a script by the step it belongs to (or ``reboot``).

    Matched against the step list first, not by searching for "reboot" — the
    firmware step is ``fwupdmgr upgrade -y --no-reboot-check``.
    """
    if script in _STEP_CMDS:
        return "step%d" % _STEP_CMDS.index(script)
    assert "reboot" in script, "unrecognized script: %r" % script
    return _REBOOT


def _run(hosts, *, local_hosts=(), fail=(), password=None, input="y\ny\n", barrier=None):
    """Invoke the command with sudo/SSH mocked, recording the dispatch order.

    *local_hosts* are the labels :func:`is_local_host` should accept — i.e. the
    ones that name this machine.  *fail* names hosts whose every step fails.
    *barrier*, when given, is awaited inside each call so a serial
    implementation cannot satisfy it.
    """
    events: list[tuple[str, str]] = []
    lock = threading.Lock()
    broken: list[str] = []

    def fake_sudo(host, script, pw, ssh_kwargs=None, timeout=None, dry_run=False):
        kind = _kind(script)
        if barrier is not None and kind != _REBOOT:
            try:
                barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                broken.append("%s/%s" % (host, kind))
        with lock:
            events.append((kind, host))
            assert pw == password, "password %r reached the dispatch" % (pw,)
        rc = 1 if host in fail else 0
        return RemoteResult(host=host, returncode=rc, stdout="done", stderr="boom" if rc else "")

    with (
        mock.patch(
            "sparkrun.cli._setup._fe_update._resolve_setup_context",
            return_value=(list(hosts), "me", {}),
        ),
        mock.patch("sparkrun.cli._setup._sudo.ensure_sudo_password", return_value=(password, None)),
        mock.patch("sparkrun.utils.is_local_host", side_effect=lambda h: h in local_hosts),
        mock.patch("sparkrun.orchestration.sudo.run_sudo_script_on_host", fake_sudo),
    ):
        result = CliRunner().invoke(setup_fe_system_update, ["--hosts", ",".join(hosts)], input=input)
    return result, events, broken


def _hosts_for(events, kind):
    return [h for k, h in events if k == kind]


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------


def test_hosts_run_a_step_concurrently():
    """All hosts must be inside the same step at once.

    The barrier only trips if three calls are in flight together, so a serial
    loop — the previous behavior — cannot pass this.
    """
    hosts = ["h1", "h2", "h3"]
    barrier = threading.Barrier(len(hosts))

    r, events, broken = _run(hosts, barrier=barrier)

    assert r.exit_code == 0, r.output
    assert not broken, "hosts did not reach the step together: %s" % broken
    assert len(events) == len(hosts) * (len(_FE_UPDATE_STEPS) + 1)


def test_steps_stay_barriered_across_hosts():
    """Every host finishes step N before any host starts step N+1.

    Lockstep is what keeps a failure interpretable: every host sits at the
    same point, and a step that fails everywhere aborts the run before the
    next one lands.
    """
    r, events, _ = _run(["h1", "h2", "h3"])
    assert r.exit_code == 0, r.output

    order = [k for k, _h in events]
    for idx in range(len(_FE_UPDATE_STEPS)):
        step = "step%d" % idx
        last = len(order) - 1 - order[::-1].index(step)
        later = order[last + 1 :]
        assert step not in later
        # ...and nothing from a later step ran before this one finished.
        assert all(k in (step, _REBOOT) or int(k[4:]) > idx for k in later if k != _REBOOT)


# ---------------------------------------------------------------------------
# Control node goes last
# ---------------------------------------------------------------------------


def test_control_node_is_updated_after_every_remote_host():
    """This machine's steps all follow the cluster's — its reboot ends the run."""
    r, events, _ = _run(["h1", "h2", "me-box"], local_hosts=("me-box",))
    assert r.exit_code == 0, r.output

    first_local = min(i for i, (_k, h) in enumerate(events) if h == "me-box")
    last_remote_update = max(i for i, (k, h) in enumerate(events) if h != "me-box" and k != _REBOOT)
    assert first_local > last_remote_update


def test_control_node_reboots_last():
    r, events, _ = _run(["h1", "h2", "me-box"], local_hosts=("me-box",))
    assert r.exit_code == 0, r.output

    reboots = _hosts_for(events, _REBOOT)
    assert reboots[-1] == "me-box"
    assert sorted(reboots[:-1]) == ["h1", "h2"]


def test_control_node_aliases_collapse_to_one_target():
    """`spark-01` and its own LAN IP are one machine, not two.

    The interactive "All" path prepends ``socket.gethostname()`` to a cluster
    list that already carries this machine by IP.  A string compare let both
    through, so the control node was updated twice and raced itself on reboot.
    """
    r, events, _ = _run(
        ["spark-01", "192.168.1.41", "192.168.1.42"],
        local_hosts=("spark-01", "192.168.1.41"),
    )
    assert r.exit_code == 0, r.output

    for idx in range(len(_FE_UPDATE_STEPS)):
        touched = _hosts_for(events, "step%d" % idx)
        assert sorted(touched) == ["192.168.1.42", "spark-01"]

    assert _hosts_for(events, _REBOOT) == ["192.168.1.42", "spark-01"]
    assert "192.168.1.41" not in {h for _k, h in events}
    assert "also listed as 192.168.1.41" in r.output


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_failed_host_is_dropped_from_later_steps_and_not_rebooted():
    r, events, _ = _run(["h1", "h2"], fail=("h2",))

    assert r.exit_code == 1, r.output
    assert _hosts_for(events, "step0") == ["h1", "h2"] or _hosts_for(events, "step0") == ["h2", "h1"]
    for idx in range(1, len(_FE_UPDATE_STEPS)):
        assert _hosts_for(events, "step%d" % idx) == ["h1"]
    assert _hosts_for(events, _REBOOT) == ["h1"]
    assert "1 updated, 1 failed (h2)" in r.output


def test_control_node_failure_does_not_block_remote_reboots():
    """A broken control node must not strand the hosts that updated fine."""
    r, events, _ = _run(["h1", "me-box"], local_hosts=("me-box",), fail=("me-box",))

    assert r.exit_code == 1, r.output
    assert _hosts_for(events, _REBOOT) == ["h1"]


def test_nopasswd_password_threads_through_unchanged():
    """A NOPASSWD cluster resolves to None and it must reach dispatch as None."""
    r, events, _ = _run(["h1", "h2"], password=None)
    assert r.exit_code == 0, r.output  # the assertion lives in the fake


def test_password_threads_through_unchanged():
    r, events, _ = _run(["h1", "h2"], password="hunter2")
    assert r.exit_code == 0, r.output


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_reboots_nothing_and_names_the_order():
    hosts = ["h1", "me-box"]
    with (
        mock.patch(
            "sparkrun.cli._setup._fe_update._resolve_setup_context",
            return_value=(hosts, "me", {}),
        ),
        mock.patch("sparkrun.utils.is_local_host", side_effect=lambda h: h == "me-box"),
    ):
        r = CliRunner().invoke(setup_fe_system_update, ["--hosts", "h1,me-box", "--dry-run"])

    assert r.exit_code == 0, r.output
    assert "me-box is this machine" in r.output
    assert "updated and rebooted last" in r.output
    assert "[dry-run] Would reboot: h1, me-box" in r.output
