"""Tests for ClusterMonitor's reconnect watchdog.

Covers the two deadlines the watchdog enforces:

- **staleness** — a host that delivered samples and then went quiet
  (``last_updated`` ages past ``interval * 5``).
- **first sample** — a host that has *never* delivered a sample since its
  connect attempt (``last_updated is None``, so nothing can go stale).  Without
  this second deadline a boot race, a half-open TCP connection, or an SSH that
  hangs before its first line wedges that host for the life of the process —
  ``cluster monitor --json`` then streams ``sample: null`` for it forever
  (spark-arena/sparkrun#236).

The watchdog is driven synchronously here: the module's ``time`` is swapped for
a fake whose ``sleep`` advances a controllable clock and stops the loop after a
fixed number of passes, so no test sleeps for real.
"""

from __future__ import annotations

import types
from unittest import mock

import pytest

from sparkrun.core import monitoring
from sparkrun.core.monitoring import FIRST_SAMPLE_TIMEOUT, ClusterMonitor, MonitorSample


class _Clock:
    """Controllable monotonic clock."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now


def _drive_watchdog(monitor, monkeypatch, *, clock, advance_per_tick, passes=1):
    """Run ``monitor._watchdog`` for exactly *passes* iterations of its body.

    Each simulated sleep advances *clock* by *advance_per_tick* seconds.  On the
    tick after the last requested pass the monitor is stopped, so the loop's
    ``if not self._started: break`` ends it without executing the body again.
    """
    ticks = {"n": 0}

    def fake_sleep(_seconds):
        ticks["n"] += 1
        clock.now += advance_per_tick
        if ticks["n"] > passes:
            monitor._started = False

    monkeypatch.setattr(
        monitoring,
        "time",
        types.SimpleNamespace(monotonic=clock.monotonic, sleep=fake_sleep),
    )
    monitor._started = True
    monitor._watchdog()


@pytest.fixture
def monitor_with_recorded_starts(monkeypatch):
    """A ClusterMonitor whose ``_start_host`` records calls instead of SSHing."""

    def _make(hosts=("h1",), interval=2):
        mon = ClusterMonitor(list(hosts), {}, interval=interval)
        started: list[str] = []

        def fake_start(host: str) -> None:
            started.append(host)
            # Mirror the real _start_host's stamp so reconnect loops behave.
            mon.states[host].connect_started = monitoring.time.monotonic()

        monkeypatch.setattr(mon, "_start_host", fake_start)
        return mon, started

    return _make


# --------------------------------------------------------------------------
# First-sample deadline — the #236 regression
# --------------------------------------------------------------------------


def test_host_with_no_first_sample_is_reconnected(monitor_with_recorded_starts, monkeypatch):
    """A host that never produces a sample is retried, not skipped forever."""
    clock = _Clock()
    mon, started = monitor_with_recorded_starts()
    state = mon.states["h1"]
    state.connect_started = clock.now
    state.last_updated = None  # never parsed a sample
    state.error = None  # hung rather than failed — no error to see either

    _drive_watchdog(mon, monkeypatch, clock=clock, advance_per_tick=FIRST_SAMPLE_TIMEOUT + 1)

    assert started == ["h1"]
    assert "reconnecting" in state.error


def test_host_within_first_sample_timeout_is_left_alone(monitor_with_recorded_starts, monkeypatch):
    """A slow-but-progressing SSH handshake is not thrashed by the watchdog."""
    clock = _Clock()
    mon, started = monitor_with_recorded_starts()
    mon.states["h1"].connect_started = clock.now

    # Five ticks of 2s = 10s elapsed, well inside the 30s floor.
    _drive_watchdog(mon, monkeypatch, clock=clock, advance_per_tick=2, passes=5)

    assert started == []
    assert mon.states["h1"].error is None


def test_first_sample_timeout_scales_with_long_intervals(monitor_with_recorded_starts, monkeypatch):
    """With interval=20 the stale threshold (100s) outranks the 30s floor."""
    clock = _Clock()
    mon, started = monitor_with_recorded_starts(interval=20)
    mon.states["h1"].connect_started = clock.now

    # 40s elapsed: past the floor, but inside this monitor's 100s threshold.
    _drive_watchdog(mon, monkeypatch, clock=clock, advance_per_tick=40)
    assert started == []

    # Another 80s (120s total) clears the threshold.
    _drive_watchdog(mon, monkeypatch, clock=clock, advance_per_tick=80)
    assert started == ["h1"]


def test_host_never_started_is_skipped(monitor_with_recorded_starts, monkeypatch):
    """No connect attempt (connect_started None) means nothing to time out."""
    clock = _Clock()
    mon, started = monitor_with_recorded_starts()
    mon.states["h1"].connect_started = None
    mon.states["h1"].last_updated = None

    _drive_watchdog(mon, monkeypatch, clock=clock, advance_per_tick=FIRST_SAMPLE_TIMEOUT * 10)

    assert started == []


def test_still_silent_after_reconnect_retries_again(monitor_with_recorded_starts, monkeypatch):
    """A host down at monitor start converges once it comes back, not never.

    The reconnect re-arms ``connect_started``, so each subsequent timeout fires
    another attempt instead of wedging after the first one.
    """
    clock = _Clock()
    mon, started = monitor_with_recorded_starts()
    mon.states["h1"].connect_started = clock.now

    _drive_watchdog(mon, monkeypatch, clock=clock, advance_per_tick=FIRST_SAMPLE_TIMEOUT + 1, passes=3)

    assert started == ["h1", "h1", "h1"]


def test_reconnect_of_sampleless_host_leaves_last_updated_none(monitor_with_recorded_starts, monkeypatch):
    """Invariant: ``last_updated`` is set only once a real sample has landed.

    Stamping it on a host that never sampled would silently move it onto the
    staleness path and fabricate a "last seen" time it never had.
    """
    clock = _Clock()
    mon, _started = monitor_with_recorded_starts()
    mon.states["h1"].connect_started = clock.now
    monkeypatch.setattr(
        monitoring,
        "time",
        types.SimpleNamespace(monotonic=clock.monotonic, sleep=lambda _s: None),
    )

    mon._reconnect_host("h1", reason="no data since connect")

    assert mon.states["h1"].last_updated is None
    assert mon.states["h1"].latest is None
    assert mon.states["h1"].error == "no data since connect — reconnecting"


# --------------------------------------------------------------------------
# Staleness deadline — pre-existing behavior, guarded against regression
# --------------------------------------------------------------------------


def test_stale_host_with_data_still_reconnects(monitor_with_recorded_starts, monkeypatch):
    """The original staleness path is unchanged, and re-arms last_updated."""
    clock = _Clock()
    mon, started = monitor_with_recorded_starts()
    state = mon.states["h1"]
    state.latest = MonitorSample(hostname="h1")
    connected_at = clock.now
    state.last_updated = connected_at

    # interval=2 → threshold 10s; advance 12s.
    _drive_watchdog(mon, monkeypatch, clock=clock, advance_per_tick=12)

    assert started == ["h1"]
    assert state.error == "stale data — reconnecting"
    # Re-armed to the reconnect instant so the host stays on the staleness path.
    assert state.last_updated == connected_at + 12


def test_fresh_host_with_data_is_not_reconnected(monitor_with_recorded_starts, monkeypatch):
    clock = _Clock()
    mon, started = monitor_with_recorded_starts()
    state = mon.states["h1"]
    state.latest = MonitorSample(hostname="h1")
    state.last_updated = clock.now

    _drive_watchdog(mon, monkeypatch, clock=clock, advance_per_tick=2, passes=3)

    assert started == []


# --------------------------------------------------------------------------
# Loss and recovery of a host that was previously healthy
# --------------------------------------------------------------------------


def test_disconnected_host_keeps_retrying_while_down(monitor_with_recorded_starts, monkeypatch):
    """A node that stays down is retried every cycle, not just once.

    One reconnect attempt would strand the host if the node happened to still be
    unreachable at that instant; the monitor must keep trying so it recovers
    whenever the node comes back.
    """
    clock = _Clock()
    mon, started = monitor_with_recorded_starts()
    state = mon.states["h1"]
    state.latest = MonitorSample(hostname="h1")
    state.last_updated = clock.now

    # interval=2 → 10s threshold; every 12s tick is one more failed retry.
    _drive_watchdog(mon, monkeypatch, clock=clock, advance_per_tick=12, passes=4)

    assert started == ["h1", "h1", "h1", "h1"]


def test_reconnect_clears_the_last_sample(monitor_with_recorded_starts, monkeypatch):
    """The stale sample is dropped, so an outage can't read as live telemetry.

    Keeping it would leave ``--json`` emitting plausible frozen numbers under
    ``sample`` for the whole outage, visible as stale only to a consumer that
    also checks ``error``.
    """
    clock = _Clock()
    mon, _started = monitor_with_recorded_starts()
    state = mon.states["h1"]
    state.latest = MonitorSample(hostname="h1", cpu_usage_pct="42")
    state.last_updated = clock.now

    _drive_watchdog(mon, monkeypatch, clock=clock, advance_per_tick=12)

    assert state.latest is None
    assert state.error == "stale data — reconnecting"


def test_cleared_sample_keeps_host_on_the_fast_staleness_cadence(monitor_with_recorded_starts, monkeypatch):
    """Clearing ``latest`` must not demote the host to the first-sample deadline.

    ``last_updated`` — not ``latest`` — is what records "this host has had data".
    Keying the re-arm off the sample would push a downed-but-known host onto the
    30s first-sample timer instead of the 10s staleness one, slowing recovery.
    """
    clock = _Clock()
    mon, started = monitor_with_recorded_starts()
    state = mon.states["h1"]
    state.latest = MonitorSample(hostname="h1")
    state.last_updated = clock.now

    _drive_watchdog(mon, monkeypatch, clock=clock, advance_per_tick=12)

    assert state.latest is None  # sample dropped ...
    assert state.last_updated is not None  # ... but still on the staleness path
    assert len(started) == 1

    # A second 12s tick trips the 10s staleness threshold again, not the 30s one.
    _drive_watchdog(mon, monkeypatch, clock=clock, advance_per_tick=12)
    assert len(started) == 2


def test_recovered_host_restores_sample_and_clears_error():
    """Once the node returns, a parsed line makes the host healthy again."""
    mon = ClusterMonitor(["h1"], {}, interval=2)
    state = mon.states["h1"]
    state.error = "stale data — reconnecting"
    state.latest = None

    line = ",".join("42" for _ in monitoring.MONITOR_COLUMNS)
    proc = mock.MagicMock()
    proc.stdout = iter([line + "\n"])
    proc.poll.return_value = 0

    mon._reader("h1", proc)

    assert state.latest is not None
    assert state.error is None
    assert state.last_updated is not None


# --------------------------------------------------------------------------
# _start_host stamps the connect time
# --------------------------------------------------------------------------


def test_start_host_stamps_connect_started(monkeypatch):
    mon = ClusterMonitor(["h1"], {}, interval=2)
    mon._script = "echo hi"

    proc = mock.MagicMock()
    proc.stdin = mock.MagicMock()
    with mock.patch("subprocess.Popen", return_value=proc), mock.patch("threading.Thread"):
        mon._start_host("h1")

    assert mon.states["h1"].connect_started is not None


def test_start_host_stamps_connect_started_even_when_spawn_fails(monkeypatch):
    """A host whose SSH can't even launch must still be retried later."""
    mon = ClusterMonitor(["h1"], {}, interval=2)
    mon._script = "echo hi"

    with mock.patch("subprocess.Popen", side_effect=OSError("no ssh binary")):
        mon._start_host("h1")

    assert mon.states["h1"].connect_started is not None
    assert "SSH failed" in mon.states["h1"].error
