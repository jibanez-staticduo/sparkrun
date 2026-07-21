"""Tests for the resource-telemetry seam + the api live-monitoring surface.

(Distinct from ``test_telemetry.py``, which covers anonymous usage analytics.)

Covers:
- TelemetryProvider discovery / scope selection (SAF).
- HostTelemetryProvider backend selection + session snapshot (SSH mocked).
- api.open_telemetry (None when no provider for the scope).
- LiveMonitorSession / api.open_live_monitor frame composition (telemetry + occupancy).
"""

from __future__ import annotations

import time
from unittest import mock

import pytest

import sparkrun.api as api
from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload
from sparkrun.core.monitoring import HostActivity, HostTelemetry, MonitorFrame, MonitorSample
from sparkrun.orchestration.telemetry import get_telemetry_provider, list_telemetry_scopes
from sparkrun.orchestration.telemetry._base import TelemetryProvider, TelemetrySession


# --------------------------------------------------------------------------
# Provider discovery
# --------------------------------------------------------------------------


def test_host_provider_registered_and_resolvable():
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    assert "host" in list_telemetry_scopes(v)
    provider = get_telemetry_provider("host", v)
    assert provider is not None
    assert provider.scope == "host"


def test_unknown_scope_returns_none():
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    # No k8s/modal telemetry provider in-tree — a substrate with no provider
    # yields None (occupancy-only monitoring), never an error.
    assert get_telemetry_provider("k8s", v) is None


def test_base_provider_open_not_implemented():
    with mock.patch.object(TelemetryProvider, "scope", "x"):
        with pytest.raises(NotImplementedError):
            TelemetryProvider().open(["h1"])


# --------------------------------------------------------------------------
# HostTelemetryProvider — backend selection + session (SSH mocked)
# --------------------------------------------------------------------------


class _FakeMonitor:
    """Stand-in for ClusterMonitor: records start/stop, exposes .states."""

    def __init__(self, hosts, ssh_kwargs, interval=2, **kw):
        from sparkrun.core.monitoring import HostMonitorState

        self.hosts = hosts
        self.interval = interval
        self.started = False
        self.stopped = False
        self.states = {h: HostMonitorState(latest=MonitorSample(hostname=h, gpu_util_pct="10")) for h in hosts}

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_host_provider_defaults_to_bash_backend(monkeypatch):
    from sparkrun.orchestration.telemetry.host import HostTelemetryProvider

    created = {}

    def _fake_cm(hosts, ssh_kwargs, interval=2, **kw):
        created["bash"] = _FakeMonitor(hosts, ssh_kwargs, interval)
        return created["bash"]

    monkeypatch.setattr("sparkrun.core.monitoring.ClusterMonitor", _fake_cm)
    session = HostTelemetryProvider().open(["h1"], ssh_kwargs={}, interval=3)
    assert created["bash"].started
    assert session.snapshot()["h1"].sample.gpu_util_pct == "10"
    session.close()
    assert created["bash"].stopped


def test_host_provider_nv_monitor_backend(monkeypatch):
    from sparkrun.orchestration.telemetry.host import HostTelemetryProvider

    created = {}

    def _fake_nv(hosts, ssh_kwargs, interval=2, **kw):
        created["nv"] = _FakeMonitor(hosts, ssh_kwargs, interval)
        return created["nv"]

    monkeypatch.setattr("sparkrun.core.monitoring.NvMonitorClusterMonitor", _fake_nv)
    HostTelemetryProvider().open(["h1"], ssh_kwargs={}, backend="nv-monitor")
    assert "nv" in created and created["nv"].started


# --------------------------------------------------------------------------
# api.open_telemetry / live-monitor composition
# --------------------------------------------------------------------------


class _FakeSession(TelemetrySession):
    def __init__(self, snap):
        self._snap = snap
        self.closed = False

    def snapshot(self):
        return self._snap

    def close(self):
        self.closed = True


class _FakeProvider:
    scope = "host"

    def __init__(self, session):
        self._session = session

    def open(self, hosts, **kw):
        return self._session


def test_open_telemetry_none_when_no_provider(monkeypatch):
    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda scope, v=None: None)
    assert api.open_telemetry(["h1"]) is None


def _drive_until(session, pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    frame = session.frame()
    while not pred(frame) and time.monotonic() < deadline:
        time.sleep(0.02)
        frame = session.frame()
    return frame


def test_live_monitor_frame_combines_telemetry_and_occupancy(monkeypatch):
    tel = {"h1": HostTelemetry(host="h1", sample=MonitorSample(hostname="h1", gpu_util_pct="42"))}
    monkeypatch.setattr(
        "sparkrun.orchestration.telemetry.get_telemetry_provider",
        lambda scope, v=None: _FakeProvider(_FakeSession(tel)),
    )
    snap = ClusterStatus(
        hosts=(HostOccupancy(host="h1", workloads=(RunningWorkload(cluster_id="sparkrun_x"),), used_slots=1, free_slots=0),),
        executor="docker",
    )
    # The background poller reads sparkrun.api.status (package attr).
    monkeypatch.setattr("sparkrun.api.status", lambda *a, **k: snap)

    session = api.open_live_monitor(["h1"], interval=1, status_interval=1)
    try:
        frame = _drive_until(session, lambda f: f.for_host("h1").used_slots == 1)
    finally:
        session.close()

    assert isinstance(frame, MonitorFrame)
    act = frame.for_host("h1")
    assert isinstance(act, HostActivity)
    assert act.telemetry.gpu_util_pct == "42"  # telemetry axis
    assert [w.cluster_id for w in act.workloads] == ["sparkrun_x"]  # occupancy axis
    assert act.used_slots == 1


def test_live_monitor_occupancy_only_when_no_telemetry(monkeypatch):
    """A substrate with no telemetry provider still yields occupancy frames."""
    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda scope, v=None: None)
    snap = ClusterStatus(hosts=(HostOccupancy(host="h1", used_slots=0, free_slots=4),), executor="docker")
    monkeypatch.setattr("sparkrun.api.status", lambda *a, **k: snap)

    session = api.open_live_monitor(["h1"], interval=1, status_interval=1)
    try:
        frame = _drive_until(session, lambda f: f.for_host("h1").free_slots == 4)
    finally:
        session.close()

    act = frame.for_host("h1")
    assert act.telemetry is None  # no telemetry provider
    assert act.free_slots == 4  # occupancy still present
